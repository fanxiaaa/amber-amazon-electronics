"""Portable Amazon model construction and batched inputs.

The archived numbers predate this entry point. See docs/REVIEW.md.
"""
import contextlib
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from amber.amazon_dmin import AmazonDMIN
from amber.amazon_kd import AuxiliaryKD
from amber.amazon_transfer import LoopOnlyVM
from amber.hstu_transfer import HSTURanker
from amber.hstu_late import LateHistoryHSTU

ARMS = ('baseline', 'kd', 'loop', 'kd_loop', 'amber_tokens', 'kd_amber',
        'hstu', 'hstu_amber_late', 'hstu_raw_late')
KD_ARMS = ('kd', 'kd_loop', 'kd_amber')


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


class TokenDMIN(LoopOnlyVM):
    def forward(self, batch):
        tokens = self.token_projection(batch['loop_history'])
        valid = torch.arange(tokens.shape[1], device=tokens.device)[None] < batch['loop_length'][:, None]
        return super().forward({**batch, 'loop_history': tokens.masked_fill(~valid[..., None], 0)})


@contextlib.contextmanager
def branch_rng(seed, device):
    # Extra branches must not advance the baseline's dropout RNG stream.
    cpu = torch.get_rng_state()
    gpu = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if gpu is not None:
            torch.cuda.set_rng_state(gpu, device)


def build_model(arm, sizes, cfg, seed, device='cpu', positive_rate=.5):
    if arm not in ARMS:
        raise ValueError(arm)
    device = torch.device(device)
    torch.manual_seed(seed)
    if arm.startswith('hstu'):
        schema = dict(cat_roles=['item', 'item'], cat_cardinalities=[sizes['item'], sizes['category']], num_roles=[])
        spec = dict(history_length=100, history_item_fields=[0, 1], clicked_only=True,
                    token_dimension=cfg['token_dimension'])
        if arm == 'hstu':
            model = HSTURanker(schema, spec, cfg['hstu'], positive_rate)
        else:
            model = LateHistoryHSTU(schema, spec, cfg['hstu'], positive_rate,
                                    raw_control=arm == 'hstu_raw_late')
        return model.to(device)
    cls = TokenDMIN if arm in ('amber_tokens', 'kd_amber') else LoopOnlyVM if arm in ('loop', 'kd_loop') else AmazonDMIN
    model = cls(sizes, cfg['dmin']).to(device)
    if cls != AmazonDMIN:
        with branch_rng(cfg['autoencoder_seed'], device):
            model.cpu()
            model.add_history_path(cfg)
            if cls == TokenDMIN:
                torch.manual_seed(cfg['projection_seed'])
                model.token_projection = nn.Linear(cfg['token_dimension'], cfg['loop_encoder']['dimension'])
            model.to(device)
    if arm in KD_ARMS:
        model = AuxiliaryKD(model, dict(cfg, auxiliary_seed=cfg['auxiliary_seeds'][str(seed)]))
    return model


def historical_references(arrays, limit=50):
    """Earlier positive stage1/2 events only; never the sibling candidate."""
    n = len(arrays['label'])
    refs = np.full((n, limit), -1, np.int32)
    order = np.lexsort((-arrays['label'], arrays['user_id'], arrays['review_prefix']))
    past = {}
    for row in order:
        if arrays['stage'][row] == 0:
            continue
        previous = past.setdefault(int(arrays['user_id'][row]), [])
        eligible = [j for j in previous if arrays['review_prefix'][j] < arrays['review_prefix'][row]][-limit:]
        refs[row, :len(eligible)] = eligible
        if arrays['stage'][row] in (1, 2) and arrays['label'][row] == 1:
            previous.append(int(row))
    return refs


def align_references(arrays, refs):
    aligned = np.full_like(arrays['history_item'], -1, dtype=np.int32)
    owner, col = np.nonzero(refs >= 0)
    past = refs[owner, col]
    position = arrays['review_prefix'][past] - arrays['review_prefix'][owner] + arrays['length'][owner]
    if not (np.all(arrays['user_id'][past] == arrays['user_id'][owner]) and
            np.all(arrays['review_prefix'][past] < arrays['review_prefix'][owner]) and
            np.all(arrays['label'][past] == 1) and np.all(np.isin(arrays['stage'][past], [1, 2])) and
            np.all((position >= 0) & (position < arrays['length'][owner]))):
        raise ValueError('History references violate the past-only information boundary')
    for field in ('item', 'category'):
        if not np.array_equal(arrays['history_' + field][owner, position], arrays[field][past]):
            raise ValueError('Cached event does not match raw history')
    aligned[owner, position] = past
    return aligned


class AmazonData:
    def __init__(self, root, device='cpu', cache=None):
        self.root = Path(root)
        self.manifest = read(self.root / 'manifest.json')
        if sha256(self.root / 'inputs.npz') != self.manifest['inputs_sha256']:
            raise ValueError('Prepared input digest mismatch')
        with np.load(self.root / 'inputs.npz', allow_pickle=False) as saved:
            self.arrays = dict(saved)
        self.device = torch.device(device)
        self.raw = {k: torch.as_tensor(v, device=self.device) for k, v in self.arrays.items()}
        self.cache = None
        if cache is not None:
            metadata = read(Path(cache).with_suffix('.json'))
            if metadata['inputs_sha256'] != self.manifest['inputs_sha256'] or sha256(cache) != metadata['cache_sha256']:
                raise ValueError('Cache/input digest mismatch')
            with np.load(cache, allow_pickle=False) as saved:
                cached = dict(saved)
            if not np.array_equal(cached['row_id'], self.arrays['row_id']):
                raise ValueError('Cache row alignment mismatch')
            aligned = align_references(self.arrays, cached['refs'])
            self.cache = {k: torch.as_tensor(v, device=self.device) for k, v in cached.items()}
            self.cache['aligned'] = torch.as_tensor(aligned, device=self.device)

    def indices(self, stage):
        return np.flatnonzero(self.arrays['stage'] == stage)

    def batch(self, arm, indices):
        idx = torch.as_tensor(indices, device=self.device)
        b = {k: self.raw[k][idx].long() for k in ('item', 'category', 'history_item', 'history_category', 'length')}
        if arm.startswith('hstu'):
            b.update(cats=torch.stack((b['item'], b['category']), -1),
                     nums=torch.empty((len(idx), 0), device=self.device),
                     history_cats=torch.stack((b['history_item'], b['history_category']), -1),
                     history_nums=torch.empty((len(idx), 100, 0), device=self.device))
        if arm not in ('baseline', 'hstu') and self.cache is None:
            raise ValueError('This arm needs --cache from prepare_features.py')
        if arm in ('loop', 'kd_loop', 'amber_tokens', 'kd_amber'):
            refs = self.cache['refs'][idx].long()
            name = 'loop_latent' if arm in ('loop', 'kd_loop') else 'amber_latent'
            b.update(loop_history=self.cache[name][refs.clamp_min(0)].masked_fill((refs < 0)[..., None], 0),
                     loop_length=(refs >= 0).sum(1))
        if arm in ('hstu_amber_late', 'hstu_raw_late'):
            refs = self.cache['aligned'][idx].long()
            b['amber_valid'] = refs >= 0
            if arm == 'hstu_amber_late':
                b['amber_history'] = self.cache['amber_latent'][refs.clamp_min(0)]
        return b
