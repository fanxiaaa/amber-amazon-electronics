"""Teacher, post-hoc LoopFM compression, and teacher-only AMBER event tokens.

Explicit stages keep all large artifacts in the supplied work directory.
No weights or dataset download is performed by this script.
"""
import argparse
import gc
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from amber.amazon_transfer import DetachedAutoencoder
from amber.amazon_pretrained import PretrainedAMBER
from amber.model import AMBERConfig, EventTokenizer
from amber.experiment import AmazonData, build_model, historical_references, align_references, read, sha256, write
from train import fit, batches


def hidden(model, batch):
    target = torch.cat((model.item(batch['item']), model.category(batch['category'])), -1)
    history = torch.cat((model.item(batch['history_item']), model.category(batch['history_category'])), -1)
    valid, allowed = model.masks(batch['length'], history.shape[1])
    interests = model.extractor(model.refiner(history, allowed), target, allowed, valid)
    return model.dnn.mlp[:3](torch.cat(interests + [target], -1))


def verify_stage(root, name, identity):
    rec = read(root / (name + '.json'))
    if rec['identity'] != identity or rec['model_sha256'] != sha256(root / (name + '.pt')):
        raise ValueError('Upstream stage identity mismatch: ' + name)


def finish_stage(root, name, state, identity):
    torch.save(state, root / (name + '.pt'))
    write(root / (name + '.json'), dict(identity=identity, model_sha256=sha256(root / (name + '.pt'))))


def teacher(data, cfg, root, identity):
    model = build_model('baseline', data.manifest['vocab_sizes'], cfg, cfg['seeds'][0], data.device)
    fit(data, model, cfg, 'baseline', .0005, stage=0)
    finish_stage(root, 'teacher', dict(model=model.state_dict()), identity)


def amber(data, cfg, root, identity, pretrained, joint):
    if pretrained is None:
        raise ValueError('--pretrained must point to the pinned Qwen snapshot')
    for name, key in [('config.json', 'config_sha256'), ('model.safetensors', 'model_safetensors_sha256')]:
        if sha256(pretrained / name) != cfg['pretrained_model'][key]:
            raise ValueError('Pretrained snapshot mismatch: ' + name)
    model = PretrainedAMBER(data.manifest['vocab_sizes'],
                            dict(seed=cfg['upstream']['seed'], pretrained_path=str(pretrained))).to(data.device)
    if joint:
        verify_stage(root, 'alignment', identity)
        model.load_state_dict(torch.load(root / 'alignment.pt', map_location='cpu', weights_only=True)['model'])
    a = cfg['upstream']
    model.set_stage(joint)
    groups = [dict(params=[p for n, p in model.named_parameters() if not n.startswith('llm.')],
                   lr=a['joint_tokenizer_lr'] if joint else a['alignment_lr'])]
    if joint:
        groups.append(dict(params=model.llm.parameters(), lr=a['joint_llm_lr']))
    optimizer = torch.optim.AdamW(groups, weight_decay=a['weight_decay'], foreach=False)
    model.train()
    for idx in batches(data.indices(0), a['batch_size']):
        batch = data.batch('baseline', idx)
        y = data.raw['label'][idx].float()
        optimizer.zero_grad(set_to_none=True)
        for lo in range(0, len(idx), a['microbatch']):
            sl = slice(lo, lo + a['microbatch'])
            loss = F.binary_cross_entropy_with_logits(model({k: v[sl] for k, v in batch.items()}), y[sl], reduction='sum') / len(idx)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
    name = 'joint' if joint else 'alignment'
    finish_stage(root, name, dict(model=model.state_dict()), identity)
    if joint:
        finish_stage(root, 'tokenizer', dict(tokenizer=model.tokenizer.state_dict(), projection=model.projection.state_dict()), identity)


def cache(data, cfg, root, identity):
    verify_stage(root, 'teacher', identity)
    verify_stage(root, 'tokenizer', identity)
    arrays = data.arrays
    teacher_model = build_model('baseline', data.manifest['vocab_sizes'], cfg, cfg['seeds'][0], data.device)
    teacher_model.load_state_dict(torch.load(root / 'teacher.pt', map_location=data.device, weights_only=True)['model'])
    teacher_model.eval().requires_grad_(False)
    torch.manual_seed(cfg['autoencoder_seed'])
    ae = DetachedAutoencoder().to(data.device)
    opt = torch.optim.Adam(ae.parameters(), lr=.0005)
    for idx in batches(data.indices(0), 256):
        with torch.no_grad():
            h = hidden(teacher_model, data.batch('baseline', idx))
        opt.zero_grad(set_to_none=True)
        ae.reconstruction_loss(h).backward()
        torch.nn.utils.clip_grad_norm_(ae.parameters(), 10., error_if_nonfinite=True)
        opt.step()
    finish_stage(root, 'autoencoder', dict(ae=ae.state_dict()), identity)
    n = len(arrays['label'])
    loop_latent = np.zeros((n, 32), np.float32)
    soft = np.full(n, np.nan, np.float32)
    sources = np.flatnonzero(np.isin(arrays['stage'], [1, 2]) & (arrays['label'] == 1))
    ae.eval()
    with torch.inference_mode():
        for idx in batches(sources, 256):
            loop_latent[idx] = ae.encoder(hidden(teacher_model, data.batch('baseline', idx))).cpu().numpy()
        for idx in batches(data.indices(1), 256):
            soft[idx] = teacher_model(data.batch('baseline', idx)).cpu().numpy()
    del teacher_model, ae, opt
    gc.collect()
    if data.device.type == 'cuda':
        torch.cuda.empty_cache()
    sizes = data.manifest['vocab_sizes']
    tokenizer = EventTokenizer(AMBERConfig(categorical_cardinalities=[sizes['item'], sizes['category'], 3],
                                          categorical_roles=['item', 'item', 'outcome']))
    projection = nn.Linear(128, cfg['token_dimension'])
    weights = torch.load(root / 'tokenizer.pt', map_location='cpu', weights_only=True)
    tokenizer.load_state_dict(weights['tokenizer'])
    projection.load_state_dict(weights['projection'])
    tokenizer.to(data.device).eval().requires_grad_(False)
    projection.to(data.device).eval().requires_grad_(False)
    fields = np.stack((arrays['item'][sources], arrays['category'][sources]), -1)
    unique, inv = np.unique(fields, axis=0, return_inverse=True)
    tokens = []
    with torch.inference_mode():
        for lo in range(0, len(unique), 512):
            f = torch.as_tensor(unique[lo:lo + 512], device=data.device)
            cats = torch.cat((f, torch.zeros((len(f), 1), device=data.device, dtype=torch.long)), -1)
            nums = torch.empty((len(f), 0), device=data.device)
            tokens.append(projection(tokenizer(cats, nums, 'ranking_context')).cpu().numpy())
    latent = np.zeros((n, cfg['token_dimension']), np.float32)
    latent[sources] = np.concatenate(tokens)[inv]
    refs = historical_references(arrays)
    align_references(arrays, refs)
    np.savez_compressed(root / 'features.npz', row_id=arrays['row_id'], refs=refs,
                        loop_latent=loop_latent, amber_latent=latent, teacher_logits=soft)
    write(root / 'features.json', dict(identity=identity, inputs_sha256=identity['inputs_sha256'],
          cache_sha256=sha256(root / 'features.npz'),
          teacher_sha256=sha256(root / 'teacher.pt'), tokenizer_sha256=sha256(root / 'tokenizer.pt'),
          token_role='Frozen ranking-context A; item/category only; no LLM hidden states',
          teacher_targets='stage1 only; NaN elsewhere', sources=len(sources)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['teacher', 'alignment', 'joint', 'cache'])
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path)
    parser.add_argument('--config', type=Path, default=Path('configs/amazon.json'))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    cfg = read(args.config)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = AmazonData(args.data, args.device)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    output = args.work_dir / ('features.npz' if args.stage == 'cache' else args.stage + '.pt')
    if output.exists():
        raise FileExistsError(output)
    identity = dict(config_sha256=sha256(args.config), inputs_sha256=data.manifest['inputs_sha256'])
    if args.stage == 'teacher':
        teacher(data, cfg, args.work_dir, identity)
    elif args.stage in ('alignment', 'joint'):
        amber(data, cfg, args.work_dir, identity, args.pretrained, args.stage == 'joint')
    else:
        cache(data, cfg, args.work_dir, identity)


if __name__ == '__main__':
    main()
