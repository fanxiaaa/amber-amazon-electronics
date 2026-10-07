"""Small synthetic checks for the newly packaged interfaces; no research replay."""
from pathlib import Path
import gc
import json
import math
import statistics

import numpy as np
import unittest
import tempfile
import inspect
import torch
from torch.nn import functional as F

from amber.experiment import ARMS, KD_ARMS, AmazonData, align_references, build_model, historical_references, read, sha256, write
from amber.amazon_kd import kd_losses
from prepare_data import encode_records
from train import fit, predict

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def fixture_arrays():
    # Four successive prefixes, paired positive/negative candidates, one user.
    records = []
    for prefix in range(1, 5):
        for label in (1, 0):
            records.append((len(records), prefix - 1, label, 7, prefix + 1 if label else 9,
                            1, tuple(range(1, prefix + 1)), (1,) * prefix))
    return encode_records(records, {'item': {i: i + 2 for i in range(1, 10)}, 'category': {1: 2}})


def fixture_data(tmp_path):
    arrays = fixture_arrays()
    root = tmp_path / 'data'
    root.mkdir()
    np.savez(root / 'inputs.npz', **arrays)
    write(root / 'manifest.json', dict(inputs_sha256=sha256(root / 'inputs.npz'),
                                      vocab_sizes={'item': 12, 'category': 3}))
    refs = historical_references(arrays)
    rng = np.random.default_rng(11)
    cache = tmp_path / 'features.npz'
    np.savez(cache, row_id=arrays['row_id'], refs=refs,
             loop_latent=rng.normal(size=(8, 32)).astype('float32'),
             amber_latent=rng.normal(size=(8, 896)).astype('float32'),
             teacher_logits=np.zeros(8, 'float32'))
    write(cache.with_suffix('.json'), dict(inputs_sha256=sha256(root / 'inputs.npz'), cache_sha256=sha256(cache)))
    return AmazonData(root, cache=cache)


def test_new_reference_builder_keeps_sibling_candidates_together():
    a = fixture_arrays()
    refs = historical_references(a)
    assert np.array_equal(refs[::2], refs[1::2])
    assert (refs[:4] == -1).all()  # Teacher events are excluded from extra branch.
    assert refs[4, 0] == 2
    assert refs[6, :2].tolist() == [2, 4]
    aligned = align_references(a, refs)
    assert aligned[6, :4].tolist() == [-1, -1, 2, 4]
    refs[6, 0] = 6
    with unittest.TestCase().assertRaisesRegex(ValueError, 'past-only'):
        align_references(a, refs)


def test_portable_kd_preserves_base_initialization_and_rng():
    cfg = read(ROOT / 'configs/amazon.json')
    sizes = {'item': 12, 'category': 3}
    base = build_model('baseline', sizes, cfg, cfg['seeds'][0])
    expected_rng = torch.get_rng_state().clone()
    kd = build_model('kd', sizes, cfg, cfg['seeds'][0])
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert all(torch.equal(v, kd.vm.state_dict()[k]) for k, v in base.state_dict().items())
    teacher = torch.tensor([.2, -.3], requires_grad=True)
    serving = torch.tensor([.1, -.2], requires_grad=True)
    auxiliary = torch.tensor([.3, -.4], requires_grad=True)
    kd_losses(serving, auxiliary, torch.tensor([1., 0.]), teacher, cfg)[2].backward()
    assert teacher.grad is None and auxiliary.grad is not None
    kd.hook.remove()


def test_every_packaged_arm_accepts_the_new_batch_contract(tmp_path):
    data = fixture_data(tmp_path)
    cfg = read(ROOT / 'configs/amazon.json')
    for arm in ARMS:
        model = build_model(arm, data.manifest['vocab_sizes'], cfg, cfg['seeds'][0])
        batch = data.batch(arm, np.array([4, 6]))
        output = model(batch)
        assert output.shape == (2,) and torch.isfinite(output).all(), arm
        F.binary_cross_entropy_with_logits(output, torch.tensor([1., 0.])).backward()
        assert any(p.grad is not None for p in model.parameters()), arm
        if arm in KD_ARMS:
            model.hook.remove()
        del model
        gc.collect()


def test_new_gather_and_late_branch_preserve_zero_initialization(tmp_path):
    data = fixture_data(tmp_path)
    cfg = read(ROOT / 'configs/amazon.json')
    batch = data.batch('hstu_amber_late', np.array([2, 6]))
    base = build_model('hstu', data.manifest['vocab_sizes'], cfg, cfg['seeds'][0]).eval()
    late = build_model('hstu_amber_late', data.manifest['vocab_sizes'], cfg, cfg['seeds'][0]).eval()
    with torch.no_grad():
        torch.testing.assert_close(late(batch), base(batch), rtol=0, atol=1e-7)
        late.extra_head.weight.normal_(std=.01)
        expected = late(batch)
        batch['amber_history'][~batch['amber_valid']] = 1e4
        torch.testing.assert_close(late(batch), expected, rtol=0, atol=1e-7)


def test_history_control_uses_covered_raw_history(tmp_path):
    data = fixture_data(tmp_path)
    cfg = read(ROOT / 'configs/amazon.json')
    batch = data.batch('hstu_raw_late', np.array([2, 6]))
    assert 'amber_history' not in batch
    model = build_model('hstu_raw_late', data.manifest['vocab_sizes'], cfg, cfg['seeds'][0]).eval()
    with torch.no_grad():
        torch.testing.assert_close(model(batch), model.base(batch), rtol=0, atol=1e-7)
        model.extra_head.weight.normal_(std=.01)
        covered = model(batch)
        batch['amber_valid'].fill_(False)
        uncovered = model(batch)
        torch.testing.assert_close(uncovered, model.base(batch), rtol=0, atol=1e-7)
        assert not torch.allclose(covered[1:], uncovered[1:])


def test_new_training_entry_on_synthetic_fixture(tmp_path):
    data = fixture_data(tmp_path)
    cfg = read(ROOT / 'configs/amazon.json')
    model = build_model('hstu_amber_late', data.manifest['vocab_sizes'], cfg, cfg['seeds'][0])
    fit(data, model, cfg, 'hstu_amber_late', .001)
    idx, logits, scores = predict(data, model, 'hstu_amber_late', 2)
    assert len(idx) == 2 and scores['n'] == 2 and np.isfinite(logits).all()
    assert set(idx).isdisjoint(data.indices(3))


def test_exported_numbers_match_all_archived_seed_summaries():
    r = read(ROOT / 'results/amazon_electronics.json')
    for arm in ARMS:
        for metric in ('log_loss', 'auc', 'ne_uncalibrated'):
            values = [r['per_seed'][str(s)][arm]['scores'][metric] for s in r['seeds']]
            assert abs(statistics.mean(values) - r['summary'][arm][metric]['mean']) < 1e-12
            assert abs(statistics.stdev(values) - r['summary'][arm][metric]['sd']) < 1e-12
        for seed in r['seeds']:
            scores = r['per_seed'][str(seed)][arm]['scores']
            assert scores['n'] == 16384 and scores['positive_rate'] == .5
            assert abs(scores['log_loss'] / math.log(2) - scores['ne_uncalibrated']) < 1e-12
    for control, contrast in r['contrasts'].items():
        delta = [r['per_seed'][str(s)]['hstu_amber_late']['scores']['log_loss'] -
                 r['per_seed'][str(s)][control]['scores']['log_loss'] for s in r['seeds']]
        assert np.allclose(delta, contrast['per_seed'], rtol=0, atol=1e-12)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, function in list(globals().items()):
        if name.startswith('test_') and callable(function):
            def run(function=function):
                if inspect.signature(function).parameters:
                    with tempfile.TemporaryDirectory(prefix='amber-release-') as directory:
                        function(Path(directory))
                else:
                    function()
            suite.addTest(unittest.FunctionTestCase(run, description=name))
    return suite
