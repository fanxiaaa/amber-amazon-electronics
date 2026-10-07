"""One downstream fit; selection and testing are explicitly separate actions."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from amber.amazon_kd import kd_losses
from amber.experiment import ARMS, KD_ARMS, AmazonData, build_model, read, sha256, write


def metrics(y, z):
    y, z = np.asarray(y, dtype=np.float64), np.asarray(z, dtype=np.float64)
    if not np.isfinite(z).all():
        raise ValueError('Nonfinite predictions')
    loss = float((np.logaddexp(0, z) - y * z).mean())
    rate = float(y.mean())
    entropy = -(rate * np.log(rate) + (1 - rate) * np.log1p(-rate)) if 0 < rate < 1 else 0
    return dict(n=len(y), positive_rate=rate, log_loss=loss,
                auc=float(roc_auc_score(y, z)) if entropy else None,
                ne_uncalibrated=loss / entropy if entropy else None)


def batches(indices, size):
    # BatchNorm cannot train on a singleton; merge it into the preceding batch.
    for lo in range(0, len(indices), size):
        if lo == len(indices) - 1 and lo > 0:
            break
        hi = min(lo + size, len(indices))
        if len(indices) - hi == 1:
            hi += 1
        yield indices[lo:hi]


def fit(data, model, cfg, arm, rate, stage=1):
    indices = data.indices(stage)
    optimizer = torch.optim.Adam(model.parameters(), lr=rate, foreach=False)
    model.train()
    for idx in batches(indices, cfg['batch_size']):
        batch = data.batch(arm, idx)
        y = data.raw['label'][idx].float()
        optimizer.zero_grad(set_to_none=True)
        if arm in KD_ARMS:
            serving, auxiliary = model.training_logits(batch)
            soft = data.cache['teacher_logits'][idx]
            if not torch.isfinite(soft).all():
                raise ValueError('Missing teacher targets for fitting rows')
            _, _, loss = kd_losses(serving, auxiliary, y, soft, cfg)
        else:
            loss = F.binary_cross_entropy_with_logits(model(batch), y)
        (loss + model.regularization_loss()).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()


@torch.inference_mode()
def predict(data, model, arm, stage):
    model.eval()
    indices = data.indices(stage)
    logits = np.concatenate([model(data.batch(arm, idx)).cpu().numpy() for idx in batches(indices, 256)])
    return indices, logits, metrics(data.arrays['label'][indices], logits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['fit', 'test'])
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--config', type=Path, default=Path('configs/amazon.json'))
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--learning-rate', type=float, help='Override for validation-only grid fits')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    cfg = read(args.config)
    if args.seed not in cfg['seeds']:
        parser.error(f"--seed must be one of {cfg['seeds']} from {args.config}")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = AmazonData(args.data, args.device, args.cache)
    model = build_model(args.arm, data.manifest['vocab_sizes'], cfg, args.seed, args.device,
                        float(data.arrays['label'][data.indices(0)].mean()))
    rate = args.learning_rate if args.learning_rate is not None else cfg['learning_rates'][args.arm]
    identity = dict(arm=args.arm, seed=args.seed, learning_rate=rate, config_sha256=sha256(args.config),
                    inputs_sha256=data.manifest['inputs_sha256'], cache_sha256=sha256(args.cache) if args.cache else None)
    if args.action == 'fit':
        args.run_dir.mkdir(parents=True, exist_ok=False)
        fit(data, model, cfg, args.arm, rate)
        torch.save(dict(model=model.state_dict()), args.run_dir / 'model.pt')
        _, _, scores = predict(data, model, args.arm, 2)
        write(args.run_dir / 'fit.json', dict(**identity, validation=scores,
              model_sha256=sha256(args.run_dir / 'model.pt'), torch_version=torch.__version__,
              cuda_version=torch.version.cuda, tf32=False, test_evaluated=False))
    else:
        if (args.run_dir / 'test.json').exists():
            raise FileExistsError('Test output already exists; no implicit replay')
        receipt = read(args.run_dir / 'fit.json')
        if any(receipt[k] != v for k, v in identity.items()) or receipt['model_sha256'] != sha256(args.run_dir / 'model.pt'):
            raise ValueError('Model/run identity mismatch')
        model.load_state_dict(torch.load(args.run_dir / 'model.pt', map_location=args.device, weights_only=True)['model'])
        idx, logits, scores = predict(data, model, args.arm, 3)
        np.savez(args.run_dir / 'test_predictions.npz', rows=data.arrays['row_id'][idx],
                 users=data.arrays['user_id'][idx], labels=data.arrays['label'][idx], logits=logits)
        write(args.run_dir / 'test.json', dict(**identity, scores=scores, model_sha256=receipt['model_sha256'],
              predictions_sha256=sha256(args.run_dir / 'test_predictions.npz')))
    if args.arm in KD_ARMS:
        model.hook.remove()


if __name__ == '__main__':
    main()
