# AMBER on Amazon Electronics

A comparison of nine recommendation configurations on four downstream seeds:
**20222023, 20222025, 20222032 and 20222034**.

## Results

**Among the nine reported configurations, AMBER+HSTU has the lowest LogLoss
and NE in each of the four reported seeds.** The cohort contains 8,192 users
and 16,384 test candidates. Values below are means with paired user/seed
bootstrap intervals. This is a descriptive comparison rather than an
equal-budget architecture leaderboard; see the protocol and result limitations.

| Method | LogLoss ↓ [paired bootstrap 95% interval] | NE ↓ [paired bootstrap 95% interval] |
|---|---|---|
| DMIN | 0.631955 [0.627741, 0.636282] | 0.911718 [0.905639, 0.917961] |
| DMIN + KD | 0.627701 [0.623217, 0.631931] | 0.905581 [0.899112, 0.911683] |
| DMIN + LoopFM | 0.643631 [0.636885, 0.648866] | 0.928563 [0.918831, 0.936116] |
| DMIN + KD + LoopFM | 0.630880 [0.624979, 0.637855] | 0.910167 [0.901654, 0.920230] |
| DMIN + AMBER | 0.629422 [0.624665, 0.634168] | 0.908063 [0.901201, 0.914911] |
| DMIN + KD + AMBER | 0.627795 [0.623287, 0.632289] | 0.905716 [0.899214, 0.912200] |
| HSTU | 0.624027 [0.619197, 0.628868] | 0.900281 [0.893312, 0.907265] |
| AMBER+HSTU | **0.619139 [0.614193, 0.624012]** | **0.893229 [0.886094, 0.900259]** |
| HSTU+history | 0.620840 [0.616173, 0.625333] | 0.895682 [0.888949, 0.902165] |

NE equals LogLoss / ln(2) on this balanced test.

[Per-seed values and paired comparisons](docs/RESULTS.md) and
[machine-readable results](results/amazon_electronics.json) are included.

## Repository

| Path | Contents |
|---|---|
| `src/amber/` | DMIN, KD, LoopFM-style features, HSTU, AMBER and the history control |
| `scripts/prepare_data.py` | Paired-prefix cohort and training-only vocabulary |
| `scripts/prepare_features.py` | Teacher training and frozen history features |
| `scripts/train.py` | Fitting, validation and explicit test evaluation |
| `configs/amazon.json` | Four downstream seeds and selected hyperparameters |
| `results/amazon_electronics.json` | Metrics, intervals and artifact digests for all 36 cells |

The portable entry points have synthetic test coverage. Results were produced
by the research runners; they have not been regenerated with these portable
entry points. See the [code review](docs/REVIEW.md).

## Running

Use Python 3.12 on a compute machine. Install with `uv sync`, or into an
existing compatible environment with `python -m pip install -e .`.

```bash
PYTHONPATH=src:scripts python -m unittest discover -s tests -v
```

The [protocol](docs/PROTOCOL.md) describes data, preprocessing, model settings
and execution commands. Keep datasets, weights and run outputs outside this
checkout; none are bundled. Training is not needed to inspect the tables.

## Attribution

[AMBER](https://arxiv.org/abs/2608.25546),
[LoopFM](https://arxiv.org/abs/2605.29280),
[HSTU](https://arxiv.org/abs/2402.17152), and
[FuxiCTR DMIN](https://github.com/reczoo/FuxiCTR/tree/b4ce3b518e49fc3d1077791a6469f3e728f07b8a/model_zoo/DMIN).
Code is Apache-2.0 with upstream notices preserved in `NOTICE` and `licenses/`.
Dataset and model weights retain their own terms and are not redistributed here.
