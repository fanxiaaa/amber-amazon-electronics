# Amazon Electronics protocol and execution

## Data

Use the [BARS AmazonElectronics_x1 dataset](https://huggingface.co/datasets/reczoo/AmazonElectronics_x1),
revision `e9cc5044d076f05fda6737e5219c9c3f2598c3dd`.
The source archive is `AmazonElectronics_x1.zip` (131,709,052 bytes), with SHA256
`83183787433dbf559744f00ef09af52e60239908946b33484576910143c8a5d3`.
Obtain it from the source under its applicable terms; it is not redistributed.
The dataset's sampled positive/negative candidates are not logged exposure CTR.

Reconstruct the prior cohort by ranking users with at least two training contexts
by SHA256 of `20261010:<decimal user ID>` and excluding the first 8,192 users.
Among remaining users with at least three contexts, rank by
`20261015:<decimal user ID>` and take the first 8,192. Break hash ties by user ID.
No held-out labels or scores enter selection. The code verifies contiguous paired
prefixes in the pinned source. It ships no cohort IDs.

For a selected user with m training contexts, put prefixes through
`max(1, min(m-2, floor(4*m/7)))` in the teacher stage, the next prefixes through
m-1 in downstream fitting, and prefix m in validation. The published test pair
is the test stage. Candidates sharing a user and prefix remain together.

| Stage | Targets |
|---|---:|
| Teacher/upstream | 54,464 |
| Downstream fitting | 39,330 |
| Validation | 16,384 |
| Test | 16,384 |

Vocabulary uses teacher-stage targets and their full supplied histories only.
Padding is 0, unseen values are 1, and observed raw IDs (including raw zero) start
at 2. Item and category embeddings are shared between query and history; user ID
is metadata only. Retain the latest 100 positive-history events. Sort training by
prefix length, user ID, positive candidate first; retain published test row order.
This is user-relative ordinal ordering, not global timestamp ordering or the
LoopFM paper's eight sequential chunks. The source negative sampling excludes
reviewed items including held-out items; this inherited preprocessing assumption
limits interpretation as a real-world prospective ranking task.

## Models and fitting

The DMIN adapter uses 64-dimensional item/category embeddings, eight interest
heads and a 1024/512/256 MLP. The KD auxiliary head branches after the first
1024-wide ReLU and is discarded for serving; soft-label scale, loss weight and
shared-gradient scale are all 1. This differs from the original LoopFM recipe.

The LoopFM-style teacher is a stage0 DMIN. Freeze it and train a post-hoc
1024→256→32→256→1024 autoencoder for one teacher-stage pass. Its 32-dimensional
historical features enter a four-head history module. This is post-hoc compression,
not simultaneous teacher/autoencoder training or teacher refresh.

AMBER uses the shared 128-dimensional, two-layer event tokenizer and a projection
to the 896-dimensional Qwen2.5-0.5B-Instruct hidden size. Train the tokenizer,
projection and head for one teacher-stage pass with Qwen frozen; then jointly
train for a second teacher-stage pass. The downstream representation is the
frozen projected A token of the historical item/category, not a contextual Qwen
hidden state. Historical B tokens are constant positives and omitted from the
downstream cache. Qwen is an adaptation of the paper's pretrained backbone.

Both history-feature methods use earlier same-user positive stage1/2 events,
latest 50 only. No current candidate or same-prefix sibling enters this branch.
For HSTU, cached events are aligned to their exact positions in its 100-event
history. Earlier validation positives can appear in the later test history;
they do not update model weights. Test labels never enter training.

DMIN + AMBER projects 896→32 and uses the same history module as the LoopFM-style
arm. HSTU uses two 128-wide blocks, four heads, positional bias, SiLU attention,
and a 128→256→1 shared head. Its AMBER branch normalizes frozen tokens, projects
896→32 with SiLU, applies target-conditioned attention and maps the pooled
features 32→128→256. The last projection starts at zero and adds before the
shared head's ReLU. Serving uses one model and one prediction head.

HSTU+history uses the same extra attention and shared head, replacing frozen
AMBER tokens with the HSTU history embeddings before the HSTU blocks. Both
branches use the same covered-event mask. HSTU+history does not consume AMBER
token values; its input projection is 128→32 instead of 896→32, so the extra
parameter counts differ. The cache supplies the shared event references.

All downstream fits use one ordered pass, Adam, batch size 256, gradient clip
10, FP32 and TF32 disabled. Original architecture-specific regularization stays
fixed. Validation LogLoss on seed 20222023 selects the learning rate separately
for each arm from {0.0001, 0.0005, 0.001}. The selected rates remain frozen.
`configs/amazon.json` contains these rates and the four packaged downstream seeds,
20222023, 20222025, 20222032 and 20222034. Upstream models are fixed across
seeds. See RESULTS.md for the reported results and interpretation.

## Commands

Run on compute storage with the project installed. Replace the paths below with
your own storage and a locally downloaded Qwen snapshot. The pinned model is
`Qwen/Qwen2.5-0.5B-Instruct`, revision
`7ae557604adf67be50417f59c2c2f167def9a775`; expected model/config digests are in
`configs/amazon.json`. No remote code execution or automatic model download is used.

```bash
WORK_ROOT=/path/to/compute-storage/amber-amazon
QWEN_SNAPSHOT=/path/to/pinned-qwen-snapshot
python scripts/prepare_data.py --archive "$WORK_ROOT/AmazonElectronics_x1.zip" --output "$WORK_ROOT/data"
python scripts/prepare_features.py teacher --data "$WORK_ROOT/data" --work-dir "$WORK_ROOT/upstream"
python scripts/prepare_features.py alignment --data "$WORK_ROOT/data" --work-dir "$WORK_ROOT/upstream" --pretrained "$QWEN_SNAPSHOT"
python scripts/prepare_features.py joint --data "$WORK_ROOT/data" --work-dir "$WORK_ROOT/upstream" --pretrained "$QWEN_SNAPSHOT"
python scripts/prepare_features.py cache --data "$WORK_ROOT/data" --work-dir "$WORK_ROOT/upstream"

python scripts/train.py fit --data "$WORK_ROOT/data" --cache "$WORK_ROOT/upstream/features.npz" --arm hstu_amber_late --seed 20222023 --run-dir "$WORK_ROOT/runs/hstu-amber-seed20222023"
python scripts/train.py test --data "$WORK_ROOT/data" --cache "$WORK_ROOT/upstream/features.npz" --arm hstu_amber_late --seed 20222023 --run-dir "$WORK_ROOT/runs/hstu-amber-seed20222023"
```

`fit` evaluates validation only. `test` is a separate explicit action for a
previously fitted checkpoint. Repeat with the nine arm names and four retained seeds in
the config if running a new full comparison. To perform new rate selection,
give `fit --learning-rate RATE` a distinct directory for each first-seed rate,
select on validation before any test evaluation, then freeze the selected rates.
The same override must be given when testing such a run.

These portable commands have synthetic interface coverage; full-data equivalence
with the research runners has not been established. The packaged results reuse
24 completed cells and add the 12 previously missing DMIN-family cells. Existing
HSTU-family intervals were reused; six DMIN-family metric/contrast intervals
were calculated for the first time. Historical computations were not replayed.
