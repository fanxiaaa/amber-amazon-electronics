# Release validation

Scope: nine Amazon models, four downstream seeds, and 36 reported cells.
This document records source, packaging, and synthetic-test validation.

## Model and interface review

The package contains DMIN, DMIN+KD, DMIN+LoopFM, DMIN+KD+LoopFM, DMIN+AMBER,
DMIN+KD+AMBER, HSTU, AMBER+HSTU and HSTU+history. The HSTU history-control
branch uses the original attention/head operations and the same event coverage
as AMBER+HSTU, with raw history embeddings in place of AMBER token values.
Both retain zero initialization of the added head projection.

The CLI reads the four allowed downstream seeds from `configs/amazon.json`.
Auxiliary-head RNG mappings are explicit. KD construction preserves the base
model's CPU/device RNG stream. The extra feature branches preserve the base
model's dropout RNG stream. Data, cache and checkpoint paths are user supplied.

Historical references must share the target user, precede its prefix, and refer
to positive stage1/2 events. Cached row identities and digests are checked.
Array loading disables pickle; checkpoint loading uses `weights_only=True`.
Pretrained loading is local-only. No datasets, weights, checkpoints, per-user
predictions, credentials or machine-local execution paths are shipped.

## Verification

Seven synthetic tests passed on Dell CPU in 2.335 seconds. Configuration/CLI
checks accept all four packaged seeds and reject an unlisted seed. A content scan
found only the four intended downstream seed IDs. Existing HSTU-family intervals
match their source exactly, and all 36 per-seed cells are present.

The synthetic suite covers paired-prefix boundaries, KD initialization and
teacher detachment, every model's forward/backward batch interface, AMBER
initialization and masking, the restored raw-history branch, a small fit and
validation step, and reported metric arithmetic. Synthetic inputs are not
observations from the research dataset.

Archived metrics were produced by the research runners, not regenerated with
the portable release scripts. Full-data equivalence of those scripts remains
untested. The additional twelve cells use the inherited research model builders
and training loop with new seed-specific initialization; the other 24 cells are
reused unchanged. No historical experiment or bootstrap was replayed.

## Provenance and scope

Retained model components originate from research commit
`ab2dbcd05ab58d2f30254d3a784b5cd3366ac662`; new experiment/result source hashes
are recorded in `source_manifest.json`. The release adapts construction,
execution and device handling, and restores the requested history-control arm.

FuxiCTR DMIN code retains its copyright and Apache-2.0 notices. The independent
HSTU implementation and the AMBER/LoopFM adaptations are attributed in the source
and README. Data and pretrained weights retain their own terms.

The report covers four downstream seeds, with identical seeds used across all
reported models.
