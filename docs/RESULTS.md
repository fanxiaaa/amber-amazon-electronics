# Amazon Electronics results

Nine models evaluated on the same four downstream seeds and test cohort. Every
reported model uses the same seeds.

## Summary

Mean [paired bootstrap 95% interval].

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

## Paired comparisons

AMBER+HSTU minus control; negative values favor AMBER.

| Control | ΔLogLoss [paired bootstrap 95% interval] | ΔNE [paired bootstrap 95% interval] | Improving seeds |
|---|---|---|---:|
| DMIN | -0.012815 [-0.016650, -0.007819] | -0.018489 [-0.024021, -0.011280] | 4/4 |
| DMIN + KD | -0.008562 [-0.011778, -0.004798] | -0.012352 [-0.016991, -0.006922] | 4/4 |
| DMIN + LoopFM | -0.024492 [-0.031262, -0.017619] | -0.035334 [-0.045102, -0.025419] | 4/4 |
| DMIN + KD + LoopFM | -0.011740 [-0.019548, -0.004715] | -0.016938 [-0.028202, -0.006802] | 4/4 |
| DMIN + AMBER | -0.010282 [-0.014339, -0.005069] | -0.014834 [-0.020686, -0.007313] | 4/4 |
| DMIN + KD + AMBER | -0.008655 [-0.013134, -0.003600] | -0.012487 [-0.018948, -0.005193] | 4/4 |
| HSTU | -0.004888 [-0.006850, -0.002556] | -0.007052 [-0.009882, -0.003687] | 4/4 |
| HSTU+history | -0.001701 [-0.002798, -0.000728] | -0.002453 [-0.004036, -0.001050] | 4/4 |

## Per-seed metrics

| Model | Seed | LogLoss ↓ | NE ↓ |
|---|---:|---:|---:|
| DMIN | 20222023 | 0.633745357 | 0.914301284 |
| DMIN | 20222025 | 0.633501289 | 0.913949168 |
| DMIN | 20222032 | 0.630754111 | 0.909985828 |
| DMIN | 20222034 | 0.629817928 | 0.908635202 |
| DMIN + KD | 20222023 | 0.629209954 | 0.907758080 |
| DMIN + KD | 20222025 | 0.628374572 | 0.906552879 |
| DMIN + KD | 20222032 | 0.625618151 | 0.902576205 |
| DMIN + KD | 20222034 | 0.627602121 | 0.905438468 |
| DMIN + LoopFM | 20222023 | 0.633814493 | 0.914401026 |
| DMIN + LoopFM | 20222025 | 0.647965288 | 0.934816308 |
| DMIN + LoopFM | 20222032 | 0.647687239 | 0.934415167 |
| DMIN + LoopFM | 20222034 | 0.645057124 | 0.930620714 |
| DMIN + KD + LoopFM | 20222023 | 0.629168565 | 0.907698369 |
| DMIN + KD + LoopFM | 20222025 | 0.640873118 | 0.924584469 |
| DMIN + KD + LoopFM | 20222032 | 0.627196222 | 0.904852879 |
| DMIN + KD + LoopFM | 20222034 | 0.626280283 | 0.903531459 |
| DMIN + AMBER | 20222023 | 0.632077718 | 0.911895389 |
| DMIN + AMBER | 20222025 | 0.630315772 | 0.909353438 |
| DMIN + AMBER | 20222032 | 0.628445856 | 0.906655719 |
| DMIN + AMBER | 20222034 | 0.626847258 | 0.904349431 |
| DMIN + KD + AMBER | 20222023 | 0.628481338 | 0.906706910 |
| DMIN + KD + AMBER | 20222025 | 0.626915812 | 0.904448334 |
| DMIN + KD + AMBER | 20222032 | 0.630171151 | 0.909144794 |
| DMIN + KD + AMBER | 20222034 | 0.625610117 | 0.902564613 |
| HSTU | 20222023 | 0.623062885 | 0.898889735 |
| HSTU | 20222025 | 0.625067738 | 0.901782126 |
| HSTU | 20222032 | 0.621330151 | 0.896389927 |
| HSTU | 20222034 | 0.626649072 | 0.904063509 |
| AMBER+HSTU | 20222023 | 0.618504689 | 0.892313647 |
| AMBER+HSTU | 20222025 | 0.617779060 | 0.891266787 |
| AMBER+HSTU | 20222032 | 0.616228248 | 0.889029437 |
| AMBER+HSTU | 20222034 | 0.624044737 | 0.900306248 |
| HSTU+history | 20222023 | 0.621346425 | 0.896413407 |
| HSTU+history | 20222025 | 0.619889125 | 0.894310967 |
| HSTU+history | 20222032 | 0.617573419 | 0.890970110 |
| HSTU+history | 20222034 | 0.624550038 | 0.901035243 |

## Interval method and scope

5,000 bootstrap draws independently resample 8,192 user clusters and four
reported seeds with replacement. Every model uses the same user and seed draws.
Intervals are percentile 2.5/97.5 bounds; LogLoss is weighted by event count.
Each user has one positive and one negative test candidate, so NE and its
interval equal LogLoss / ln(2). For a paired comparison, subtract model losses
within each draw; do not subtract endpoints of separate model intervals.



## Provenance

The package contains 36 cells: 24 completed cells reused unchanged, plus 12
new fits for the six DMIN-family models on seeds 20222032 and 20222034. Model
and prediction digests are retained in the JSON. HSTU-family intervals are
reused unchanged; the six DMIN-family intervals and paired contrasts were
calculated for the first time from saved predictions. No historical experiment
or bootstrap was replayed. Numerical-source hashes are in `source_manifest.json`;
the corresponding source artifacts are not distributed in this package.

The JSON also retains sample SD and secondary metrics from the source records.
Those SD values are not confidence intervals. No per-user predictions are bundled.
