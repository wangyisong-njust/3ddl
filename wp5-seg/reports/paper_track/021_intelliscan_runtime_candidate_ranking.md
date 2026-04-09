# 021 Intelliscan Runtime Candidate Ranking

## Objective

Push the retained runtime candidates from `020_runtime_backend_calibration_compile_energy.md` into the full `intelliscan` pipeline and rank them under the real deployment gate.

This note answers a stricter question than patch-level or full-case `wp5-seg` evaluation:

- which runtime backend is actually fastest inside the whole single-scan pipeline
- whether backend choice changes downstream teacher drift materially
- how much patch-level acceleration survives after detection, merging, metrology, GLTF, and report generation are included

## Candidates

Reference:

- teacher / eager PyTorch

Whole-pipeline candidates:

- `r=0.375` / eager PyTorch
- `r=0.375` / TRT FP16
- `r=0.375` / TRT INT8 fit-only
- `r=0.5` / eager PyTorch
- `r=0.5` / `torch.compile(reduce-overhead)`
- `r=0.5` / TRT FP16

The retained runtime candidates from `020` are:

- `r=0.375` / TRT FP16
- `r=0.5` / TRT FP16
- `r=0.5` / `torch.compile(reduce-overhead)`

The fit-only INT8 path is included here as a validated-with-caveat row because calibration optimization was part of this pass.

## Samples

- `SN002`: `../../../intelliscan/output_runtime_rank/SN002_teacher_eager`
- `SN009`: `../../../intelliscan/output_runtime_rank/SN009_teacher_eager`
- `SN010`: `../../../intelliscan/output_runtime_rank/SN010_teacher_eager`

## Experimental Setup

- Runner: `../../../intelliscan/scripts/benchmark_runtime_candidates.py`
- GPU: physical `GPU 1`
- Output base: `../../../intelliscan/output_runtime_rank/`
- Summary JSON:
  - `../../../intelliscan/output_runtime_rank/analysis/runtime_candidate_ranking_summary.json`
- Aggregate CSV table:
  - `../../../intelliscan/output_runtime_rank/analysis/runtime_candidate_ranking_table.csv`

The runner measured:

- whole-pipeline wall clock
- `metrics.json` stage timings
- GPU power / energy from `nvidia-smi`
- teacher-relative segmentation and metrology drift

## Aggregate Ranking Table

| Variant | Status | Patch infer (ms) | Mean total (s) | Mean seg+met (s) | Total speedup vs teacher | Mean energy (J) | Mean power (W) | Mean Dice c3 vs teacher | Mean Dice c4 vs teacher | Pad flips | Solder flips |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher / eager | reference | `17.40` | `34.96` | `22.09` | `0.00%` | `4945.14` | `108.06` | - | - | - | - |
| `r=0.375` / eager | context-only | `16.47` | `33.38` | `21.35` | `4.53%` | `4833.24` | `109.47` | `0.5391` | `0.2690` | `52` | `117` |
| `r=0.375` / TRT FP16 | retained | `3.15` | `30.41` | `17.97` | `13.02%` | `3745.68` | `91.46` | `0.5389` | `0.2690` | `52` | `117` |
| `r=0.375` / TRT INT8 fit-only | validated with caveat | `2.79` | `29.57` | `17.64` | `15.41%` | `3667.74` | `91.41` | `0.4649` | `0.2695` | `50` | `116` |
| `r=0.5` / eager | context-only | `8.87` | `30.39` | `18.62` | `13.08%` | `3815.77` | `94.80` | `0.2159` | `0.3049` | `70` | `140` |
| `r=0.5` / compile | retained | `4.80` | `50.16` | `37.54` | `-43.49%` | `5524.06` | `90.78` | `0.2159` | `0.3049` | `70` | `140` |
| `r=0.5` / TRT FP16 | retained | `1.62` | `29.45` | `17.29` | `15.78%` | `3498.51` | `87.65` | `0.2158` | `0.3049` | `71` | `141` |

## Sample-Level Observations

### Runtime

- `r=0.5` / TRT FP16 is the fastest retained runtime candidate on all three samples.
- `r=0.5` / compile is the clearest negative result of this pass:
  - patch-level compile is strong
  - but whole-pipeline cold-start latency is much worse because every CLI run recompiles the model
- `r=0.375` / TRT FP16 is consistently faster than `r=0.375` / eager, but the end-to-end gain stays much smaller than the patch-level gain.

This is the key systems result:

- very large patch-level acceleration shrinks to about `13%` to `16%` total pipeline speedup
- the rest of the pipeline still matters

### Quality

- backend choice barely changes teacher-relative drift inside the same student family
- `r=0.375` / eager and `r=0.375` / TRT FP16 are almost identical in:
  - voxel agreement
  - class `3/4` Dice
  - defect flips
- `r=0.5` / eager, `r=0.5` / compile, and `r=0.5` / TRT FP16 are also effectively identical in downstream quality

This means:

- backend acceleration does not fix the student model drift
- the dominant deployment problem is still the compressed model itself, not the runtime backend

### Calibration-Specific Note

The fit-only INT8 row is interesting but not strong enough to retain as the preferred backend:

- it is slightly faster than `r=0.375` / TRT FP16 at whole-pipeline level
- it has slightly fewer total defect flips on this three-sample panel
- but it loses more class `3` Dice relative to the teacher
- and `020` already showed that full-case `wp5-seg` quality remains below the FP16 path

So this row stays “validated with caveat,” not retained.

## Important Sample-Level Numbers

### `SN002`

| Variant | Total (s) | Seg+Met (s) | Energy (J) |
| --- | ---: | ---: | ---: |
| teacher | `25.83` | `14.25` | `3391.9` |
| `r=0.375` TRT FP16 | `24.32` | `12.77` | `3026.0` |
| `r=0.5` compile | `54.27` | `42.75` | `5801.5` |
| `r=0.5` TRT FP16 | `24.89` | `12.49` | `2939.6` |

### `SN009`

| Variant | Total (s) | Seg+Met (s) | Energy (J) |
| --- | ---: | ---: | ---: |
| teacher | `50.23` | `35.78` | `7634.4` |
| `r=0.375` TRT FP16 | `40.06` | `26.67` | `4977.3` |
| `r=0.5` compile | `58.61` | `44.39` | `6482.6` |
| `r=0.5` TRT FP16 | `38.64` | `25.64` | `4532.5` |

### `SN010`

| Variant | Total (s) | Seg+Met (s) | Energy (J) |
| --- | ---: | ---: | ---: |
| teacher | `28.82` | `16.24` | `3809.1` |
| `r=0.375` TRT FP16 | `26.84` | `14.48` | `3233.7` |
| `r=0.5` compile | `37.61` | `25.48` | `4288.1` |
| `r=0.5` TRT FP16 | `24.81` | `13.75` | `3023.4` |

## Interpretation

### 1. Patch-level backend ranking is not enough

`torch.compile(reduce-overhead)` looked strong at the patch level for `r=0.5`, but it fails badly in the current single-scan CLI deployment because the compile warmup cost is paid on every run.

So:

- keep compile as a PyTorch fallback backend in `wp5-seg`
- do not treat it as a strong current `intelliscan` single-sample backend

### 2. TRT FP16 is the strongest retained runtime backend

Across this three-sample deployment panel:

- `r=0.5` / TRT FP16 is the fastest and lowest-energy retained runtime candidate
- `r=0.375` / TRT FP16 is the cleaner higher-quality retained runtime candidate

### 3. Backend optimization does not solve model drift

The `r=0.375` and `r=0.5` teacher gaps remain almost unchanged when switching:

- eager -> TRT FP16
- eager -> compile

So the next research move should not be “another backend trick.”

It should be:

- better student training / selection
- or downstream-aware compression criteria

## Decision

Keep:

- `r=0.375` / TRT FP16 as the preferred runtime backend for the safer retained student
- `r=0.5` / TRT FP16 as the preferred runtime backend for the faster retained student

Do not keep as a preferred `intelliscan` single-sample backend:

- `r=0.5` / compile

Reason:

- patch-level compile gains do not survive current cold-start whole-pipeline execution

Do not upgrade to retained:

- `r=0.375` / TRT INT8 fit-only

Reason:

- it remains a mixed result with weaker class `3` fidelity than FP16

## Bottom Line

This pass gives a clear deployment ranking:

1. best speed / energy: `r=0.5` / TRT FP16
2. best quality among retained runtime candidates: `r=0.375` / TRT FP16
3. best PyTorch fallback only at patch level, not in current CLI whole pipeline: `r=0.5` / compile

But none of these candidates clears the final teacher-relative deployment gate yet, because model-induced metrology drift still dominates.

## Next Step

Do not search for another runtime trick first.

The next meaningful move is to improve the student model or its selection rule, then re-run the same whole-pipeline ranking with:

- the current `r=0.375` / TRT FP16 line as the quality runtime reference
- the current `r=0.5` / TRT FP16 line as the speed runtime reference
