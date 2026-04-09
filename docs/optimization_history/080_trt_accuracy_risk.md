# 080 TRT Accuracy Risk

## Objective

Record why current TRT FP16 acceleration cannot yet be described as lossless for defect-sensitive use.

## Target Files

- `intelliscan/output_formal/SN009_current_gpu/segmentation.nii.gz`
- `intelliscan/output_formal/SN009_current_gpu_trt/segmentation.nii.gz`
- `intelliscan/output_formal/SN009_current_gpu/metrology/metrology.csv`
- `intelliscan/output_formal/SN009_current_gpu_trt/metrology/metrology.csv`

## Code Change Summary

- No new code in this note. This is a comparison note.

## Why This Change Was Made

- End-to-end deployment safety depends on segmentation and metrology stability, not only speed.

## Baseline Behavior

- SN009 PyTorch baseline run under `SN009_current_gpu`

## Optimized Behavior

- SN009 TRT FP16 run under `SN009_current_gpu_trt`

## How It Was Tested

- Computed in this pass from saved segmentation volumes and metrology CSVs

## Sample(s) Used

- `SN009`

## Key Timing Numbers

- Baseline total: `93.23s`
- TRT FP16 total: `81.15s`
- Baseline segmentation: `33.79s`
- TRT FP16 segmentation: `24.15s`

## Accuracy / Quality Impact

Computed in this pass from the two saved segmentation volumes:

| Metric | Value |
| --- | --- |
| voxel accuracy | `0.989006` |
| class 0 Dice | `0.997720` |
| class 1 Dice | `0.855279` |
| class 2 Dice | `0.787414` |
| class 3 Dice | `0.192758` |
| class 4 Dice | `0.422984` |

Computed in this pass from the two metrology CSVs:

| Metric | Value |
| --- | --- |
| mean abs delta BLT | `1.259` |
| mean abs delta Pad_misalignment | `4.112` |
| mean abs delta Void_to_solder_ratio | `0.0245` |
| mean abs delta pillar_width | `0.651` |
| mean abs delta pillar_height | `2.671` |
| pad_misalignment defect flips | `67` |
| solder_extrusion defect flips | `12` |
| void_ratio defect flips | `0` |

Defect count summary:

| Variant | pad_misalignment | solder_extrusion | void_ratio |
| --- | --- | --- | --- |
| Baseline | `70` | `3` | `0` |
| TRT FP16 | `17` | `13` | `0` |

## Risks / Caveats

- Background dominates voxel count, so voxel accuracy is not a safe deployment metric by itself.
- Current risk is largest on defect-sensitive classes and downstream metrology decisions.

## Conclusion

- Current TRT FP16 should be described as `speed-positive, quality-risk present`.
- Average or background-dominated metrics are not enough for deployment claims.

## Evidence Paths

- `intelliscan/output_formal/SN009_current_gpu/segmentation.nii.gz`
- `intelliscan/output_formal/SN009_current_gpu_trt/segmentation.nii.gz`
- `intelliscan/output_formal/SN009_current_gpu/metrology/metrology.csv`
- `intelliscan/output_formal/SN009_current_gpu_trt/metrology/metrology.csv`
