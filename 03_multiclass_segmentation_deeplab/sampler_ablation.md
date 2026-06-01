# Stage D Sampler Ablation Equivalence

Stage D changes only the train sampler:

- `default`: the same shuffled DataLoader behavior as Stage A;
- `weighted`: `WeightedRandomSampler` with inverse-frequency metadata `class_name` weights.

## Stage A and Stage D default

The standalone Stage D runner defaults now match the Stage A all-modalities seed-series settings:

| Setting | Stage A | Stage D default |
|---|---|---|
| config | `configs/archaeology_5class_research_split_v1.yaml` | same |
| modalities | `Li,Ae,SpOr` | same |
| seed | `101` | same |
| batch size | `16` | same |
| num workers | `2` | same |
| split | frozen CSV split | same |
| selection metric | `weighted_competition_f1` | same |
| object IoU threshold | `0.3` | same |
| train shuffle | enabled | enabled |
| sampler | default PyTorch sampler | same |
| drop last | `False` | same |
| explicit DataLoader generator | not set | not set |
| augmentations | none | none |

The Stage D output directory differs intentionally so the ablation does not overwrite Stage A artifacts.

## Reproducibility limit

The recipe is equivalent, but bit-for-bit reconstruction of a previously trained GPU checkpoint is not
guaranteed. The original Stage A run did not enable PyTorch deterministic-algorithms mode for CUDA kernels.
Enabling it now would change the established recipe, so Stage D keeps the existing training behavior and
documents the limitation.

Use `--num-workers 0` only as an explicit notebook stability fallback. That fallback no longer exactly
matches the original Stage A worker configuration.
