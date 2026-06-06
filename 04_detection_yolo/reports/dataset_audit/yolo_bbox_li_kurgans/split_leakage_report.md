# Split Leakage Check

Leakage is checked at two levels:

- `region`
- `source_id = region | modality | raster_file`

| Check | Train unique | Val unique | Overlap |
|---|---:|---:|---:|
| Region | 19 | 5 | 0 |
| Source ID | 19 | 5 | 0 |

No split leakage found by `region` or derived `source_id`.
