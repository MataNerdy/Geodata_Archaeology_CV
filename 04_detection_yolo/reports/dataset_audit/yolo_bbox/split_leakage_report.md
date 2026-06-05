# Split Leakage Check

Leakage is checked at two levels:

- `region`
- `source_id = region | modality | raster_file`

| Check | Train unique | Val unique | Overlap |
|---|---:|---:|---:|
| Region | 88 | 21 | 0 |
| Source ID | 111 | 24 | 0 |

No split leakage found by `region` or derived `source_id`.
