# Filter Impact Table

| dataset_version | step_name | images_before | images_after | positive_before | positive_after | bbox_before | bbox_after | removed_images | removed_positive_images | removed_bbox |
|---|---|---|---|---|---|---|---|---|---|---|
| v3a_minimal | modality_filter | 1693 | 625 | 416 | 186 | 4889 | 1886 | 1068 | 230 | 3003 |
| v3a_minimal | valid_fraction_gte_0.8 | 625 | 499 | 186 | 146 | 1886 | 591 | 126 | 40 | 1295 |
| v3a_minimal | bbox_area_filter_disabled | 499 | 499 | 146 | 146 | 591 | 591 | 0 | 0 | 0 |
| v3a_minimal | bbox_edge_filter_disabled | 499 | 499 | 146 | 146 | 591 | 591 | 0 | 0 | 0 |
| v3a_minimal | n_objects_cutoff_disabled | 499 | 499 | 146 | 146 | 591 | 591 | 0 | 0 | 0 |
| v3a_minimal | negative_sampling_ratio_1 | 499 | 292 | 146 | 146 | 591 | 591 | 207 | 0 | 0 |
| v3b_medium | modality_filter | 1693 | 625 | 416 | 186 | 4889 | 1886 | 1068 | 230 | 3003 |
| v3b_medium | valid_fraction_gte_0.85 | 625 | 481 | 186 | 142 | 1886 | 582 | 144 | 44 | 1304 |
| v3b_medium | bbox_area_filter | 481 | 481 | 142 | 142 | 582 | 579 | 0 | 0 | 3 |
| v3b_medium | bbox_edge_filter_disabled | 481 | 481 | 142 | 142 | 579 | 579 | 0 | 0 | 0 |
| v3b_medium | n_objects_lte_50 | 481 | 481 | 142 | 142 | 579 | 579 | 0 | 0 | 0 |
| v3b_medium | negative_sampling_ratio_1 | 481 | 284 | 142 | 142 | 579 | 579 | 197 | 0 | 0 |
| v3c_strict | modality_filter | 1693 | 625 | 416 | 186 | 4889 | 1886 | 1068 | 230 | 3003 |
| v3c_strict | valid_fraction_gte_0.9 | 625 | 454 | 186 | 135 | 1886 | 567 | 171 | 51 | 1319 |
| v3c_strict | bbox_area_filter | 454 | 444 | 135 | 125 | 567 | 487 | 10 | 10 | 80 |
| v3c_strict | bbox_edge_filter | 444 | 396 | 125 | 77 | 487 | 350 | 48 | 48 | 137 |
| v3c_strict | n_objects_lte_20 | 396 | 390 | 77 | 73 | 350 | 231 | 6 | 4 | 119 |
| v3c_strict | negative_sampling_ratio_1 | 390 | 146 | 73 | 73 | 231 | 231 | 244 | 0 | 0 |
| v3d_li_ae_medium | modality_filter | 1693 | 1455 | 416 | 368 | 4889 | 4288 | 238 | 48 | 601 |
| v3d_li_ae_medium | valid_fraction_gte_0.85 | 1455 | 1311 | 368 | 324 | 4288 | 2984 | 144 | 44 | 1304 |
| v3d_li_ae_medium | bbox_area_filter | 1311 | 1311 | 324 | 324 | 2984 | 2952 | 0 | 0 | 32 |
| v3d_li_ae_medium | bbox_edge_filter_disabled | 1311 | 1311 | 324 | 324 | 2952 | 2952 | 0 | 0 | 0 |
| v3d_li_ae_medium | n_objects_lte_50 | 1311 | 1298 | 324 | 311 | 2952 | 1207 | 13 | 13 | 1745 |
| v3d_li_ae_medium | negative_sampling_ratio_1 | 1298 | 622 | 311 | 311 | 1207 | 1207 | 676 | 0 | 0 |
