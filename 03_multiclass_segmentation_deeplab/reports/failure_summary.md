## Representative failure cases

The collage contains the five lowest-scoring validation patches for the final Stage C pipeline, ranked by patch-level object F1 and weighted score.

![Representative failure cases](reports/figures/failure_cases.png)

## Failure analysis

- The analysis contains `2476` GT objects whose dominant predicted class differs from the annotation: `2058` map to `background`, while `418` map to another foreground class.
- `kurgany_povrezhdennye` contributes the largest number of failed GT objects: `1110`.
- The strongest foreground-to-foreground confusion is `kurgany_tselye -> kurgany_povrezhdennye` with `231` objects.
- Full or near-full misses dominate: `83.1%` of failed GT objects are classified as `background` inside the annotated component.
- In the visual Top-5 worst patches, all selected samples have `object_f1=0.000` and `weighted=0.000`; four of the five GT labels are `fortifikatsii`, concentrated in `005_ЛУБНО` and `006_МОСКОВИТЫ`.
- Most frequent observed failure transitions:
  - `kurgany_povrezhdennye -> background`: `1052` objects.
  - `kurgany_tselye -> background`: `624` objects.
  - `kurgany_tselye -> kurgany_povrezhdennye`: `231` objects.
  - `fortifikatsii -> background`: `170` objects.
  - `arkhitektury -> background`: `157` objects.
