# Research Summary

Эта таблица объединяет `18` строк и показывает путь от диагностических экспериментов к итоговому pipeline.

Части таблицы имеют разный смысл:

- строки `Encoder comparison` относятся к раннему диагностическому split;
- строки `Research Split V1` являются сопоставимыми benchmark-запусками до Stage C;
- строки `Stage C` — это улучшенные pipeline-варианты ранее обученных checkpoints, а не новые нейросети.

| Этап | Модель | Модальности | Seed | Состояние pipeline | Weighted F1 | Object F1 | Precision | Recall |
|---|---|---|---:|---|---:|---:|---:|---:|
| Encoder comparison | ResNet34 | `Li` | 42 | diagnostic split | 0.3421 | 0.4195 | 0.2684 | 0.9604 |
| Encoder comparison | ResNet50 | `Li` | 42 | diagnostic split | 0.4832 | 0.6058 | 0.4460 | 0.9441 |
| Encoder comparison | ResNet34 | все | 42 | diagnostic split | **0.6603** | **0.7790** | **0.7018** | 0.8754 |
| Encoder comparison | ResNet50 | все | 42 | diagnostic split | 0.6299 | 0.7306 | 0.6068 | **0.9180** |
| Research Split V1 | ResNet34 | `Li` | 13 | before Stage C | 0.6902 | 0.7522 | 0.7427 | 0.7620 |
| Research Split V1 | ResNet34 | `Li` | 21 | before Stage C | 0.7151 | 0.7783 | 0.8333 | 0.7300 |
| Research Split V1 | ResNet34 | `Li` | 42 | before Stage C | 0.6341 | 0.6176 | 0.6409 | 0.5960 |
| Research Split V1 | ResNet34 | `Li` | 77 | before Stage C | 0.6727 | 0.7886 | 0.7527 | **0.8280** |
| Research Split V1 | ResNet34 | `Li` | 101 | before Stage C | **0.7310** | **0.8276** | **0.8395** | 0.8160 |
| Research Split V1 | ResNet34 | `Li`, `Ae`, `SpOr` | 13 | before Stage C | 0.6620 | 0.6484 | **0.9609** | 0.4893 |
| Research Split V1 | ResNet34 | `Li`, `Ae`, `SpOr` | 21 | before Stage C | **0.6811** | 0.6802 | 0.9456 | 0.5312 |
| Research Split V1 | ResNet34 | `Li`, `Ae`, `SpOr` | 42 | before Stage C | 0.6778 | 0.6104 | 0.9252 | 0.4555 |
| Research Split V1 | ResNet34 | `Li`, `Ae`, `SpOr` | 77 | before Stage C | 0.6725 | 0.7072 | 0.9490 | 0.5635 |
| Research Split V1 | ResNet34 | `Li`, `Ae`, `SpOr` | 101 | before Stage C | 0.6700 | **0.7480** | 0.9540 | **0.6152** |
| Stage C | ResNet34 | `Li` | 101 | conf `0.3`, area `8`, opening `False` | 0.7316 | **0.8288** | 0.8398 | **0.8180** |
| Stage C | ResNet34 | `Li`, `Ae`, `SpOr` | 21 | conf `0.3`, area `8`, opening `True` | 0.7246 | 0.7413 | 0.9039 | 0.6283 |
| Stage C | ResNet34 | `Li`, `Ae`, `SpOr` | 77 | conf `0.3`, area `8`, opening `True` | 0.6949 | 0.7606 | **0.9167** | 0.6499 |
| **Stage C: final** | **ResNet34** | **`Li`, `Ae`, `SpOr`** | **101** | **conf `0.3`, area `8`, opening `True`** | **0.7457** | **0.7995** | 0.9114 | 0.7120 |

## Interpretation

- Первые четыре строки используются только для выбора направления исследования.
- На Research Split V1 лучшим checkpoint до Stage C стала модель `resnet34_li_seed_101`.
- После настройки объектного постпроцессинга победил мультимодальный pipeline `resnet34_all_seed_101`.
