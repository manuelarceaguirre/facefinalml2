# Results Summary

## Evaluation protocol

Final reported results use the repaired official available-image, pair-safe protocol:

- Exact canonical image mapping: `BMI/Data/Images/<name>`
- Official `is_training` split for held-out test
- Pair grouping: `pair_id = row_id // 2`
- Validation split carved only from official training pool
- Group-overlap audit: train/val/test pair overlap is zero

Available clean subset:

| Split | Images | Pair groups | BMI mean | BMI std |
| --- | ---: | ---: | ---: | ---: |
| Train | 2,565 | 1,295 | 32.38 | 7.83 |
| Validation | 642 | 324 | 32.44 | 8.16 |
| Test | 748 | 383 | 33.46 | 8.62 |

## Paper baseline

| Model | Male r | Female r | Overall r |
| --- | ---: | ---: | ---: |
| Paper VGG-Net + SVR | 0.58 | 0.36 | 0.47 |
| Paper VGG-Face + SVR | 0.71 | 0.57 | 0.65 |

## Final v4 results

Best validation-selected ensemble:

`SLSQP decorrelated ensemble` over:

- `arcface_dinov2_convnext__quantile_ridge`
- `concat_all_reference__pca_svr`
- `arcface_dinov2__pca_svr`
- `arcface_convnext__ridge`

| Metric | Test value |
| --- | ---: |
| Pearson r | 0.7216 |
| Spearman rho | 0.7459 |
| MAE | 4.4345 |
| RMSE | 6.1989 |
| R2 | 0.4816 |
| Bias | -1.5519 |
| Within 2 BMI | 0.3422 |
| Within 5 BMI | 0.6818 |

Best simple single model:

| Model | Test Pearson r | Test MAE |
| --- | ---: | ---: |
| `arcface_dinov2_convnext__ridge` | 0.7104 | 4.5083 |
| `arcface_dinov2__pca_svr` | 0.7097 | 4.4882 |
| `arcface_dinov2_convnext__quantile_ridge` | 0.7070 | 4.5252 |

## Gender subgroup results for best ensemble

| Subgroup | n | Pearson r | Spearman rho | MAE | RMSE | Bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Female | 322 | 0.7183 | 0.7464 | 4.3693 | 6.0896 | -0.9512 |
| Male | 426 | 0.7268 | 0.7504 | 4.4837 | 6.2803 | -2.0059 |

## Main conclusion

On the official available-image, pair-safe split, the ArcFace + DINOv2 + ConvNeXt ensemble achieved Pearson r = 0.7216, outperforming the paper's VGG-Face + SVR overall benchmark of r = 0.65. The improvement was consistent across gender subgroups: female r = 0.7183 and male r = 0.7268.

## Remaining limitation

The model still regresses toward the mean. In the final ensemble, obesity_3 remains underpredicted and healthy BMI remains overpredicted, so the report should include residual and BMI-category error analysis.
