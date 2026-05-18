# FaceFinalML2 Handoff Notes

Last updated: 2026-05-18

## One-sentence status

We built a Google Colab pipeline for a Face-to-BMI replication/extension and achieved a final official available-image, pair-safe test Pearson **r = 0.7216**, beating the paper's reported VGG-Face + SVR overall baseline of **r = 0.65**.

## Repository

Public GitHub repo:

```text
https://github.com/manuelarceaguirre/facefinalml2
```

Current local project path:

```text
/Users/manuelarce/Library/CloudStorage/SeaDrive-manuel(100.73.58.1)/My Libraries/My Library/UChicago/Spring 2026/ml2/final/facefinalml2
```

Colab artifacts downloaded from Google Drive are stored locally outside the repo at:

```text
/Users/manuelarce/Library/CloudStorage/SeaDrive-manuel(100.73.58.1)/My Libraries/My Library/UChicago/Spring 2026/ml2/final/colab_artifacts/
```

## Assignment goal

Replicate and improve the paper:

> Kocabey et al., **Face-to-BMI: Using Computer Vision to Infer Body Mass Index on Social Media**, ICWSM 2017.

Deliverables:

1. Simple web API / demo to predict BMI from face image in real time.
2. 10-page implementation write-up.
3. 10-minute final presentation or recorded demo.

Paper baseline metrics:

| Model | Male r | Female r | Overall r |
| --- | ---: | ---: | ---: |
| VGG-Net features + SVR | 0.58 | 0.36 | 0.47 |
| VGG-Face features + SVR | 0.71 | 0.57 | 0.65 |

The project target is to beat **overall Pearson r = 0.65** without identity/person leakage.

## Dataset

Source file in Google Drive:

```text
Copy of BMI.zip
Drive file ID: 16XA-MCnTG8ONdgxK0uPfFXWnA5oF3bFa
Size: about 987 MB
```

Inside the extracted zip, the canonical metadata was:

```text
/content/facefinalml2/data/extracted/BMI/Data/data.csv
```

Metadata columns:

```text
unnamed:_0, bmi, gender, is_training, name
```

Canonical image directory:

```text
/content/facefinalml2/data/extracted/BMI/Data/Images
```

Canonical image mapping used in the final protocol:

```text
image_path = BMI/Data/Images/<name>
```

## Important methodological correction

The first run grouped by filename and was treated as exploratory only because adjacent rows are likely before/after images from the same person.

Final repaired protocol:

- Use exact canonical image paths, not fuzzy recursive matching.
- Use `is_training` as the official paper-comparable test split.
- Use pair grouping:

```python
pair_id = row_id // 2
```

where `row_id` comes from metadata column `unnamed:_0`.

- Split validation only from the official training pool.
- Enforce zero pair/group overlap across train, validation, and test.

## Final data audit

From the v2/v3 repaired run:

```text
Raw metadata rows: 4206
Exact paths found: 3962
Missing exact paths: 244
Clean canonical rows after verification/filtering: 3955
Pairs audited: 2003
Pairs with n != 2: 44
Pairs with gender mismatch: 0
Pairs split across train/test: 0
```

Final official available-image split:

| Split | Images | Pair groups | BMI mean | BMI std | BMI min | BMI max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 2,565 | 1,295 | 32.3800 | 7.8319 | 17.7162 | 69.7282 |
| Validation | 642 | 324 | 32.4387 | 8.1642 | 17.7190 | 68.2029 |
| Test | 748 | 383 | 33.4566 | 8.6152 | 18.6510 | 68.6184 |

Split category distribution:

| Split | Healthy | Obesity 1 | Obesity 2 | Obesity 3 | Overweight | Underweight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Test | 0.160 | 0.207 | 0.150 | 0.225 | 0.258 | 0.000 |
| Train | 0.166 | 0.225 | 0.166 | 0.156 | 0.284 | 0.002 |
| Validation | 0.181 | 0.240 | 0.134 | 0.185 | 0.257 | 0.003 |

Face detection rate in the repaired run:

```text
0.998
```

## Experiment chronology

### v1 exploratory run

Pipeline:

- Fuzzy/recursive image mapping.
- Grouped by filename.
- Frozen embeddings:
  - FaceNet/VGGFace2 tight crop
  - ConvNeXt loose crop
  - DINOv2 loose crop
  - concatenation
- Regressors: Ridge, ElasticNet, PCA-SVR, RandomForest, XGBoost.

Result:

```text
Validation Pearson r: 0.6511
Test Pearson r:       0.6470
Test MAE:             4.4942
Test RMSE:            5.9072
```

Interpretation: close to paper, but not scientifically final because filename grouping was not pair-safe.

### v2 repaired official/pair-safe run

Pipeline correction:

- Exact image mapping.
- `is_training` official split.
- Pair grouping by `row_id // 2`.
- Validation split only from official training pool.

Features:

```text
facenet_vggface2_tight: (3955, 512)
convnext_loose:         (3955, 768)
dinov2_vits14_loose:    (3955, 384)
concat_all:             (3955, 1664)
```

Key result:

```text
Final v2 ensemble validation Pearson r: 0.6906
Final v2 ensemble test Pearson r:       0.6391
```

Interpretation: rigorous but did not beat paper. FaceNet branch collapsed/produced constant predictions.

### v3 ArcFace + attempted OOF stacking

Changes:

- Added InsightFace ArcFace `buffalo_l` embeddings.
- Added `quantile_ridge` target-transform regressor.
- Attempted full OOF stacking.

Issue:

- Full OOF was too slow because it trained many base models across five feature sets and folds.
- We interrupted it after seeing that ArcFace was working and validation results improved strongly.

Key partial evidence:

```text
ArcFace alone was useful:
arcface pca_svr val r ≈ 0.692
arcface ridge val r ≈ 0.661

concat_all + quantile ridge val r ≈ 0.749
```

### v4 targeted ArcFace ensemble

Purpose: fast targeted run using cached ArcFace, DINOv2, and ConvNeXt features.

Features used:

```text
arcface:  (3955, 512), finite fraction 0.8096
DINOv2:   (3955, 384), finite fraction 1.0
ConvNeXt: (3955, 768), finite fraction 1.0
```

Targeted feature sets:

- `arcface`
- `dinov2`
- `convnext`
- `arcface_dinov2`
- `arcface_convnext`
- `dinov2_convnext`
- `arcface_dinov2_convnext`
- `concat_all_reference`

Targeted regressors:

- RidgeCV
- Quantile-target RidgeCV
- PCA + SVR

Skipped in v4:

- Full OOF stacking
- FaceNet
- ElasticNetCV
- RandomForest
- XGBoost

## Final headline result

Best validation-selected ensemble:

```text
slsqp_decorrelated_cap050
```

Members and weights:

| Member | Weight |
| --- | ---: |
| `arcface_dinov2_convnext__quantile_ridge` | 0.5000 |
| `concat_all_reference__pca_svr` | 0.2402 |
| `arcface_dinov2__pca_svr` | 0.2598 |
| `arcface_convnext__ridge` | ~0.0000 |

Official pair-safe test metrics:

| Metric | Value |
| --- | ---: |
| Pearson r | 0.721594 |
| Spearman rho | 0.745947 |
| MAE | 4.434457 |
| RMSE | 6.198936 |
| R2 | 0.481581 |
| Bias | -1.551855 |
| Within 2 BMI | 0.342246 |
| Within 5 BMI | 0.681818 |

Comparison to paper:

| Model | Overall r |
| --- | ---: |
| Paper VGG-Face + SVR | 0.6500 |
| Our best single model | 0.7104 |
| Our best v4 ensemble | 0.7216 |

Best simple single model:

```text
arcface_dinov2_convnext__ridge
```

Official test metrics:

| Metric | Value |
| --- | ---: |
| Pearson r | 0.710393 |
| Spearman rho | 0.733425 |
| MAE | 4.508252 |
| RMSE | 6.172029 |
| R2 | 0.486072 |
| Bias | -0.959169 |
| Within 2 BMI | 0.324866 |
| Within 5 BMI | 0.655080 |

## Final subgroup results

For best ensemble `slsqp_decorrelated_cap050`:

| Subgroup | n | Pearson r | Spearman rho | MAE | RMSE | Bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Female | 322 | 0.718295 | 0.746392 | 4.369274 | 6.089566 | -0.951158 |
| Male | 426 | 0.726810 | 0.750364 | 4.483726 | 6.280343 | -2.005903 |

Paper subgroup comparison:

| Subgroup | Paper r | Our r |
| --- | ---: | ---: |
| Female | 0.57 | 0.7183 |
| Male | 0.71 | 0.7268 |
| Overall | 0.65 | 0.7216 |

BMI category results for best ensemble:

| BMI category | n | Pearson r | MAE | RMSE | Bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| Healthy | 120 | 0.0695 | 3.5057 | 4.3235 | +3.2103 |
| Overweight | 193 | 0.2017 | 3.0955 | 4.0953 | +1.5592 |
| Obesity 1 | 155 | 0.2462 | 2.5719 | 3.2270 | -1.0158 |
| Obesity 2 | 112 | 0.2412 | 3.7496 | 4.7330 | -2.3867 |
| Obesity 3 | 168 | 0.2544 | 8.8112 | 10.6738 | -8.4654 |

Main remaining failure mode: regression to the mean. The model still overpredicts lower BMI and underpredicts extreme high BMI, especially obesity_3.

## Files in the repo

```text
.gitignore
README.md
PROJECT_PLAN.md
RESULTS.md
HANDOFF.md
requirements.txt

notebooks/facefinalml2_colab_runner.ipynb
notebooks/v4_targeted_arcface_ensemble_cell.py

report/.gitkeep
data/.gitkeep
models/.gitkeep

chatgpt_pro_master_prompt.txt
chatgpt_pro_results_followup_prompt.txt
chatgpt_pro_v2_results_prompt.txt
```

Notes:

- `notebooks/facefinalml2_colab_runner.ipynb` is the large single-cell Colab runner. It evolved to v3 and includes ArcFace/OOF code.
- `notebooks/v4_targeted_arcface_ensemble_cell.py` is the final fast targeted cell that produced the best results.
- `RESULTS.md` contains a concise final result summary.
- `PROJECT_PLAN.md` contains the initial technical plan.
- The `chatgpt_pro_*.txt` files contain prompts used for external planning. They are not needed for final deliverables.

## Private Colab artifacts

Downloaded locally from Google Drive:

```text
../colab_artifacts/facefinalml2_artifacts_20260518_171727.zip
../colab_artifacts/facefinalml2_artifacts_20260518_171727/
```

Extracted artifact files:

```text
face_bmi_pipeline_v3.out
outputs/audit_image_rows_v2_canonical.csv
outputs/split_v2_with_crops.csv
outputs/metrics/model_results_long.csv
outputs/metrics/targeted_v4_model_results_long.csv
outputs/metrics/targeted_v4_ensemble_results_long.csv
outputs/metrics/targeted_v4_summary.json
outputs/metrics/targeted_v4_subgroup_metrics.csv
```

Do **not** push these private artifacts to the public repo without reviewing privacy concerns. They contain image filenames, BMI labels, gender labels, and split assignments.

## What to do next

### Highest priority

Create the final report and presentation from the v4 result.

Report headline:

> On the official available-image, pair-safe split, our ArcFace + DINOv2 + ConvNeXt validation-selected ensemble achieved Pearson r = 0.7216, outperforming the original VGG-Face + SVR benchmark of r = 0.65. The improvement was consistent across gender subgroups, with female r = 0.7183 and male r = 0.7268.

### Report sections to write

1. Introduction and paper baseline.
2. Dataset and audit.
3. Leakage issue found in v1 and repaired protocol.
4. Models:
   - DINOv2
   - ConvNeXt
   - ArcFace
   - fused embeddings
   - Ridge / quantile Ridge / PCA-SVR
   - validation-selected ensemble
5. Results table vs paper.
6. Ablations.
7. Subgroup and error analysis.
8. Regression-to-the-mean limitation.
9. Demo/API.
10. Ethics and limitations.

### Important figures/tables to generate

Use private artifact CSVs to generate:

- Dataset split table.
- Main comparison table.
- Validation/test ablation table.
- Gender subgroup table.
- BMI category residual table.
- Predicted vs true BMI scatter.
- Residual vs true BMI plot.
- BMI distribution by split.
- Prediction distribution vs true distribution.
- Ensemble member/weight table.

### Recommended claim language

Safe strong claim:

> We modestly but clearly outperform the original paper on the provided official available-image subset under pair-safe evaluation.

Avoid:

> This is a production-ready BMI estimator.

Ethical language:

> This is an academic replication and critique. BMI prediction from face images is noisy, privacy-sensitive, and potentially biased. It should not be used for medical, employment, insurance, or personal decisions.

## Known caveats

1. The final evaluation uses the official **available-image subset**, not the full 4,206 rows, because 244 canonical image paths were missing and some images failed verification/filtering.
2. The exact paper split was 3,368 train / 838 test; our clean official available split is smaller after filtering.
3. ArcFace detections are not available for every image; missing embeddings were imputed in the regressors.
4. The ensemble was selected on validation. Report the simple single model too because it is easier to defend and already beats the paper at r = 0.7104.
5. Regression-to-the-mean remains substantial for extreme BMI categories.

## Suggested final result table

| Model | Features | Regressor/ensemble | Test Pearson r | Test Spearman | MAE | RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Paper VGG-Net | VGG-Net fc6 | SVR | 0.470 | - | - | - |
| Paper VGG-Face | VGG-Face fc6 | SVR | 0.650 | - | - | - |
| v2 repaired baseline | DINOv2 + ConvNeXt | weighted ensemble | 0.639 | 0.656 | 5.007 | 6.839 |
| Best single v4 | ArcFace + DINOv2 + ConvNeXt | RidgeCV | 0.710 | 0.733 | 4.508 | 6.172 |
| Best final v4 | ArcFace + DINOv2 + ConvNeXt | validation-selected ensemble | 0.722 | 0.746 | 4.434 | 6.199 |

## Suggested presentation story

1. The original Face-to-BMI paper reached r = 0.65 with VGG-Face + SVR.
2. Our first run got r = 0.647 but had a protocol problem: filename grouping was not pair-safe.
3. We fixed the protocol using exact image mapping, official `is_training`, and pair grouping.
4. Under the repaired protocol, generic DINOv2/ConvNeXt reached r = 0.639.
5. Adding a modern face-recognition embedding, ArcFace, was the key improvement.
6. Final ArcFace + DINOv2 + ConvNeXt ensemble reached r = 0.722.
7. We also improved both gender subgroup correlations relative to the paper.
8. However, the model still regresses to the mean and is not a diagnostic tool.

## If another model continues this work

Start by reading:

1. `README.md`
2. `RESULTS.md`
3. `HANDOFF.md`
4. `notebooks/v4_targeted_arcface_ensemble_cell.py`
5. Private artifact `targeted_v4_summary.json`
6. Private artifact `targeted_v4_model_results_long.csv`
7. Private artifact `targeted_v4_subgroup_metrics.csv`

Do not rerun training unless necessary; the final result is already sufficient for the class project. Focus on report, figures, demo, and presentation.
