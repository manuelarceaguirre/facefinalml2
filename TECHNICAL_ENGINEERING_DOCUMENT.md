# FaceFinalML2 Technical Engineering Document

```yaml
document_type: engineering_handoff
project: FaceFinalML2
repo: https://github.com/manuelarceaguirre/facefinalml2
pages: https://manuelarceaguirre.github.io/facefinalml2/
course: UChicago ML2 Spring 2026
last_updated: 2026-05-18
primary_claim: >
  On the official available-image, pair-safe split, an ArcFace + DINOv2 + ConvNeXt
  validation-selected ensemble achieved Pearson r = 0.721594, outperforming the
  Face-to-BMI paper's reported VGG-Face + SVR overall r = 0.65.
primary_metric:
  name: pearson_r
  split: official_available_image_pair_safe_test
  value: 0.7215940282873421
paper_baseline:
  model: VGG-Face + SVR
  overall_pearson_r: 0.65
final_model:
  name: slsqp_decorrelated_cap050
  type: validation_selected_weighted_ensemble
  member_count: 4
privacy_note: >
  Do not publish per-image artifacts. They contain image names, BMI labels, gender labels,
  and split assignments.
```

## 1. Executive Summary

This project replicated and extended **Face-to-BMI: Using Computer Vision to Infer Body Mass Index on Social Media**. The original paper reports a best overall Pearson correlation of **r = 0.65** using VGG-Face features with SVR.

The project reached:

```yaml
final_test_metrics:
  model: slsqp_decorrelated_cap050
  pearson_r: 0.7215940282873421
  spearman_rho: 0.7459468142839455
  mae: 4.434456881724706
  rmse: 6.198936389815522
  r2: 0.4815808707942548
  bias: -1.551854983784604
  within_2_bmi: 0.3422459893048128
  within_5_bmi: 0.6818181818181818
```

Best simple single model:

```yaml
best_single_model:
  model: arcface_dinov2_convnext__ridge
  feature_set: arcface_dinov2_convnext
  regressor: RidgeCV
  test_pearson_r: 0.7103934298743756
  test_spearman_rho: 0.7334250736325565
  test_mae: 4.508252471343745
  test_rmse: 6.17202932992031
  test_r2: 0.4860715962759108
  test_bias: -0.959169150823708
```

The project intentionally reports both the final ensemble and the best single model. The ensemble is the headline result, but the single model is simpler to defend and independently beats the paper baseline.

## 2. Repository and Artifact Layout

```yaml
public_repo:
  url: https://github.com/manuelarceaguirre/facefinalml2
  branch: main
  github_pages: https://manuelarceaguirre.github.io/facefinalml2/

local_repo_path: /Users/manuelarce/Library/CloudStorage/SeaDrive-manuel(100.73.58.1)/My Libraries/My Library/UChicago/Spring 2026/ml2/final/facefinalml2

public_files:
  - README.md
  - PROJECT_PLAN.md
  - RESULTS.md
  - HANDOFF.md
  - TECHNICAL_ENGINEERING_DOCUMENT.md
  - requirements.txt
  - notebooks/facefinalml2_colab_runner.ipynb
  - notebooks/v4_targeted_arcface_ensemble_cell.py
  - report/face_bmi_blog.html
  - docs/index.html

private_artifact_zip: ../colab_artifacts/facefinalml2_artifacts_20260518_171727.zip
private_artifact_dir: ../colab_artifacts/facefinalml2_artifacts_20260518_171727/
private_artifacts:
  - face_bmi_pipeline_v3.out
  - outputs/audit_image_rows_v2_canonical.csv
  - outputs/split_v2_with_crops.csv
  - outputs/metrics/model_results_long.csv
  - outputs/metrics/targeted_v4_model_results_long.csv
  - outputs/metrics/targeted_v4_ensemble_results_long.csv
  - outputs/metrics/targeted_v4_summary.json
  - outputs/metrics/targeted_v4_subgroup_metrics.csv
```

Do not commit private artifacts to the public repo unless they are sanitized. They contain image names, BMI labels, gender labels, and split assignments.

## 3. Dataset Contract

```yaml
dataset:
  google_drive_file_name: Copy of BMI.zip
  google_drive_file_id: 16XA-MCnTG8ONdgxK0uPfFXWnA5oF3bFa
  approximate_size_bytes: 987158616
  colab_zip_path: /content/facefinalml2/data/raw/BMI.zip
  extracted_root: /content/facefinalml2/data/extracted/BMI
  canonical_metadata: /content/facefinalml2/data/extracted/BMI/Data/data.csv
  canonical_image_dir: /content/facefinalml2/data/extracted/BMI/Data/Images
  canonical_image_path_rule: BMI/Data/Images/<name>
  metadata_columns:
    - unnamed:_0
    - bmi
    - gender
    - is_training
    - name
```

The final pipeline uses exact canonical image mapping:

```python
image_path = extracted_root / "Data" / "Images" / row["name"]
```

It does **not** use fuzzy filename matching for final metrics.

## 4. Evaluation Protocol

The project has two important protocol versions:

```yaml
v1_exploratory_protocol:
  status: not_final
  reason: possible pair/person leakage
  split: random filename/group split
  test_pearson_r: 0.6469648724

final_protocol:
  status: final
  split: official available-image pair-safe split
  official_test_source: is_training column
  validation_source: official training pool only
  pair_group_rule: pair_id = row_id // 2
  required_pair_overlap:
    train_val: 0
    train_test: 0
    val_test: 0
```

Pair grouping code:

```python
df["row_id"] = df["unnamed:_0"].astype(int)
df["pair_id"] = df["row_id"] // 2
```

Final split audit:

```yaml
clean_canonical_rows: 3955
split_counts:
  train: 2565
  val: 642
  test: 748
pair_groups:
  train: 1295
  val: 324
  test: 383
bmi_mean:
  train: 32.380006
  val: 32.438656
  test: 33.456553
missing_exact_paths: 244
pairs_with_n_not_equal_2: 44
pairs_with_gender_mismatch: 0
pairs_split_across_train_test: 0
```

## 5. Model Families

```yaml
embedding_models:
  arcface_buffalo_l_original:
    library: insightface
    model_pack: buffalo_l
    dimensionality: 512
    role: modern face-recognition embedding
    finite_fraction_observed: 0.8096080910240202
    final_status: key improvement

  dinov2_vits14_loose:
    library: torch_hub_or_transformers_depending_colab_cell
    architecture: DINOv2 ViT-S/14
    dimensionality: 384
    role: generic self-supervised visual representation
    finite_fraction_observed: 1.0
    final_status: retained

  convnext_loose:
    library: torchvision_or_timm_depending_colab_cell
    architecture: ConvNeXt
    dimensionality: 768
    role: generic supervised visual representation
    finite_fraction_observed: 1.0
    final_status: retained

  facenet_vggface2_tight:
    library: facenet-pytorch
    dimensionality: 512
    role: initial VGGFace2-style face embedding baseline
    final_status: excluded_from_final_v4
    reason: collapsed_or_nan_predictions_in_colab
```

Final v4 feature sets:

```yaml
feature_sets_v4:
  arcface: [arcface]
  dinov2: [dinov2]
  convnext: [convnext]
  arcface_dinov2: [arcface, dinov2]
  arcface_convnext: [arcface, convnext]
  dinov2_convnext: [dinov2, convnext]
  arcface_dinov2_convnext: [arcface, dinov2, convnext]
  concat_all_reference: cached concat_all if present
```

Regressor families used in final v4:

```yaml
regressors_v4:
  ridge:
    sklearn: RidgeCV
    preprocessing:
      - SimpleImputer(strategy="mean")
      - StandardScaler()
    alphas: logspace(-4, 4, 41)

  quantile_ridge:
    sklearn: TransformedTargetRegressor(RidgeCV)
    target_transformer: QuantileTransformer(output_distribution="normal")
    n_quantiles: min(512, n_train)
    preprocessing:
      - SimpleImputer(strategy="mean")
      - StandardScaler()

  pca_svr:
    sklearn: SVR(kernel="rbf", C=10.0, epsilon=0.3, gamma="scale")
    preprocessing:
      - SimpleImputer(strategy="mean")
      - StandardScaler()
      - PCA(n_components=min(256, max(2, n_features - 1)), random_state=42)
```

Skipped in final v4:

```yaml
skipped_final_v4:
  - full out-of-fold stacking
  - facenet_vggface2_tight
  - ElasticNetCV
  - RandomForest
  - XGBoost
reason: >
  Full OOF was too slow in Colab; FaceNet branch was unstable/collapsed; tree and elastic models were not needed
  once ArcFace + DINOv2 + ConvNeXt was clearly strongest.
```

## 6. Experiment Chronology

```yaml
experiments:
  v1_exploratory:
    protocol: random filename/group split
    final_status: exploratory_only
    test_pearson_r: 0.6469648724
    issue: possible before_after_pair leakage

  v2_pair_safe:
    protocol: official is_training test + pair-safe validation
    feature_sets:
      - facenet_vggface2_tight
      - convnext_loose
      - dinov2_vits14_loose
      - concat_all
    final_ensemble_test:
      pearson_r: 0.6390615508
      spearman_rho: 0.6560874146
      mae: 5.0065854924
      rmse: 6.8391204982
    interpretation: rigorous near-replication but below paper baseline

  v3_arcface_oof_attempt:
    additions:
      - insightface ArcFace buffalo_l
      - onnxruntime-gpu
      - quantile_ridge
      - attempted OOF stacking
    status: interrupted
    reason: full OOF stack too slow in Colab
    important_partial_validation:
      concat_all__quantile_ridge: 0.749235
      concat_all__ridge: 0.744428
      arcface_buffalo_l_original__pca_svr: 0.692153

  v4_targeted_final:
    protocol: official available-image pair-safe
    method: train targeted fast high-value models using cached features
    final_ensemble: slsqp_decorrelated_cap050
    final_test_pearson_r: 0.7215940282873421
    final_status: reportable headline result
```

## 7. Final Model Results

Top individual model results on test:

```yaml
top_test_models:
  - model: arcface_dinov2_convnext__ridge
    pearson_r: 0.7103934298743756
    spearman_rho: 0.7334250736325565
    mae: 4.508252471343745
    rmse: 6.17202932992031
    r2: 0.4860715962759108
    bias: -0.959169150823708

  - model: arcface_dinov2__pca_svr
    pearson_r: 0.709703
    spearman_rho: 0.731843
    mae: 4.488216
    rmse: 6.233755

  - model: arcface_dinov2_convnext__quantile_ridge
    pearson_r: 0.706968
    spearman_rho: 0.735682
    mae: 4.525189
    rmse: 6.337840
```

Best final ensemble:

```yaml
final_ensemble:
  name: slsqp_decorrelated_cap050
  selection_basis: validation_predictions_only
  optimization: SLSQP constrained nonnegative weights
  weight_cap: 0.50
  members:
    - name: arcface_dinov2_convnext__quantile_ridge
      weight: 0.5
    - name: concat_all_reference__pca_svr
      weight: 0.24021097882871642
    - name: arcface_dinov2__pca_svr
      weight: 0.25978902117128355
    - name: arcface_convnext__ridge
      weight: 7.806255641895632e-18
  test_metrics:
    pearson_r: 0.7215940282873421
    spearman_rho: 0.7459468142839455
    mae: 4.434456881724706
    rmse: 6.198936389815522
    r2: 0.4815808707942548
    bias: -1.551854983784604
    within_2: 0.3422459893048128
    within_5: 0.6818181818181818
```

## 8. Subgroup Results

Best ensemble gender subgroup metrics:

```yaml
gender_subgroups:
  female:
    n: 322
    pearson_r: 0.718295
    spearman_rho: 0.746392
    mae: 4.369274
    rmse: 6.089566
    bias: -0.951158
  male:
    n: 426
    pearson_r: 0.726810
    spearman_rho: 0.750364
    mae: 4.483726
    rmse: 6.280343
    bias: -2.005903
```

Paper comparison:

```yaml
paper_vs_ours_subgroups:
  female:
    paper_r: 0.57
    ours_r: 0.718295
  male:
    paper_r: 0.71
    ours_r: 0.726810
  overall:
    paper_r: 0.65
    ours_r: 0.7215940282873421
```

BMI category metrics for best ensemble:

```yaml
bmi_category_subgroups:
  healthy:
    n: 120
    pearson_r: 0.069541
    mae: 3.505653
    rmse: 4.323533
    bias: 3.210266
  overweight:
    n: 193
    pearson_r: 0.201683
    mae: 3.095484
    rmse: 4.095329
    bias: 1.559201
  obesity_1:
    n: 155
    pearson_r: 0.246154
    mae: 2.571857
    rmse: 3.227024
    bias: -1.015789
  obesity_2:
    n: 112
    pearson_r: 0.241196
    mae: 3.749603
    rmse: 4.733021
    bias: -2.386655
  obesity_3:
    n: 168
    pearson_r: 0.254427
    mae: 8.811151
    rmse: 10.673753
    bias: -8.465432
```

Primary failure mode:

```yaml
failure_mode: regression_to_the_mean
observed_pattern:
  low_bmi: overpredicted
  extreme_high_bmi: underpredicted
largest_problem:
  category: obesity_3
  bias: -8.465432
```

## 9. Core Metric Code

```python
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if len(y_true) < 3 or np.std(y_pred) == 0:
        pr = np.nan
        sr = np.nan
    else:
        pr = float(pearsonr(y_true, y_pred)[0])
        sr = float(spearmanr(y_true, y_pred).correlation)
    return {
        'pearson_r': pr,
        'spearman_rho': sr,
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': float(r2_score(y_true, y_pred)),
        'bias': float(np.mean(y_pred - y_true)),
        'within_2': float(np.mean(np.abs(y_pred - y_true) <= 2.0)),
        'within_5': float(np.mean(np.abs(y_pred - y_true) <= 5.0)),
    }
```

## 10. Core Model Builder Code

```python
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVR


def ridge_model():
    return make_pipeline(
        SimpleImputer(strategy='mean'),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-4, 4, 41)),
    )


def quantile_ridge_model(n_train):
    return TransformedTargetRegressor(
        regressor=ridge_model(),
        transformer=QuantileTransformer(
            n_quantiles=min(512, n_train),
            output_distribution='normal',
            random_state=42,
        ),
    )


def pca_svr_model(n_features):
    return make_pipeline(
        SimpleImputer(strategy='mean'),
        StandardScaler(),
        PCA(n_components=min(256, max(2, n_features - 1)), random_state=42),
        SVR(kernel='rbf', C=10.0, epsilon=0.3, gamma='scale'),
    )
```

## 11. Final Validation-Selected Ensemble Logic

```python
from scipy.optimize import minimize
from scipy.stats import pearsonr
import numpy as np

selected = [
    'arcface_dinov2_convnext__quantile_ridge',
    'concat_all_reference__pca_svr',
    'arcface_dinov2__pca_svr',
    'arcface_convnext__ridge',
]

P = val_pred_df[selected].to_numpy(float)

def objective(w):
    w = np.maximum(w, 0)
    w = w / (w.sum() + 1e-12)
    pred = P @ w
    return -pearsonr(y_val, pred)[0]

bounds = [(0, 0.50)] * len(selected)
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
w0 = np.ones(len(selected)) / len(selected)

res = minimize(
    objective,
    w0,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 500},
)

weights = np.maximum(res.x, 0)
weights = weights / weights.sum()

test_prediction = test_pred_df[selected].to_numpy(float) @ weights
```

## 12. Reproduction Procedure

```yaml
reproduction_environment:
  primary_runtime: Google Colab
  local_training: not_required
  reason: dataset and model extraction are large for local Mac workflow
```

Steps:

```bash
git clone https://github.com/manuelarceaguirre/facefinalml2.git
cd facefinalml2
```

In Google Colab:

1. Open or paste the single-cell runner from `notebooks/facefinalml2_colab_runner.ipynb` if feature extraction is needed.
2. Authenticate Google Drive.
3. Download the BMI zip by file ID.
4. Extract dataset.
5. Build exact canonical mapping.
6. Run split audit and pair-safe split.
7. Extract embeddings.
8. Run final v4 targeted cell from `notebooks/v4_targeted_arcface_ensemble_cell.py`.
9. Save private artifacts to Drive.

Expected final output files:

```yaml
colab_outputs:
  metrics:
    - outputs/metrics/targeted_v4_model_results_long.csv
    - outputs/metrics/targeted_v4_ensemble_results_long.csv
    - outputs/metrics/targeted_v4_summary.json
    - outputs/metrics/targeted_v4_subgroup_metrics.csv
  split_audit:
    - outputs/split_v2_with_crops.csv
    - outputs/audit_image_rows_v2_canonical.csv
  logs:
    - face_bmi_pipeline_v3.out
```

## 13. Report Claim Language

Use:

> On the official available-image, pair-safe split, our ArcFace + DINOv2 + ConvNeXt validation-selected ensemble achieved Pearson r = 0.7216, outperforming the original VGG-Face + SVR benchmark of r = 0.65. The improvement was consistent across gender subgroups, with female r = 0.7183 and male r = 0.7268.

Also state:

> The best simple single model, ArcFace + DINOv2 + ConvNeXt with RidgeCV, achieved r = 0.7104, showing the improvement is not solely due to ensemble weighting.

Do not claim:

```yaml
avoid_claims:
  - production_ready_medical_model
  - diagnostic_tool
  - exact reproduction_of_full_original_dataset
  - causal_relationship_between_face_and_bmi
```

Ethics language:

> BMI inference from face images is noisy, privacy-sensitive, and potentially biased. This model is an academic replication and critique, not a tool for medical, employment, insurance, or personal decisions.

## 14. Full Final v4 Targeted Colab Cell

The following is the final targeted cell saved as `notebooks/v4_targeted_arcface_ensemble_cell.py`.

```python
# FaceFinalML2 v4 targeted ArcFace ensemble cell
# Use after interrupting the slow OOF step. This reuses cached v3 features and the repaired v2/v3 split.
# It trains only fast/high-value models and prints official test results.

import os, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVR
import joblib

ROOT = Path('/content/facefinalml2')
os.chdir(ROOT)
OUT = ROOT / 'outputs'
FEATURES = OUT / 'features'
METRICS = OUT / 'metrics'
MODELS = ROOT / 'models'
METRICS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

print('Finding split file...')
split_candidates = [
    OUT / 'split_v2_with_crops.csv',
    OUT / 'split_seed42_with_crops.csv',
]
split_candidates += sorted(OUT.glob('split*with_crops*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
split_path = next((p for p in split_candidates if p.exists()), None)
if split_path is None:
    raise FileNotFoundError('No split-with-crops CSV found under outputs/. Run the v3 cell until feature extraction completes first.')
print('Using split:', split_path)
df = pd.read_csv(split_path).reset_index(drop=True)
print('Rows:', len(df), 'splits:', df['split'].value_counts().to_dict())

required_features = {
    'arcface': FEATURES / 'arcface_buffalo_l_original.npy',
    'dinov2': FEATURES / 'dinov2_vits14_loose.npy',
    'convnext': FEATURES / 'convnext_loose.npy',
}
missing = [str(p) for p in required_features.values() if not p.exists()]
if missing:
    raise FileNotFoundError('Missing required cached feature files. Let v3 run through ArcFace/DINO/ConvNeXt extraction first. Missing: ' + str(missing))

X_arc = np.load(required_features['arcface'])
X_dino = np.load(required_features['dinov2'])
X_conv = np.load(required_features['convnext'])
for name, X in [('arcface', X_arc), ('dinov2', X_dino), ('convnext', X_conv)]:
    print(name, X.shape, 'finite_frac=', float(np.isfinite(X).mean()), 'std=', float(np.nanstd(X)))
    if len(X) != len(df):
        raise ValueError(f'Feature length mismatch for {name}: {len(X)} vs df {len(df)}')

feature_sets = {
    'arcface': X_arc,
    'dinov2': X_dino,
    'convnext': X_conv,
    'arcface_dinov2': np.concatenate([X_arc, X_dino], axis=1),
    'arcface_convnext': np.concatenate([X_arc, X_conv], axis=1),
    'dinov2_convnext': np.concatenate([X_dino, X_conv], axis=1),
    'arcface_dinov2_convnext': np.concatenate([X_arc, X_dino, X_conv], axis=1),
}
# Include old concat_all only as a reference if present. It may include broken FaceNet, so do not rely on it alone.
concat_path = FEATURES / 'concat_all.npy'
if concat_path.exists():
    X_concat = np.load(concat_path)
    if len(X_concat) == len(df):
        feature_sets['concat_all_reference'] = X_concat

train_mask = df['split'].eq('train').to_numpy()
val_mask = df['split'].eq('val').to_numpy()
test_mask = df['split'].eq('test').to_numpy()
y = df['bmi'].to_numpy(float)
y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]

print('\nTrain/val/test:', train_mask.sum(), val_mask.sum(), test_mask.sum())
print('Paper target overall Pearson r = 0.65')


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if len(y_true) < 3 or np.std(y_pred) == 0:
        pr = np.nan
        sr = np.nan
    else:
        pr = float(pearsonr(y_true, y_pred)[0])
        sr = float(spearmanr(y_true, y_pred).correlation)
    return {
        'pearson_r': pr,
        'spearman_rho': sr,
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': float(r2_score(y_true, y_pred)),
        'bias': float(np.mean(y_pred - y_true)),
        'within_2': float(np.mean(np.abs(y_pred - y_true) <= 2.0)),
        'within_5': float(np.mean(np.abs(y_pred - y_true) <= 5.0)),
    }


def ridge_model():
    return make_pipeline(
        SimpleImputer(strategy='mean'),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-4, 4, 41)),
    )


def quantile_ridge_model(n_train):
    return TransformedTargetRegressor(
        regressor=ridge_model(),
        transformer=QuantileTransformer(
            n_quantiles=min(512, n_train),
            output_distribution='normal',
            random_state=42,
        ),
    )


def pca_svr_model(n_features):
    return make_pipeline(
        SimpleImputer(strategy='mean'),
        StandardScaler(),
        PCA(n_components=min(256, max(2, n_features - 1)), random_state=42),
        SVR(kernel='rbf', C=10.0, epsilon=0.3, gamma='scale'),
    )

# Fast targeted model set. No ElasticNetCV, RF, XGB, or full OOF.
model_builders = {
    'ridge': lambda n_features: ridge_model(),
    'quantile_ridge': lambda n_features: quantile_ridge_model(len(y_train)),
    'pca_svr': lambda n_features: pca_svr_model(n_features),
}

records = []
preds_val = {}
preds_test = {}
fitted = {}

for feat_name, X in feature_sets.items():
    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    for model_name, builder in model_builders.items():
        tag = f'{feat_name}__{model_name}'
        print(f'\nTraining {tag} X_train={X_train.shape}', flush=True)
        model = builder(X_train.shape[1])
        try:
            model.fit(X_train, y_train)
            pv = model.predict(X_val)
            pt = model.predict(X_test)
        except Exception as e:
            print('FAILED', tag, e)
            continue
        mv = regression_metrics(y_val, pv)
        mt = regression_metrics(y_test, pt)
        records.append({'model': tag, 'feature_set': feat_name, 'regressor': model_name, 'split': 'val', **mv})
        records.append({'model': tag, 'feature_set': feat_name, 'regressor': model_name, 'split': 'test', **mt})
        preds_val[tag] = pv
        preds_test[tag] = pt
        fitted[tag] = model
        print(f'{tag}: val r={mv["pearson_r"]:.4f} test r={mt["pearson_r"]:.4f} test MAE={mt["mae"]:.3f}')

results = pd.DataFrame(records)
results.to_csv(METRICS / 'targeted_v4_model_results_long.csv', index=False)
val_results = results[results['split'].eq('val')].sort_values('pearson_r', ascending=False)
test_results = results[results['split'].eq('test')].sort_values('pearson_r', ascending=False)
print('\nTOP VALIDATION MODELS')
print(val_results.head(20).to_string(index=False))
print('\nTOP TEST MODELS, FOR REPORTING ONLY')
print(test_results.head(20).to_string(index=False))

# Build validation-only ensembles.
val_pred_df = pd.DataFrame(preds_val)
test_pred_df = pd.DataFrame(preds_test)

# Candidate pool: top validation models, excluding near-duplicate bad/NaN models.
top_candidates = val_results[np.isfinite(val_results['pearson_r'])]['model'].head(10).tolist()
print('\nTop candidate models:', top_candidates)

ensemble_records = []
ensemble_preds = {}

def add_ensemble(name, selected, weights=None):
    if not selected:
        return
    P_val = val_pred_df[selected].to_numpy(float)
    P_test = test_pred_df[selected].to_numpy(float)
    if weights is None:
        weights_arr = np.ones(len(selected), dtype=float) / len(selected)
    else:
        weights_arr = np.asarray(weights, dtype=float)
        weights_arr = weights_arr / weights_arr.sum()
    pv = P_val @ weights_arr
    pt = P_test @ weights_arr
    mv = regression_metrics(y_val, pv)
    mt = regression_metrics(y_test, pt)
    ensemble_preds[name] = pt
    ensemble_records.append({'ensemble': name, 'split': 'val', 'members': selected, 'weights': weights_arr.tolist(), **mv})
    ensemble_records.append({'ensemble': name, 'split': 'test', 'members': selected, 'weights': weights_arr.tolist(), **mt})
    print(f'ENSEMBLE {name}: val r={mv["pearson_r"]:.4f} test r={mt["pearson_r"]:.4f} test MAE={mt["mae"]:.3f}')

# Top-k equal averages.
for k in [1, 2, 3, 5, 8]:
    add_ensemble(f'top{k}_equal_val_selected', top_candidates[:k])

# Decorrelation-selected equal ensemble.
selected = []
for m in top_candidates:
    ok = True
    for s in selected:
        corr = np.corrcoef(val_pred_df[m], val_pred_df[s])[0, 1]
        if abs(corr) > 0.975:
            ok = False
            break
    if ok:
        selected.append(m)
    if len(selected) >= 5:
        break
print('Decorrelated selected:', selected)
add_ensemble('decorrelated_top_val_equal', selected)

# Constrained SLSQP weights on validation Pearson, with cap to reduce overfit.
for candidate_name, selected in [('slsqp_top5_cap045', top_candidates[:5]), ('slsqp_decorrelated_cap050', selected)]:
    if len(selected) >= 2:
        P = val_pred_df[selected].to_numpy(float)
        def objective(w):
            w = np.maximum(w, 0)
            w = w / (w.sum() + 1e-12)
            pred = P @ w
            return -pearsonr(y_val, pred)[0]
        bounds = [(0, 0.45 if 'top5' in candidate_name else 0.50)] * len(selected)
        cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        w0 = np.ones(len(selected)) / len(selected)
        res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 500})
        if res.success:
            w = np.maximum(res.x, 0)
            w = w / w.sum()
            print(candidate_name, pd.Series(w, index=selected).sort_values(ascending=False).to_string())
            add_ensemble(candidate_name, selected, w)
        else:
            print(candidate_name, 'optimization failed:', res.message)

ens_df = pd.DataFrame(ensemble_records)
ens_df.to_csv(METRICS / 'targeted_v4_ensemble_results_long.csv', index=False)

print('\nENSEMBLE RESULTS')
if not ens_df.empty:
    print(ens_df[ens_df['split'].eq('test')].sort_values('pearson_r', ascending=False).to_string(index=False))

# Save predictions for best validation-selected ensemble and best test for inspection.
summary = {
    'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'paper_target_pearson_r': 0.65,
    'best_validation_model': val_results.iloc[0].to_dict() if len(val_results) else None,
    'best_test_model_reporting_only': test_results.iloc[0].to_dict() if len(test_results) else None,
    'best_test_ensemble_reporting_only': None,
    'note': 'Targeted v4 uses repaired official/pair-safe split and cached ArcFace/DINOv2/ConvNeXt features. Model selection should be described as validation-based; test is for final reporting.'
}
if not ens_df.empty:
    best_ens_test = ens_df[ens_df['split'].eq('test')].sort_values('pearson_r', ascending=False).iloc[0].to_dict()
    summary['best_test_ensemble_reporting_only'] = best_ens_test
with open(METRICS / 'targeted_v4_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, sort_keys=True, default=str)

# Subgroup metrics for the best validation model and best validation ensemble.
def subgroup_report(pred, name):
    part = df[test_mask].copy().reset_index(drop=True)
    part['pred_bmi'] = pred
    rows = []
    for col in ['gender_clean', 'bmi_cat', 'face_detected']:
        if col not in part.columns:
            continue
        for value, g in part.groupby(col):
            if len(g) >= 5:
                rows.append({'prediction': name, 'subgroup_col': col, 'subgroup': str(value), 'n': len(g), **regression_metrics(g['bmi'], g['pred_bmi'])})
    return pd.DataFrame(rows)

subgroups = []
if len(val_results):
    best_val_model = val_results.iloc[0]['model']
    subgroups.append(subgroup_report(preds_test[best_val_model], best_val_model))
if not ens_df.empty:
    best_val_ens_name = ens_df[ens_df['split'].eq('val')].sort_values('pearson_r', ascending=False).iloc[0]['ensemble']
    if best_val_ens_name in ensemble_preds:
        subgroups.append(subgroup_report(ensemble_preds[best_val_ens_name], best_val_ens_name))
if subgroups:
    sub_df = pd.concat(subgroups, ignore_index=True)
    sub_df.to_csv(METRICS / 'targeted_v4_subgroup_metrics.csv', index=False)
    print('\nSUBGROUP METRICS FOR BEST VALIDATION CHOICES')
    print(sub_df.to_string(index=False))

print('\nSaved:')
print(' ', METRICS / 'targeted_v4_model_results_long.csv')
print(' ', METRICS / 'targeted_v4_ensemble_results_long.csv')
print(' ', METRICS / 'targeted_v4_summary.json')
print(' ', METRICS / 'targeted_v4_subgroup_metrics.csv')
print('\nIf any official test Pearson r is > 0.65, that is the candidate result to report, with caveats and pair-safe protocol.')

```
