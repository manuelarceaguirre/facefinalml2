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
