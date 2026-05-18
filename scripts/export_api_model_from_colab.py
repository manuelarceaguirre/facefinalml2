"""Export the minimal FaceFinalML2 API model bundle from cached Colab features.

Run this in Colab after feature extraction has produced:
  outputs/features/arcface_buffalo_l_original.npy
  outputs/features/dinov2_vits14_loose.npy
  outputs/features/convnext_loose.npy
  outputs/split_v2_with_crops.csv

It trains a simple deployment model on all non-test rows (train + validation) and
writes models/face_bmi_api_bundle.joblib. The public reported metrics still come
from the held-out v4 evaluation; this bundle is for the live demo/API.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/content/facefinalml2") if Path("/content/facefinalml2").exists() else Path.cwd()
OUT = ROOT / "outputs"
FEATURES = OUT / "features"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

split_path = OUT / "split_v2_with_crops.csv"
if not split_path.exists():
    candidates = sorted(OUT.glob("split*with_crops*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No split-with-crops CSV found under outputs/.")
    split_path = candidates[0]

feature_paths = {
    "arcface": FEATURES / "arcface_buffalo_l_original.npy",
    "dinov2": FEATURES / "dinov2_vits14_loose.npy",
    "convnext": FEATURES / "convnext_loose.npy",
}
missing = [str(p) for p in feature_paths.values() if not p.exists()]
if missing:
    raise FileNotFoundError(f"Missing cached feature files: {missing}")

print("Using split:", split_path)
df = pd.read_csv(split_path).reset_index(drop=True)
X_arc = np.load(feature_paths["arcface"])
X_dino = np.load(feature_paths["dinov2"])
X_conv = np.load(feature_paths["convnext"])
X = np.concatenate([X_arc, X_dino, X_conv], axis=1)
y = df["bmi"].to_numpy(float)

if len(X) != len(df):
    raise ValueError(f"Feature/data length mismatch: X={len(X)} df={len(df)}")

# Deployment fit: use all non-test data. The test set remains untouched for the
# reported numbers in RESULTS.md / HANDOFF.md.
train_mask = ~df["split"].eq("test").to_numpy()
print("Training deployment model on non-test rows:", int(train_mask.sum()))
print("Feature matrix:", X.shape)

model = make_pipeline(
    SimpleImputer(strategy="mean"),
    StandardScaler(),
    RidgeCV(alphas=np.logspace(-4, 4, 41)),
)
model.fit(X[train_mask], y[train_mask])

bundle = {
    "version": "facefinalml2_api_v1_arcface_dinov2_convnext_ridge",
    "kind": "single_sklearn_regressor",
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "feature_set": "arcface_dinov2_convnext",
    "feature_order": ["arcface", "dinov2", "convnext"],
    "feature_dimensions": {"arcface": 512, "dinov2": 384, "convnext": 768, "total": int(X.shape[1])},
    "model": model,
    "training_rows": int(train_mask.sum()),
    "training_source": "train+val rows from official available-image pair-safe split",
    "reported_test_metrics_from_v4_train_only_model": {
        "model": "arcface_dinov2_convnext__ridge",
        "pearson_r": 0.7103934298743756,
        "spearman_rho": 0.7334250736325565,
        "mae": 4.508252471343745,
        "rmse": 6.17202932992031,
        "r2": 0.4860715962759108,
        "bias": -0.959169150823708,
    },
    "ethics_warning": "Academic demo only. Not for medical, employment, insurance, or personal decisions.",
}

out_path = MODELS / "face_bmi_api_bundle.joblib"
joblib.dump(bundle, out_path)
print("Saved:", out_path)
print("Bundle metadata:")
print(json.dumps({k: v for k, v in bundle.items() if k != "model"}, indent=2))
