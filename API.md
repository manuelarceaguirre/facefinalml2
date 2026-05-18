# Minimal FaceFinalML2 Web API

This is the simplest functional API/demo for the final model family.

## What it serves

- `GET /` webcam-only browser demo.
- `POST /detect` lightweight YuNet CPU face-box tracking endpoint.
- `POST /predict` heavier BMI endpoint using ArcFace when available, plus DINOv2 and ConvNeXt.
- `GET /health` model-bundle status.

The API uses a private sklearn deployment bundle:

```text
models/face_bmi_api_bundle.joblib
```

This file is not committed because model/data artifacts are ignored.

## Model used

For deployment simplicity, the API uses the strongest simple model family:

```text
ArcFace + DINOv2 + ConvNeXt features -> RidgeCV BMI regressor
```

The evaluated v4 single-model test result was:

```text
Pearson r:    0.7104
Spearman rho: 0.7334
MAE:          4.5083
RMSE:         6.1720
```

The final report also includes the validation-selected ensemble result `r = 0.7216`, but this API intentionally avoids extra complexity and uses the simpler deployment model.

## Create the model bundle in Colab

After the feature extraction pipeline has produced cached `.npy` features, run:

```bash
python scripts/export_api_model_from_colab.py
```

Expected inputs:

```text
outputs/features/arcface_buffalo_l_original.npy
outputs/features/dinov2_vits14_loose.npy
outputs/features/convnext_loose.npy
outputs/split_v2_with_crops.csv
```

Expected output:

```text
models/face_bmi_api_bundle.joblib
```

Copy that `.joblib` file wherever the API will run.

## Run locally

```bash
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000
```

The page is webcam-only: click **Start webcam**, center your face, then click **Calculate BMI**. The green box is tracked with YuNet for responsiveness; the BMI calculation uses the heavier model.

## Call the BMI endpoint

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/face.jpg"
```

Example response:

```json
{
  "predicted_bmi": 31.42,
  "face_detected": true,
  "bbox": {"x": 140.0, "y": 80.0, "w": 180.0, "h": 180.0},
  "model_version": "facefinalml2_api_v1_arcface_dinov2_convnext_ridge",
  "warning": "Academic demo only. Not for medical or personal decisions."
}
```

## Important limitation

This is an academic demo only. BMI prediction from face images is noisy, privacy-sensitive, and potentially biased. Do not use it for medical, employment, insurance, or personal decisions.
