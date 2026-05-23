# Backend and Streamlit Integration Handoff

This document explains how the FaceFinalML2 backend works, how we merged Rui Gong's Streamlit demo frontend, and how to run the complete demo locally.

## Repositories

Original Streamlit demo frontend:

```text
https://github.com/RuiGong01/ML2_final_face-to-bmi-streamlit-demo
```

Integrated final project repository:

```text
https://github.com/manuelarceaguirre/facefinalml2
```

## High-level architecture

The final demo has two layers:

1. **FastAPI backend** in `api.py`
   - owns all model inference
   - loads the private BMI model bundle
   - extracts features with ArcFace, DINOv2, and ConvNeXt
   - tracks faces with YuNet
   - optionally enrolls/recognizes names with ArcFace embeddings

2. **Streamlit frontend** in `app.py`
   - adapted from Rui Gong's Streamlit app
   - provides the polished report/demo layout
   - embeds the live FastAPI webcam page as the primary experience
   - keeps upload/camera snapshot/sample image support as fallback
   - talks to the backend through `predictor.py`

The live webcam UI is served by FastAPI at `GET /`. Streamlit embeds that page in an iframe, so the demo keeps our multi-person live webcam behavior while still using the teammate's Streamlit presentation shell.

## Files added or modified for the merge

```text
api.py                         FastAPI model server and live webcam UI
app.py                         Streamlit frontend adapted from Rui's repo
predictor.py                   Adapter from Streamlit to FastAPI backend
API.md                         API usage notes
demo_samples/                  Backup demo images from Rui's frontend
models/face_bmi_api_bundle.joblib   Private local model bundle, not committed
models/known_faces.joblib           Optional local name-recognition registry, not committed
```

`app.py`, `predictor.py`, and `demo_samples/` came from Rui's Streamlit demo structure and were adapted to call our live backend rather than a placeholder predictor.

## Backend endpoints

### `GET /health`

Checks backend readiness.

Example:

```bash
curl http://127.0.0.1:8000/health
```

Example response:

```json
{
  "ok": true,
  "model_path": "models/face_bmi_api_bundle.joblib",
  "known_faces_path": "models/known_faces.joblib",
  "known_faces_loaded": true,
  "message": "ready"
}
```

### `GET /`

Serves the live webcam demo page.

Features:

- mirrored webcam view
- YuNet face boxes for up to six people
- stable client-side track IDs so colors do not swap between people
- optional live name recognition
- BMI calculation on button click
- BMI labels rendered over the moving face boxes

### `POST /detect`

Lightweight real-time detector endpoint. It uses OpenCV YuNet on CPU and returns up to six face boxes. This is intentionally separate from BMI prediction so the webcam stays responsive.

Input: multipart image upload named `file`.

Example:

```bash
curl -X POST http://127.0.0.1:8000/detect \
  -F "file=@frame.jpg"
```

### `POST /recognize_multi`

Runs ArcFace recognition only, not BMI prediction. The browser calls this periodically while the webcam is running so enrolled names can appear before pressing the BMI button.

Input: multipart image upload named `file`.

Output: up to six people with optional names.

### `POST /predict_multi`

Runs the heavier BMI model for up to six people. This is called when the user presses **Calculate BMI for everyone**.

For each detected person, the backend:

1. detects/crops the face with InsightFace/ArcFace when available
2. extracts ArcFace embedding
3. extracts DINOv2 features
4. extracts ConvNeXt features
5. concatenates the frozen feature vectors
6. predicts BMI with the trained Ridge model
7. optionally recognizes the person from `models/known_faces.joblib`

### `POST /predict`

Backwards-compatible single-person endpoint. Used by the Streamlit still-image fallback through `predictor.py`.

### `POST /enroll`

Enrolls a person for local name recognition.

Input:

- `name`: form field
- `file`: webcam image/frame

The backend extracts an ArcFace embedding and appends it to `models/known_faces.joblib`. Raw enrollment images are not stored by this endpoint.

## Model bundle

The backend expects:

```text
models/face_bmi_api_bundle.joblib
```

This file is private and is not committed to GitHub. It contains the trained deployment model:

```text
ArcFace + DINOv2 + ConvNeXt features -> RidgeCV BMI regressor
```

The evaluated single-model test performance for this model family was:

```text
Pearson r:    0.7104
Spearman rho: 0.7334
MAE:          4.5083
RMSE:         6.1720
```

The final report ensemble reached Pearson `r = 0.7216`, but the deployed API uses the simpler Ridge model because it is easier to package and still beats the original paper's `r = 0.65` target.

## Name recognition registry

Optional name recognition uses:

```text
models/known_faces.joblib
```

This file is also private and ignored by git. It stores ArcFace embeddings for enrolled people. It does not need ground-truth BMI values.

There are two enrollment methods:

1. **Webcam enrollment in the live demo**
   - type a name
   - press **Capture for recognition**
   - the backend saves the embedding

2. **Folder-based enrollment**
   - place consented images under `known_faces/<Person Name>/`
   - run:

```bash
python scripts/enroll_known_faces.py
```

## How we merged Rui's Streamlit frontend

Rui's repo provided a polished Streamlit app structure with:

- app header
- styled panels/cards
- upload input
- webcam snapshot input
- sample images
- result card
- model summary cards
- reference-paper section

We copied/adapted the Streamlit layer into this repo as:

```text
app.py
predictor.py
demo_samples/
```

Then we changed the predictor boundary:

- Rui's frontend expected a local/pluggable predictor.
- We replaced that with `predictor.py`, which sends images to our FastAPI backend.
- We made the live webcam API page the primary Streamlit content by embedding `http://127.0.0.1:8000/` inside the Streamlit app.
- We kept Rui's upload/webcam snapshot/demo sample interface as a fallback expander.

This preserves the polished Streamlit presentation while keeping our original live API behavior.

## Local setup

From the repo root:

```bash
git clone https://github.com/manuelarceaguirre/facefinalml2.git
cd facefinalml2
python3 -m pip install -r requirements.txt
```

Make sure the private model bundle exists:

```text
models/face_bmi_api_bundle.joblib
```

If missing, build it in Colab with:

```bash
python scripts/build_api_bundle_colab.py
```

or download the saved bundle from Google Drive into `models/`.

## Run the full demo locally

Use two terminals.

### Terminal 1: start the FastAPI backend

```bash
cd facefinalml2
uvicorn api:app --host 127.0.0.1 --port 8000
```

Check health:

```bash
curl http://127.0.0.1:8000/health
```

Open the raw live API page if desired:

```text
http://127.0.0.1:8000/
```

### Terminal 2: start the Streamlit frontend

```bash
cd facefinalml2
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open:

```text
http://127.0.0.1:8501
```

The Streamlit app embeds the FastAPI webcam page and should be used for the final presentation.

## Environment variables

Optional variables:

```bash
FACEBMI_API_URL="http://127.0.0.1:8000"       # Streamlit -> FastAPI endpoint
FACEBMI_MODEL="models/face_bmi_api_bundle.joblib"
FACEBMI_KNOWN_FACES="models/known_faces.joblib"
FACEBMI_RECOGNITION_THRESHOLD="0.32"
FACEBMI_API_TIMEOUT="120"
```

## Demo workflow

1. Start FastAPI.
2. Start Streamlit.
3. Open `http://127.0.0.1:8501`.
4. Click **Start webcam** in the embedded live panel.
5. If doing name recognition:
   - type a name
   - click **Capture for recognition**
   - wait for the live overlay to recognize the person automatically
6. Put up to six people in frame.
7. Click **Calculate BMI for everyone**.
8. The overlay should show stable colored boxes and BMI/name labels.

## Troubleshooting

### Backend health says model missing

Copy the model bundle into:

```text
models/face_bmi_api_bundle.joblib
```

### Streamlit loads but embedded webcam does not work

Open the backend directly:

```text
http://127.0.0.1:8000/
```

If the browser blocks camera access inside the Streamlit iframe, use the backend page directly for the live webcam demo.

### Names do not appear until BMI is calculated

The current backend should call `/recognize_multi` periodically from the browser. Check that `models/known_faces.joblib` exists and that `/health` reports:

```json
"known_faces_loaded": true
```

### Face boxes swap colors between people

The browser includes a simple tracker that matches detections by IoU and center distance. If people cross or overlap heavily, IDs can still swap. Ask people to stand separated for the cleanest demo.

## Privacy and ethics note

This is an academic demo. BMI prediction from face images is noisy, privacy-sensitive, and potentially biased. It should not be used for medical, employment, insurance, or personal decisions. Enrollment for name recognition should only be done with consent or public/demo-safe images.
