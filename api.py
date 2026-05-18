"""Minimal real-time Face-to-BMI API.

Run:
    pip install -r requirements.txt
    python scripts/export_api_model_from_colab.py  # in Colab, after features exist
    uvicorn api:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000.

The API expects a private model bundle at models/face_bmi_api_bundle.joblib by
default. The bundle is intentionally not committed to git.
"""

from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

MODEL_PATH = Path(os.getenv("FACEBMI_MODEL", "models/face_bmi_api_bundle.joblib"))

app = FastAPI(title="FaceFinalML2 BMI API", version="1.0")


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bundle not found: {MODEL_PATH}. "
            "Create it in Colab with scripts/export_api_model_from_colab.py and copy it into models/."
        )
    return joblib.load(MODEL_PATH)


class FeatureExtractor:
    """Lazy ArcFace + DINOv2 + ConvNeXt feature extractor.

    The sklearn model handles missing ArcFace detections through its imputer, so
    if no face is detected we return NaNs for ArcFace and still compute generic
    DINOv2/ConvNeXt features on the full image.
    """

    def __init__(self) -> None:
        import torch
        import torchvision.transforms as T
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ArcFace is best when available, but local demo environments may not
        # have insightface/onnxruntime installed. If unavailable, the API still
        # runs by feeding NaNs for ArcFace; the sklearn imputer handles them.
        self.face_app = None
        try:
            from insightface.app import FaceAnalysis

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
            self.face_app = FaceAnalysis(name="buffalo_l", providers=providers)
            self.face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
        except Exception as exc:
            print(f"ArcFace disabled for this demo process: {exc}")

        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(self.device).eval()
        self.convnext = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).features.to(self.device).eval()

        self.transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @staticmethod
    def _loose_crop(image: Image.Image, bbox: np.ndarray, margin: float = 0.25) -> Image.Image:
        w, h = image.size
        x1, y1, x2, y2 = bbox.astype(float)
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(bw, bh) * (1.0 + 2.0 * margin)
        nx1 = max(0, int(round(cx - side / 2)))
        ny1 = max(0, int(round(cy - side / 2)))
        nx2 = min(w, int(round(cx + side / 2)))
        ny2 = min(h, int(round(cy + side / 2)))
        if nx2 <= nx1 or ny2 <= ny1:
            return image
        return image.crop((nx1, ny1, nx2, ny2))

    def _detect_face(self, image: Image.Image) -> Tuple[np.ndarray, Image.Image, bool]:
        if self.face_app is None:
            return np.full(512, np.nan, dtype=np.float32), image, False

        # InsightFace expects BGR uint8.
        rgb = np.asarray(image.convert("RGB"))
        bgr = rgb[:, :, ::-1].copy()
        faces = self.face_app.get(bgr)
        if not faces:
            return np.full(512, np.nan, dtype=np.float32), image, False

        face = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
        arcface = np.asarray(face.embedding, dtype=np.float32)
        crop = self._loose_crop(image, np.asarray(face.bbox))
        return arcface, crop, True

    def _vision_features(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            dino = self.dino(x)
            if isinstance(dino, (tuple, list)):
                dino = dino[0]
            dino = dino.detach().cpu().numpy().reshape(-1).astype(np.float32)

            conv = self.convnext(x)
            conv = torch.nn.functional.adaptive_avg_pool2d(conv, 1).flatten(1)
            conv = conv.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return dino, conv

    def extract(self, image_bytes: bytes) -> Tuple[Dict[str, np.ndarray], bool]:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arcface, crop, face_detected = self._detect_face(image)
        dinov2, convnext = self._vision_features(crop)
        features = {
            "arcface": arcface,
            "dinov2": dinov2,
            "convnext": convnext,
            "arcface_dinov2": np.concatenate([arcface, dinov2]),
            "arcface_convnext": np.concatenate([arcface, convnext]),
            "dinov2_convnext": np.concatenate([dinov2, convnext]),
            "arcface_dinov2_convnext": np.concatenate([arcface, dinov2, convnext]),
        }
        return features, face_detected


@lru_cache(maxsize=1)
def get_extractor() -> FeatureExtractor:
    return FeatureExtractor()


def predict_from_features(bundle: dict, features: Dict[str, np.ndarray]) -> float:
    kind = bundle.get("kind", "single_sklearn_regressor")

    if kind == "single_sklearn_regressor":
        feature_set = bundle["feature_set"]
        x = features[feature_set].reshape(1, -1)
        return float(bundle["model"].predict(x)[0])

    if kind == "weighted_ensemble":
        preds = []
        weights = []
        for member in bundle["members"]:
            x = features[member["feature_set"]].reshape(1, -1)
            preds.append(float(member["model"].predict(x)[0]))
            weights.append(float(member["weight"]))
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
        return float(np.dot(weights, np.asarray(preds, dtype=float)))

    raise ValueError(f"Unsupported model bundle kind: {kind}")


@app.get("/health")
def health() -> dict:
    return {
        "ok": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "message": "ready" if MODEL_PATH.exists() else "missing model bundle",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    try:
        bundle = load_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    image_bytes = await file.read()
    try:
        features, face_detected = get_extractor().extract(image_bytes)
        bmi = predict_from_features(bundle, features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {
        "predicted_bmi": round(bmi, 2),
        "face_detected": face_detected,
        "model_version": bundle.get("version", "unknown"),
        "warning": "Academic demo only. Not for medical or personal decisions.",
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FaceFinalML2 BMI Demo</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; max-width: 760px; margin: 40px auto; padding: 0 18px; line-height: 1.5; }
    h1 { margin-bottom: 4px; }
    .muted { color: #666; }
    .card { border: 1px solid #ddd; padding: 18px; border-radius: 10px; margin: 16px 0; }
    button, input::file-selector-button { padding: 9px 12px; border: 1px solid #222; background: white; border-radius: 7px; cursor: pointer; }
    video, canvas { width: 100%; max-width: 420px; border-radius: 8px; border: 1px solid #ddd; }
    #result { font-size: 22px; font-weight: 700; }
    .small { font-size: 13px; color: #666; }
  </style>
</head>
<body>
  <h1>FaceFinalML2 BMI Demo</h1>
  <p class="muted">Upload a face image or use your webcam. Academic demo only.</p>

  <div class="card">
    <h2>Upload image</h2>
    <input id="file" type="file" accept="image/*">
    <button onclick="predictFile()">Predict BMI</button>
  </div>

  <div class="card">
    <h2>Webcam</h2>
    <p><button onclick="startCam()">Start webcam</button> <button onclick="captureAndPredict()">Capture and predict</button></p>
    <video id="video" autoplay playsinline></video>
    <canvas id="canvas" style="display:none"></canvas>
  </div>

  <div class="card">
    <div id="result">No prediction yet.</div>
    <p id="details" class="small"></p>
  </div>

  <p class="small">Privacy note: this local demo sends the selected image to the running API server only. Do not use for medical, employment, insurance, or personal decisions.</p>

<script>
async function sendBlob(blob, name='image.jpg') {
  const form = new FormData();
  form.append('file', blob, name);
  document.getElementById('result').textContent = 'Predicting... first run may download/load models.';
  document.getElementById('details').textContent = '';
  const res = await fetch('/predict', { method: 'POST', body: form });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Prediction failed');
  document.getElementById('result').textContent = `Predicted BMI: ${data.predicted_bmi}`;
  document.getElementById('details').textContent = `Face detected: ${data.face_detected}. ${data.warning}`;
}

async function predictFile() {
  try {
    const f = document.getElementById('file').files[0];
    if (!f) return alert('Choose an image first.');
    await sendBlob(f, f.name);
  } catch (e) { document.getElementById('result').textContent = e.message; }
}

async function startCam() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  document.getElementById('video').srcObject = stream;
}

async function captureAndPredict() {
  try {
    const video = document.getElementById('video');
    if (!video.videoWidth) return alert('Start webcam first.');
    const canvas = document.getElementById('canvas');
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    canvas.toBlob(blob => sendBlob(blob, 'webcam.jpg').catch(e => document.getElementById('result').textContent = e.message), 'image/jpeg', 0.92);
  } catch (e) { document.getElementById('result').textContent = e.message; }
}
</script>
</body>
</html>
    """
