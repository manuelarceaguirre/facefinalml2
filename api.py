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
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

MODEL_PATH = Path(os.getenv("FACEBMI_MODEL", "models/face_bmi_api_bundle.joblib"))
YUNET_PATH = Path(os.getenv("FACEBMI_YUNET", "models/face_detection_yunet_2023mar.onnx"))
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

app = FastAPI(title="FaceFinalML2 BMI API", version="1.0")


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bundle not found: {MODEL_PATH}. "
            "Create it in Colab with scripts/export_api_model_from_colab.py and copy it into models/."
        )
    bundle = joblib.load(MODEL_PATH)
    _patch_sklearn_compat(bundle)
    return bundle


def _patch_sklearn_compat(obj) -> None:
    """Patch small sklearn pickle incompatibilities across minor versions.

    The demo bundle was exported in Colab with sklearn 1.6.1. Some local
    environments may run newer sklearn versions whose SimpleImputer expects a
    private `_fill_dtype` attribute. Reconstruct it from `_fit_dtype` so the
    public demo remains runnable without forcing an exact sklearn pin.
    """
    if isinstance(obj, dict):
        for value in obj.values():
            _patch_sklearn_compat(value)
        return
    steps = getattr(obj, "steps", None)
    if steps is not None:
        for _, step in steps:
            _patch_sklearn_compat(step)
    if obj.__class__.__name__ == "SimpleImputer" and not hasattr(obj, "_fill_dtype"):
        obj._fill_dtype = getattr(obj, "_fit_dtype", np.float64)


def _bbox_dict_xyxy(bbox: np.ndarray, width: int, height: int) -> dict:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    return {"x": x1, "y": y1, "w": max(0.0, x2 - x1), "h": max(0.0, y2 - y1)}


class YuNetDetector:
    """Small CPU face detector for real-time webcam box tracking."""

    def __init__(self) -> None:
        import cv2

        self.cv2 = cv2
        YUNET_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not YUNET_PATH.exists():
            print(f"Downloading YuNet face detector to {YUNET_PATH}")
            urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
        self.detector = cv2.FaceDetectorYN_create(
            str(YUNET_PATH),
            "",
            (320, 320),
            score_threshold=0.7,
            nms_threshold=0.3,
            top_k=5000,
        )

    def detect(self, image_bytes: bytes) -> dict:
        cv2 = self.cv2
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image")
        height, width = bgr.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return {"face_detected": False, "bbox": None, "image_width": width, "image_height": height}
        face = max(faces, key=lambda f: float(f[14]) if len(f) > 14 else float(f[2] * f[3]))
        x, y, w, h = [float(v) for v in face[:4]]
        bbox = _bbox_dict_xyxy(np.array([x, y, x + w, y + h], dtype=float), width, height)
        return {"face_detected": True, "bbox": bbox, "image_width": width, "image_height": height}


@lru_cache(maxsize=1)
def get_yunet() -> YuNetDetector:
    return YuNetDetector()


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

    def _detect_face(self, image: Image.Image) -> Tuple[np.ndarray, Image.Image, bool, Optional[dict]]:
        if self.face_app is None:
            return np.full(512, np.nan, dtype=np.float32), image, False, None

        # InsightFace expects BGR uint8.
        rgb = np.asarray(image.convert("RGB"))
        bgr = rgb[:, :, ::-1].copy()
        faces = self.face_app.get(bgr)
        if not faces:
            return np.full(512, np.nan, dtype=np.float32), image, False, None

        face = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
        raw_bbox = np.asarray(face.bbox)
        arcface = np.asarray(face.embedding, dtype=np.float32)
        crop = self._loose_crop(image, raw_bbox)
        bbox = _bbox_dict_xyxy(raw_bbox, image.width, image.height)
        return arcface, crop, True, bbox

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

    def extract(self, image_bytes: bytes) -> Tuple[Dict[str, np.ndarray], bool, Optional[dict]]:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arcface, crop, face_detected, bbox = self._detect_face(image)
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
        return features, face_detected, bbox


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


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")
    image_bytes = await file.read()
    try:
        return get_yunet().detect(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}") from exc


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
        features, face_detected, bbox = get_extractor().extract(image_bytes)
        bmi = predict_from_features(bundle, features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {
        "predicted_bmi": round(bmi, 2),
        "face_detected": face_detected,
        "bbox": bbox,
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
  <title>FaceFinalML2 Webcam BMI Demo</title>
  <style>
    :root { --ink:#111; --muted:#666; --line:#ddd; --accent:#1f6b45; --paper:#fffdf7; }
    * { box-sizing: border-box; }
    body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; max-width: 860px; margin: 32px auto; padding: 0 18px; line-height: 1.45; color: var(--ink); background: #f3f1eb; }
    h1 { margin: 0 0 6px; letter-spacing: -0.03em; }
    .muted, .small { color: var(--muted); }
    .small { font-size: 13px; }
    .card { border: 1px solid var(--line); background: var(--paper); padding: 18px; border-radius: 12px; margin: 16px 0; }
    button { padding: 10px 14px; border: 1px solid #222; background: white; border-radius: 8px; cursor: pointer; font-weight: 650; }
    button.primary { background: var(--ink); color: white; }
    button:disabled { opacity: .55; cursor: wait; }
    .stage { position: relative; width: min(100%, 720px); margin: 14px 0; background: #111; border-radius: 12px; overflow: hidden; border: 1px solid #222; }
    video, canvas.overlay { display: block; width: 100%; height: auto; }
    canvas.overlay { position: absolute; inset: 0; pointer-events: none; }
    #result { font-size: clamp(30px, 6vw, 54px); font-weight: 800; letter-spacing: -0.05em; margin: 8px 0 0; }
    #status { min-height: 20px; }
    .row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
    .pill { font-size: 12px; color: var(--muted); border: 1px solid var(--line); padding: 4px 8px; border-radius: 999px; background: white; }
  </style>
</head>
<body>
  <h1>FaceFinalML2 Webcam BMI Demo</h1>
  <p class="muted">Webcam-only demo. YuNet tracks the face box in real time; clicking calculate runs the heavier ArcFace + DINOv2 + ConvNeXt BMI model.</p>

  <div class="card">
    <div class="row">
      <button onclick="startCam()">Start webcam</button>
      <button id="calcBtn" class="primary" onclick="calculateBMI()">Calculate BMI</button>
      <span id="trackerPill" class="pill">tracker idle</span>
    </div>

    <div class="stage">
      <video id="video" autoplay playsinline muted></video>
      <canvas id="overlay" class="overlay"></canvas>
    </div>

    <canvas id="capture" style="display:none"></canvas>
    <div id="result">BMI --</div>
    <p id="status" class="small">Start the webcam, center your face in the frame, then press calculate.</p>
  </div>

  <p class="small">Academic demo only. BMI prediction from face images is noisy, privacy-sensitive, and not for medical or personal decisions.</p>

<script>
let stream = null;
let tracking = false;
let busyDetect = false;
let lastBox = null;
let lastBMI = null;
let lastFaceDetected = false;

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const capture = document.getElementById('capture');
const result = document.getElementById('result');
const statusEl = document.getElementById('status');
const trackerPill = document.getElementById('trackerPill');
const calcBtn = document.getElementById('calcBtn');

async function startCam() {
  stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 1280, height: 720 }, audio: false });
  video.srcObject = stream;
  await video.play();
  resizeOverlay();
  tracking = true;
  statusEl.textContent = 'Tracking face box. Press calculate when ready.';
  requestAnimationFrame(drawLoop);
  setInterval(trackFace, 220);
}

function resizeOverlay() {
  if (!video.videoWidth) return;
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
}

function frameBlob(quality = 0.75) {
  if (!video.videoWidth) throw new Error('Webcam is not ready.');
  capture.width = video.videoWidth;
  capture.height = video.videoHeight;
  capture.getContext('2d').drawImage(video, 0, 0, capture.width, capture.height);
  return new Promise(resolve => capture.toBlob(resolve, 'image/jpeg', quality));
}

async function trackFace() {
  if (!tracking || busyDetect || !video.videoWidth) return;
  busyDetect = true;
  try {
    const blob = await frameBlob(0.55);
    const form = new FormData();
    form.append('file', blob, 'frame.jpg');
    const res = await fetch('/detect', { method: 'POST', body: form });
    const data = await res.json();
    if (res.ok && data.face_detected) {
      lastBox = data.bbox;
      trackerPill.textContent = 'face tracked';
    } else {
      lastBox = null;
      trackerPill.textContent = 'no face box';
    }
  } catch (e) {
    trackerPill.textContent = 'tracker error';
  } finally {
    busyDetect = false;
  }
}

function drawLoop() {
  resizeOverlay();
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (lastBox) {
    const {x, y, w, h} = lastBox;
    ctx.lineWidth = Math.max(3, overlay.width / 240);
    ctx.strokeStyle = '#00ff88';
    ctx.strokeRect(x, y, w, h);

    const label = lastBMI !== null ? `BMI ${lastBMI}` : 'face';
    ctx.font = `${Math.max(20, overlay.width / 28)}px system-ui, sans-serif`;
    const pad = 8;
    const metrics = ctx.measureText(label);
    const boxH = Math.max(32, overlay.width / 22);
    const labelY = Math.max(0, y - boxH - 6);
    ctx.fillStyle = 'rgba(0,0,0,0.78)';
    ctx.fillRect(x, labelY, metrics.width + pad * 2, boxH);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x + pad, labelY + boxH - 10);
  }

  requestAnimationFrame(drawLoop);
}

async function calculateBMI() {
  try {
    if (!video.videoWidth) return alert('Start webcam first.');
    calcBtn.disabled = true;
    calcBtn.textContent = 'Calculating...';
    result.textContent = 'BMI ...';
    statusEl.textContent = 'Running ArcFace + DINOv2 + ConvNeXt. This can take a few seconds on CPU.';

    const blob = await frameBlob(0.92);
    const form = new FormData();
    form.append('file', blob, 'webcam.jpg');
    const res = await fetch('/predict', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Prediction failed');

    lastBMI = data.predicted_bmi;
    lastFaceDetected = data.face_detected;
    if (data.bbox) lastBox = data.bbox;
    result.textContent = `BMI ${data.predicted_bmi}`;
    statusEl.textContent = `ArcFace face detected: ${data.face_detected}. ${data.warning}`;
  } catch (e) {
    statusEl.textContent = e.message;
    result.textContent = 'BMI --';
  } finally {
    calcBtn.disabled = false;
    calcBtn.textContent = 'Calculate BMI';
  }
}

window.addEventListener('resize', resizeOverlay);
</script>
</body>
</html>
    """
