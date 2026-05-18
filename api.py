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

    def _detect_bgr(self, bgr: np.ndarray, max_faces: int = 6) -> dict:
        height, width = bgr.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return {"face_detected": False, "faces": [], "bbox": None, "image_width": width, "image_height": height}

        rows = []
        for face in faces:
            x, y, w, h = [float(v) for v in face[:4]]
            score = float(face[14]) if len(face) > 14 else 1.0
            rows.append({
                "bbox": _bbox_dict_xyxy(np.array([x, y, x + w, y + h], dtype=float), width, height),
                "score": score,
            })
        rows.sort(key=lambda r: r["score"], reverse=True)
        rows = rows[:max_faces]
        return {
            "face_detected": bool(rows),
            "faces": rows,
            "bbox": rows[0]["bbox"] if rows else None,
            "image_width": width,
            "image_height": height,
        }

    def detect(self, image_bytes: bytes, max_faces: int = 6) -> dict:
        cv2 = self.cv2
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image")
        return self._detect_bgr(bgr, max_faces=max_faces)

    def detect_pil(self, image: Image.Image, max_faces: int = 6) -> dict:
        rgb = np.asarray(image.convert("RGB"))
        bgr = rgb[:, :, ::-1].copy()
        return self._detect_bgr(bgr, max_faces=max_faces)


@lru_cache(maxsize=1)
def get_yunet() -> YuNetDetector:
    return YuNetDetector()


class FeatureExtractor:
    """Lazy ArcFace + DINOv2 + ConvNeXt feature extractor."""

    def __init__(self) -> None:
        import torch
        import torchvision.transforms as T
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def _detect_faces(self, image: Image.Image, max_faces: int = 6) -> list[dict]:
        if self.face_app is not None:
            rgb = np.asarray(image.convert("RGB"))
            bgr = rgb[:, :, ::-1].copy()
            faces = self.face_app.get(bgr)
            faces = sorted(faces, key=lambda f: float(getattr(f, "det_score", 0.0)), reverse=True)[:max_faces]
            out = []
            for face in faces:
                raw_bbox = np.asarray(face.bbox)
                out.append({
                    "arcface": np.asarray(face.embedding, dtype=np.float32),
                    "crop": self._loose_crop(image, raw_bbox),
                    "face_detected": True,
                    "bbox": _bbox_dict_xyxy(raw_bbox, image.width, image.height),
                })
            if out:
                return out

        # Fallback: YuNet gives boxes and crops, but ArcFace embedding is missing.
        det = get_yunet().detect_pil(image, max_faces=max_faces)
        out = []
        for row in det.get("faces", []):
            b = row["bbox"]
            raw_bbox = np.array([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]], dtype=float)
            out.append({
                "arcface": np.full(512, np.nan, dtype=np.float32),
                "crop": self._loose_crop(image, raw_bbox),
                "face_detected": False,
                "bbox": b,
            })
        if out:
            return out

        return []

    def _vision_features_batch(self, images: list[Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        x = torch.stack([self.transform(im.convert("RGB")) for im in images]).to(self.device)
        with torch.inference_mode():
            dino = self.dino(x)
            if isinstance(dino, (tuple, list)):
                dino = dino[0]
            dino = dino.detach().cpu().numpy().astype(np.float32)

            conv = self.convnext(x)
            conv = torch.nn.functional.adaptive_avg_pool2d(conv, 1).flatten(1)
            conv = conv.detach().cpu().numpy().astype(np.float32)
        return dino, conv

    def _feature_dict(self, arcface: np.ndarray, dinov2: np.ndarray, convnext: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "arcface": arcface,
            "dinov2": dinov2,
            "convnext": convnext,
            "arcface_dinov2": np.concatenate([arcface, dinov2]),
            "arcface_convnext": np.concatenate([arcface, convnext]),
            "dinov2_convnext": np.concatenate([dinov2, convnext]),
            "arcface_dinov2_convnext": np.concatenate([arcface, dinov2, convnext]),
        }

    def extract_many(self, image_bytes: bytes, max_faces: int = 6) -> list[Tuple[Dict[str, np.ndarray], bool, Optional[dict]]]:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        detected = self._detect_faces(image, max_faces=max_faces)
        if not detected:
            return []
        dino, conv = self._vision_features_batch([d["crop"] for d in detected])
        rows = []
        for i, d in enumerate(detected):
            rows.append((self._feature_dict(d["arcface"], dino[i], conv[i]), bool(d["face_detected"]), d["bbox"]))
        return rows

    def extract(self, image_bytes: bytes) -> Tuple[Dict[str, np.ndarray], bool, Optional[dict]]:
        rows = self.extract_many(image_bytes, max_faces=1)
        if rows:
            return rows[0]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        dino, conv = self._vision_features_batch([image])
        arcface = np.full(512, np.nan, dtype=np.float32)
        return self._feature_dict(arcface, dino[0], conv[0]), False, None


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
        return get_yunet().detect(image_bytes, max_faces=6)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}") from exc


@app.post("/predict_multi")
async def predict_multi(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    try:
        bundle = load_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    image_bytes = await file.read()
    try:
        extracted = get_extractor().extract_many(image_bytes, max_faces=6)
        people = []
        for idx, (features, face_detected, bbox) in enumerate(extracted, start=1):
            bmi = predict_from_features(bundle, features)
            people.append({
                "person_id": idx,
                "predicted_bmi": round(bmi, 2),
                "face_detected": face_detected,
                "bbox": bbox,
            })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {
        "people": people,
        "count": len(people),
        "model_version": bundle.get("version", "unknown"),
        "warning": "Academic demo only. Not for medical or personal decisions.",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    result = await predict_multi(file)
    first = result["people"][0] if result["people"] else {"predicted_bmi": None, "face_detected": False, "bbox": None}
    return {
        "predicted_bmi": first["predicted_bmi"],
        "face_detected": first["face_detected"],
        "bbox": first["bbox"],
        "model_version": result.get("model_version", "unknown"),
        "warning": result["warning"],
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
    :root { --ink:#111; --muted:#666; --line:#ddd; --paper:#fffdf7; }
    * { box-sizing: border-box; }
    body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; max-width: 900px; margin: 32px auto; padding: 0 18px; line-height: 1.45; color: var(--ink); background: #f3f1eb; }
    h1 { margin: 0 0 6px; letter-spacing: -0.03em; }
    .muted, .small { color: var(--muted); }
    .small { font-size: 13px; }
    .card { border: 1px solid var(--line); background: var(--paper); padding: 18px; border-radius: 12px; margin: 16px 0; }
    button { padding: 10px 14px; border: 1px solid #222; background: white; border-radius: 8px; cursor: pointer; font-weight: 650; }
    button.primary { background: var(--ink); color: white; }
    button:disabled { opacity: .55; cursor: wait; }
    .stage { position: relative; width: min(100%, 760px); margin: 14px 0; background: #111; border-radius: 12px; overflow: hidden; border: 1px solid #222; }
    video, canvas.overlay { display: block; width: 100%; height: auto; }
    canvas.overlay { position: absolute; inset: 0; pointer-events: none; }
    #result { font-size: clamp(24px, 5vw, 46px); font-weight: 800; letter-spacing: -0.05em; margin: 8px 0 0; }
    #status { min-height: 20px; }
    .row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
    .pill { font-size: 12px; color: var(--muted); border: 1px solid var(--line); padding: 4px 8px; border-radius: 999px; background: white; }
    .people { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; margin-top: 12px; max-width: 760px; }
    .person { border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; background: white; font-weight: 700; }
    .person span { display: block; color: var(--muted); font-size: 12px; font-weight: 500; }
  </style>
</head>
<body>
  <h1>FaceFinalML2 Webcam BMI Demo</h1>
  <p class="muted">Webcam-only demo. YuNet tracks up to six face boxes in real time; clicking calculate runs the heavier ArcFace + DINOv2 + ConvNeXt BMI model for each detected person.</p>

  <div class="card">
    <div class="row">
      <button onclick="startCam()">Start webcam</button>
      <button id="calcBtn" class="primary" onclick="calculateBMI()">Calculate BMI for everyone</button>
      <span id="trackerPill" class="pill">tracker idle</span>
    </div>

    <div class="stage">
      <video id="video" autoplay playsinline muted></video>
      <canvas id="overlay" class="overlay"></canvas>
    </div>

    <canvas id="capture" style="display:none"></canvas>
    <div id="result">BMI --</div>
    <div id="people" class="people"></div>
    <p id="status" class="small">Start the webcam, get everyone in frame, then press calculate.</p>
  </div>

  <p class="small">Academic demo only. BMI prediction from face images is noisy, privacy-sensitive, and not for medical or personal decisions.</p>

<script>
let tracking = false;
let busyDetect = false;
let faces = [];
let predictions = [];

const colors = ['#00ff88', '#00c2ff', '#ffcc00', '#ff5c8a', '#b26cff', '#ff7a00'];
const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const capture = document.getElementById('capture');
const result = document.getElementById('result');
const statusEl = document.getElementById('status');
const trackerPill = document.getElementById('trackerPill');
const calcBtn = document.getElementById('calcBtn');
const peopleEl = document.getElementById('people');

async function startCam() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 1280, height: 720 }, audio: false });
  video.srcObject = stream;
  await video.play();
  resizeOverlay();
  tracking = true;
  statusEl.textContent = 'Tracking up to six faces. Press calculate when ready.';
  requestAnimationFrame(drawLoop);
  setInterval(trackFaces, 220);
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

async function trackFaces() {
  if (!tracking || busyDetect || !video.videoWidth) return;
  busyDetect = true;
  try {
    const blob = await frameBlob(0.55);
    const form = new FormData();
    form.append('file', blob, 'frame.jpg');
    const res = await fetch('/detect', { method: 'POST', body: form });
    const data = await res.json();
    if (res.ok) {
      faces = (data.faces || []).slice(0, 6).map((f, i) => ({...f, person_id: i + 1}));
      trackerPill.textContent = faces.length ? `${faces.length} face${faces.length === 1 ? '' : 's'} tracked` : 'no faces';
    }
  } catch (e) {
    trackerPill.textContent = 'tracker error';
  } finally {
    busyDetect = false;
  }
}

function bmiForPerson(id) {
  const p = predictions.find(x => x.person_id === id);
  return p ? p.predicted_bmi : null;
}

function drawLoop() {
  resizeOverlay();
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const drawFaces = predictions.length ? predictions : faces;
  drawFaces.slice(0, 6).forEach((row, i) => {
    const b = row.bbox || (row.bbox === null ? null : row.bbox);
    if (!b) return;
    const id = row.person_id || (i + 1);
    const color = colors[(id - 1) % colors.length];
    const bmi = row.predicted_bmi ?? bmiForPerson(id);
    const label = bmi !== null ? `P${id} BMI ${bmi}` : `P${id}`;

    ctx.lineWidth = Math.max(3, overlay.width / 240);
    ctx.strokeStyle = color;
    ctx.strokeRect(b.x, b.y, b.w, b.h);

    ctx.font = `${Math.max(18, overlay.width / 34)}px system-ui, sans-serif`;
    const pad = 8;
    const metrics = ctx.measureText(label);
    const boxH = Math.max(30, overlay.width / 26);
    const labelY = Math.max(0, b.y - boxH - 6);
    ctx.fillStyle = 'rgba(0,0,0,0.78)';
    ctx.fillRect(b.x, labelY, metrics.width + pad * 2, boxH);
    ctx.fillStyle = color;
    ctx.fillText(label, b.x + pad, labelY + boxH - 9);
  });

  requestAnimationFrame(drawLoop);
}

function renderPeople() {
  if (!predictions.length) { peopleEl.innerHTML = ''; return; }
  peopleEl.innerHTML = predictions.map((p, i) => `
    <div class="person" style="border-color:${colors[i % colors.length]}">
      Person ${p.person_id}: BMI ${p.predicted_bmi}
      <span>ArcFace detected: ${p.face_detected}</span>
    </div>`).join('');
}

async function calculateBMI() {
  try {
    if (!video.videoWidth) return alert('Start webcam first.');
    calcBtn.disabled = true;
    calcBtn.textContent = 'Calculating...';
    result.textContent = 'BMI ...';
    peopleEl.innerHTML = '';
    predictions = [];
    statusEl.textContent = 'Running ArcFace + DINOv2 + ConvNeXt for up to six people. This can take several seconds on CPU.';

    const blob = await frameBlob(0.92);
    const form = new FormData();
    form.append('file', blob, 'webcam.jpg');
    const res = await fetch('/predict_multi', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Prediction failed');

    predictions = (data.people || []).slice(0, 6);
    if (!predictions.length) {
      result.textContent = 'BMI --';
      statusEl.textContent = 'No faces detected. Try better lighting and center everyone in frame.';
      return;
    }
    result.textContent = predictions.map(p => `P${p.person_id}: ${p.predicted_bmi}`).join('   ');
    statusEl.textContent = `${predictions.length} prediction${predictions.length === 1 ? '' : 's'} complete. ${data.warning}`;
    renderPeople();
  } catch (e) {
    statusEl.textContent = e.message;
    result.textContent = 'BMI --';
  } finally {
    calcBtn.disabled = false;
    calcBtn.textContent = 'Calculate BMI for everyone';
  }
}

window.addEventListener('resize', resizeOverlay);
</script>
</body>
</html>
    """
