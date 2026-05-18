"""Enroll known faces for the local classroom demo.

Input layout:
  known_faces/
    Person Name/
      image1.jpg
      image2.png

Output:
  models/known_faces.joblib

The output stores ArcFace embeddings, not raw images. Use only with consent or
for public/demo-safe images.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
KNOWN_DIR = ROOT / "known_faces"
OUT_PATH = ROOT / "models/known_faces.joblib"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def load_face_app():
    import torch
    from insightface.app import FaceAnalysis

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
    return app


def iter_people():
    for person_dir in sorted(KNOWN_DIR.iterdir()):
        if not person_dir.is_dir() or person_dir.name.startswith("."):
            continue
        images = [p for p in sorted(person_dir.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
        if images:
            yield person_dir.name, images


def main() -> None:
    if not KNOWN_DIR.exists():
        raise FileNotFoundError(f"Missing {KNOWN_DIR}")

    people_inputs = list(iter_people())
    if not people_inputs:
        raise SystemExit(
            "No enrollment images found. Add images under known_faces/<Person Name>/ first."
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    face_app = load_face_app()
    enrolled = []

    for name, paths in people_inputs:
        embeddings = []
        print(f"Enrolling {name}: {len(paths)} image(s)")
        for path in paths:
            try:
                rgb = np.asarray(Image.open(path).convert("RGB"))
                bgr = rgb[:, :, ::-1].copy()
                faces = face_app.get(bgr)
                if not faces:
                    print(f"  skip no face: {path}")
                    continue
                face = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
                embeddings.append(normalize(face.embedding))
                print(f"  ok: {path}")
            except Exception as exc:
                print(f"  skip error {path}: {exc}")

        if embeddings:
            enrolled.append({
                "name": name,
                "embeddings": np.vstack(embeddings).astype(np.float32),
                "n_images": len(embeddings),
            })

    if not enrolled:
        raise SystemExit("No faces could be enrolled.")

    bundle = {
        "version": "facefinalml2_known_faces_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "threshold_default": 0.32,
        "people": enrolled,
    }
    joblib.dump(bundle, OUT_PATH)
    print(f"Saved {len(enrolled)} people to {OUT_PATH}")
    for person in enrolled:
        print(f"  {person['name']}: {person['n_images']} embedding(s)")


if __name__ == "__main__":
    main()
