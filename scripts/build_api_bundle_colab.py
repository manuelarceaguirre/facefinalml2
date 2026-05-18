"""Build the minimal FaceFinalML2 API model bundle from scratch in Colab.

This is for the case where the previous Colab runtime was disconnected and the
cached feature files were lost.

What it does:
  1. Installs only the packages needed for the API model bundle.
  2. Authenticates to Google Drive and downloads the BMI zip by file ID.
  3. Extracts the dataset.
  4. Rebuilds the official available-image split table.
  5. Extracts ArcFace, DINOv2, and ConvNeXt features.
  6. Trains the minimal deployment RidgeCV model.
  7. Saves models/face_bmi_api_bundle.joblib.
  8. Copies the bundle to Google Drive and triggers a browser download.

Run in Colab as a notebook cell, not as a shell-only Python process:
  !git clone https://github.com/manuelarceaguirre/facefinalml2.git /content/facefinalml2
  %cd /content/facefinalml2
  %run scripts/build_api_bundle_colab.py

Important: use `%run`, not `!python`, because Google Colab's interactive
Drive authentication requires the IPython kernel context.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path("/content/facefinalml2") if Path("/content").exists() else Path.cwd()
DRIVE_FILE_ID = "16XA-MCnTG8ONdgxK0uPfFXWnA5oF3bFa"
ZIP_PATH = ROOT / "data/raw/BMI.zip"
EXTRACT_DIR = ROOT / "data/extracted"
OUT = ROOT / "outputs"
FEATURES = OUT / "features"
MODELS = ROOT / "models"


def run(cmd: list[str], check: bool = True) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=check)


def in_colab() -> bool:
    return Path("/content").exists()


def install_deps() -> None:
    # Keep this minimal. Torch/torchvision are usually already present in Colab.
    pkgs = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "joblib",
        "pillow",
        "tqdm",
        "opencv-python-headless",
        "insightface",
        "onnxruntime-gpu",
        "google-api-python-client",
    ]
    run([sys.executable, "-m", "pip", "install", "-q", *pkgs])


def download_drive_file() -> None:
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 100_000_000:
        print("BMI zip already exists:", ZIP_PATH)
        return

    if not in_colab():
        raise RuntimeError("Google Drive download helper expects Colab.")

    from google.colab import auth
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    print("Authenticating Google Drive...")
    auth.authenticate_user()
    service = build("drive", "v3")

    meta = service.files().get(fileId=DRIVE_FILE_ID, fields="id,name,size").execute()
    print("Downloading:", meta)

    request = service.files().get_media(fileId=DRIVE_FILE_ID)
    with io.FileIO(str(ZIP_PATH), "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=64 * 1024 * 1024)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"download {status.progress() * 100:.1f}%", flush=True)

    print("Downloaded:", ZIP_PATH, ZIP_PATH.stat().st_size, "bytes")


def extract_zip() -> Path:
    marker = EXTRACT_DIR / ".extract_done"
    if marker.exists():
        data_csvs = list(EXTRACT_DIR.rglob("data.csv"))
        if data_csvs:
            print("Dataset already extracted:", EXTRACT_DIR)
            return data_csvs[0]

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    print("Extracting zip. This can take a few minutes...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))

    data_csvs = list(EXTRACT_DIR.rglob("data.csv"))
    if not data_csvs:
        raise FileNotFoundError("Could not find data.csv after extraction.")
    print("Found metadata:", data_csvs[0])
    return data_csvs[0]


def norm_col(c: str) -> str:
    return str(c).strip().lower().replace(" ", "_")


def parse_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    return str(x).strip().lower() in {"true", "1", "yes", "y", "train", "training"}


def build_split(data_csv: Path):
    import pandas as pd
    from PIL import Image
    from tqdm.auto import tqdm

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_csv)
    original_cols = list(df.columns)
    df.columns = [norm_col(c) for c in df.columns]

    row_col = None
    for c in df.columns:
        if c.startswith("unnamed"):
            row_col = c
            break
    if row_col is None:
        row_col = "row_id"
        df[row_col] = range(len(df))

    required = ["bmi", "gender", "is_training", "name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Original columns: {original_cols}; normalized: {list(df.columns)}")

    image_dir = data_csv.parent / "Images"
    rows = []
    print("Verifying exact canonical image paths...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        path = image_dir / str(row["name"])
        if not path.exists():
            continue
        try:
            with Image.open(path) as im:
                im.verify()
        except Exception:
            continue
        row_id = int(row[row_col])
        is_train = parse_bool(row["is_training"])
        rows.append({
            "source_index": int(idx),
            "row_id": row_id,
            "pair_id": row_id // 2,
            "bmi": float(row["bmi"]),
            "gender": row["gender"],
            "is_training": is_train,
            "name": row["name"],
            "image_path": str(path),
            "split": "train" if is_train else "test",
        })

    clean = pd.DataFrame(rows).reset_index(drop=True)
    out_path = OUT / "split_v2_with_crops.csv"
    clean.to_csv(out_path, index=False)
    print("Saved split:", out_path)
    print("Rows:", clean.shape)
    print("Split counts:", clean["split"].value_counts().to_dict())
    print("Pair groups:", clean.groupby("split")["pair_id"].nunique().to_dict())
    return clean


class Extractor:
    def __init__(self):
        import torch
        import torchvision.transforms as T
        from insightface.app import FaceAnalysis
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Feature device:", self.device)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        self.face_app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))

        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(self.device).eval()
        self.convnext = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).features.to(self.device).eval()

        self.transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @staticmethod
    def loose_crop(image, bbox, margin=0.25):
        import numpy as np

        w, h = image.size
        x1, y1, x2, y2 = np.asarray(bbox, dtype=float)
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

    def detect_and_crop(self, image):
        import numpy as np

        rgb = np.asarray(image.convert("RGB"))
        bgr = rgb[:, :, ::-1].copy()
        faces = self.face_app.get(bgr)
        if not faces:
            return np.full(512, np.nan, dtype=np.float32), image, False
        face = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
        arc = np.asarray(face.embedding, dtype=np.float32)
        crop = self.loose_crop(image, face.bbox)
        return arc, crop, True

    def embed_vision_batch(self, images, batch_size=64):
        import numpy as np
        import torch
        from tqdm.auto import tqdm

        dino_all = []
        conv_all = []
        for start in tqdm(range(0, len(images), batch_size), desc="DINOv2/ConvNeXt batches"):
            batch = images[start:start + batch_size]
            x = torch.stack([self.transform(im.convert("RGB")) for im in batch]).to(self.device)
            with torch.inference_mode():
                dino = self.dino(x)
                if isinstance(dino, (tuple, list)):
                    dino = dino[0]
                conv = self.convnext(x)
                conv = torch.nn.functional.adaptive_avg_pool2d(conv, 1).flatten(1)
            dino_all.append(dino.detach().cpu().numpy().astype(np.float32))
            conv_all.append(conv.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(dino_all, axis=0), np.concatenate(conv_all, axis=0)


def extract_features(clean):
    import numpy as np
    from PIL import Image
    from tqdm.auto import tqdm

    FEATURES.mkdir(parents=True, exist_ok=True)
    paths = {
        "arcface": FEATURES / "arcface_buffalo_l_original.npy",
        "dinov2": FEATURES / "dinov2_vits14_loose.npy",
        "convnext": FEATURES / "convnext_loose.npy",
    }
    if all(p.exists() and len(np.load(p, mmap_mode="r")) == len(clean) for p in paths.values()):
        print("Cached features already exist.")
        return {k: np.load(v) for k, v in paths.items()}

    extractor = Extractor()
    arcfaces = []
    crops = []
    face_detected = []

    print("Detecting faces and extracting ArcFace embeddings...")
    for p in tqdm(clean["image_path"].tolist()):
        img = Image.open(p).convert("RGB")
        arc, crop, ok = extractor.detect_and_crop(img)
        arcfaces.append(arc)
        crops.append(crop)
        face_detected.append(ok)

    X_arc = np.vstack(arcfaces).astype(np.float32)
    X_dino, X_conv = extractor.embed_vision_batch(crops, batch_size=64)

    np.save(paths["arcface"], X_arc)
    np.save(paths["dinov2"], X_dino)
    np.save(paths["convnext"], X_conv)
    clean = clean.copy()
    clean["face_detected"] = face_detected
    clean.to_csv(OUT / "split_v2_with_crops.csv", index=False)

    print("Saved features:")
    for k, p in paths.items():
        x = np.load(p, mmap_mode="r")
        print(" ", k, x.shape, p)
    return {"arcface": X_arc, "dinov2": X_dino, "convnext": X_conv}


def train_bundle(clean, feats):
    import joblib
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    MODELS.mkdir(parents=True, exist_ok=True)
    X = np.concatenate([feats["arcface"], feats["dinov2"], feats["convnext"]], axis=1)
    y = clean["bmi"].to_numpy(float)
    train_mask = clean["split"].eq("train").to_numpy()

    model = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-4, 4, 41)),
    )
    print("Training RidgeCV deployment model:", X[train_mask].shape)
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
        "training_source": "official training rows from available-image pair-safe split",
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

    out = MODELS / "face_bmi_api_bundle.joblib"
    joblib.dump(bundle, out)
    print("Saved bundle:", out, out.stat().st_size, "bytes")
    return out


def save_to_drive_and_download(bundle_path: Path) -> None:
    if not in_colab():
        return
    from google.colab import drive, files

    drive.mount("/content/drive")
    drive_dir = Path("/content/drive/MyDrive/facefinalml2")
    drive_dir.mkdir(parents=True, exist_ok=True)
    dst = drive_dir / bundle_path.name
    shutil.copy2(bundle_path, dst)
    print("Saved to Drive:", dst)
    print("Triggering browser download...")
    files.download(str(bundle_path))


def main() -> None:
    os.chdir(ROOT)
    install_deps()
    download_drive_file()
    data_csv = extract_zip()
    clean = build_split(data_csv)
    feats = extract_features(clean)
    bundle = train_bundle(clean, feats)
    save_to_drive_and_download(bundle)
    print("DONE. Copy this file into the repo's models/ directory:", bundle)


if __name__ == "__main__":
    main()
