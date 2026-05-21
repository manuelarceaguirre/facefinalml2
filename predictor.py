from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import requests
from PIL import Image


class PredictionError(Exception):
    """Raised when an input image cannot be processed for prediction."""


class PredictorUnavailableError(Exception):
    """Raised when the live FastAPI predictor cannot be reached."""


@dataclass(frozen=True)
class PredictionResult:
    bmi: float
    category: str
    model_name: str
    metrics: dict[str, Any]
    warning: str | None = None


@dataclass(frozen=True)
class Predictor:
    model_name: str
    status_label: str
    metrics: dict[str, Any]
    api_url: str
    timeout_seconds: float
    is_placeholder: bool = False
    health: dict[str, Any] | None = None


API_URL = os.getenv("FACEBMI_API_URL", "http://localhost:8000").rstrip("/")
API_TIMEOUT_SECONDS = float(os.getenv("FACEBMI_API_TIMEOUT", "120"))

DEPLOYED_MODEL_METRICS = {
    "pearson_r": 0.7104,
    "spearman_rho": 0.7334,
    "mae": 4.5083,
    "rmse": 6.1720,
    "test_n": 748,
}

MODEL_SUMMARY = [
    {
        "Model": "Paper VGG-Face + SVR",
        "Pearson r": 0.650,
        "MAE": "Not reported",
        "RMSE": "Not reported",
        "Test N": 838,
        "Status": "Paper baseline",
    },
    {
        "Model": "Deployed API: ArcFace + DINOv2 + ConvNeXt Ridge",
        "Pearson r": 0.710,
        "MAE": 4.508,
        "RMSE": 6.172,
        "Test N": 748,
        "Status": "Live API model",
    },
    {
        "Model": "Best validation-selected ensemble",
        "Pearson r": 0.722,
        "MAE": 4.435,
        "RMSE": 6.199,
        "Test N": 748,
        "Status": "Final report result",
    },
]


def _api_url(path: str, base_url: str = API_URL) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def load_predictor() -> Predictor:
    """Return metadata for the live API-backed predictor.

    The Streamlit UI is intentionally thin: all model inference lives in the
    FastAPI service in api.py. Start that service first with:

        uvicorn api:app --host 0.0.0.0 --port 8000

    Override the endpoint with FACEBMI_API_URL if the API is hosted elsewhere.
    """
    health: dict[str, Any] | None = None
    status_label = "Live API configured"
    try:
        response = requests.get(_api_url("/health"), timeout=5)
        if response.ok:
            health = response.json()
            status_label = "Live API ready" if health.get("ok") else "API missing model bundle"
        else:
            status_label = f"API health HTTP {response.status_code}"
    except requests.RequestException:
        status_label = "Live API offline"

    return Predictor(
        model_name="FaceFinalML2 API: ArcFace + DINOv2 + ConvNeXt Ridge",
        status_label=status_label,
        metrics=DEPLOYED_MODEL_METRICS,
        api_url=API_URL,
        timeout_seconds=API_TIMEOUT_SECONDS,
        health=health,
    )


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal weight"
    if bmi < 30:
        return "Overweight"
    if bmi < 35:
        return "Obesity class I"
    if bmi < 40:
        return "Obesity class II"
    return "Obesity class III"


def validate_image(image: Image.Image) -> Image.Image:
    if image is None:
        raise PredictionError("No image was provided.")
    if not isinstance(image, Image.Image):
        raise PredictionError("Input must be a PIL image.")
    if image.width < 32 or image.height < 32:
        raise PredictionError("Image is too small for prediction.")
    return image.convert("RGB")


def _image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=94)
    return buffer.getvalue()


def _extract_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"
    detail = payload.get("detail") if isinstance(payload, dict) else None
    return str(detail or payload or f"HTTP {response.status_code}")


def predict_bmi(image: Image.Image, predictor: Predictor | None = None) -> PredictionResult:
    """Predict BMI by sending the selected image to the live FastAPI model.

    Public interface kept compatible with the Streamlit skeleton:
    PIL image -> PredictionResult.
    """
    active_predictor = predictor or load_predictor()
    rgb = validate_image(image)
    image_bytes = _image_to_jpeg_bytes(rgb)

    try:
        response = requests.post(
            _api_url("/predict", active_predictor.api_url),
            files={"file": ("face.jpg", image_bytes, "image/jpeg")},
            timeout=active_predictor.timeout_seconds,
        )
    except requests.RequestException as exc:
        raise PredictorUnavailableError(
            f"Could not reach the live FaceFinalML2 API at {active_predictor.api_url}. "
            "Start it with `uvicorn api:app --host 0.0.0.0 --port 8000` or set FACEBMI_API_URL."
        ) from exc

    if not response.ok:
        message = _extract_error_message(response)
        raise PredictionError(f"Live API prediction failed: {message}")

    payload = response.json()
    bmi = payload.get("predicted_bmi")
    if bmi is None:
        raise PredictionError("No face was detected. Try a clearer, front-facing image with better lighting.")

    bmi_value = float(bmi)
    warning = payload.get("warning") or "Prediction completed through the live FastAPI model."
    if payload.get("face_detected") is False:
        warning = f"{warning} ArcFace did not detect a face; the API used fallback visual features."

    return PredictionResult(
        bmi=bmi_value,
        category=bmi_category(bmi_value),
        model_name=str(payload.get("model_version") or active_predictor.model_name),
        metrics=active_predictor.metrics,
        warning=warning,
    )
