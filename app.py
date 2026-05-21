from __future__ import annotations

import html
from io import BytesIO
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, UnidentifiedImageError

from predictor import (
    MODEL_SUMMARY,
    PredictionError,
    PredictorUnavailableError,
    load_predictor,
    predict_bmi,
)


APP_TITLE = "FaceFinalML2 Live Webcam Demo"
APP_SUBTITLE = "Live multi-person webcam BMI prediction backed by the FaceFinalML2 API."
PAPER_TITLE = "Face-to-BMI: Using Computer Vision to Infer Body Mass Index on Social Media"
PAPER_URL = "https://ojs.aaai.org/index.php/ICWSM/article/view/14923"
PAPER_PDF_PATH = Path(__file__).with_name("14923-Article Text-18442-1-2-20201228 (1).pdf")
PAPER_DOWNLOAD_NAME = "Face-to-BMI_Kocabey_et_al_ICWSM_2017.pdf"
DEMO_SAMPLE_DIR = Path(__file__).with_name("demo_samples")
DEMO_SAMPLES = [
    ("Sample A", DEMO_SAMPLE_DIR / "sample_1.bmp"),
    ("Sample B", DEMO_SAMPLE_DIR / "sample_2.bmp"),
    ("Sample C", DEMO_SAMPLE_DIR / "sample_3.bmp"),
]


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f6f8fb;
            --panel: #ffffff;
            --ink: #172033;
            --muted: #667085;
            --line: #d9e1ec;
            --blue: #2457d6;
            --green: #157f5b;
            --amber: #b76e00;
            --red: #b42318;
        }

        .stApp {
            background:
                linear-gradient(180deg, #f9fbff 0%, #f3f6fa 42%, #eef3f7 100%);
            color: var(--ink);
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--line);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .app-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1.25rem;
        }

        .title-stack h1 {
            margin: 0;
            font-size: 2.2rem;
            line-height: 1.05;
            font-weight: 760;
            letter-spacing: 0;
            color: #111827;
        }

        .title-stack p {
            margin: 0.5rem 0 0;
            color: var(--muted);
            font-size: 1rem;
            max-width: 680px;
        }

        .status-row {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 0.5rem;
            min-width: 260px;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            min-height: 2rem;
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: #ffffff;
            color: #344054;
            font-size: 0.82rem;
            font-weight: 650;
            white-space: nowrap;
        }

        .pill-blue {
            border-color: #b8c8ff;
            background: #eef3ff;
            color: #1746b7;
        }

        .pill-green {
            border-color: #afe2cf;
            background: #eafaf3;
            color: #08704f;
        }

        .panel {
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 16px 36px rgba(15, 23, 42, 0.07);
            padding: 1.1rem;
        }

        .panel h2 {
            margin: 0 0 0.4rem;
            font-size: 1rem;
            font-weight: 760;
            color: #111827;
        }

        .panel p {
            margin: 0;
            color: var(--muted);
            font-size: 0.9rem;
        }

        .result-empty {
            border: 1px dashed #c7d2e5;
            border-radius: 8px;
            background: #f8fbff;
            padding: 2rem 1rem;
            text-align: center;
            color: var(--muted);
        }

        .bmi-card {
            border-radius: 8px;
            padding: 1rem;
            background: #111827;
            color: #ffffff;
            margin-bottom: 0.8rem;
        }

        .bmi-label {
            font-size: 0.8rem;
            color: #cbd5e1;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0;
        }

        .bmi-value {
            font-size: 3.1rem;
            line-height: 1;
            font-weight: 800;
            margin-top: 0.35rem;
        }

        .bmi-category {
            display: inline-flex;
            margin-top: 0.75rem;
            padding: 0.35rem 0.6rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.12);
            color: #ffffff;
            font-size: 0.85rem;
            font-weight: 700;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.6rem;
            margin-top: 0.8rem;
        }

        .metric-tile {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #ffffff;
            padding: 0.75rem;
        }

        .metric-tile span {
            display: block;
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0;
        }

        .metric-tile strong {
            display: block;
            margin-top: 0.25rem;
            color: #111827;
            font-size: 1.1rem;
        }

        .model-card-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
        }

        .model-card {
            min-height: 13rem;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
            padding: 1rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .model-card-featured {
            border-color: #9fc3ff;
            background: linear-gradient(180deg, #ffffff 0%, #f3f7ff 100%);
        }

        .model-card-pending {
            border-style: dashed;
            background: #fbfdff;
        }

        .model-card-title {
            color: #111827;
            font-size: 1rem;
            font-weight: 760;
            line-height: 1.25;
            margin-bottom: 0.4rem;
        }

        .model-card-status {
            display: inline-flex;
            width: fit-content;
            border-radius: 999px;
            padding: 0.28rem 0.55rem;
            background: #eef3ff;
            color: #1746b7;
            font-size: 0.75rem;
            font-weight: 700;
        }

        .model-card-r {
            margin-top: 0.9rem;
        }

        .model-card-r span {
            display: block;
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0;
        }

        .model-card-r strong {
            display: block;
            margin-top: 0.15rem;
            color: #111827;
            font-size: 2rem;
            line-height: 1;
        }

        .model-card-detail-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.45rem;
            margin-top: 0.85rem;
        }

        .model-card-detail {
            border-top: 1px solid #e8edf5;
            padding-top: 0.5rem;
        }

        .model-card-detail span {
            display: block;
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .model-card-detail strong {
            display: block;
            color: #111827;
            font-size: 0.88rem;
            margin-top: 0.18rem;
            overflow-wrap: anywhere;
        }

        .note {
            border-left: 3px solid #2457d6;
            background: #eef3ff;
            color: #263a66;
            padding: 0.75rem 0.9rem;
            border-radius: 6px;
            font-size: 0.88rem;
            margin-top: 0.8rem;
        }

        .ethics-note {
            border: 1px solid #f0d8a8;
            background: #fff8e8;
            color: #5c4206;
            border-radius: 8px;
            padding: 0.9rem;
            font-size: 0.88rem;
        }

        .live-frame-wrap {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
            background: #ffffff;
            box-shadow: 0 16px 36px rgba(15, 23, 42, 0.07);
            margin-top: 0.8rem;
        }

        .fallback-heading {
            margin-top: 1rem;
            color: var(--muted);
            font-size: 0.88rem;
        }

        .reference-card {
            border: 1px solid var(--line);
            background: #ffffff;
            color: #172033;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.05);
            margin-top: 0.8rem;
        }

        .reference-eyebrow {
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 760;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }

        .reference-title {
            color: #111827;
            font-size: 1rem;
            line-height: 1.3;
            font-weight: 760;
            margin-bottom: 0.5rem;
        }

        .reference-meta {
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.45;
        }

        div[data-testid="stFileUploader"] {
            border: 1px solid #d7dfed;
            border-radius: 8px;
            padding: 0.65rem;
            background: #fbfdff;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 8px;
            min-height: 3rem;
            font-weight: 760;
            border: 0;
            background: #2457d6;
            color: #ffffff;
        }

        div.stButton > button:hover {
            background: #1c46ad;
            color: #ffffff;
            border: 0;
        }

        div[data-testid="stDownloadButton"] > button {
            width: 100%;
            border-radius: 8px;
            min-height: 2.65rem;
            font-weight: 760;
        }

        div[data-testid="stLinkButton"] > a {
            width: 100%;
            border-radius: 8px;
            min-height: 2.65rem;
            font-weight: 760;
        }

        @media (max-width: 760px) {
            .app-header {
                display: block;
            }

            .status-row {
                justify-content: flex-start;
                margin-top: 0.9rem;
            }

            .title-stack h1 {
                font-size: 1.75rem;
            }

            .metric-grid {
                grid-template-columns: 1fr;
            }

            .model-card-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def decode_image(file_obj: Any) -> Image.Image:
    try:
        raw = file_obj.getvalue()
        image = Image.open(BytesIO(raw))
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise PredictionError("The selected file could not be read as an image.") from exc


def get_file_signature(file_obj: Any) -> str:
    raw = file_obj.getvalue()
    name = getattr(file_obj, "name", "camera")
    return f"{name}:{len(raw)}:{hash(raw)}"


def clear_active_input() -> None:
    st.session_state.active_image = None
    st.session_state.active_source = None
    st.session_state.active_image_signature = None
    st.session_state.prediction = None


def set_active_image(image: Image.Image, source: str, signature: str) -> None:
    if signature != st.session_state.active_image_signature:
        st.session_state.prediction = None
    st.session_state.active_image = image
    st.session_state.active_source = source
    st.session_state.active_image_signature = signature


def load_sample_image(path: Path) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise PredictionError(f"Could not read demo sample: {path.name}") from exc


def get_sample_signature(path: Path) -> str:
    stat = path.stat()
    return f"sample:{path.name}:{stat.st_size}:{int(stat.st_mtime)}"


def render_header(predictor_status: str) -> None:
    st.markdown(
        f"""
        <div class="app-header">
            <div class="title-stack">
                <h1>{APP_TITLE}</h1>
                <p>{APP_SUBTITLE}</p>
            </div>
            <div class="status-row">
                <span class="pill pill-blue">Model: {predictor_status}</span>
                <span class="pill pill-green">Target: r &gt; 0.65</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel_start(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="panel">
            <h2>{title}</h2>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_tile(label: str, value: Any) -> str:
    return f"""
    <div class="metric-tile">
        <span>{label}</span>
        <strong>{value}</strong>
    </div>
    """


def format_metric(value: Any) -> str:
    if value is None:
        return "Pending"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_prediction() -> None:
    prediction = st.session_state.get("prediction")

    if prediction is None:
        st.markdown(
            """
            <div class="result-empty">
                Upload or capture a face image, then run prediction.
                Results will appear here.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    metric_html = "".join(
        [
            render_metric_tile("Pearson r", format_metric(prediction.metrics.get("pearson_r"))),
            render_metric_tile("MAE", format_metric(prediction.metrics.get("mae"))),
            render_metric_tile("RMSE", format_metric(prediction.metrics.get("rmse"))),
            render_metric_tile("Test N", format_metric(prediction.metrics.get("test_n"))),
        ]
    )

    st.markdown(
        f"""
        <div class="bmi-card">
            <div class="bmi-label">Predicted BMI</div>
            <div class="bmi-value">{prediction.bmi:.1f}</div>
            <div class="bmi-category">{prediction.category}</div>
        </div>
        <div class="metric-grid">{metric_html}</div>
        <div class="note">
            Model used: <strong>{prediction.model_name}</strong><br>
            {prediction.warning or "Prediction completed successfully."}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_summary() -> None:
    cards = []
    for index, model in enumerate(MODEL_SUMMARY):
        card_class = "model-card"
        if index == 1:
            card_class += " model-card-featured"
        if "Pending" in str(model["Pearson r"]):
            card_class += " model-card-pending"

        model_name = html.escape(str(model["Model"]))
        status = html.escape(str(model["Status"]))
        pearson_r = html.escape(format_metric(model["Pearson r"]))
        mae = html.escape(format_metric(model["MAE"]))
        rmse = html.escape(format_metric(model["RMSE"]))
        test_n = html.escape(format_metric(model["Test N"]))

        cards.append(
            f'<div class="{card_class}">'
            '<div>'
            f'<div class="model-card-title">{model_name}</div>'
            f'<div class="model-card-status">{status}</div>'
            '<div class="model-card-r">'
            '<span>Pearson r</span>'
            f'<strong>{pearson_r}</strong>'
            '</div>'
            '</div>'
            '<div class="model-card-detail-grid">'
            '<div class="model-card-detail"><span>MAE</span>'
            f'<strong>{mae}</strong></div>'
            '<div class="model-card-detail"><span>RMSE</span>'
            f'<strong>{rmse}</strong></div>'
            '<div class="model-card-detail"><span>Test N</span>'
            f'<strong>{test_n}</strong></div>'
            '</div>'
            '</div>'
        )

    st.markdown(
        f'<div class="model-card-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def render_live_webcam(api_url: str) -> None:
    safe_url = html.escape(api_url.rstrip("/"))
    components.html(
        f"""
        <div class="live-frame-wrap">
          <iframe
            src="{safe_url}/"
            title="FaceFinalML2 live webcam API"
            allow="camera; microphone; autoplay; clipboard-write"
            style="width:100%;height:760px;border:0;background:white;"
          ></iframe>
        </div>
        """,
        height=790,
        scrolling=True,
    )



def render_reference_paper() -> None:
    st.markdown(
        f"""
        <div class="reference-card">
            <div class="reference-eyebrow">Reference Paper</div>
            <div class="reference-title">{html.escape(PAPER_TITLE)}</div>
            <div class="reference-meta">
                Kocabey et al., ICWSM 2017<br>
                DOI: 10.1609/icwsm.v11i1.14923
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    paper_actions = st.columns(2, gap="small")
    with paper_actions[0]:
        st.link_button("Open article page", PAPER_URL, width="stretch")
    with paper_actions[1]:
        if PAPER_PDF_PATH.exists():
            st.download_button(
                "Download PDF",
                data=PAPER_PDF_PATH.read_bytes(),
                file_name=PAPER_DOWNLOAD_NAME,
                mime="application/pdf",
                width="stretch",
            )
        else:
            st.button("PDF missing", disabled=True, width="stretch")


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_styles()

    predictor = load_predictor()
    render_header(predictor.status_label)

    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "active_image" not in st.session_state:
        st.session_state.active_image = None
    if "active_source" not in st.session_state:
        st.session_state.active_source = None
    if "active_image_signature" not in st.session_state:
        st.session_state.active_image_signature = None

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        render_panel_start(
            "Live Webcam",
            "This is the primary demo path: live webcam tracking, up to six people, optional name enrollment, and BMI labels over the video. Upload is only a fallback below.",
        )
        render_live_webcam(predictor.api_url)

        st.markdown('<div class="fallback-heading">Fallback single-image predictor</div>', unsafe_allow_html=True)
        with st.expander("Open upload / camera snapshot / demo samples fallback", expanded=False):
            upload_tab, webcam_tab, sample_tab = st.tabs(["Webcam Snapshot", "Upload", "Demo Samples"])

            uploaded_file = None
            camera_file = None
            selected_sample: tuple[str, Path] | None = None

            with webcam_tab:
                camera_file = st.camera_input("Capture one face image")
                st.caption("This is a fallback still-image path. Use the live webcam panel above for the final demo.")

            with upload_tab:
                uploaded_file = st.file_uploader(
                    "Choose a face image",
                    type=["jpg", "jpeg", "png", "bmp"],
                    accept_multiple_files=False,
                )
                st.caption("Supported formats: JPG, PNG, BMP.")

            with sample_tab:
                st.caption("Use these backup samples if webcam access is unavailable during the live demo.")
                sample_columns = st.columns(3, gap="small")
                for index, ((label, sample_path), column) in enumerate(zip(DEMO_SAMPLES, sample_columns)):
                    with column:
                        if sample_path.exists():
                            st.image(str(sample_path), caption=label, width="stretch")
                            if st.button(f"Use {label}", key=f"use_demo_sample_{index}"):
                                selected_sample = (label, sample_path)
                        else:
                            st.warning(f"{label} missing")

            selected_file = camera_file or uploaded_file
            source_label = "Webcam snapshot" if camera_file is not None else "Uploaded image"

            if selected_sample is not None:
                try:
                    sample_label, sample_path = selected_sample
                    image = load_sample_image(sample_path)
                    set_active_image(
                        image=image,
                        source=f"Demo sample: {sample_label}",
                        signature=get_sample_signature(sample_path),
                    )
                except PredictionError as exc:
                    clear_active_input()
                    st.error(str(exc))
            elif selected_file is not None:
                try:
                    signature = get_file_signature(selected_file)
                    image = decode_image(selected_file)
                    set_active_image(image=image, source=source_label, signature=signature)
                except PredictionError as exc:
                    clear_active_input()
                    st.error(str(exc))
            elif not str(st.session_state.active_image_signature).startswith("sample:"):
                clear_active_input()

            if st.session_state.active_image is not None:
                st.image(
                    st.session_state.active_image,
                    caption=st.session_state.active_source,
                    width="stretch",
                )
            else:
                st.info("No fallback image selected yet.")

            predict_clicked = st.button("Predict fallback image BMI", type="primary")
            if predict_clicked:
                image = st.session_state.get("active_image")
                if image is None:
                    st.warning("Please capture, upload, or select an image before predicting.")
                else:
                    try:
                        with st.spinner("Running BMI prediction..."):
                            st.session_state.prediction = predict_bmi(image, predictor)
                        st.success("Prediction ready.")
                    except PredictorUnavailableError as exc:
                        st.error(str(exc))
                    except PredictionError as exc:
                        st.error(str(exc))
                    except Exception as exc:  # Defensive UI boundary for live demos.
                        st.error(f"Prediction failed unexpectedly: {exc}")

    with right:
        render_panel_start(
            "Fallback Result",
            "This panel shows results only for the optional still-image fallback. The main live webcam result appears inside the embedded demo on the left.",
        )
        st.write("")
        render_prediction()

    st.write("")
    bottom_left, bottom_right = st.columns([1.1, 0.9], gap="large")

    with bottom_left:
        st.subheader("Model Summary")
        render_model_summary()

    with bottom_right:
        st.subheader("Demo Notes")
        st.markdown(
            """
            <div class="ethics-note">
                This app is for an academic demonstration only. BMI inferred from a face image is noisy
                at the individual level and should not be used for medical, hiring, insurance, or
                identity-related decisions.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="note">
                Live setup: start <code>uvicorn api:app --host 0.0.0.0 --port 8000</code>,
                then run this Streamlit app. Set <code>FACEBMI_API_URL</code> if the API is remote.
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_reference_paper()


if __name__ == "__main__":
    main()
