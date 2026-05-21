# FaceFinalML2: Face-to-BMI Replication and API

Course project for ML2, Spring 2026. This repository will replicate and extend **Face-to-BMI: Using Computer Vision to Infer Body Mass Index on Social Media** using the BMI image dataset provided for the assignment.

## Project goal

Build a real-time BMI prediction system from a face image using transfer learning, then expose it through a simple web API or demo interface. The target is to match or beat the paper's reported performance.

Project page / technical blog:

- https://manuelarceaguirre.github.io/facefinalml2/

See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the full technical plan.
See [`RESULTS.md`](RESULTS.md) and [`HANDOFF.md`](HANDOFF.md) for final metrics and handoff notes.

Fast-start Colab notebook:

- [`notebooks/facefinalml2_colab_runner.ipynb`](notebooks/facefinalml2_colab_runner.ipynb)

Minimal API/demo:

- [`api.py`](api.py)
- [`app.py`](app.py) Streamlit frontend for the live API
- [`predictor.py`](predictor.py) Streamlit-to-FastAPI adapter
- [`API.md`](API.md)

The Colab notebook is intentionally a single-cell runner: it clones/pulls this repo, downloads the Google Drive BMI zip, audits the data, creates leakage-free splits, extracts frozen FaceNet/VGGFace2 + ConvNeXt + optional DINOv2 embeddings, trains regularized regressors, evaluates an ensemble, and writes metrics under `outputs/metrics/`.

## Reference paper baseline

The paper reports Pearson correlation on the test set for BMI prediction:

| Model | Male | Female | Overall |
| --- | ---: | ---: | ---: |
| VGG-Net features + SVR | 0.58 | 0.36 | 0.47 |
| VGG-Face features + SVR | 0.71 | 0.57 | 0.65 |

Our primary benchmark target is therefore **overall Pearson r > 0.65** on a held-out split with no person leakage between train and test.

## Planned approach

1. Use Google Colab for training and experimentation.
2. Load the BMI zip from Google Drive.
3. Build a clean metadata table from the dataset.
4. Split by person/slug so images of the same person do not appear in both train and test.
5. Train transfer-learning baselines:
   - baseline: frozen pre-trained CNN embeddings + regression head
   - improved: fine-tuned image model with augmentation
6. Evaluate with:
   - Pearson correlation
   - MAE
   - RMSE
   - gender-stratified metrics if metadata permits
7. Deploy a simple prediction interface:
   - FastAPI REST endpoint
   - Streamlit UI with upload, webcam capture, demo samples, BMI result display, model summary, and paper reference
8. Prepare the final write-up and 10-minute presentation/demo.

## Data

The dataset is not committed to this repository because it is large and may contain sensitive images.

Google Drive source provided for the project:

- File: `Copy of BMI.zip`
- Drive file id: `16XA-MCnTG8ONdgxK0uPfFXWnA5oF3bFa`
- Size: approximately 987 MB

Expected local/Colab layout after downloading and extracting:

```text
data/
  raw/
    BMI.zip
  extracted/
    images/
    images.csv
```

## Repository layout

```text
facefinalml2/
  README.md              Project overview and instructions
  requirements.txt       Python dependencies for Colab/API/Streamlit work
  api.py                 FastAPI model server
  app.py                 Streamlit frontend
  predictor.py           Adapter that calls the live API from Streamlit
  demo_samples/          Backup images for live presentation fallback
  notebooks/             Colab notebooks will go here
  report/                Final write-up sources/figures will go here
  models/                Trained model artifacts, ignored by git
  data/                  Local data, ignored by git
```

## Colab workflow

The intended workflow is:

1. Open a notebook from `notebooks/` in Google Colab.
2. Mount Google Drive or download the shared zip by file id.
3. Install dependencies from `requirements.txt`.
4. Run preprocessing, training, evaluation, and export a model artifact.
5. Push notebook updates and final documentation back to GitHub.

## Run the Streamlit frontend against the live API

Terminal 1:

```bash
cd final/facefinalml2
uvicorn api:app --host 0.0.0.0 --port 8000
```

Terminal 2:

```bash
cd final/facefinalml2
streamlit run app.py
```

If the API is hosted somewhere else, point the frontend at it:

```bash
FACEBMI_API_URL="https://your-api-host" streamlit run app.py
```

## Deliverables checklist

- [x] Real-time BMI prediction API or Streamlit demo
- [ ] Trained/fine-tuned model and saved inference artifact
- [ ] Evaluation table comparing against paper baseline
- [ ] 10-page implementation write-up
- [ ] 10-minute presentation or recorded demo

## Ethical note

BMI inference from face images is sensitive and can be biased. This project is for academic replication only. The final report should include limitations, privacy concerns, bias analysis, and appropriate use boundaries.
