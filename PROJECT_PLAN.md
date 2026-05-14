# Project Plan: Face-to-BMI Replication and Modern Extension

## Executive recommendation

Build this as a rigorous replication plus modern extension of the ICWSM 2017 Face-to-BMI paper, not as a simple fine-tuning demo.

**Final target system:** leakage-free face detection/alignment, tight and loose face crops, frozen modern embeddings from face and foundation models, regularized regressors, validation-weighted ensemble, and an educational API/demo.

The strongest practical route is:

1. Reproduce the paper-style baseline with a VGGFace2/FaceNet-style embedding plus SVR/Ridge.
2. Improve it with modern frozen embeddings: ArcFace/InsightFace if available, DINOv2, ConvNeXt, and possibly CLIP.
3. Train regularized regressors and a validation-weighted ensemble.
4. Add professor-impact analysis: subgroup metrics, leakage checks, residual plots, uncertainty, and explainability.
5. Deploy a small educational FastAPI or Streamlit demo.

The original paper reports **overall Pearson r = 0.65** for VGG-Face fc6 features plus SVR on a leakage-free held-out split. The project target is to beat that result honestly, with no person/slug overlap between train, validation, and test.

## Original paper baseline

Reference paper:

> Kocabey et al., "Face-to-BMI: Using Computer Vision to Infer Body Mass Index on Social Media," ICWSM 2017.

The paper used the VisualBMI dataset:

- Started from 16,483 social-media images.
- Manually cleaned/cropped to 2,103 before/after face pairs.
- Produced 4,206 face images with gender, height, weight, and BMI labels.
- Used transfer learning: pretrained CNN features plus epsilon-SVR.
- Train/test split: 3,368 train images and 838 test images.
- Important: the same individual was not allowed in both train and test.

Reported Pearson correlations:

| Model | Male | Female | Overall |
| --- | ---: | ---: | ---: |
| VGG-Net features + SVR | 0.58 | 0.36 | 0.47 |
| VGG-Face features + SVR | 0.71 | 0.57 | 0.65 |

## Technical strategy

### Track A: faithful reproduction baseline

Purpose: scientific credibility and direct comparison against the paper.

Recommended implementation:

- Face detection/cropping with MTCNN or RetinaFace-style detector.
- Embedding with `facenet-pytorch` InceptionResnetV1 pretrained on VGGFace2 as a practical modern substitute for old VGG-Face tooling.
- Regressors:
  - RidgeCV
  - ElasticNetCV
  - SVR with RBF kernel
  - optional XGBoost
- Metrics:
  - Pearson r
  - Spearman rho
  - MAE
  - RMSE
  - R2

### Track B: modern strong baseline

Purpose: best chance to beat the paper quickly and robustly.

Recommended feature families:

1. **Face embedding:** FaceNet/VGGFace2 or ArcFace/InsightFace.
2. **Foundation embedding:** DINOv2 ViT-S/14 or ViT-B/14.
3. **Modern CNN embedding:** ConvNeXt-Tiny or ConvNeXt-Small from `timm`.
4. **Optional:** CLIP image embeddings.

Recommended model:

```text
Input image
  -> face detector
  -> tight aligned crop for face-recognition embedding
  -> loose face/jaw/neck crop for foundation/CNN embedding
  -> frozen embeddings
  -> standardized features
  -> Ridge/SVR/XGBoost regressors
  -> validation-weighted ensemble
  -> BMI estimate + uncertainty proxy
```

This is the primary final model because it is strong, fast, ablatable, and much less likely to overfit than end-to-end training on a small dataset.

### Track C: ambitious extension

Purpose: make the project look research-grade without making it dependent on risky training.

Possible additions:

- Multi-task ConvNeXt model with BMI regression plus BMI-category classification.
- Huber loss plus a small Pearson-correlation loss term.
- Label-distribution or ordinal loss over BMI bins.
- Test-time augmentation.
- Ensemble uncertainty from model disagreement.
- Grad-CAM or attention visualizations.
- Fairness/error analysis by gender and BMI category.
- Optional MediaPipe FaceMesh geometric features.

Track C should be treated as an advanced experiment. The frozen-embedding ensemble should remain the locked fallback.

## Data pipeline

Dataset source:

- Google Drive file name: `Copy of BMI.zip`
- Drive file ID: `16XA-MCnTG8ONdgxK0uPfFXWnA5oF3bFa`
- Approximate size: 987 MB

Expected layout after extraction:

```text
data/
  raw/
    BMI.zip
  extracted/
    images/
      <slug>/
        <slug>_001.jpg
        <slug>_002.jpg
    images.csv
```

Expected metadata columns may include:

```text
Name, slug, num of Photo, Weight (kg), Height (m), Actual BMI, Type
```

The pipeline must:

1. Download the zip in Colab.
2. Unzip and inspect all files.
3. Find the metadata CSV.
4. Normalize column names.
5. Construct image paths from slug folders or filename columns.
6. Compute BMI from weight/height if necessary.
7. Validate BMI, height, and weight ranges.
8. Remove corrupted images.
9. Create BMI categories for stratification.
10. Split by person/slug, not by image.
11. Save a deterministic split CSV.

The most important methodological rule is:

> All images from the same person/slug must remain in exactly one split.

## Preprocessing

Use two crops per image:

| Crop | Size | Purpose |
| --- | ---: | --- |
| tight face crop | 160x160 or 112x112 | face-recognition embeddings |
| loose face/jaw/neck crop | 224x224 or 320x320 | DINOv2, ConvNeXt, CLIP, fine-tuning |

Face handling policy:

- If multiple faces are detected, choose the largest high-confidence face near the image center.
- If no face is detected, use a centered square fallback crop and mark `face_detected = false`.
- Report metrics both on all images and on face-detected images if possible.

Safe augmentations for later fine-tuning:

- Horizontal flip.
- Small rotation.
- Mild brightness/contrast.
- Mild blur/noise/JPEG artifacts.

Avoid transforms that change body/face geometry aggressively.

## Evaluation protocol

Primary metric:

- Pearson correlation between predicted BMI and true BMI.

Secondary metrics:

- Spearman rho.
- MAE.
- RMSE.
- R2.
- Bias / mean signed error.
- Within 2 BMI units.
- Within 5 BMI units.

Required result tables:

1. Paper baseline table.
2. Reproduction baseline table.
3. Modern embedding comparison table.
4. Ensemble result table.
5. Ablation table.
6. Subgroup metrics by gender/BMI category if metadata allows.

Suggested final comparison table:

| Model | Features | Regressor | Leakage-free split | Overall r | MAE | RMSE | R2 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Paper VGG-Net + SVR | VGG-Net fc6 | SVR | yes | 0.47 | - | - | - |
| Paper VGG-Face + SVR | VGG-Face fc6 | SVR | yes | 0.65 | - | - | - |
| Reproduction | FaceNet/VGGFace2 | SVR/Ridge | yes | TBD | TBD | TBD | TBD |
| Modern CNN | ConvNeXt | Ridge/SVR | yes | TBD | TBD | TBD | TBD |
| Foundation | DINOv2 | Ridge/SVR | yes | TBD | TBD | TBD | TBD |
| Final ensemble | Face + foundation + CNN | weighted ensemble | yes | TBD | TBD | TBD | TBD |

## Deployment plan

The demo should be educational only.

Recommended demo:

- Streamlit first, because it is easiest for a live class demo.
- FastAPI endpoint if time allows.

The demo should:

1. Accept an uploaded image or webcam frame.
2. Detect/crop the face.
3. Extract embeddings.
4. Predict BMI with the saved regressor ensemble.
5. Display BMI estimate and uncertainty proxy.
6. Show a clear disclaimer.

Required disclaimer:

> This is an academic ML demo, not a medical diagnostic tool. BMI-from-face prediction is noisy, biased, and privacy-sensitive. It must not be used for health, employment, insurance, or personal judgments.

## Report outline

Suggested title:

**Beyond VGG-Face: Modern Face and Foundation Embeddings for Leakage-Free Face-to-BMI Estimation**

Ten-page structure:

1. Abstract.
2. Introduction.
3. Related work.
4. Dataset and audit.
5. Methods.
6. Experiments.
7. Results.
8. Ablations and error analysis.
9. Deployment/API.
10. Bias, ethics, limitations, and conclusion.

Figures to include:

- Pipeline diagram.
- BMI histogram by split.
- Predicted vs true BMI scatter.
- Residuals vs BMI.
- Residuals by subgroup.
- Example Grad-CAM/attention maps if available.
- Demo screenshot.

## Presentation outline

Ten-minute structure:

1. Problem and ethical caveat.
2. Original Face-to-BMI paper and baseline.
3. Data audit and leakage-free splitting.
4. Methods: reproduction, modern embeddings, ensemble.
5. Results vs paper.
6. Ablations.
7. Bias/error analysis.
8. Live demo or recording.
9. Limitations.
10. Takeaways.

## Risk register

| Risk | Mitigation |
| --- | --- |
| Person leakage | Split by slug/person and assert disjoint groups. |
| Wrong image-label mapping | Visual audit random samples and print source rows. |
| Dataset is small | Prefer frozen embeddings and regularized regressors. |
| Label noise | Use robust losses/regressors and report uncertainty. |
| Face detection failures | Save fallback crops and report detection rate. |
| Gender or BMI-category bias | Report subgroup metrics and limitations. |
| Colab instability | Save intermediate features, metrics, and split CSVs. |
| Over-tuning test set | Select models on validation only; evaluate test once. |
| Ethical misuse | Educational-only demo and strong disclaimers. |

## Definition of success

Minimum successful project:

- Clean Colab pipeline.
- Leakage-free split.
- Reproduction baseline.
- At least one modern embedding model.
- Honest comparison to paper.
- Streamlit or FastAPI demo.
- Ten-page write-up and presentation.

Excellent project:

- Final ensemble beats Pearson r = 0.65.
- Strong ablation table.
- Subgroup/fairness analysis.
- Uncertainty estimate.
- Explainability figure.
- Live demo using the same preprocessing and model code as evaluation.
