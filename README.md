# DermalScan AI

AI-assisted facial skin-image analysis project. It provides a Flask API that detects faces and classifies visible skin-related image patterns using an EfficientNetB0 model.

> **Important:** This is an experimental software project, not a medical device. Its predictions are not a diagnosis, treatment recommendation, or clinically validated result. Do not use it for medical decisions.

## Problem statement

Inspecting facial images manually can be subjective and time-consuming. DermalScan explores a reproducible computer-vision workflow that detects faces in an image and classifies each detected face into one of four project labels: **clear skin**, **dark spots**, **puffy eyes**, or **wrinkles**. The goal is to demonstrate an end-to-end image-inference pipeline, not to establish medical or cosmetic ground truth.

## Solution

The application accepts an image through an HTTP endpoint, converts it to RGB, detects faces with MTCNN, expands each face crop slightly for context, and passes a resized, normalized crop to a trained TensorFlow/Keras EfficientNetB0 classifier. It returns the predicted class and model confidence for each face, along with a base64-encoded annotated JPEG.

## Architecture

```mermaid
flowchart LR
    A[Client image] -->|POST /predict| B[Flask API]
    B --> C[Pillow: decode and convert to RGB]
    C --> D[MTCNN: detect faces]
    D --> E[Expand crop, resize to 224x224, normalize]
    E --> F[TensorFlow/Keras EfficientNetB0]
    F --> G[Four-class prediction and confidence]
    G --> H[JSON response and annotated image]
```

Inference details:

1. `app.py` accepts a multipart form upload in the `image` field. The request limit is 16 MB.
2. `scripts/inference_model.py` uses MTCNN to detect faces, processes the detected faces largest-first, and expands each crop by 20% around its bounding box.
3. Each crop is resized to 224 × 224 and pixel values are scaled to `[0, 1]` before classification.
4. The best model predicts one of the four labels. The response includes the face box, label, confidence, and annotated image.

## Project structure

```text
app.py                         Flask API entry point
scripts/
  inference_model.py           Face detection and model inference
  prepare_dataset.py           TensorFlow dataset loading and augmentation
  train_model.py               EfficientNetB0 training and fine-tuning
  evaluate_model.py            Model evaluation
  crop_faces_mtcnn.py          Face-cropping preprocessing
  convert_to_rgb.py            Image format preprocessing
  apply_color_correction.py    Color correction preprocessing
  clean_dataset_mtcnn_300.py   Dataset cleaning
  augment_to_300.py             Dataset augmentation
  split_dataset.py              Train/validation/test split
models/
  efficientnetb0_skin_model_best.h5
  efficientnetb0_base_notop.weights.h5
```

The facial-image datasets, generated logs, and virtual environments are not included in this repository. Training requires a locally prepared dataset; see **Training** below. Please only use images and datasets that you have permission to process.

## Quick start

Use Python with the project dependencies installed. The repository does not currently pin dependency versions in a `requirements.txt` file.

```bash
python -m venv .venv
# Activate the environment, then install the inference dependencies:
pip install flask pillow tensorflow numpy mtcnn opencv-python
python app.py
```

The development server listens on `http://127.0.0.1:5000/`. The root route returns a short health/status message. To submit an image:

```bash
curl -X POST -F "image=@face.jpg" http://127.0.0.1:5000/predict
```

Successful `/predict` response shape:

```json
{
  "status": "success",
  "predictions": [
    {
      "box": [10, 20, 210, 220],
      "dermalscan_label": "wrinkles",
      "dermalscan_confidence": 0.84,
      "predicted_age": "42"
    }
  ],
  "annotated_image": "<base64-encoded JPEG>"
}
```

`box` contains the expanded face-crop coordinates. `annotated_image` is a base64-encoded JPEG. A request without an image receives an error response; an image without a detected face returns a no-face message.

## Training

`prepare_dataset.py` expects a `data_split_fixed/` folder containing `train/`, `val/`, and `test/` directories, each organized into class-named subfolders. The script applies augmentation to training images and normalizes the data. The training script builds an EfficientNetB0 transfer-learning model, first training a classifier head with the base frozen, then fine-tuning the final layers. The best checkpoint is selected using validation accuracy.

From the repository root, after preparing the dataset and installing the additional training/evaluation dependencies, run:

```bash
pip install matplotlib scikit-learn seaborn tqdm
python scripts/train_model.py
```

The scripts may create model checkpoints, logs, and training plots locally. Dataset acquisition and preparation are the user's responsibility; no facial-image dataset is distributed here.

## Model files

- `models/efficientnetb0_skin_model_best.h5` is the model loaded by the inference code.
- `models/efficientnetb0_base_notop.weights.h5` is the local EfficientNet base weights file used by training when available. Training falls back to Keras ImageNet weights if it is absent.
- `efficientnetb0_skin_model_final_epoch.h5` is an additional 26.2 MB checkpoint available as an asset in the [models-v1 release](https://github.com/Praneeth8845/DermalScan-AI/releases/tag/models-v1). It is kept out of the Git file tree because it exceeds GitHub's 25 MB browser-upload limit; download it into `models/` only if you need that checkpoint. Inference uses the best checkpoint by default.

## Limitations and privacy

- The classes are project labels for image classification, not medical diagnoses. Model performance depends on its training data and has not been independently validated here.
- The API includes a `predicted_age` field for demonstration only. The current code generates a random value from 20 to 70; it does **not** estimate age from the image. Do not interpret or rely on this field.
- The Flask app runs with debug mode enabled for local development. Do not expose this development server directly to the public internet; use a production server and add appropriate authentication, validation, and operational safeguards before deployment.
- Facial images are sensitive. The datasets are intentionally not included in this public repository. Obtain consent and permission before processing or sharing any face images.

## License

No license is currently specified. Contact the repository owner before reusing or redistributing this code or model files.
