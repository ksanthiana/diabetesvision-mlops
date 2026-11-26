import csv
import subprocess
import sys
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile


import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from fastapi.middleware.cors import CORSMiddleware

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        BASE_DIR / "models" / "diabetesvision_mobilenetv2_final.keras"
    )
)

IMAGE_SIZE = (224, 224)

GRADE_LABELS = {
    0: "No diabetic retinopathy",
    1: "Mild diabetic retinopathy",
    2: "Moderate diabetic retinopathy",
    3: "Severe diabetic retinopathy",
    4: "Proliferative diabetic retinopathy",
}

# --------------------------------------------------
# Application
# --------------------------------------------------

app = FastAPI(
    
    title="DiabetesVision API",
    description="Retinal image classification research API.",
    version="1.0.0",
)

app.openapi_version = "3.0.2"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


@app.on_event("startup")
def load_model():
    global model

    if MODEL_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded successfully: {MODEL_PATH}")
    else:
        print(f"Model not found yet: {MODEL_PATH}")


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to DiabetesVision API",
        "status": "running",
        "model_loaded": model is not None,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")

    image_array = np.array(image)

    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )

    image_tensor = tf.image.resize_with_pad(
        image_tensor,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1]
    )

    image_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_tensor
    )

    return tf.expand_dims(image_tensor, axis=0)


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please add the trained model."
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    try:
        image = Image.open(file.file).convert("RGB")
        processed_image = preprocess_image(image)

        predictions = model.predict(processed_image, verbose=0)[0]

        predicted_grade = int(np.argmax(predictions))

        probabilities = {
            GRADE_LABELS[i]: round(float(predictions[i]), 4)
            for i in range(len(GRADE_LABELS))
        }

        return {
            "filename": file.filename,
            "predicted_grade": predicted_grade,
            "prediction": GRADE_LABELS[predicted_grade],
            "confidence": round(float(predictions[predicted_grade]), 4),
            "probabilities": probabilities,
            "notice": (
                "Research demonstration only. This model is not a "
                "medical diagnostic tool."
            ),
        }

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be read as an image."
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )



# --------------------------------------------------
# Upload labelled images for retraining
# --------------------------------------------------

UPLOAD_DIR = BASE_DIR / "data" / "uploaded"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

LABELS_FILE = UPLOAD_DIR / "labels.csv"
RETRAIN_SCRIPT = BASE_DIR / "src" / "retraining.py"
RETRAINED_MODEL_PATH = BASE_DIR / "models" / "diabetesvision_mobilenetv2_retrained.keras"


@app.post("/retrain/upload")
async def upload_training_images(
    grade: int = Form(...),
    files: list[UploadFile] = File(...)
):
    if grade not in GRADE_LABELS:
        raise HTTPException(status_code=400, detail="Grade must be between 0 and 4.")

    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one image.")

    saved_files = []

    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image files are accepted.")

        try:
            image = Image.open(file.file)
            image.verify()
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid image: {file.filename}")

        extension = Path(file.filename or "").suffix.lower()
        if extension not in [".jpg", ".jpeg", ".png"]:
            raise HTTPException(status_code=400, detail="Use JPG, JPEG, or PNG images.")

        filename = f"{uuid.uuid4().hex}{extension}"
        destination = UPLOAD_DIR / filename

        file.file.seek(0)
        destination.write_bytes(await file.read())

        saved_files.append(filename)

    write_header = not LABELS_FILE.exists()

    with LABELS_FILE.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["filename", "diagnosis"])

        for filename in saved_files:
            writer.writerow([filename, grade])

    return {
        "message": "Images uploaded and labelled successfully.",
        "grade": grade,
        "uploaded_count": len(saved_files),
        "files": saved_files
    }


# --------------------------------------------------
# Trigger model retraining
# --------------------------------------------------

@app.post("/retrain")
def retrain_model():
    global model

    if not RETRAIN_SCRIPT.exists():
        raise HTTPException(status_code=500, detail="Retraining script not found.")

    try:
        result = subprocess.run(
            [sys.executable, str(RETRAIN_SCRIPT)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=3600
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Retraining failed: {result.stderr[-3000:]}"
            )

        if not RETRAINED_MODEL_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail="Retraining finished but the new model was not found."
            )

        model = tf.keras.models.load_model(RETRAINED_MODEL_PATH)

        return {
            "message": "Retraining completed successfully.",
            "model_loaded": True,
            "model_path": str(RETRAINED_MODEL_PATH),
            "training_output": result.stdout[-2000:]
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Retraining timed out.")
