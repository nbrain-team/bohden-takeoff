import cv2
import torch
import joblib
import shutil
import os
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO

# Load trained YOLOv8 model
yolo_model = YOLO("models/best_2.pt")

# Load trained XGBoost model
ml_model = joblib.load("models/risk_model.pkl")

# FastAPI App
app = FastAPI()

# Function to extract safety violations
def extract_risk_features(image_path):
    results = yolo_model(image_path)
    detections = results[0].boxes.cls.cpu().numpy()
    hazard_counts = {cls: (detections == cls).sum() for cls in range(len(results[0].names))}
    return hazard_counts

@app.get("/")
async def home():
    return {"message": "Risk Analysis API is running. Use /predict_risk/ for predictions."}

@app.post("/predict_risk/")
async def predict_risk(file: UploadFile = File(...)):
    try:
        # Save uploaded image
        image_path = f"temp/{file.filename}"
        os.makedirs("temp", exist_ok=True)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract risk features from image
        features = extract_risk_features(image_path)

        # Convert extracted features to DataFrame (Placeholder for real integration)
        df = pd.DataFrame([features])

        # Predict construction delay risk
        risk_prediction = ml_model.predict(df)

        # Cleanup
        os.remove(image_path)

        return {"risk_assessment": features, "delay_risk": int(risk_prediction[0])}

    except Exception as e:
        return {"error": str(e)}
