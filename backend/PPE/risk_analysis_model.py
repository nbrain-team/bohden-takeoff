import cv2
import torch
import pandas as pd
import joblib
from ultralytics import YOLO
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score
from fastapi import FastAPI, File, UploadFile
import shutil
import os
from sklearn.model_selection import GridSearchCV
import numpy as np

# Load trained YOLOv8 model
yolo_model = YOLO("models/best_2.pt")

# Function to extract safety violations
def extract_risk_features(image_path):
    results = yolo_model(image_path)
    detections = results[0].boxes.cls.cpu().numpy()  # Extract class IDs
    hazard_counts = {cls: (detections == cls).sum() for cls in range(len(results[0].names))}
    
    print("🛑 Detected Hazards:", hazard_counts)  # Print detected hazards
    
    return hazard_counts

# Load construction delay dataset
df = pd.read_csv("construction_delay_india.csv")

# Feature Engineering
df = pd.get_dummies(df, columns=["City", "Project Type", "External Factor", "Delay Reason"], drop_first=True)

# Convert "Milestone" column to numeric
if "Milestone" in df.columns:
    df["Milestone"] = pd.to_numeric(df["Milestone"], errors="coerce").fillna(0)

# Drop non-numeric columns before computing correlation
numeric_df = df.select_dtypes(include=[np.number])

# Compute correlation on numeric columns only
corr_matrix = numeric_df.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
drop_cols = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Train-test split
X = df.drop(columns=["Delayed", "Project ID", "Start Date", "End Date"])  # Drop non-numeric & target
y = df["Delayed"]  # Target: Delayed (1) or Not (0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Hyperparameter tuning for XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_lambda': [0.1, 1, 10],
    'reg_alpha': [0.1, 1, 10]
}

grid_search = GridSearchCV(XGBClassifier(enable_categorical=True), param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

joblib.dump(X_train.columns.tolist(), "train_columns.pkl")
print("✅ Saved training feature columns: train_columns.pkl")

# Save the trained model properly
model_path = "best_xgb_model.json"
best_model.save_model(model_path)
print(f"✅ Model saved successfully as {model_path}")

# Ensure the model is loaded properly
if os.path.exists(model_path):
    loaded_model = XGBClassifier()
    loaded_model.load_model(model_path)
    print("✅ Model loaded successfully.")
else:
    print("❌ Model file not found. Training might have failed.")

# Evaluate accuracy and R² score
y_pred = loaded_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Cross-validation for better generalization
cv_scores = cross_val_score(loaded_model, X, y, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
cv_mean_accuracy = np.mean(cv_scores)

print("Best Parameters:", grid_search.best_params_)
print("Model Accuracy (Test Set):", accuracy)
print("Cross-Validation Accuracy:", cv_mean_accuracy)
print("R² Score:", r2)
