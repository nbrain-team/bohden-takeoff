import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import gc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

# Load dataset
file_path = "metro_projects_risk_analysis.csv"
df = pd.read_csv(file_path)

# Drop non-numeric columns
df.drop(columns=["Project ID"], inplace=True, errors="ignore")

# Encode categorical variables
label_encoders = {}
categorical_cols = ["City", "Risk Level"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for future use

# Define features (X) and target variables (y)
X = df.drop(columns=["Risk Level", "Delay time (days)"])  # Features
y_classification = df["Risk Level"]  # Target for classification
y_regression = df["Delay time (days)"]  # Target for regression

# Split data into training and testing sets
X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.2, stratify=y_classification, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)

# Initialize XGBoost model for classification
xgb_classifier = xgb.XGBClassifier(objective="multi:softmax", num_class=3, eval_metric="mlogloss")

# Define hyperparameter grid for classification
param_grid_class = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [100, 200, 300],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

# Hyperparameter tuning using GridSearchCV (Classification)
grid_search_class = GridSearchCV(xgb_classifier, param_grid_class, cv=5, scoring="accuracy", n_jobs=1, verbose=2)
grid_search_class.fit(X_train, y_train_class)

# Best model for classification
best_model_class = grid_search_class.best_estimator_
print(f"✅ Best Parameters (Classification): {grid_search_class.best_params_}")

# Make predictions (Classification)
y_pred_class = best_model_class.predict(X_test)

# Evaluate classification model
accuracy = accuracy_score(y_test_class, y_pred_class)
print(f"✅ Classification Model Accuracy: {accuracy:.4f}")
print("\n📊 Classification Report:\n", classification_report(y_test_class, y_pred_class))

# Confusion matrix visualization
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test_class, y_pred_class), annot=True, fmt="d", cmap="Blues", xticklabels=label_encoders["Risk Level"].classes_, yticklabels=label_encoders["Risk Level"].classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Risk Level")
plt.show()



# Save trained models and label encoders
joblib.dump(best_model_class, "xgboost_metro_classifier.pkl")

joblib.dump(label_encoders, "label_encoders.pkl")

print("✅ Models saved successfully.")

# Explicit cleanup to prevent joblib resource_tracker warnings
gc.collect()
