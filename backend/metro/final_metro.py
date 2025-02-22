
# Import libraries
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

# Step 1: Load and preprocess the data
def load_and_preprocess_data(file_path):
    # Load dataset
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

    return X_train, X_test, y_train_class, y_test_class, X_train_reg, X_test_reg, y_train_reg, y_test_reg, label_encoders

# Step 2: Build and tune the classification model
def build_classification_model(X_train, y_train_class):
    # Initialize XGBoost classifier
    xgb_classifier = xgb.XGBClassifier(objective="multi:softmax", num_class=3, eval_metric="mlogloss")

    # Define hyperparameter grid for classification
    param_grid_class = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [100, 200, 300],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    # Perform hyperparameter tuning using GridSearchCV
    grid_search_class = GridSearchCV(xgb_classifier, param_grid_class, cv=5, scoring="accuracy", n_jobs=1, verbose=2)
    grid_search_class.fit(X_train, y_train_class)

    # Select the best model for classification
    best_model_class = grid_search_class.best_estimator_
    print(f"✅ Best Parameters (Classification): {grid_search_class.best_params_}")

    return best_model_class

# Step 3: Build and tune the regression model
def build_regression_model(X_train_reg, y_train_reg):
    # Initialize XGBoost regressor
    xgb_regressor = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse")

    # Define hyperparameter grid for regression
    param_grid_reg = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [100, 200, 300],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    # Perform hyperparameter tuning using GridSearchCV
    grid_search_reg = GridSearchCV(xgb_regressor, param_grid_reg, cv=5, scoring="neg_mean_squared_error", n_jobs=1, verbose=2)
    grid_search_reg.fit(X_train_reg, y_train_reg)

    # Select the best model for regression
    best_model_reg = grid_search_reg.best_estimator_
    print(f"✅ Best Parameters (Regression): {grid_search_reg.best_params_}")

    return best_model_reg

# Step 4: Evaluate models
def evaluate_models(best_model_class, X_test, y_test_class, best_model_reg, X_test_reg, y_test_reg, label_encoders):
    # Evaluate classification model
    y_pred_class = best_model_class.predict(X_test)
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

    # Evaluate regression model
    y_pred_reg = best_model_reg.predict(X_test_reg)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)

    print(f"✅ Regression Model Metrics:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.4f}")

# Step 5: Save models and encoders
def save_models_and_encoders(best_model_class, best_model_reg, label_encoders):
    joblib.dump(best_model_class, "xgboost_metro_classifier.pkl")
    joblib.dump(best_model_reg, "xgboost_metro_regressor.pkl")
    joblib.dump(label_encoders, "label_encoders.pkl")
    print("✅ Models saved successfully.")

# Step 6: Feature importance analysis
def analyze_feature_importance(best_model_class, best_model_reg):
    # Classification feature importance
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(best_model_class, importance_type="weight")
    plt.title("Feature Importance - Classification")
    plt.show()

    # Regression feature importance
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(best_model_reg, importance_type="weight")
    plt.title("Feature Importance - Regression")
    plt.show()

# Step 7: Scenario analysis
def perform_scenario_analysis(best_model_class, best_model_reg, label_encoders, X_test):
    scenario_data = X_test.iloc[0:1].copy()  # Take a sample project
    scenario_data["Equipment Factor (%)"] = 90  # Simulate improved equipment

    risk_level = best_model_class.predict(scenario_data)[0]
    delay_days = best_model_reg.predict(scenario_data)[0]

    print(f"Scenario Analysis:")
    print(f"Risk Level: {label_encoders['Risk Level'].inverse_transform([risk_level])[0]}")
    print(f"Delay Time (days): {delay_days:.2f}")

# Main function
def main():
    # File path
    file_path = "backend\metro\metro_projects_risk_analysis.csv"

    # Step 1: Load and preprocess data
    X_train, X_test, y_train_class, y_test_class, X_train_reg, X_test_reg, y_train_reg, y_test_reg, label_encoders = load_and_preprocess_data(file_path)

    # Step 2: Build and tune classification model
    best_model_class = build_classification_model(X_train, y_train_class)

    # Step 3: Build and tune regression model
    best_model_reg = build_regression_model(X_train_reg, y_train_reg)

    # Step 4: Evaluate models
    evaluate_models(best_model_class, X_test, y_test_class, best_model_reg, X_test_reg, y_test_reg, label_encoders)

    # Step 5: Save models and encoders
    save_models_and_encoders(best_model_class, best_model_reg, label_encoders)

    # Step 6: Feature importance analysis
    analyze_feature_importance(best_model_class, best_model_reg)

    # Step 7: Scenario analysis
    perform_scenario_analysis(best_model_class, best_model_reg, label_encoders, X_test)

    # Clean up resources
    gc.collect()

# Run the main function
if __name__ == "__main__":
    main()