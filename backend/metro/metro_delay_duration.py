import pandas as pd
import numpy as np
import joblib
import optuna
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime

def load_and_preprocess_data(filepath):
    """Loads and preprocesses the metro construction delay dataset."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()  # Remove extra spaces in column names

    # Ensure target column exists
    target_column = "Delay time (days)"
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset!")

    # Drop low-importance features
    drop_columns = ["Project ID", "City", "Risk Level"]
    df.drop(columns=drop_columns, inplace=True, errors="ignore")

    # Handle missing values
    df.fillna(df.select_dtypes(include=["number"]).mean(), inplace=True)

    # Apply Log Transformation to Target Variable
    df[target_column] = np.log1p(df[target_column])  # log(1 + x)

    # Feature Scaling (Normalize large numerical columns)
    scaler = StandardScaler()
    numerical_cols = ["Tunnel Length (km)", "Num of Stations", "Num of Workers", 
                      "Budget (in Crores)", "Equipment Factor (%)", 
                      "Signal System Complexity (%)", "Material Factor (%)", 
                      "Gov Regulation Factor (%)", "Urban Congestion Impact (%)"]
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Define features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler, X.columns

def objective(trial):
    """Hyperparameter tuning with Optuna."""
    params = {
    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    "max_depth": trial.suggest_int("max_depth", 3, 12),
    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10, log=True),
    "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True),
    "objective": "reg:squarederror"
}

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)

def train_model(X_train, y_train):
    """Trains an XGBoost Regressor with Optuna hyperparameter tuning."""
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    best_params = study.best_params

    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)

    print(f"\n✅ Best Parameters: {best_params}")
    return best_model

def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model on test data."""
    y_pred = model.predict(X_test)

    # Convert log-transformed predictions back to original scale
    y_test_original = np.expm1(y_test)
    y_pred_original = np.expm1(y_pred)

    r2 = r2_score(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)

    print("\n📊 Model Performance:")
    print(f"✅ R² Score: {r2:.4f}")
    print(f"✅ MAE: {mae:.2f} days")
    print(f"✅ RMSE: {rmse:.2f} days")

    return r2, mae, rmse

def save_model(model, scaler, feature_names, r2, mae, rmse):
    """Saves the trained model, scaler, and feature names using joblib."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model_filename = f"xgboost_metro_regressor_r2-{r2:.4f}_rmse-{rmse:.2f}_time-{timestamp}.pkl"
    scaler_filename = f"scaler_{timestamp}.pkl"
    feature_names_filename = f"feature_names_{timestamp}.pkl"

    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(feature_names, feature_names_filename)

    print(f"\n💾 Model saved as '{model_filename}' ✅")
    print(f"💾 Scaler saved as '{scaler_filename}' ✅")
    print(f"💾 Feature names saved as '{feature_names_filename}' ✅")

if __name__ == "__main__":
    filepath = "metro_projects_risk_analysis.csv"
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data(filepath)

    print("\n✅ Data preprocessing complete.")
    print(f"🔹 X_train shape: {X_train.shape}")
    print(f"🔹 X_test shape: {X_test.shape}")

    # Train and tune the model
    best_model = train_model(X_train, y_train)

    # Evaluate on test set
    r2, mae, rmse = evaluate_model(best_model, X_test, y_test)

    # Save model and preprocessing tools
    save_model(best_model, scaler, feature_names, r2, mae, rmse)

    # Feature Importance Visualization
    plt.figure(figsize=(10, 6))
    importances = best_model.feature_importances_
    features = list(feature_names)
    plt.barh(features, importances)
    plt.xlabel("Feature Importance Score")
    plt.title("XGBoost Feature Importance - Metro Delay Prediction")
    plt.gca().invert_yaxis()
    plt.show()
