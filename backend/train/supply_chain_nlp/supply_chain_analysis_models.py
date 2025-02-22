import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error
from imblearn.over_sampling import SMOTE
import numpy as np
import joblib

# Step 1: Load and Preprocess Data

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ["Climate_Risk", "Delivery_Mode", "Inventory_Buffer_Level"]
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Feature Engineering
    df["Reliability_Adjusted_Score"] = df["Reliability_Score (0-1)"] * df["Quality_Rating (1-5)"]
    df["Geo_Proximity_Impact"] = df["Geo_Proximity (km)"] / (df["Lead_Time (days)"] + 1)
    df["Financial_Reliability_Index"] = df["Financial_Stability_Score"] * df["Reliability_Score (0-1)"]
    df["Historical_Delay_Trend"] = df["Historical_Delay (days)"].rolling(window=3, min_periods=1).mean()
    df["Supplier_Score"] = MinMaxScaler().fit_transform(
        df[["Quality_Rating (1-5)", "Financial_Stability_Score", "Reliability_Score (0-1)"]]
    ).mean(axis=1)
    
    df.fillna(df.median(), inplace=True)
    
    return df, label_encoders

# Step 2: Feature Importance Analysis

def feature_importance_analysis(model, feature_names):
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='coolwarm')
    plt.title('Feature Importance')
    plt.show()

# Step 3: Predictive Analytics with XGBoost and SMOTE

def train_model(df):
    features = [
        "Lead_Time (days)", "Quality_Rating (1-5)", "Financial_Stability_Score",
        "Geo_Proximity (km)", "Climate_Risk", "Inventory_Buffer_Level",
        "Reliability_Adjusted_Score", "Geo_Proximity_Impact", "Financial_Reliability_Index",
        "Historical_Delay_Trend"
    ]
    target = "Historical_Delay (days)"
    
    df["Delay_Risk"] = (df[target] >= 5).astype(int)
    X = df[features]
    y = df["Delay_Risk"]
    
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    # Hyperparameter tuning
    params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2]
    }
    xgb = XGBClassifier(random_state=42)
    grid = GridSearchCV(xgb, params, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Feature Importance Analysis
    feature_importance_analysis(best_model, features)
    
    # Save model
    joblib.dump(best_model, "xgboost_model.pkl")
    print("XGBoost model saved as xgboost_model.pkl")
    
    return best_model

# Step 4: Forecasting Lead Time with XGBoost Regressor

def forecast_supply(df):
    df["Month"] = np.random.randint(1, 13, size=len(df))  # Placeholder for actual dates
    X = df[["Month", "Supplier_Score"]]
    y = df["Lead_Time (days)"]
    
    model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X, y)
    future_months = np.array([[i, df["Supplier_Score"].mean()] for i in range(1, 13)])
    predictions = model.predict(future_months)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 13), predictions, marker='o', linestyle='-', color='green')
    plt.title("Forecasted Supply Lead Time by Month")
    plt.xlabel("Month")
    plt.ylabel("Predicted Lead Time (days)")
    plt.show()
    
    # Save model
    joblib.dump(model, "xgboost_regressor.pkl")
    print("XGBoost Regressor model saved as xgboost_regressor.pkl")
    
    return model

# Main Execution
if __name__ == "__main__":
    file_path = "C:\\Users\\Dell\\Downloads\\construction-optimisation\\backend\\train\\supply_chain_nlp\\construction_supply_chain_new.csv"  # Update with actual file path
    df, encoders = load_and_preprocess_data(file_path)
    model = train_model(df)
    forecast_supply(df)
