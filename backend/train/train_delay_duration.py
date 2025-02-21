import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def load_and_preprocess_data(filepath):
    """Loads and preprocesses the construction delay dataset."""
    df = pd.read_csv(filepath)

    df.columns = df.columns.str.strip()

    
    if "Delayed" in df.columns:
        df.drop(columns=["Delayed"], inplace=True)

    
    df.fillna(df.select_dtypes(include=["number"]).mean(), inplace=True)

    
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

   
    target_column = "Delay Duration (days)" 
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset!")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoders

def train_model(X_train, y_train):
    """Trains an XGBoost Regressor and performs hyperparameter tuning using GridSearchCV."""
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200, 300],
        'reg_alpha': [0, 1, 5],
        'reg_lambda': [1, 5, 10]
    }

    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    grid_search = GridSearchCV(model, param_grid, scoring='r2', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"\n✅ Best Parameters: {grid_search.best_params_}")
    print(f"✅ Best Cross-Validation R²: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model on test data."""
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print("\n📊 Model Performance:")
    print(f"✅ R² Score: {r2:.4f}")

def save_model(model, filename="model.pkl"):
    """Saves the trained model using joblib."""
    joblib.dump(model, filename)
    print(f"\n💾 Model saved as '{filename}' ✅")



  

if __name__ == "__main__":
    filepath = "C:\\Users\\Dell\\Downloads\\construction-optimisation\\construction_delay_india.csv"  # Update with your file path
    X_train, X_test, y_train, y_test, encoders = load_and_preprocess_data(filepath)

    print("\n✅ Data preprocessing complete.")
    print(f"🔹 X_train shape: {X_train.shape}")
    print(f"🔹 X_test shape: {X_test.shape}")

    # Train and tune the model
    best_model = train_model(X_train, y_train)

    # Evaluate on test set
    evaluate_model(best_model, X_test, y_test)

    save_model(best_model)


