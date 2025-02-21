import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load trained model & feature names
model_filename = "xgboost_metro_regressor_r2--0.1995_rmse-114.58_time-2025-02-21_14-43-17.pkl"  # Update if using timestamped filenames
feature_names_filename = "feature_names_2025-02-21_14-43-17.pkl"

# Load model and features
model = joblib.load(model_filename)
feature_names = joblib.load(feature_names_filename)

# Get feature importance
importances = model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

# Print feature importance ranking
print("\n📊 Feature Importance Ranking:")
for i in sorted_indices:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh([feature_names[i] for i in sorted_indices], importances[sorted_indices])
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("XGBoost Feature Importance - Metro Delay Prediction")
plt.gca().invert_yaxis()  # Flip for better readability
plt.show()
