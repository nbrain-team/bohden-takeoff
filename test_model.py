import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler
model = joblib.load('/c:/Users/NAVYA/Documents/LNT- Hackathon/project_delay_duration_model.pkl')
scaler = joblib.load('/c:/Users/NAVYA/Documents/LNT- Hackathon/scaler.pkl')

# New data for prediction
new_data = pd.DataFrame({
    'Project Size (sq. m)': [50000],
    'Labor Count': [1200],
    'Equipment Count': [30],
    'Avg Temperature (°C)': [25],
    'Rainfall (mm)': [200],
    'Humidity': [60],
    'Wind Speed': [15],
    'Weather Conditions': ['Clear'],
    'Milestone': ['Inspection'],
    'External Factor': ['Funding issues'],
    'Snowfall (cm)': [0],
    'Storms': [0]
})

# Preprocess the new data
new_data = pd.get_dummies(new_data, drop_first=True)
# Ensure all columns match the training data
missing_cols = set(X.columns) - set(new_data.columns)
for col in missing_cols:
    new_data[col] = 0
new_data = new_data[X.columns]

# Scale the features
new_data_scaled = scaler.transform(new_data)

# Make predictions
prediction = model.predict(new_data_scaled)
print(f"Predicted Delay Duration (days): {prediction[0]}")
