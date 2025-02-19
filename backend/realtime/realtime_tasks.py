import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the saved model and scaler
model = joblib.load('/c:/Users/NAVYA/Documents/LNT- Hackathon/backend/models/project_delay_duration_model.pkl')
scaler = joblib.load('/c:/Users/NAVYA/Documents/LNT- Hackathon/backend/models/scaler.pkl')

@app.route('/realtime_predict', methods=['POST'])
def realtime_predict():
    data = request.json
    logging.debug(f"Received data: {data}")
    
    input_data = pd.DataFrame({
        'Project Size (sq. m)': [data['project_size']],
        'Labor Count': [data['labor_count']],
        'Equipment Count': [data['equipment_count']],
        'Avg Temperature (°C)': [data['avg_temp']],
        'Rainfall (mm)': [data['rainfall']],
        'Humidity': [data['humidity']],
        'Wind Speed': [data['wind_speed']],
        'Weather Conditions': [data['weather_conditions']],
        'Milestone': [data['milestone']],
        'External Factor': [data['external_factor']]
    })
    
    logging.debug(f"Input data before preprocessing: {input_data}")
    
    # One-hot encode categorical variables
    input_data = pd.get_dummies(input_data, drop_first=True)
    
    # Ensure all columns are present
    missing_cols = set(scaler.feature_names_in_) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0
    input_data = input_data[scaler.feature_names_in_]
    
    logging.debug(f"Preprocessed input data: {input_data}")
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    logging.debug(f"Scaled input data: {input_data_scaled}")
    
    # Predict delay duration
    prediction = model.predict(input_data_scaled)
    
    logging.debug(f"Prediction: {prediction}")
    
    return jsonify({'predicted_delay_duration': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
