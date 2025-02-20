from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS
import logging
from ultralytics import YOLO
import os

app = Flask(__name__)
CORS(app)  # Enable CORS

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the saved cost overrun model and scaler
cost_overrun_model = joblib.load('C:\\Users\\NAVYA\\Documents\\LNT- Hackathon\\backend\\models\\project_cost_overrun_model.pkl')
scaler = joblib.load('C:\\Users\\NAVYA\\Documents\\LNT- Hackathon\\backend\\models\\scaler.pkl')

# Load the YOLO model
yolo_model = YOLO("best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    logging.debug(f"Received data: {data}")
    
    input_data = pd.DataFrame({
        'Project Size (sq. m)': [data['project_size']],
        'Labor Count': [data['labor_count']],
        'Equipment Count': [data['equipment_count']],
        'Avg Temperature (°C)': [data['avg_temp']],
        'Rainfall (mm)': [data['rainfall']],
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
    
    # Predict cost overrun
    prediction = cost_overrun_model.predict(input_data_scaled)
    
    logging.debug(f"Prediction: {prediction}")
    
    return jsonify({'prediction': 'Yes' if prediction[0] == 1 else 'No'})

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        logging.error("No file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logging.error("No file selected")
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save the uploaded file to a temporary directory
        temp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)
        
        # Perform inference using the YOLO model
        save_dir = os.path.join(os.getcwd(), "runs", "detect", "predict")
        os.makedirs(save_dir, exist_ok=True)
        results = yolo_model.predict(file_path, save=True, save_dir=save_dir)
        
        # Return the path to the saved annotated image
        annotated_image_path = os.path.join(save_dir, file.filename)
        return jsonify({'annotated_image_path': annotated_image_path})
    except Exception as e:
        logging.error(f"Error during detection: {e}")
        return jsonify({'error': 'Detection failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)