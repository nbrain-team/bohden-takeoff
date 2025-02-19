import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import schedule
import time
from selenium import webdriver
from selenium.webdriver.common.by import By

# Load the saved model and scaler
model = joblib.load('/c:/Users/NAVYA/Documents/LNT- Hackathon/backend/models/project_delay_duration_model.pkl')
scaler = joblib.load('/c:/Users/NAVYA/Documents/LNT- Hackathon/backend/models/scaler.pkl')

# Function to automate data collection
def collect_data():
    # Example: Using Selenium to scrape data from a website
    driver = webdriver.Chrome()
    driver.get('https://example.com/construction-data')
    
    # Extract data (this is just an example, adjust selectors as needed)
    project_size = driver.find_element(By.ID, 'project-size').text
    labor_count = driver.find_element(By.ID, 'labor-count').text
    equipment_count = driver.find_element(By.ID, 'equipment-count').text
    avg_temp = driver.find_element(By.ID, 'avg-temp').text
    rainfall = driver.find_element(By.ID, 'rainfall').text
    humidity = driver.find_element(By.ID, 'humidity').text
    wind_speed = driver.find_element(By.ID, 'wind-speed').text
    weather_conditions = driver.find_element(By.ID, 'weather-conditions').text
    milestone = driver.find_element(By.ID, 'milestone').text
    external_factor = driver.find_element(By.ID, 'external-factor').text
    
    driver.quit()
    
    # Create a DataFrame with the collected data
    new_data = pd.DataFrame({
        'Project Size (sq. m)': [project_size],
        'Labor Count': [labor_count],
        'Equipment Count': [equipment_count],
        'Avg Temperature (°C)': [avg_temp],
        'Rainfall (mm)': [rainfall],
        'Humidity': [humidity],
        'Wind Speed': [wind_speed],
        'Weather Conditions': [weather_conditions],
        'Milestone': [milestone],
        'External Factor': [external_factor]
    })
    
    return new_data

# Function to automate data preprocessing
def preprocess_data(new_data):
    new_data = pd.get_dummies(new_data, drop_first=True)
    missing_cols = set(scaler.feature_names_in_) - set(new_data.columns)
    for col in missing_cols:
        new_data[col] = 0
    new_data = new_data[scaler.feature_names_in_]
    new_data_scaled = scaler.transform(new_data)
    return new_data_scaled

# Function to automate prediction
def predict_delay_duration(new_data):
    new_data_scaled = preprocess_data(new_data)
    prediction = model.predict(new_data_scaled)
    return prediction[0]

# Function to automate the entire process
def automate_process():
    new_data = collect_data()
    predicted_delay = predict_delay_duration(new_data)
    print(f"Predicted Delay Duration (days): {predicted_delay}")

# Schedule the automation to run every day at a specific time
schedule.every().day.at("10:00").do(automate_process)

# Keep the script running
while True:
    schedule.run_pending()
    time.sleep(1)
