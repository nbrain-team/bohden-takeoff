import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Load the saved model and scaler
model = joblib.load('/c:/Users/NAVYA/Documents/LNT- Hackathon/project_delay_duration_model.pkl')
scaler = joblib.load('/c:/Users/NAVYA/Documents/LNT- Hackathon/scaler.pkl')

# Function to preprocess data
def preprocess_data(new_data):
    new_data = pd.get_dummies(new_data, drop_first=True)
    missing_cols = set(X.columns) - set(new_data.columns)
    for col in missing_cols:
        new_data[col] = 0
    new_data = new_data[X.columns]
    new_data_scaled = scaler.transform(new_data)
    return new_data_scaled

# Function to predict delay duration
def predict_delay_duration(new_data):
    new_data_scaled = preprocess_data(new_data)
    prediction = model.predict(new_data_scaled)
    return prediction[0]

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Real-Time Project Monitoring Dashboard"),
    dcc.Input(id='input-project-size', type='number', placeholder='Project Size (sq. m)'),
    dcc.Input(id='input-labor-count', type='number', placeholder='Labor Count'),
    dcc.Input(id='input-equipment-count', type='number', placeholder='Equipment Count'),
    dcc.Input(id='input-avg-temp', type='number', placeholder='Avg Temperature (°C)'),
    dcc.Input(id='input-rainfall', type='number', placeholder='Rainfall (mm)'),
    dcc.Input(id='input-humidity', type='number', placeholder='Humidity'),
    dcc.Input(id='input-wind-speed', type='number', placeholder='Wind Speed'),
    dcc.Input(id='input-weather-conditions', type='text', placeholder='Weather Conditions'),
    dcc.Input(id='input-milestone', type='text', placeholder='Milestone'),
    dcc.Input(id='input-external-factor', type='text', placeholder='External Factor'),
    html.Button('Predict Delay', id='predict-button'),
    html.Div(id='output-prediction')
])

@app.callback(
    Output('output-prediction', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input('input-project-size', 'value'),
     Input('input-labor-count', 'value'),
     Input('input-equipment-count', 'value'),
     Input('input-avg-temp', 'value'),
     Input('input-rainfall', 'value'),
     Input('input-humidity', 'value'),
     Input('input-wind-speed', 'value'),
     Input('input-weather-conditions', 'value'),
     Input('input-milestone', 'value'),
     Input('input-external-factor', 'value')]
)
def update_output(n_clicks, project_size, labor_count, equipment_count, avg_temp, rainfall, humidity, wind_speed, weather_conditions, milestone, external_factor):
    if n_clicks is None:
        return ""
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
    predicted_delay = predict_delay_duration(new_data)
    return f"Predicted Delay Duration (days): {predicted_delay}"

if __name__ == '__main__':
    app.run_server(debug=True)
