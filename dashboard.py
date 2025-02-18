import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import requests

# Load the saved models and scalers
delay_model = joblib.load('/c:/Users/NAVYA/Documents/LNT- Hackathon/project_delay_duration_model.pkl')
cost_overrun_model = joblib.load('/c:/Users/NAVYA/Documents/LNT- Hackathon/project_cost_overrun_model.pkl')
risk_factor_model = joblib.load('/c:/Users/NAVYA/Documents/LNT- Hackathon/project_risk_factor_model.pkl')
supply_chain_model = joblib.load('/c:/Users/NAVYA/Documents/LNT- Hackathon/project_supply_chain_model.pkl')
workforce_optimization_model = joblib.load('/c:/Users/NAVYA/Documents/LNT- Hackathon/project_workforce_optimization_model.pkl')
completion_time_model = joblib.load('/c:/Users/NAVYA/Documents/LNT- Hackathon/project_completion_time_model.pkl')
cost_efficiency_model = joblib.load('/c:/Users/NAVYA/Documents/LNT- Hackathon/project_cost_efficiency_model.pkl')
scaler = joblib.load('/c:/Users/NAVYA/Documents/LNT- Hackathon/scaler.pkl')

# Function to fetch real-time weather data
def fetch_weather_data(city):
    api_key = 'your_api_key'
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'
    response = requests.get(url)
    data = response.json()
    return data['main']['temp'], data['main']['humidity'], data['wind']['speed']

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
    prediction = delay_model.predict(new_data_scaled)
    return prediction[0]

# Function to predict cost overrun
def predict_cost_overrun(new_data):
    new_data_scaled = preprocess_data(new_data)
    prediction = cost_overrun_model.predict(new_data_scaled)
    return 'Yes' if prediction[0] else 'No'

# Function to predict risk factors
def predict_risk_factors(new_data):
    new_data_scaled = preprocess_data(new_data)
    prediction = risk_factor_model.predict(new_data_scaled)
    return 'Yes' if prediction[0] else 'No'

# Function to predict supply chain disruptions
def predict_supply_chain_disruption(new_data):
    new_data_scaled = preprocess_data(new_data)
    prediction = supply_chain_model.predict(new_data_scaled)
    return 'Yes' if prediction[0] else 'No'

# Function to predict workforce optimization
def predict_workforce_optimization(new_data):
    new_data_scaled = preprocess_data(new_data)
    prediction = workforce_optimization_model.predict(new_data_scaled)
    return prediction[0]

# Function to predict completion time
def predict_completion_time(new_data):
    new_data_scaled = preprocess_data(new_data)
    prediction = completion_time_model.predict(new_data_scaled)
    return prediction[0]

# Function to predict cost efficiency
def predict_cost_efficiency(new_data):
    new_data_scaled = preprocess_data(new_data)
    prediction = cost_efficiency_model.predict(new_data_scaled)
    return prediction[0]

# Function to provide solutions for handling delays
def provide_delay_solutions(delay_duration):
    if delay_duration > 30:
        return "Consider increasing labor count or extending working hours."
    elif delay_duration > 15:
        return "Review project milestones and adjust timelines."
    else:
        return "Monitor project closely and address any emerging issues promptly."

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Project Management Dashboard"),
    dcc.Input(id='input-project-size', type='number', placeholder='Project Size (sq. m)'),
    dcc.Input(id='input-labor-count', type='number', placeholder='Labor Count'),
    dcc.Input(id='input-equipment-count', type='number', placeholder='Equipment Count'),
    dcc.Input(id='input-city', type='text', placeholder='City'),
    dcc.Input(id='input-weather-conditions', type='text', placeholder='Weather Conditions'),
    dcc.Input(id='input-milestone', type='text', placeholder='Milestone'),
    dcc.Input(id='input-external-factor', type='text', placeholder='External Factor'),
    html.Button('Predict', id='predict-button'),
    html.Div(id='output-delay-duration'),
    html.Div(id='output-cost-overrun'),
    html.Div(id='output-risk-factors'),
    html.Div(id='output-supply-chain-disruption'),
    html.Div(id='output-workforce-optimization'),
    html.Div(id='output-completion-time'),
    html.Div(id='output-cost-efficiency'),
    html.Div(id='output-delay-solutions'),
    dcc.Graph(id='prediction-graph')
])

@app.callback(
    [Output('output-delay-duration', 'children'),
     Output('output-cost-overrun', 'children'),
     Output('output-risk-factors', 'children'),
     Output('output-supply-chain-disruption', 'children'),
     Output('output-workforce-optimization', 'children'),
     Output('output-completion-time', 'children'),
     Output('output-cost-efficiency', 'children'),
     Output('output-delay-solutions', 'children'),
     Output('prediction-graph', 'figure')],
    [Input('predict-button', 'n_clicks')],
    [Input('input-project-size', 'value'),
     Input('input-labor-count', 'value'),
     Input('input-equipment-count', 'value'),
     Input('input-city', 'value'),
     Input('input-weather-conditions', 'value'),
     Input('input-milestone', 'value'),
     Input('input-external-factor', 'value')]
)
def update_output(n_clicks, project_size, labor_count, equipment_count, city, weather_conditions, milestone, external_factor):
    if n_clicks is None:
        return "", "", "", "", "", "", "", "", {}
    
    # Fetch real-time weather data
    avg_temp, humidity, wind_speed = fetch_weather_data(city)
    
    new_data = pd.DataFrame({
        'Project Size (sq. m)': [project_size],
        'Labor Count': [labor_count],
        'Equipment Count': [equipment_count],
        'Avg Temperature (°C)': [avg_temp],
        'Rainfall (mm)': [0],  # Placeholder, replace with actual data if available
        'Humidity': [humidity],
        'Wind Speed': [wind_speed],
        'Weather Conditions': [weather_conditions],
        'Milestone': [milestone],
        'External Factor': [external_factor]
    })
    
    delay_duration = predict_delay_duration(new_data)
    cost_overrun = predict_cost_overrun(new_data)
    risk_factors = predict_risk_factors(new_data)
    supply_chain_disruption = predict_supply_chain_disruption(new_data)
    workforce_optimization = predict_workforce_optimization(new_data)
    completion_time = predict_completion_time(new_data)
    cost_efficiency = predict_cost_efficiency(new_data)
    
    # Provide solutions for handling delays
    delay_solutions = provide_delay_solutions(delay_duration)
    
    # Create a bar chart for the predictions
    predictions = {
        'Delay Duration (days)': delay_duration,
        'Cost Overrun': cost_overrun,
        'Risk Factors': risk_factors,
        'Supply Chain Disruption': supply_chain_disruption,
        'Workforce Optimization': workforce_optimization,
        'Completion Time (days)': completion_time,
        'Cost Efficiency': cost_efficiency
    }
    fig = px.bar(x=list(predictions.keys()), y=list(predictions.values()), title='Project Predictions')
    
    return (f"Predicted Delay Duration (days): {delay_duration}",
            f"Cost Overrun: {cost_overrun}",
            f"Risk Factors: {risk_factors}",
            f"Supply Chain Disruption: {supply_chain_disruption}",
            f"Workforce Optimization: {workforce_optimization}",
            f"Completion Time (days): {completion_time}",
            f"Cost Efficiency: {cost_efficiency}",
            delay_solutions,
            fig)

if __name__ == '__main__':
    app.run_server(debug=True)
