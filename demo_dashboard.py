import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the saved cost overrun model and scaler
cost_overrun_model = joblib.load('C:\\Users\\NAVYA\\Documents\\LNT- Hackathon\\project_cost_overrun_model.pkl')
scaler = joblib.load('C:\\Users\\NAVYA\\Documents\\LNT- Hackathon\\scaler.pkl')

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Cost Overrun Prediction Dashboard"),
    dcc.Input(id='project-size', type='number', placeholder='Project Size (sq. m)'),
    dcc.Input(id='labor-count', type='number', placeholder='Labor Count'),
    dcc.Input(id='equipment-count', type='number', placeholder='Equipment Count'),
    dcc.Input(id='avg-temp', type='number', placeholder='Avg Temperature (°C)'),
    dcc.Input(id='rainfall', type='number', placeholder='Rainfall (mm)'),
    dcc.Input(id='milestone', type='text', placeholder='Milestone'),
    dcc.Input(id='external-factor', type='text', placeholder='External Factor'),
    html.Button('Predict', id='predict-button', n_clicks=0),
    html.Div(id='prediction-output')
])

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    Input('project-size', 'value'),
    Input('labor-count', 'value'),
    Input('equipment-count', 'value'),
    Input('avg-temp', 'value'),
    Input('rainfall', 'value'),
    Input('milestone', 'value'),
    Input('external-factor', 'value')
)
def predict_cost_overrun(n_clicks, project_size, labor_count, equipment_count, avg_temp, rainfall, milestone, external_factor):
    if n_clicks > 0:
        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'Project Size (sq. m)': [project_size],
            'Labor Count': [labor_count],
            'Equipment Count': [equipment_count],
            'Avg Temperature (°C)': [avg_temp],
            'Rainfall (mm)': [rainfall],
            'Milestone': [milestone],
            'External Factor': [external_factor]
        })
        
        # One-hot encode categorical variables
        input_data = pd.get_dummies(input_data, drop_first=True)
        
        # Ensure all columns are present
        missing_cols = set(scaler.feature_names_in_) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0
        input_data = input_data[scaler.feature_names_in_]
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Predict cost overrun
        prediction = cost_overrun_model.predict(input_data_scaled)
        
        return f"Predicted Cost Overrun: {'Yes' if prediction[0] == 1 else 'No'}"
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)
