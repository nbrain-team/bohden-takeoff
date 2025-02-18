import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
data = pd.read_csv('/c:/Users/NAVYA/Documents/LNT- Hackathon/construction_delay_india.csv')

# Preprocess the data
X = data.drop(columns=['Project ID', 'City', 'Project Type', 'Start Date', 'End Date', 'Delayed', 'Delay Duration (days)', 'Delay Reason'])
y_delay_duration = data['Delay Duration (days)']
y_cost_overrun = data['Delayed'].astype(int)

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train_delay, y_test_delay = train_test_split(X, y_delay_duration, test_size=0.2, random_state=42)
_, _, y_train_cost, y_test_cost = train_test_split(X, y_cost_overrun, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the delay duration model
delay_model = LinearRegression()
delay_model.fit(X_train_scaled, y_train_delay)

# Train the cost overrun model
cost_overrun_model = LogisticRegression()
cost_overrun_model.fit(X_train_scaled, y_train_cost)

# Save the models and scaler
joblib.dump(delay_model, '/c:/Users/NAVYA/Documents/LNT- Hackathon/project_delay_duration_model.pkl')
joblib.dump(cost_overrun_model, '/c:/Users/NAVYA/Documents/LNT- Hackathon/project_cost_overrun_model.pkl')
joblib.dump(scaler, '/c:/Users/NAVYA/Documents/LNT- Hackathon/scaler.pkl')

print("Models trained and saved successfully.")
