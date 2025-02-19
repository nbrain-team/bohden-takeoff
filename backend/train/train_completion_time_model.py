import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
data = pd.read_csv('/c:/Users/NAVYA/Documents/LNT- Hackathon/construction_delay_india.csv')

# Preprocess the data
X = data.drop(columns=['Project ID', 'City', 'Project Type', 'Start Date', 'End Date', 'Delayed', 'Delay Duration (days)', 'Delay Reason'])
y = data['Delay Duration (days)']  # Assuming 'Delay Duration (days)' is the target for completion time

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the completion time model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(model, 'C:\\Users\\NAVYA\\Documents\\LNT- Hackathon\\project_completion_time_model.pkl')
joblib.dump(scaler, 'C:\\Users\\NAVYA\\Documents\\LNT- Hackathon\\scaler.pkl')

print("Completion time model trained and saved successfully.")
