import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
data = pd.read_csv('/c:/Users/NAVYA/Documents/LNT- Hackathon/data/construction_delay_india.csv')

# Preprocess the data
X = data[['Project Size (sq. m)', 'Labor Count', 'Equipment Count', 'Avg Temperature (°C)', 'Rainfall (mm)', 'Milestone', 'External Factor']]
y = data['Delay Reason']

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)
y = pd.get_dummies(y, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Save the trained model and scaler
joblib.dump(model, '/c:/Users/NAVYA/Documents/LNT- Hackathon/backend/models/project_risk_factors_model.pkl')
joblib.dump(scaler, '/c:/Users/NAVYA/Documents/LNT- Hackathon/backend/models/scaler.pkl')

# Evaluate the model
accuracy = model.score(X_test_scaled, y_test)
print(f"Model Accuracy: {accuracy}")
