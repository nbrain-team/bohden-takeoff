import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
data = pd.read_csv('/c:/Users/NAVYA/Documents/LNT- Hackathon/construction_delay_india.csv')

# Preprocess the data
data['Start Date'] = pd.to_datetime(data['Start Date'])
data['End Date'] = pd.to_datetime(data['End Date'])
data['Cost Overrun'] = data['Cost Overrun'].fillna(0)  # Assuming a column 'Cost Overrun' exists

# Feature selection
features = ['Project Size (sq. m)', 'Labor Count', 'Equipment Count', 'Avg Temperature (°C)', 'Rainfall (mm)', 'Milestone', 'External Factor']
X = pd.get_dummies(data[features], drop_first=True)
y = data['Cost Overrun']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 6, 8, 10],
    'criterion': ['mse', 'mae']
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# Train the best model
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
cross_val_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_absolute_error')

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Cross-Validation Scores: {cross_val_scores}")
print(f"Mean Cross-Validation Score: {cross_val_scores.mean()}")

# Save the model and scaler
joblib.dump(best_model, '/c:/Users/NAVYA/Documents/LNT- Hackathon/project_cost_efficiency_model.pkl')
joblib.dump(scaler, '/c:/Users/NAVYA/Documents/LNT- Hackathon/scaler.pkl')
