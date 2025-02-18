import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
data = pd.read_csv('/c:/Users/NAVYA/Documents/LNT- Hackathon/construction_delay_india.csv')

# Preprocess the data
data['Start Date'] = pd.to_datetime(data['Start Date'])
data['End Date'] = pd.to_datetime(data['End Date'])
data['Risk Factor'] = data['Delay Reason'].apply(lambda x: 1 if x in ['Labor strike', 'Equipment failure', 'Regulatory approval delay'] else 0)

# Feature selection
features = ['Project Size (sq. m)', 'Labor Count', 'Equipment Count', 'Avg Temperature (°C)', 'Rainfall (mm)', 'Milestone', 'External Factor']
X = pd.get_dummies(data[features], drop_first=True)
y = data['Risk Factor']

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
    'criterion': ['gini', 'entropy']
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Train the best model
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cross_val_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
print(f"Cross-Validation Scores: {cross_val_scores}")
print(f"Mean Cross-Validation Score: {cross_val_scores.mean()}")

# Save the model and scaler
joblib.dump(best_model, '/c:/Users/NAVYA/Documents/LNT- Hackathon/project_risk_factor_model.pkl')
joblib.dump(scaler, '/c:/Users/NAVYA/Documents/LNT- Hackathon/scaler.pkl')
