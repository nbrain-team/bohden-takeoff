import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load test dataset

df_test = pd.read_csv("../../construction_test_dataset.csv")

# Load trained model

model = joblib.load("model.pkl")
print("✅ Model loaded successfully.")

# Preprocess test data
if "Delay Duration (days)" in df_test.columns:
    y_test = df_test["Delay Duration (days)"]
    X_test = df_test.drop(columns=["Delay Duration (days)"])
else:
    y_test = None  # No actual delay values in test set
    X_test = df_test

# Drop non-numeric columns that are not needed
drop_cols = ["Delayed"]
X_test = X_test.drop(columns=drop_cols, errors="ignore")

# Convert categorical columns using Label Encoding
categorical_cols = X_test.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    le = LabelEncoder()
    X_test[col] = le.fit_transform(X_test[col].astype(str))

# Ensure numeric data type
X_test = X_test.astype(float)

# Make predictions
y_pred = model.predict(X_test)

# Compute metrics if actual values exist
if y_test is not None:
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
else:
    r2, mae, rmse = "N/A", "N/A", "N/A"

# Generate visualizations
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.xlabel("Actual Delay Duration (days)")
plt.ylabel("Predicted Delay Duration (days)")
plt.title("Actual vs Predicted Delay Duration")
plt.savefig("actual_vs_predicted.png")
plt.close()

# Feature Importance
plt.figure(figsize=(10, 6))
feature_importance = pd.Series(model.feature_importances_, index=X_test.columns).nlargest(10)
feature_importance.plot(kind='barh', color='blue')
plt.xlabel("Importance Score")
plt.title("Top 10 Feature Importances")
plt.savefig("feature_importance.png")
plt.close()

# Generate PDF report
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(200, 10, "Construction Delay Regression Report", ln=True, align="C")
pdf.ln(10)

pdf.set_font("Arial", size=12)

# Summary section
pdf.cell(0, 10, "Summary of Model Performance", ln=True, align="L")
pdf.ln(5)
pdf.cell(0, 10, f"Total Projects Evaluated: {len(df_test)}", ln=True)
pdf.cell(0, 10, f"R² Score: {r2}", ln=True)
pdf.cell(0, 10, f"Mean Absolute Error (MAE): {mae} days", ln=True)
pdf.cell(0, 10, f"Root Mean Squared Error (RMSE): {rmse} days", ln=True)
pdf.ln(10)

# Add Visualizations
pdf.cell(0, 10, "Actual vs Predicted Delay Duration", ln=True, align="L")
pdf.image("actual_vs_predicted.png", x=10, w=180)
pdf.ln(10)

pdf.cell(0, 10, "Feature Importance Analysis", ln=True, align="L")
pdf.image("feature_importance.png", x=10, w=180)
pdf.ln(10)

# Detailed Report
pdf.cell(0, 10, "Sample Predictions", ln=True, align="L")
pdf.ln(5)
for i in range(min(10, len(df_test))):
    pdf.cell(0, 10, f"Project {i+1}: Predicted Delay Duration: {y_pred[i]:.2f} days", ln=True)
pdf.ln(10)

pdf.output("regression_report.pdf")
print("✅ PDF report generated: regression_report.pdf")
