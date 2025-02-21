import pandas as pd
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from fpdf import FPDF
import os

# Load test dataset
df_test = pd.read_csv("test_construction_delay_large.csv")

# Load training dataset columns for alignment
train_columns = joblib.load("train_columns.pkl")  # Ensure this file exists

# Preprocess test data
df_test = pd.get_dummies(df_test, columns=["City", "Project Type", "External Factor", "Delay Reason"], drop_first=True)
if "Milestone" in df_test.columns:
    df_test["Milestone"] = pd.to_numeric(df_test["Milestone"], errors="coerce").fillna(0)

# Align columns with training data
for col in train_columns:
    if col not in df_test.columns:
        df_test[col] = 0  # Add missing columns with default value 0

# Ensure only relevant features are used
df_test = df_test.reindex(columns=train_columns, fill_value=0)

# Load trained model
model_path = "best_xgb_model.json"
if os.path.exists(model_path):
    loaded_model = XGBClassifier()
    loaded_model.load_model(model_path)
    print("✅ Model loaded successfully.")
else:
    raise FileNotFoundError("❌ Model file not found. Retrain the model or check the path.")

# Make predictions
y_pred = loaded_model.predict(df_test)
df_test["Predicted Delay"] = y_pred

# Save predictions
df_test.to_csv("predicted_report.csv", index=False)
print("✅ Predictions saved to predicted_report.csv")

# Ensure "Delay Duration (days)" exists before calculating average
if "Delay Duration (days)" in df_test.columns:
    average_delay = df_test.loc[df_test["Predicted Delay"] == 1, "Delay Duration (days)"].mean()
else:
    average_delay = "N/A"  # Handle missing data

# Common delay reasons
if "Delay Reason" in df_test.columns:
    common_reasons = df_test["Delay Reason"].value_counts()
else:
    common_reasons = {}

# Generate Bar Chart for Delay Distribution
plt.figure(figsize=(8, 5))
df_test["Predicted Delay"].value_counts().plot(kind='bar', color=['green', 'red'])
plt.xticks(ticks=[0, 1], labels=['On-Time', 'Delayed'], rotation=0)
plt.xlabel("Project Status")
plt.ylabel("Number of Projects")
plt.title("Construction Delay Distribution")
plt.savefig("delay_distribution.png")
plt.close()

# Generate PDF report
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(200, 10, "Construction Delay Prediction Report", ln=True, align="C")
pdf.ln(10)

pdf.set_font("Arial", size=12)

# Summary section
pdf.cell(0, 10, "Summary of Predictions", ln=True, align="L")
pdf.ln(5)
pdf.cell(0, 10, f"Total Projects: {len(df_test)}", ln=True)
pdf.cell(0, 10, f"Delayed Projects: {df_test['Predicted Delay'].sum()}", ln=True)
pdf.cell(0, 10, f"On-Time Projects: {len(df_test) - df_test['Predicted Delay'].sum()}", ln=True)
pdf.cell(0, 10, f"Average Delay Duration: {average_delay if average_delay != 'N/A' else 'N/A'} days", ln=True)
pdf.ln(10)

# Common Delay Reasons
pdf.cell(0, 10, "Most Common Delay Reasons", ln=True, align="L")
pdf.ln(5)
if common_reasons:
    for reason, count in common_reasons.items():
        pdf.cell(0, 10, f"{reason}: {count} occurrences", ln=True)
else:
    pdf.cell(0, 10, "No delay reasons available.", ln=True)
pdf.ln(10)

# Add Bar Chart to PDF
pdf.image("delay_distribution.png", x=10, w=180)
pdf.ln(10)

# Detailed Report
pdf.cell(0, 10, "Detailed Project Analysis", ln=True, align="L")
pdf.ln(5)
if "Project ID" in df_test.columns:
    for i, row in df_test.iterrows():
        pdf.cell(0, 10, f"Project ID: {row['Project ID']}", ln=True)
        pdf.cell(0, 10, f"Predicted Delay: {'Yes' if row['Predicted Delay'] else 'No'}", ln=True)
        pdf.cell(0, 10, f"Delay Duration (if delayed): {row.get('Delay Duration (days)', 'N/A')}", ln=True)
        pdf.cell(0, 10, f"External Factor: {row.get('External Factor', 'N/A')}", ln=True)
        pdf.cell(0, 10, f"Delay Reason: {row.get('Delay Reason', 'N/A')}", ln=True)
        pdf.ln(10)
else:
    for i, row in df_test.iterrows():
        pdf.cell(0, 10, f"Row {i+1} - Predicted Delay: {'Yes' if row['Predicted Delay'] else 'No'}", ln=True)
        pdf.ln(5)

pdf.output("predicted_report.pdf")
print("✅ PDF report generated: predicted_report.pdf")


