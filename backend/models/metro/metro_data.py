import numpy as np
import pandas as pd

# Define the number of Metro projects
num_projects = 20000 

# Generate Metro-specific project data
np.random.seed(42)  # For reproducibility
df_metro = pd.DataFrame({
    "Project ID": [f"METRO-{i+1:04d}" for i in range(num_projects)],
    "City": np.random.choice(["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai", "Kolkata"], num_projects),
    "Tunnel Length (km)": np.random.uniform(1, 50, size=num_projects).round(2),
    "Num of Stations": np.random.randint(5, 100, size=num_projects),
    "Num of Workers": np.random.randint(100, 5000, size=num_projects),
    "Equipment Factor (%)": np.random.randint(0, 100, size=num_projects),
    "Signal System Complexity (%)": np.random.randint(0, 100, size=num_projects),
    "Material Factor (%)": np.random.randint(0, 100, size=num_projects),
    "Gov Regulation Factor (%)": np.random.randint(0, 100, size=num_projects),
    "Urban Congestion Impact (%)": np.random.randint(0, 100, size=num_projects),
    "Expected Completion Time (months)": np.random.randint(24, 120, size=num_projects),
    "Budget (in Crores)": np.random.uniform(500, 5000, size=num_projects).round(2),
    "Delay time (days)": np.random.randint(0, 365, size=num_projects),
})

# Generate a 'Risk Level' column based on weighted factors
risk_score_metro = (
    df_metro["Equipment Factor (%)"] * 0.2 +
    df_metro["Signal System Complexity (%)"] * 0.2 +
    df_metro["Material Factor (%)"] * 0.2 +
    df_metro["Gov Regulation Factor (%)"] * 0.25 +
    df_metro["Urban Congestion Impact (%)"] * 0.15
)

df_metro["Risk Level"] = pd.cut(risk_score_metro, bins=[0, 40, 70, 100], labels=["Low", "Medium", "High"])

# Save the new Metro-specific dataset
metro_file_path = "metro_projects_risk_analysis.csv"
df_metro.to_csv(metro_file_path, index=False)
print(f"✅ Metro risk analysis dataset saved as '{metro_file_path}'")
