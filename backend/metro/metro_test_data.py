# Re-run the test data generation since execution state was reset

import numpy as np
import pandas as pd

# Generate test dataset for Metro project risk analysis
num_test_projects = 5000  # Define the number of test projects

# Create test dataset with realistic Metro project attributes
np.random.seed(99)  # For reproducibility
df_test_metro = pd.DataFrame({
    "Project ID": [f"TEST-METRO-{i+1:04d}" for i in range(num_test_projects)],
    "City": np.random.choice(["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai", "Kolkata"], num_test_projects),
    "Tunnel Length (km)": np.random.uniform(1, 50, size=num_test_projects).round(2),
    "Num of Stations": np.random.randint(5, 100, size=num_test_projects),
    "Num of Workers": np.random.randint(100, 5000, size=num_test_projects),
    "Equipment Factor (%)": np.random.randint(0, 100, size=num_test_projects),
    "Signal System Complexity (%)": np.random.randint(0, 100, size=num_test_projects),
    "Material Factor (%)": np.random.randint(0, 100, size=num_test_projects),
    "Gov Regulation Factor (%)": np.random.randint(0, 100, size=num_test_projects),
    "Urban Congestion Impact (%)": np.random.randint(0, 100, size=num_test_projects),
    "Expected Completion Time (months)": np.random.randint(24, 120, size=num_test_projects),
    "Budget (in Crores)": np.random.uniform(500, 5000, size=num_test_projects).round(2),
})

# Save the test dataset
test_file_path = "metro_test_data.csv"
df_test_metro.to_csv(test_file_path, index=False)
test_file_path
