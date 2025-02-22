import pandas as pd
import numpy as np
import random

# Define possible values for categorical fields
locations = ["Mumbai", "Delhi", "Chennai", "Kolkata", "Bangalore", "Hyderabad"]
materials = ["Steel", "Cement", "Bricks", "Wood", "Glass", "Concrete"]
delivery_modes = ["Road", "Rail", "Ship", "Air"]
climate_risk_levels = ["Low", "Medium", "High"]

# Generate synthetic supplier data
num_suppliers = 15000  # Adjust for dataset size
data = []

for i in range(1, num_suppliers + 1):
    supplier_id = f"S{i:03d}"
    supplier_name = f"Supplier_{i}"
    location = random.choice(locations)
    material_type = random.choice(materials)
    lead_time = random.randint(5, 30)  # Delivery time in days
    historical_delay = random.randint(0, 10)  # Past delays in days
    quality_rating = round(random.uniform(3, 5), 1)  # Rating between 3 and 5
    avg_cost_per_unit = random.randint(2000, 10000)  # Cost in INR
    financial_stability = round(random.uniform(5, 10), 1)  # Score from 5 to 10
    past_contract_failures = random.randint(0, 3)
    alternative_suppliers = ", ".join([f"S{random.randint(1, num_suppliers):03d}" for _ in range(2)])
    climate_risk = random.choice(climate_risk_levels)
    delivery_mode = random.choice(delivery_modes)
    reliability_score = round(random.uniform(0.7, 1.0), 2)
    geo_proximity = random.randint(10, 100)  # Distance in km
    inventory_buffer_level = random.choice(["Low", "Medium", "High"])

    data.append([
        supplier_id, supplier_name, location, material_type, lead_time, historical_delay, quality_rating,
        avg_cost_per_unit, financial_stability, past_contract_failures, alternative_suppliers,
        climate_risk, delivery_mode, reliability_score, geo_proximity, inventory_buffer_level
    ])

# Create DataFrame
columns = [
    "Supplier_ID", "Supplier_Name", "Location", "Material_Type", "Lead_Time (days)", "Historical_Delay (days)",
    "Quality_Rating (1-5)", "Avg_Cost_per_Unit (INR)", "Financial_Stability_Score", "Past_Contract_Failures",
    "Alternative_Suppliers", "Climate_Risk", "Delivery_Mode", "Reliability_Score (0-1)", "Geo_Proximity (km)",
    "Inventory_Buffer_Level"
]
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv("construction_supply_chain_new.csv", index=False)

print("Synthetic dataset 'construction_supply_chain_new.csv' generated successfully!")
