import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load trained model
model_path = "xgboost_model.pkl"
model = joblib.load(model_path)

# Load dataset
file_path = "construction_supply_chain_new.csv"
df = pd.read_csv(file_path)

# 🔹 Ensure Feature Engineering Matches Training
def preprocess_data(df):
    # Compute new feature columns
    if all(col in df.columns for col in ["Reliability_Score (0-1)", "Quality_Rating (1-5)"]):
        df["Reliability_Adjusted_Score"] = df["Reliability_Score (0-1)"] * df["Quality_Rating (1-5)"]

    if all(col in df.columns for col in ["Geo_Proximity (km)", "Lead_Time (days)"]):
        df["Geo_Proximity_Impact"] = df["Geo_Proximity (km)"] / (df["Lead_Time (days)"] + 1)

    if all(col in df.columns for col in ["Financial_Stability_Score", "Reliability_Score (0-1)"]):
        df["Financial_Reliability_Index"] = df["Financial_Stability_Score"] * df["Reliability_Score (0-1)"]

    if "Historical_Delay (days)" in df.columns:
        df["Historical_Delay_Trend"] = df["Historical_Delay (days)"].rolling(window=3, min_periods=1).mean()

    # Normalize scores
    scaler = MinMaxScaler()
    normalized_features = ["Quality_Rating (1-5)", "Financial_Stability_Score", "Reliability_Score (0-1)"]
    df["Supplier_Score"] = scaler.fit_transform(df[normalized_features]).mean(axis=1)

    # Handle cost normalization
    if "Avg_Cost_per_Unit (INR)" in df.columns:
        df["Cost_Impact"] = 1 / (df["Avg_Cost_per_Unit (INR)"] + 1)  # Lower cost is better
        df["Cost_Impact"] = MinMaxScaler().fit_transform(df[["Cost_Impact"]])
    else:
        df["Cost_Impact"] = 0  # Default impact if cost data is missing

    # Fill missing values
    df.fillna(df.median(), inplace=True)

    return df

df = preprocess_data(df)

# 🔹 Function to Recommend Suppliers
def recommend_supplier(top_n=5, weights=None):
    if weights is None:
        weights = {
            "Reliability_Adjusted_Score": 0.4,
            "Geo_Proximity_Impact": 0.2,
            "Financial_Reliability_Index": 0.2,
            "Historical_Delay_Trend": 0.15,
            "Cost_Impact": 0.05  # Default cost weight
        }

    # Compute final score dynamically using UI weights
    df["Final_Score"] = (
        df["Reliability_Adjusted_Score"] * weights["Reliability_Adjusted_Score"] +
        df["Geo_Proximity_Impact"] * weights["Geo_Proximity_Impact"] +
        df["Financial_Reliability_Index"] * weights["Financial_Reliability_Index"] +
        df["Historical_Delay_Trend"] * weights["Historical_Delay_Trend"] +
        df["Cost_Impact"] * weights["Cost_Impact"]
    )

    # Get top N suppliers
    top_suppliers = df.sort_values(by="Final_Score", ascending=False).head(top_n)

    return top_suppliers[
        ["Supplier_Name", "Final_Score", "Avg_Cost_per_Unit (INR)", "Reliability_Score (0-1)",
         "Quality_Rating (1-5)", "Financial_Stability_Score", "Geo_Proximity (km)", "Lead_Time (days)"]
    ]

# ---- STREAMLIT UI ----
st.set_page_config(page_title="Supplier Recommendation System", layout="wide")
st.title("🔍 Supplier Recommendation System (Structured Data)")

# 📌 **Understanding Parameters**
with st.sidebar.expander("ℹ️ Understanding Parameters", expanded=False):
    st.markdown("""
    - **Reliability Score** → Measures supplier reliability (0-1). Higher is better.
    - **Quality Rating** → Supplier's product/service quality (1-5). Higher is better.
    - **Financial Stability** → Measures financial health. Higher values indicate less risk.
    - **Proximity Impact** → Distance vs. lead time. Lower distance & faster lead time get higher scores.
    - **Historical Delay** → Average past delays in project deliveries. Lower is better.
    - **Cost Impact** → Adjusts ranking to favor lower-cost suppliers.
    """)

# User-defined weights
st.sidebar.header("🔧 Weight Adjustments")
weights = {
    "Reliability_Adjusted_Score": st.sidebar.slider("Reliability Weight", 0.0, 1.0, 0.4),
    "Geo_Proximity_Impact": st.sidebar.slider("Proximity Weight", 0.0, 1.0, 0.2),
    "Financial_Reliability_Index": st.sidebar.slider("Financial Stability Weight", 0.0, 1.0, 0.2),
    "Historical_Delay_Trend": st.sidebar.slider("Historical Delay Weight", 0.0, 1.0, 0.15),
    "Cost_Impact": st.sidebar.slider("Cost Weight", 0.0, 1.0, 0.05)
}

# Ensure total weight does not exceed 1
total_weight = sum(weights.values())
if total_weight > 1:
    st.sidebar.warning("⚠️ Total weight exceeds 1. Adjust sliders.")

top_n = st.sidebar.slider("Number of Suppliers", 1, 10, 5)

# Run recommendation
if st.button("Find Best Suppliers"):
    recommendations = recommend_supplier(top_n, weights)
    st.write("### 🏆 Top Recommended Suppliers")
    st.dataframe(recommendations.reset_index(drop=True))

    # 📌 **Understanding Final Score**
    st.markdown("## 📊 Understanding Final Score")
    st.info(f"""
    The **Final Score** is computed based on a weighted combination of the following factors:
    - **Reliability Score** ({weights['Reliability_Adjusted_Score'] * 100:.0f}%)
    - **Proximity Impact** ({weights['Geo_Proximity_Impact'] * 100:.0f}%)
    - **Financial Stability** ({weights['Financial_Reliability_Index'] * 100:.0f}%)
    - **Historical Delay Trend** ({weights['Historical_Delay_Trend'] * 100:.0f}%)
    - **Cost Impact** ({weights['Cost_Impact'] * 100:.0f}%)
    
    Each factor is **normalized and scaled** before computing the final ranking. Adjust the sliders to change their impact on supplier selection.
    """)

st.sidebar.header("📜 About")
st.sidebar.info("This system ranks suppliers based on structured data features such as reliability, proximity, financial stability, and cost.")
