import streamlit as st
import requests
import os
import joblib
import tempfile
import pandas as pd
from ultralytics import YOLO
from PIL import Image

# Load the trained metro risk classifier
METRO_MODEL_PATH = "../backend/metro/xgboost_metro_classifier.pkl"
ENCODER_PATH = "../backend/metro/label_encoders.pkl"

# Load the supplier recommendation model & vectorizer
SUPPLIER_MODEL_PATH = "../backend/train/supply_chain_nlp/supplier_recommendation_model.pkl"
VECTORIZER_PATH = "../backend/train/supply_chain_nlp/vectorizer.pkl"

# Load models and encoders
metro_model = joblib.load(METRO_MODEL_PATH)
label_encoders = joblib.load(ENCODER_PATH)

supplier_model = joblib.load(SUPPLIER_MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Set up the Streamlit page
st.set_page_config(page_title="BuildSmart", page_icon="🏗️", layout="wide")
st.title("🏗️ BuildSmart - Construction AI")

# Sidebar for Navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Select a Page",
    ["Dashboard", "Metro Risk Prediction", "Chatbot", "YOLO Detection", "Supply Chain Copilot"],
    index=0
)

### 🏗️ DASHBOARD SECTION ###
if menu == "Dashboard":
    st.subheader("📊 Project Cost Overrun Prediction & Safety Analysis")

    with st.expander("🔍 Predict Cost Overrun"):
        col1, col2 = st.columns(2)
        with col1:
            project_size = st.number_input("Project Size (sq. m)", min_value=0)
            labor_count = st.number_input("Labor Count", min_value=0)
            equipment_count = st.number_input("Equipment Count", min_value=0)
        with col2:
            avg_temp = st.number_input("Avg Temperature (°C)", min_value=-50, max_value=50)
            rainfall = st.number_input("Rainfall (mm)", min_value=0)
            milestone = st.text_input("Milestone")
            external_factor = st.text_input("External Factor")

        if st.button("🔮 Predict Cost Overrun"):
            with st.spinner("Predicting..."):
                response = requests.post(f"http://127.0.0.1:5000/predict", json={
                    "project_size": project_size,
                    "labor_count": labor_count,
                    "equipment_count": equipment_count,
                    "avg_temp": avg_temp,
                    "rainfall": rainfall,
                    "milestone": milestone,
                    "external_factor": external_factor
                })
                if response.status_code == 200:
                    prediction = response.json().get("prediction", "N/A")
                    st.success(f"**Prediction:** {prediction}")
                else:
                    st.error("Failed to get a prediction. Please try again.")

### 🚆 METRO RISK PREDICTION SECTION ###
elif menu == "Metro Risk Prediction":
    st.subheader("🚆 Metro Project Risk Level Prediction")

    with st.form("metro_risk_form"):
        city = st.selectbox("City", ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai", "Kolkata"])
        tunnel_length = st.number_input("Tunnel Length (km)", min_value=1.0, max_value=50.0, step=0.1)
        num_stations = st.number_input("Number of Stations", min_value=5, max_value=100)
        num_workers = st.number_input("Number of Workers", min_value=100, max_value=5000)
        equipment_factor = st.slider("Equipment Factor (%)", 0, 100)
        signal_complexity = st.slider("Signal System Complexity (%)", 0, 100)
        material_factor = st.slider("Material Factor (%)", 0, 100)
        gov_regulation = st.slider("Government Regulation Factor (%)", 0, 100)
        urban_congestion = st.slider("Urban Congestion Impact (%)", 0, 100)
        budget = st.number_input("Budget (in Crores)", min_value=500.0, max_value=5000.0, step=10.0)
        expected_completion_time = st.number_input("Expected Completion Time (months)", min_value=24, max_value=120)

        submitted = st.form_submit_button("🚀 Predict Risk Level")

    if submitted:
        with st.spinner("Predicting risk level..."):
            city_encoded = label_encoders["City"].transform([city])[0]

            input_data = pd.DataFrame([[
                city_encoded, tunnel_length, num_stations, num_workers,
                equipment_factor, signal_complexity, material_factor,
                gov_regulation, urban_congestion, expected_completion_time, budget
            ]], columns=["City", "Tunnel Length (km)", "Num of Stations", "Num of Workers",
                         "Equipment Factor (%)", "Signal System Complexity (%)", "Material Factor (%)",
                         "Gov Regulation Factor (%)", "Urban Congestion Impact (%)",
                         "Expected Completion Time (months)", "Budget (in Crores)"])

            risk_prediction = metro_model.predict(input_data)[0]
            risk_label = label_encoders["Risk Level"].inverse_transform([risk_prediction])[0]
            st.success(f"✅ **Predicted Risk Level:** {risk_label}")

### 🔍 SUPPLY CHAIN COPILOT ###
elif menu == "Supply Chain Copilot":
    st.subheader("🔗 Construction Supplier Copilot")

    query = st.text_input("Enter Material Type or Supplier Need (e.g., 'Steel Supplier in Mumbai')")

    if st.button("Get Recommendations"):
        with st.spinner("Fetching recommendations..."):
            query_vector = vectorizer.transform([query])
            distances, indices = supplier_model.kneighbors(query_vector)

            st.write("### 🔍 Recommended Suppliers:")
            for i in range(len(indices[0])):
                supplier_idx = indices[0][i]
                st.write(f"**Supplier {i+1}:**")
                st.write(f"🔹 **Supplier Name:** {supplier_idx}")
                st.write(f"🔹 **Similarity Score:** {round(distances[0][i], 3)}")
                st.write("---")

### 🤖 AI CHATBOT SECTION ###
elif menu == "Chatbot":
    st.subheader("🤖 AI Construction Chatbot")
    st.write("💬 Ask anything about construction risks, safety, and cost optimization!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            response = requests.post(f"http://127.0.0.1:5000/chatbot", json={"message": user_input})
            if response.status_code == 200:
                bot_response = response.json().get("response", "No response")
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                with st.chat_message("assistant"):
                    st.markdown(bot_response)
            else:
                st.error("Chatbot failed to respond. Please try again.")
