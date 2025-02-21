import streamlit as st
import requests
import os
import joblib
import tempfile
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import together
from together import Together
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
together.api_key = os.getenv("TOGETHER_API_KEY1")  # Set your Together AI API key

# Load models and encoders
METRO_MODEL_PATH = "../backend/metro/xgboost_metro_classifier.pkl"
ENCODER_PATH = "../backend/metro/label_encoders.pkl"
classifier = joblib.load(METRO_MODEL_PATH)
regressor = joblib.load("../backend/metro/xgboost_metro_regressor.pkl")
label_encoders = joblib.load(ENCODER_PATH)
client = Together(api_key=together.api_key)
# Define expected features (must match training data)
expected_columns = [
    "City", "Tunnel Length (km)", "Num of Stations", "Num of Workers", 
    "Equipment Factor (%)", "Signal System Complexity (%)", "Material Factor (%)", 
    "Gov Regulation Factor (%)", "Urban Congestion Impact (%)", 
    "Expected Completion Time (months)", "Budget (in Crores)"
]

def generate_ai_explanation(risk_level, delay_days, input_data):
    """Generates an AI-based explanation using Together AI (LLaMA 2-70B)"""
    
    prompt = f"""
    A metro project has the following characteristics:
    - City: {input_data['City'].values[0]}
    - Tunnel Length: {input_data['Tunnel Length (km)'].values[0]} km
    - Number of Stations: {input_data['Num of Stations'].values[0]}
    - Number of Workers: {input_data['Num of Workers'].values[0]}
    - Equipment Factor: {input_data['Equipment Factor (%)'].values[0]}%
    - Signal System Complexity: {input_data['Signal System Complexity (%)'].values[0]}%
    - Material Factor: {input_data['Material Factor (%)'].values[0]}%
    - Government Regulation Factor: {input_data['Gov Regulation Factor (%)'].values[0]}%
    - Urban Congestion Impact: {input_data['Urban Congestion Impact (%)'].values[0]}%
    - Expected Completion Time: {input_data['Expected Completion Time (months)'].values[0]} months
    - Budget: ₹{input_data['Budget (in Crores)'].values[0]} crores

    The project has been classified as having a **{risk_level}** risk level and is expected to face a delay of **{delay_days:.2f}** days.

    Explain the results in simple terms, highlighting the key factors contributing to the risk level and delay. Also, suggest mitigation strategies.
    """

    # Call Together AI with LLaMA 2-70B
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    

    # Debug: Print the raw API response
    st.write(response.choices[0].message.content)

    # Extract the generated text from the response
    if "choices" in response and len(response["choices"]) > 0:
        return response["choices"][0]["text"]
    else:
        return "❌ Error: Unable to generate an explanation. Please check the API response."

# Load the supplier recommendation model & vectorizer
SUPPLIER_MODEL_PATH = "../backend/train/supply_chain_nlp/supplier_recommendation_model.pkl"
VECTORIZER_PATH = "../backend/train/supply_chain_nlp/vectorizer.pkl"
supplier_model = joblib.load(SUPPLIER_MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Set up the Streamlit page
st.set_page_config(page_title="BuildSmart", page_icon="🏗️", layout="wide")
st.title("🏗️ BuildSmart - Construction AI")

# Sidebar for Navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Select a Page",
    ["Dashboard", "Metro Risk Prediction", "Chatbot", "YOLO Detection", "Supply Chain Copilot", "Risk Detection"],
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

            # Predict risk level
            risk_prediction = classifier.predict(input_data)[0]
            risk_label = label_encoders["Risk Level"].inverse_transform([risk_prediction])[0]
            st.success(f"✅ **Predicted Risk Level:** {risk_label}")

            # Predict delay (if you have a regression model for delay)
            delay_days = 45.23  # Replace with actual delay prediction if available

            # Generate AI Explanation
            st.subheader("🤖 AI Explanation")
            explanation = generate_ai_explanation(risk_label, delay_days, input_data)
            st.write(explanation)

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

### 🔍 YOLO IMAGE DETECTION SECTION ###
elif menu == "YOLO Detection":
    st.subheader("📸 Upload an Image for Safety Detection")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file and st.button("🚀 Detect Safety Issues"):
        with st.spinner("Running YOLO detection..."):
            # Save uploaded file temporarily
            temp_dir = tempfile.mkdtemp()
            image_path = os.path.join(temp_dir, uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load YOLO model and perform inference
            model = YOLO("../backend/blueprint.pt")  # Ensure blueprint.pt exists
            results = model.predict(image_path, save=True)

            # Display results
            if results:
                st.success("✅ Detection completed! See detected objects below:")
                
                for result in results:
                    st.image(Image.open(os.path.join(result.save_dir, uploaded_file.name)), caption="Annotated Image", use_column_width=True)

                    # Show detected objects
                    st.write("### 📌 Detected Objects:")
                    for box in result.boxes:
                        class_id = int(box.cls)
                        class_name = result.names[class_id]
                        confidence = float(box.conf)  # Convert tensor to float
                        bbox = box.xyxy.tolist()  # Convert tensor to list
                        st.write(f"🔹 *Class:* {class_name}, *Confidence:* {confidence:.2f}, *Bounding Box:* {bbox}")
            else:
                st.error("No objects detected. Try another image.")

### 🔍 YOLO IMAGE DETECTION SECTION ###
elif menu == "Risk Detection":
    st.subheader("📸 Upload an Image for Safety Detection")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file and st.button("🚀 Perform Safety Analysis"):
        with st.spinner("Running YOLO detection..."):
            # Save uploaded file temporarily
            temp_dir = tempfile.mkdtemp()
            image_path = os.path.join(temp_dir, uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load YOLO model and perform inference
            model = YOLO("../backend/PPE/models/best_2.pt")  # Ensure blueprint.pt exists
            results = model.predict(image_path, save=True)

            # Display results
            if results:
                st.success("✅ Detection completed! See detected objects below:")
                
                for result in results:
                    st.image(Image.open(os.path.join(result.save_dir, uploaded_file.name)), caption="Annotated Image", use_column_width=True)

                    # Show detected objects
                    st.write("### 📌 Detected Objects:")
                    for box in result.boxes:
                        class_id = int(box.cls)
                        class_name = result.names[class_id]
                        confidence = float(box.conf)  # Convert tensor to float
                        bbox = box.xyxy.tolist()  # Convert tensor to list
                        st.write(f"🔹 *Class:* {class_name}, *Confidence:* {confidence:.2f}, *Bounding Box:* {bbox}")
            else:
                st.error("No objects detected. Try another image.")