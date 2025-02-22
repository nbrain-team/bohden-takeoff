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
import uuid
import time
import torch
from main_css import apply_custom_css  # Import the custom CSS function

# Set up the Streamlit page
st.set_page_config(page_title="BuildSmart", page_icon="🏗️", layout="wide")

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
    # if "choices" in response and len(response["choices"]) > 0:
    #     return response["choices"][0]["text"]
    # else:
    #     return "❌ Error: Unable to generate an explanation. Please check the API response."

def generate_detection_explanation(detected_objects, detection_type):
    """Generates an AI-based explanation for detected objects using Together AI (LLaMA 2-70B)"""
    
    prompt = f"Explain the following detected objects in simple terms for {detection_type}:\n\n"
    for obj in detected_objects:
        prompt += f"- Class: {obj['class_name']}, Confidence: {obj['confidence']:.2f}, Bounding Box: {obj['bbox']}\n"

    if detection_type == "Blueprint Detection":
        prompt += "\n Basically purpose of Blueprint detection is to help builders save time, cost and delays which happen due to problems in the blueprint itself. Explain what each term means, why it is important, and how it relates to blueprint analysis. Find faults in the blueprint based on the detection  or appreciate if something is right. Also, suggest any necessary actions or precautions."
    elif detection_type == "Risk Detection":
        prompt += "\nExplain what each term means, why it is important, and what exactly are the risks in the image. Also, suggest any necessary actions or precautions."

    # Call Together AI with LLaMA 2-70B
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    # Extract the generated text from the response
    return response.choices[0].message.content

# Load the supplier recommendation model & vectorizer
SUPPLIER_MODEL_PATH = "../backend/train/supply_chain_nlp/supplier_recommendation_model.pkl"
VECTORIZER_PATH = "../backend/train/supply_chain_nlp/vectorizer.pkl"
supplier_model = joblib.load(SUPPLIER_MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Apply custom CSS
st.markdown(apply_custom_css(), unsafe_allow_html=True)

st.title("🏗️ BuildSmart - Construction AI")


if "page" not in st.session_state:
    st.session_state.page = "Home" # Default page

# Sidebar Buttons with Streamlit's st.button and icons
if st.sidebar.button(":material/Home: Home", key="home"):
    st.session_state.page = "Home"

if st.sidebar.button(":material/Dashboard: Dashboard", key="dashboard"):
    st.session_state.page = "Dashboard"

if st.sidebar.button(":material/Train: Metro Risk Prediction", key="metro_risk"):
    st.session_state.page = "Metro Risk Prediction"

if st.sidebar.button(":material/Chat: Chatbot", key="chatbot"):
    st.session_state.page = "Chatbot"

if st.sidebar.button(":material/Square_Foot: Blueprint Detection and Analysis", key="blueprint"):
    st.session_state.page = "Blueprint Detection and Analysis"

if st.sidebar.button(":material/Local_Shipping: Supplier Recommendation", key="supply_chain"):
    st.session_state.page = "Supplier Recommendation"

if st.sidebar.button(":material/Warning: Risk Detection", key="risk_detection"):
    st.session_state.page = "Risk Detection"

if st.sidebar.button(":material/Warning: Delay Report Generator", key="delay_report"):
    st.session_state.page = "Delay Report Generator"

# Home (Landing Page)
if st.session_state.page == "Home":
    st.markdown('<div class="landing-container">', unsafe_allow_html=True)
    st.image("https://source.unsplash.com/1600x600/?construction,building", use_container_width=True)
    st.markdown('<h1 class="main-title">🏗️ BuildSmart - AI for Smarter Construction</h1>', unsafe_allow_html=True)
    st.write("🚀 AI-powered solutions for project risk assessment, cost optimization, and safety monitoring.")
    st.markdown('<a href="?page=dashboard"><button class="landing-button">Get Started</button></a>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

### 🏗️ DASHBOARD SECTION ###
elif st.session_state.page == "Dashboard":
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
elif st.session_state.page == "Metro Risk Prediction":
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

### 🔍 SUPPLY CHAIN RECOMMENDATION ###
elif st.session_state.page == "Supplier Recommendation":
     from sklearn.preprocessing import MinMaxScaler

     # Load trained model
     model_path = "../backend/train/supply_chain_nlp/xgboost_model.pkl"
     model = joblib.load(model_path)

     # Load dataset
     file_path = "../backend/train/supply_chain_nlp/construction_supply_chain_new.csv"
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
     st.sidebar.markdown(
    '<p style="color: black; background-color: black; padding: 10px; border-radius: 5px;">'
    'This system ranks suppliers based on structured data features such as reliability, proximity, financial stability, and cost.'
    '</p>',
    unsafe_allow_html=True
    )




    ### 🤖 AI CHATBOT SECTION ###
elif st.session_state.page == "Chatbot":
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
elif st.session_state.page == "Blueprint Detection and Analysis":
    st.subheader("📸 Upload an Image for Blueprint Detection")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    # Define a fixed prediction folder
    predict_folder = "runs/detect/predictBlueprint"
    os.makedirs(predict_folder, exist_ok=True)  # Ensure the folder exists

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("🚀 Detect Blueprint"):
            with st.spinner("Running detection..."):
                # Save uploaded file temporarily
                temp_dir = tempfile.mkdtemp()
                image_path = os.path.join(temp_dir, uploaded_file.name)
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Generate a unique name for the output image
                unique_id = uuid.uuid4().hex[:8]  # Short UUID
                timestamp = int(time.time())  # Timestamp
                new_image_name = f"BuildSmart_detected_blueprint_{unique_id}.png"  # Unique image name
                new_image_path = os.path.join(predict_folder, new_image_name)

                # Load YOLO model and perform inference
                model = YOLO("../backend/blueprint.pt")
                results = model.predict(image_path, save=True)  # Save images in default location

                # Find the latest prediction folder
                saved_images = sorted(
                    [f for f in os.listdir("runs/detect") if f.startswith("predict")], 
                    key=lambda x: os.path.getctime(os.path.join("runs/detect", x)), 
                    reverse=True
                )

                if saved_images:
                    latest_predict_folder = os.path.join("runs/detect", saved_images[0])
                    detected_images = [f for f in os.listdir(latest_predict_folder) if f.endswith((".png", ".jpg", ".jpeg"))]

                    if detected_images:
                        old_image_path = os.path.join(latest_predict_folder, detected_images[0])
                        os.rename(old_image_path, new_image_path)  # Move and rename file

                        # Display annotated image
                        st.image(Image.open(new_image_path), caption="Annotated Image", use_container_width=True)

                        # Show detected objects
                        st.write("### 📌 Detected Objects:")
                        detected_objects = []
                        for result in results:
                            for box in result.boxes:
                                class_id = int(box.cls)
                                class_name = result.names[class_id]
                                confidence = float(box.conf)
                                bbox = box.xyxy.tolist()
                                detected_objects.append({
                                    "class_name": class_name,
                                    "confidence": confidence,
                                    "bbox": bbox
                                })
                                st.write(f"🔹 *Class:* {class_name}, *Confidence:* {confidence:.2f}, *Bounding Box:* {bbox}")

                        # Generate AI Explanation for detected objects
                        st.subheader("🤖 AI Explanation for Detected Objects")
                        explanation = generate_detection_explanation(detected_objects, "Blueprint Detection")
                        st.write(explanation)
                    else:
                        st.error("Error: Could not find the saved prediction image.")
                else:
                    st.error("⚠️ No objects detected. Try another image.")

### 🔍 YOLO IMAGE DETECTION SECTION ###
elif st.session_state.page == "Risk Detection":
    st.subheader("📸 Upload a Construction Site Image for Risk Detection")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    # Define a fixed prediction folder
    predict_folder = "runs/detect/predictRisk"
    os.makedirs(predict_folder, exist_ok=True)  # Ensure the folder exists

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("🚀 Detect Risk"):
            with st.spinner("Running detection..."):
                # Save uploaded file temporarily
                temp_dir = tempfile.mkdtemp()
                image_path = os.path.join(temp_dir, uploaded_file.name)
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Generate a unique name for the output image
                unique_id = uuid.uuid4().hex[:8]  # Short UUID
                timestamp = int(time.time())  # Timestamp
                new_image_name = f"BuildSmart_detected_risk_{unique_id}.png"  # Unique image name
                new_image_path = os.path.join(predict_folder, new_image_name)

                # Load YOLO model and perform inference
                model = YOLO("../backend/PPE/models/best_2.pt")
                results = model.predict(image_path, save=True)  # Save images in default location

                # Find the latest prediction folder
                saved_images = sorted(
                    [f for f in os.listdir("runs/detect") if f.startswith("predict")], 
                    key=lambda x: os.path.getctime(os.path.join("runs/detect", x)), 
                    reverse=True
                )

                if saved_images:
                    latest_predict_folder = os.path.join("runs/detect", saved_images[0])
                    detected_images = [f for f in os.listdir(latest_predict_folder) if f.endswith((".png", ".jpg", ".jpeg"))]

                    if detected_images:
                        old_image_path = os.path.join(latest_predict_folder, detected_images[0])
                        os.rename(old_image_path, new_image_path)  # Move and rename file

                        # Display annotated image
                        st.image(Image.open(new_image_path), caption="Annotated Image", use_container_width=True)

                        # Show detected objects
                        st.write("### 📌 Detected Objects:")
                        detected_objects = []
                        for result in results:
                            for box in result.boxes:
                                class_id = int(box.cls)
                                class_name = result.names[class_id]
                                confidence = float(box.conf)
                                bbox = box.xyxy.tolist()
                                detected_objects.append({
                                    "class_name": class_name,
                                    "confidence": confidence,
                                    "bbox": bbox
                                })
                                st.write(f"🔹 *Class:* {class_name}, *Confidence:* {confidence:.2f}, *Bounding Box:* {bbox}")

                        # Generate AI Explanation for detected objects
                        st.subheader("🤖 AI Explanation for Detected Objects")
                        explanation = generate_detection_explanation(detected_objects, "Risk Detection")
                        st.write(explanation)
                    else:
                        st.error("Error: Could not find the saved prediction image.")
                else:
                    st.error("⚠️ No objects detected. Try another image.")

    st.subheader("📹 Real-Time Risk Detection using Webcam")

    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Webcam Detection"):
            st.session_state.camera_active = True

    with col2:
        if st.button("Stop Webcam Detection"):
            st.session_state.camera_active = False

    if st.session_state.camera_active:
        import os
        import cv2
        from ultralytics import YOLO

        # Load trained YOLOv8 model
        model = YOLO("../backend/PPE/models/best_2.pt")  # Correct path to the trained model

        cap = cv2.VideoCapture(0)  # Open webcam

        frame_placeholder = st.empty()

        while cap.isOpened() and st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO model on GPU
            # Auto-detect device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Use the detected device
            results = model(frame, device=device)
             

            # Draw bounding boxes and labels
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = box.cls[0]
                    confidence = box.conf[0]
                    label_text = f"{model.names[int(label)]}: {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert the frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame in Streamlit
            frame_placeholder.image(frame_rgb, channels="RGB")

        cap.release()
        cv2.destroyAllWindows()
 

elif st.session_state.page == "Delay Report Generator":

    from fpdf import FPDF
    from PIL import Image
    import matplotlib.pyplot as plt
    from xgboost import XGBClassifier

       # Streamlit page title
    st.title("🏗️ Construction Delay Prediction System")

    # File uploader for dataset
    uploaded_file = st.file_uploader("Upload your construction project dataset (CSV)", type=["csv"])

    # Image uploader (optional)
    uploaded_image = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])

    # Load trained model
    model_path2 = "../backend/PPE/best_xgb_model.json"
    train_columns_path2 = "../backend/PPE/train_columns.pkl"

    if os.path.exists(model_path2) and os.path.exists(train_columns_path2):
        loaded_model = XGBClassifier()
        loaded_model.load_model(model_path2)
        train_columns = joblib.load(train_columns_path2)
        st.success("✅ Model loaded successfully.")
    else:
        st.error("❌ Model or training columns file not found. Please check the paths.")
        st.stop()

    if uploaded_file:
        df_test = pd.read_csv(uploaded_file)

        # Preprocessing
        df_test = pd.get_dummies(df_test, columns=["City", "Project Type", "External Factor", "Delay Reason"], drop_first=True)
        if "Milestone" in df_test.columns:
            df_test["Milestone"] = pd.to_numeric(df_test["Milestone"], errors="coerce").fillna(0)

        # Ensure consistency with training columns
        for col in train_columns:
            if col not in df_test.columns:
                df_test[col] = 0  # Add missing columns

        df_test = df_test.reindex(columns=train_columns, fill_value=0)

        # Make predictions
        y_pred = loaded_model.predict(df_test)
        df_test["Predicted Delay"] = y_pred

        # Display results
        st.write("### 📊 Prediction Results")
        st.dataframe(df_test.head())

        # Save predictions
        df_test.to_csv("predicted_report.csv", index=False)
        st.success("✅ Predictions saved as CSV.")
        st.download_button("📥 Download Predictions CSV", data=open("predicted_report.csv", "rb"), file_name="predicted_report.csv", mime="text/csv")

        # Ensure "Delay Duration (days)" exists
        average_delay = df_test["Delay Duration (days)"].mean() if "Delay Duration (days)" in df_test.columns else "N/A"

        # Delay distribution visualization
        plt.figure(figsize=(8, 5))
        df_test["Predicted Delay"].value_counts().plot(kind='bar', color=['green', 'red'])
        plt.xticks(ticks=[0, 1], labels=['On-Time', 'Delayed'], rotation=0)
        plt.xlabel("Project Status")
        plt.ylabel("Number of Projects")
        plt.title("Construction Delay Distribution")
        plt.savefig("delay_distribution.png")
        st.image("delay_distribution.png")

        # Generate PDF button
        if st.button("📄 Generate PDF Report"):
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(200, 10, "Construction Delay Prediction Report", ln=True, align="C")
            pdf.ln(10)

            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, f"Total Projects: {len(df_test)}", ln=True)
            pdf.cell(0, 10, f"Delayed Projects: {df_test['Predicted Delay'].sum()}", ln=True)
            pdf.cell(0, 10, f"On-Time Projects: {len(df_test) - df_test['Predicted Delay'].sum()}", ln=True)
            pdf.cell(0, 10, f"Average Delay Duration: {average_delay if average_delay != 'N/A' else 'N/A'} days", ln=True)
            pdf.ln(10)

            pdf.image("delay_distribution.png", x=10, w=180)

            # If user uploaded an image, add it to the PDF
            if uploaded_image:
                image_path = "uploaded_image.png"
                with open(image_path, "wb") as f:
                    f.write(uploaded_image.getbuffer())  # Save image
                pdf.add_page()
                pdf.cell(200, 10, "User Uploaded Image", ln=True, align="C")
                pdf.image(image_path, x=10, w=180)
                st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

            pdf.output("predicted_report.pdf")

            st.success("✅ PDF report generated.")
            with open("predicted_report.pdf", "rb") as f:
                st.download_button("📥 Download Report PDF", f, file_name="predicted_report.pdf", mime="application/pdf")


       


