import streamlit as st
import pandas as pd
from utils import generate_ai_explanation, generate_detection_explanation
from fpdf import FPDF
from PIL import Image
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import os
import joblib
import tempfile
import cv2  # Add this import for webcam functionality
import torch
from ultralytics import YOLO
import plotly.express as px
import plotly.graph_objects as go
import requests
import uuid
import time
from pathlib import Path

# Determine the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))
print(base_dir)
def home_page():
    st.markdown('<div class="landing-container">', unsafe_allow_html=True)
    image_path = os.path.join(base_dir, 'banner.png')
    banner = Image.open(image_path)
    st.image(banner, use_container_width=True)
    st.markdown('<h1 class="main-title">🏗️ BuildSmart - AI for Smarter Construction</h1>', unsafe_allow_html=True)
    st.write("🚀 AI-powered solutions for project risk assessment, cost optimization, and safety monitoring.")
    st.markdown('Get Started Now!', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def dashboard_page():
    st.subheader("📊 Project Cost Overrun Prediction & Safety Analysis")
    with st.expander("🔍 Predict Cost Overrun"):
        col1, col2 = st.columns(2)
        with col1:
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

def metro_risk_prediction_page(classifier, label_encoders, client):
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
            risk_prediction = classifier.predict(input_data)[0]
            risk_label = label_encoders["Risk Level"].inverse_transform([risk_prediction])[0]
            st.success(f"✅ **Predicted Risk Level:** {risk_label}")
            delay_days = 45.23  # Replace with actual delay prediction if available
            st.subheader("🤖 AI Explanation")
            explanation = generate_ai_explanation(client, risk_label, delay_days, input_data)
            st.write(explanation)

def supplier_recommendation_page():
    from sklearn.preprocessing import MinMaxScaler
    import joblib
    import pandas as pd
    model_path = os.path.join(base_dir, "../backend/train/supply_chain_nlp/xgboost_model.pkl")
    model = joblib.load(model_path)
    file_path = os.path.join(base_dir, "../backend/train/supply_chain_nlp/construction_supply_chain_new.csv")
    df = pd.read_csv(file_path)
    def preprocess_data(df):
        if all(col in df.columns for col in ["Reliability_Score (0-1)", "Quality_Rating (1-5)"]):
            df["Reliability_Adjusted_Score"] = df["Reliability_Score (0-1)"] * df["Quality_Rating (1-5)"]
        if all(col in df.columns for col in ["Geo_Proximity (km)", "Lead_Time (days)"]):
            df["Geo_Proximity_Impact"] = df["Geo_Proximity (km)"] / (df["Lead_Time (days)"] + 1)
        if all(col in df.columns for col in ["Financial_Stability_Score", "Reliability_Score (0-1)"]):
            df["Financial_Reliability_Index"] = df["Financial_Stability_Score"] * df["Reliability_Score (0-1)"]
        if "Historical_Delay (days)" in df.columns:
            df["Historical_Delay_Trend"] = df["Historical_Delay (days)"].rolling(window=3, min_periods=1).mean()
        scaler = MinMaxScaler()
        normalized_features = ["Quality_Rating (1-5)", "Financial_Stability_Score", "Reliability_Score (0-1)"]
        df["Supplier_Score"] = scaler.fit_transform(df[normalized_features]).mean(axis=1)
        if "Avg_Cost_per_Unit (INR)" in df.columns:
            df["Cost_Impact"] = 1 / (df["Avg_Cost_per_Unit (INR)"] + 1)
            df["Cost_Impact"] = MinMaxScaler().fit_transform(df[["Cost_Impact"]])
        else:
            df["Cost_Impact"] = 0
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        return df
    df = preprocess_data(df)
    def recommend_supplier(top_n=5, weights=None):
        if weights is None:
            weights = {
                "Reliability_Adjusted_Score": 0.4,
                "Geo_Proximity_Impact": 0.2,
                "Financial_Reliability_Index": 0.2,
                "Historical_Delay_Trend": 0.15,
                "Cost_Impact": 0.05
            }
        df["Final_Score"] = (
            df["Reliability_Adjusted_Score"] * weights["Reliability_Adjusted_Score"] +
            df["Geo_Proximity_Impact"] * weights["Geo_Proximity_Impact"] +
            df["Financial_Reliability_Index"] * weights["Financial_Reliability_Index"] +
            df["Historical_Delay_Trend"] * weights["Historical_Delay_Trend"] +
            df["Cost_Impact"] * weights["Cost_Impact"]
        )
        top_suppliers = df.sort_values(by="Final_Score", ascending=False).head(top_n)
        return top_suppliers[
            ["Supplier_Name", "Final_Score", "Avg_Cost_per_Unit (INR)", "Reliability_Score (0-1)",
            "Quality_Rating (1-5)", "Financial_Stability_Score", "Geo_Proximity (km)", "Lead_Time (days)"]
        ]
    st.title("🔍 Supplier Recommendation System (Structured Data)")
    right_sidebar = """
        <div class="right-sidebar">
            <h4>ℹ️ About Supplier Recommendation</h4>
            <p>This tool helps in selecting the best suppliers based on multiple criteria such as reliability, cost, financial stability, and historical delays. Adjust the weights to prioritize factors that matter most to your business.</p>
            
        </div>
            """
    st.markdown(right_sidebar, unsafe_allow_html=True)
    weights = {}  # Initialize dictionary before using it

    weights = {}  # Initialize dictionary

    st.subheader("Adjust Weights")

    col1, _ = st.columns([9, 4])  # First column takes space, second is empty

    with col1:  
        st.write("**Reliability Weight:** Measures supplier's past performance and consistency.")
        weights["Reliability_Adjusted_Score"] = st.slider(
            "Reliability Weight", 0.0, 1.0, 0.4, key="reliability", label_visibility="collapsed"
        )

        st.write("**Proximity Weight:** Considers geographical closeness to reduce logistics delays.")
        weights["Geo_Proximity_Impact"] = st.slider(
            "Proximity Weight", 0.0, 1.0, 0.2, key="proximity", label_visibility="collapsed"
        )

        st.write("**Financial Stability Weight:** Assesses supplier's financial strength to ensure long-term reliability.")
        weights["Financial_Reliability_Index"] = st.slider(
            "Financial Stability Weight", 0.0, 1.0, 0.2, key="financial", label_visibility="collapsed"
        )

        st.write("**Historical Delay Weight:** Evaluates past delivery delays and their impact.")
        weights["Historical_Delay_Trend"] = st.slider(
            "Historical Delay Weight", 0.0, 1.0, 0.15, key="delay", label_visibility="collapsed"
        )

        st.write("**Cost Weight:** Prioritizes cost efficiency in supplier selection.")
        weights["Cost_Impact"] = st.slider(
            "Cost Weight", 0.0, 1.0, 0.05, key="cost", label_visibility="collapsed"
        )

    total_weight = sum(weights.values())
    if total_weight > 1:
        st.warning("⚠️ Total weight exceeds 1. Adjust sliders.")
    top_n = st.slider("Number of Suppliers", 1, 10, 5)
    if st.button("Find Best Suppliers"):
        recommendations = recommend_supplier(top_n, weights)
        st.write("### 🏆 Top Recommended Suppliers")
        st.dataframe(recommendations.reset_index(drop=True))
        st.markdown("""
        <div class="midbar">
            <h4>📊 Understanding Final Score</h4>
            <p>The **Final Score** is calculated using weighted factors:</p>
            <ul>
                <li>Reliability: {:.0f}%</li>
                <li>Proximity: {:.0f}%</li>
                <li>Financial Stability: {:.0f}%</li>
                <li>Historical Delay: {:.0f}%</li>
                <li>Cost Impact: {:.0f}%</li>
            </ul>
        </div>
        """.format(
            weights['Reliability_Adjusted_Score'] * 100,
            weights['Geo_Proximity_Impact'] * 100,
            weights['Financial_Reliability_Index'] * 100,
            weights['Historical_Delay_Trend'] * 100,
            weights['Cost_Impact'] * 100
        ), unsafe_allow_html=True)

def chatbot_page():
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
            try:
                response = requests.post(
                    "http://127.0.0.1:5000/chatbot",
                    json={"message": user_input},
                    timeout=30  # Timeout after 10 seconds
                )
                response.raise_for_status()  # Raises an HTTPError for bad responses
                data = response.json()
                bot_response = data.get("response", "No response")
            except requests.exceptions.RequestException as e:
                bot_response = f"Error: {e}"
                st.error("Chatbot failed to respond. Please try again.")
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            with st.chat_message("assistant"):
                st.markdown(bot_response)

def blueprint_detection_page(client):
    st.subheader("📸 Upload an Image for Blueprint Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    predict_folder = os.path.join(base_dir, "runs/detect/predictBlueprint")
    os.makedirs(predict_folder, exist_ok=True)
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        if st.button("🚀 Detect Blueprint"):
            with st.spinner("Running detection..."):
                temp_dir = tempfile.mkdtemp()
                image_path = os.path.join(temp_dir, uploaded_file.name)
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                unique_id = uuid.uuid4().hex[:8]
                timestamp = int(time.time())
                new_image_name = f"BuildSmart_detected_blueprint_{unique_id}.png"
                new_image_path = os.path.join(predict_folder, new_image_name)
                model = YOLO(os.path.join(base_dir, "../backend/blueprint.pt"))
                results = model.predict(image_path, save=True)
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
                        os.rename(old_image_path, new_image_path)
                        st.image(Image.open(new_image_path), caption="Annotated Image", use_container_width=True)
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
                        st.subheader("🤖 AI Explanation for Detected Objects")
                        explanation = generate_detection_explanation(client, detected_objects, "Blueprint Detection")
                        st.write(explanation)
                    else:
                        st.error("Error: Could not find the saved prediction image.")
                else:
                    st.error("⚠️ No objects detected. Try another image.")

def risk_detection_page(client):
    st.subheader("📸 Upload a Construction Site Image for Risk Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    predict_folder = os.path.join(base_dir, "runs/detect/predictRisk")
    os.makedirs(predict_folder, exist_ok=True)
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        if st.button("🚀 Detect Risk"):
            with st.spinner("Running detection..."):
                temp_dir = tempfile.mkdtemp()
                image_path = os.path.join(temp_dir, uploaded_file.name)
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                unique_id = uuid.uuid4().hex[:8]
                timestamp = int(time.time())
                new_image_name = f"BuildSmart_detected_risk_{unique_id}.png"
                new_image_path = os.path.join(predict_folder, new_image_name)
                model = YOLO(os.path.join(base_dir, "../backend/PPE/models/best_2.pt"))
                results = model.predict(image_path, save=True)
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
                        os.rename(old_image_path, new_image_path)
                        st.image(Image.open(new_image_path), caption="Annotated Image", use_container_width=True)
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
                        st.subheader("🤖 AI Explanation for Detected Objects")
                        explanation = generate_detection_explanation(client, detected_objects, "Risk Detection")
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
        cap = cv2.VideoCapture(0)  # Open webcam
        frame_placeholder = st.empty()
        while cap.isOpened() and st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                break
            model = YOLO(os.path.join(base_dir, "../backend/PPE/models/best_2.pt"))
            results = model(frame)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = box.cls[0]
                    confidence = box.conf[0]
                    label_text = f"{model.names[int(label)]}: {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")
        cap.release()
        cv2.destroyAllWindows()

def delay_report_generator_page():
    st.subheader("📄 Delay Report Generator")
    st.write("Generate a detailed delay report for your construction project.")
    project_name = st.text_input("Project Name")
    project_manager = st.text_input("Project Manager")
    delay_reason = st.text_area("Reason for Delay")
    delay_duration = st.number_input("Delay Duration (days)", min_value=1)
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Delay Report", ln=True, align="C")
            pdf.cell(200, 10, txt=f"Project Name: {project_name}", ln=True)
            pdf.cell(200, 10, txt=f"Project Manager: {project_manager}", ln=True)
            pdf.cell(200, 10, txt=f"Reason for Delay: {delay_reason}", ln=True)
            pdf.cell(200, 10, txt=f"Delay Duration: {delay_duration} days", ln=True)
            report_path = f"delay_report_{project_name}.pdf"
            pdf.output(report_path)
            st.success(f"Report generated: {report_path}")
            st.download_button(label="Download Report", data=open(report_path, "rb").read(), file_name=report_path, mime="application/pdf")

def predict_delay_page():
    st.title("🏗️ Construction Delay Prediction System")

    uploaded_file = st.file_uploader("Upload your construction project dataset (CSV)", type=["csv"])
    uploaded_image = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])

    model_path2 = os.path.join(base_dir, "../backend/PPE/best_xgb_model.json")
    train_columns_path2 = os.path.join(base_dir, "../backend/PPE/train_columns.pkl")

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

        df_test = pd.get_dummies(df_test, columns=["City", "Project Type", "External Factor", "Delay Reason"], drop_first=True)
        if "Milestone" in df_test.columns:
            df_test["Milestone"] = pd.to_numeric(df_test["Milestone"], errors="coerce").fillna(0)

        for col in train_columns:
            if col not in df_test.columns:
                df_test[col] = 0

        df_test = df_test.reindex(columns=train_columns, fill_value=0)

        y_pred = loaded_model.predict(df_test)
        df_test["Predicted Delay"] = y_pred

        st.write("### 📊 Prediction Results")
        st.dataframe(df_test.head())

        df_test.to_csv("predicted_report.csv", index=False)
        st.success("✅ Predictions saved as CSV.")
        st.download_button("📥 Download Predictions CSV", data=open("predicted_report.csv", "rb"), file_name="predicted_report.csv", mime="text/csv")

        average_delay = df_test["Delay Duration (days)"].mean() if "Delay Duration (days)" in df_test.columns else "N/A"

        plt.figure(figsize=(8, 5))
        df_test["Predicted Delay"].value_counts().plot(kind='bar', color=['green', 'red'])
        plt.xticks(ticks=[0, 1], labels=['On-Time', 'Delayed'], rotation=0)
        plt.xlabel("Project Status")
        plt.ylabel("Number of Projects")
        plt.title("Construction Delay Distribution")
        plt.savefig("delay_distribution.png")
        st.image("delay_distribution.png")

        if st.button("📄 Generate PDF Report"):
            class PDF(FPDF):
                def header(self):
                    self.image("logo2.png", 10, 8, 33)
                    self.set_font("Arial", "B", 12)
                    self.cell(0, 10, "Construction Delay Prediction Report", 0, 1, "C")
                    self.ln(10)

            pdf = PDF()
            pdf.set_auto_page_break(auto=True, margin=15)

            # Front Page
            pdf.add_page()
            pdf.set_font("Arial", "B", 24)
            pdf.cell(0, 60, "Construction Project Report", ln=True, align="C")
            pdf.ln(20)
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "BuildSmart - AI for Smarter Construction", ln=True, align="C")
            pdf.ln(10)
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, "Generated by BuildSmart", ln=True, align="C")
            pdf.ln(10)
            pdf.image("logo.png", x=80, w=50, h=50, type='PNG')
            pdf.ln(20)

            # Main Content
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, f"Total Projects: {len(df_test)}", ln=True)
            pdf.cell(0, 10, f"Delayed Projects: {df_test['Predicted Delay'].sum()}", ln=True)
            pdf.cell(0, 10, f"On-Time Projects: {len(df_test) - df_test['Predicted Delay'].sum()}", ln=True)
            pdf.cell(0, 10, f"Average Delay Duration: {average_delay if average_delay != 'N/A' else 'N/A'} days", ln=True)
            pdf.ln(10)

            pdf.image("delay_distribution.png", x=10, w=180)

            if uploaded_image:
                image = Image.open(uploaded_image)
                image_path = "uploaded_image.png"
                image.save(image_path, format="PNG")
                pdf.add_page()
                pdf.cell(200, 10, "User Uploaded Image", ln=True, align="C")
                pdf.image(image_path, x=10, w=180)
                st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

            pdf.output("predicted_report.pdf")

            st.success("✅ PDF report generated.")
            with open("predicted_report.pdf", "rb") as f:
                st.download_button("📥 Download Report PDF", f, file_name="predicted_report.pdf", mime="application/pdf")

