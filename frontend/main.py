import streamlit as st
import requests

st.set_page_config(page_title="AI Construction Dashboard", page_icon="🏗️")

st.title("🏗️ AI Construction Dashboard")

# API URLs
FLASK_BACKEND_URL = "http://127.0.0.1:5000"

# Sidebar for Navigation
menu = st.sidebar.selectbox("Menu", ["Dashboard", "Chatbot"])

if menu == "Dashboard":
    st.subheader("📊 Project Risk Prediction & YOLO Detection")

    # Cost Overrun Prediction Form
    st.write("### Predict Cost Overrun")
    project_size = st.number_input("Project Size (sq. m)", min_value=0)
    labor_count = st.number_input("Labor Count", min_value=0)
    equipment_count = st.number_input("Equipment Count", min_value=0)
    avg_temp = st.number_input("Avg Temperature (°C)", min_value=-50, max_value=50)
    rainfall = st.number_input("Rainfall (mm)", min_value=0)
    milestone = st.text_input("Milestone")
    external_factor = st.text_input("External Factor")

    if st.button("Predict Cost Overrun"):
        response = requests.post(f"{FLASK_BACKEND_URL}/predict", json={
            "project_size": project_size,
            "labor_count": labor_count,
            "equipment_count": equipment_count,
            "avg_temp": avg_temp,
            "rainfall": rainfall,
            "milestone": milestone,
            "external_factor": external_factor
        })
        prediction = response.json().get("prediction")
        st.write(f"Prediction: {prediction}")

    # YOLO Detection Upload
    st.write("### Upload an Image for Safety Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file and st.button("Detect"):
        files = {'file': uploaded_file.getvalue()}
        response = requests.post(f"{FLASK_BACKEND_URL}/detect", files=files)
        if response.status_code == 200:
            annotated_path = response.json().get("annotated_image_path")
            st.image(annotated_path, caption="Detected Image", use_column_width=True)
        else:
            st.write("Detection failed.")

elif menu == "Chatbot":
    st.subheader("🤖 AI Construction Chatbot")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Get user input
    if user_input := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        response = requests.post(f"{FLASK_BACKEND_URL}/chatbot", json={"message": user_input})
        bot_response = response.json().get("response")

        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant"):
            st.markdown(bot_response)
