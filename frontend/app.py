import streamlit as st
import requests
import os

# Set the title of the Streamlit app
st.title("Blueprint Error Detection and Workforce Optimization")

# Define the URLs of the Flask backend
backend_url_predict = "http://127.0.0.1:5000/predict"
backend_url_detect = "http://127.0.0.1:5000/detect"

# Create a form for user input
with st.form("prediction_form"):
    st.header("Predict Cost Overrun")
    
    project_size = st.number_input("Project Size (sq. m)", min_value=0)
    labor_count = st.number_input("Labor Count", min_value=0)
    equipment_count = st.number_input("Equipment Count", min_value=0)
    avg_temp = st.number_input("Avg Temperature (°C)", min_value=-50, max_value=50)
    rainfall = st.number_input("Rainfall (mm)", min_value=0)
    milestone = st.selectbox("Milestone", ["Foundation", "Framing", "Roofing", "Finishing"])
    external_factor = st.selectbox("External Factor", ["None", "Weather", "Supply Chain", "Labor Strike"])
    
    # Submit button
    submit_button = st.form_submit_button(label="Predict")

# Handle form submission
if submit_button:
    # Prepare the data for the backend
    data = {
        "project_size": project_size,
        "labor_count": labor_count,
        "equipment_count": equipment_count,
        "avg_temp": avg_temp,
        "rainfall": rainfall,
        "milestone": milestone,
        "external_factor": external_factor
    }
    
    # Send a POST request to the Flask backend
    response = requests.post(backend_url_predict, json=data)
    
    # Display the prediction result
    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: {result['prediction']}")
    else:
        st.error("Error: Could not get a prediction from the backend.")

# Create a file uploader for blueprint images
st.header("Blueprint Error Detection")
uploaded_file = st.file_uploader("Choose a blueprint image...", type=["jpg", "jpeg", "png"])

# Handle file upload
if uploaded_file is not None:
    # Save the uploaded file to a temporary directory
    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Send the file to the backend for inference
    with open(file_path, "rb") as f:
        response = requests.post(backend_url_detect, files={"file": f})
    
    # Display the annotated image
    if response.status_code == 200:
        result = response.json()
        annotated_image_path = result['annotated_image_path']
        st.image(annotated_image_path, caption="Annotated Blueprint", use_column_width=True)
    else:
        st.error("Error: Could not get a detection from the backend.")
