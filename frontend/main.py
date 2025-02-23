import streamlit as st
from dotenv import load_dotenv
from main_css import apply_custom_css
from utils import get_base64_image
from models import load_models_and_encoders
from pages import (
    home_page,
    dashboard_page,
    metro_risk_prediction_page,
    supplier_recommendation_page,
    chatbot_page,
    blueprint_detection_page,
    risk_detection_page,
    delay_report_generator_page,
    predict_delay_page  # Add this import
)

# Set up the Streamlit page
st.set_page_config(page_title="BuildSmart", page_icon="🏗️", layout="wide")
logo2 = Image.open('logo2.png')
st.sidebar.image(logo2, use_container_width=True)



# Load environment variables
load_dotenv()

# Load models and encoders
classifier, regressor, label_encoders, client = load_models_and_encoders()

# Apply custom CSS
st.markdown(apply_custom_css(), unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "Home"  # Default page

# Sidebar Buttons with Streamlit's st.button and icons
if st.sidebar.button(":material/Home: Home", key="home"):
    st.session_state.page = "Home"


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

if st.sidebar.button(":material/Assessment: Predict Delay Report", key="predict_delay"):  # Add this button
    st.session_state.page = "Predict Delay"

if st.sidebar.button(":material/Description: Delay Data Collection", key="delay_report"):
    st.session_state.page = "Delay Report Generator"



# Page routing
if st.session_state.page == "Home":
    home_page()

elif st.session_state.page == "Metro Risk Prediction":
    metro_risk_prediction_page(classifier, label_encoders, client)
elif st.session_state.page == "Supplier Recommendation":
    supplier_recommendation_page()
elif st.session_state.page == "Chatbot":
    chatbot_page()
elif st.session_state.page == "Blueprint Detection and Analysis":
    blueprint_detection_page(client)
elif st.session_state.page == "Risk Detection":
    risk_detection_page(client)
elif st.session_state.page == "Delay Report Generator":
    delay_report_generator_page()
elif st.session_state.page == "Predict Delay":  # Add this condition
    predict_delay_page()





