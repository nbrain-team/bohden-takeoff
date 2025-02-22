import streamlit as st

# Set page configuration
st.set_page_config(page_title="BuildSmart - Construction AI", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
        /* Make the whole background white */
        body, [data-testid="stAppViewContainer"] {
            background-color: white !important;
            color: black !important;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #FFD700; /* Yellow */
            padding: 1rem;
        }

        /* Adjust main content */
        .block-container {
            padding: 3rem;
            background-color: white;
            color: black;
            text-align: center;
            border-radius: 10px;
            margin: 2rem;
        }

        /* Heading Styling */
        h1, h2, h3 {
            color: black !important;
        }

        /* Button Styling */
        .stButton > button {
            background-color: #FFD700;
            color: black;
            border-radius: 8px;
            font-size: 18px;
            font-weight: bold;
            padding: 12px 24px;
            border: none;
            cursor: pointer;
            width: 100%;
        }
        .stButton > button:hover {
            background-color: #E6C200;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Sidebar
st.sidebar.title("🚧 BuildSmart")
st.sidebar.markdown("### **Navigation**")
st.sidebar.button("🏠 Home", key="home")
st.sidebar.button("📊 Dashboard", key="dashboard")
st.sidebar.button("🛡️ Metro Risk Prediction", key="metro_risk")
st.sidebar.button("💬 Chatbot", key="chatbot")
st.sidebar.button("📜 Blueprint Detection", key="blueprint")
st.sidebar.button("🚚 Supply Chain Copilot", key="supply_chain")
st.sidebar.button("⚠️ Risk Detection", key="risk_detection")

# Main content
st.markdown("<div class='block-container'>", unsafe_allow_html=True)
st.title("BuildSmart - Construction AI")
st.subheader("🏗️ AI for Smarter Construction")
st.write("🚀 AI-powered solutions for project risk assessment, cost optimization, and safety monitoring.")

# Get Started Button
st.markdown('<div style="margin-top: 20px;">', unsafe_allow_html=True)
st.button("Get Started", key="get_started")
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
