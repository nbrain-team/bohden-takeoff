
def apply_custom_css():
    custom_css = """
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #ffffff !important; /* White background */
        color: #212529; /* Dark grey text */
    }
    .block-container {
        padding-top: 35px !important;
        background-color: #ffffff !important; /* White background */
    }
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #007bff; /* Blue */
    }
    .landing-container {
        text-align: center;
        margin-top: 50px;
        background-color: #ffffff !important; /* White background */
    }
    .landing-button {
        font-size: 20px;
        padding: 12px 25px;
        background-color: #007bff; /* Blue */
        color: #ffffff;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: 0.3s;
    }
    .landing-button:hover {
        background-color: #0056b3; /* Darker blue */
    }
    </style>
    """
    sidebar_css = """
    <style>
        [data-testid="stSidebar"] {
            padding-top: 35px !important; 
            background-color: #ffffff !important; /* White background */
            width: 250px !important;
            min-width: 250px !important;
            max-width: 250px !important;
        }
        /* Sidebar Title */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: #212529 !important; /* Dark grey text */
            text-align: center;
        }
        [data-testid="stSidebar"] hr {
            border-top: 2px solid #212529 !important; /* Dark grey separator */
        }
        /* Sidebar Items */
        .sidebar-item {
            font-size: 18px;
            padding: 12px 15px;
            background-color: #007bff !important; /* Blue button */
            color: #ffffff !important; /* White text */
            border-radius: 8px;
            text-align: center;
            margin: 8px 0;
            cursor: pointer;
            transition: 0.3s;
            font-weight: 500;
        }
        .sidebar-item:hover {
            background-color: #0056b3 !important; /* Darker blue on hover */
        }
    </style>
    """
    return custom_css + sidebar_css
