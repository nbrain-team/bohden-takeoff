def apply_custom_css():
    custom_css = """
    <style>
        /* Import Google Icons */
        @import url('https://fonts.googleapis.com/icon?family=Material+Icons');

        /* Make the whole background white */
        body {
            background-color: white !important;
            color: black !important;
            margin-top: -30px !important; /* Remove all margins */
            padding-top: -30px !important; /* Remove all padding */
            overflow-x: hidden !important; /* Disable horizontal scrolling */
        }
        [data-testid="stAppViewContainer"] {
            background-color: white !important;
            color: black !important;
            margin: 0px !important; /* Remove all margins */
            padding: 0px !important; /* Remove all padding */
            overflow-x: hidden !important; /* Disable horizontal scrolling */
        }
        .logo-container {
            position: absolute;
            top: -10px;
            left: -10px;
            width: 120px; /* Adjust size */
            height: auto;
        }
        .logo-container img {
            width: 100%;
            height: auto;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #f1c232; /* Yellow */
            border-right: 2px solid #d4a017; /* Slightly darker yellow border */
            width: 300px !important; /* Reduced width */
            min-width: 300px !important; /* Reduced width */
            max-width: 300px !important; /* Reduced width */
            color: black !important; /* Make text black */
            margin-top: -20px !important;
            overflow-y: hidden !important; /* Disable vertical scrolling */
        }

        /* Sidebar Item Styling */
        .sidebar-item {
            margin-bottom: 1rem; /* Add spacing between sidebar items */
            padding: 0.5rem;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s, color 0.3s;
            color: black !important; /* Make text black */
        }
        .sidebar-item:hover {
            background-color: #d4a017;
            color: white !important; /* Make text white on hover */
        }

        /* Adjust main content */
        .block-container {
            padding: 3rem;
            background-color: white;
            color: black !important; /* Make text black */
            text-align: center;
            border-radius: 10px;
            margin: 2rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow */
        }

        /* Heading Styling */
        h1, h2, h3 {
            color: black !important;
            font-family: 'Arial', sans-serif;
        }

        /* Button Styling */
        .stButton > button {
            background-color: #f1c232;
            color: black !important; /* Make text black */
            border-radius: 8px;
            font-size: 18px;
            font-weight: bold;
            padding: 12px 24px;
            border: none;
            cursor: pointer;
            width: 100%;
            transition: background-color 0.3s, color 0.3s;
        }
        .stButton > button:hover {
            background-color: #d4a017;
            color: white !important; /* Make text white on hover */
        }
        .stImage > img {
            width: 100% !important;
            height: auto !important;
            display: block;
        }

        /* Right Sidebar Styling */
        .right-sidebar {
            position: absolute;
            right: 20px;
            top: 80px;
            width: 300px;
            background: #f9f9f9;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            font-size: 14px;
            color: black !important; /* Make text black */
        }
        
        @media (max-width: 768px) {
            .right-sidebar {
                position: relative;
                width: 100%;
                right: 0;
            }
        }
        
        .midbar {
            position: absolute;
            
            top: 80px;
            width: 100%;
            background: #f9f9f9;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            font-size: 14px;
            color: black !important; /* Make text black */
        }
        
        @media (max-width: 768px) {
            .midbar {
                position: relative;
                width: 100%;
                right: 0;
            }
        }


        /* Icon Styling */
        .material-icons {
            vertical-align: middle;
            margin-right: 8px;
            color: black !important; /* Make icons black */
        }

        /* Logo Styling */
        .logo {
            display: block;
            margin: 0 auto;
            width: 150px;
            height: auto;
        }
    </style>
    """
    return custom_css
