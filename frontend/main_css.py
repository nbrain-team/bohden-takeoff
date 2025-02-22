def apply_custom_css():
    custom_css = """
    <style>
        /* Import Google Icons */
        @import url('https://fonts.googleapis.com/icon?family=Material+Icons');

        /* Make the whole background white */
        body, [data-testid="stAppViewContainer"] {
            background-color: white !important;
            color: black !important;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #f1c232; /* Yellow */
            
            border-right: 2px solid #d4a017; /* Slightly darker yellow border */
            width: 250px !important; /* Reduced width */
            min-width: 250px !important; /* Reduced width */
            max-width: 250px !important; /* Reduced width */
        }

        /* Sidebar Item Styling */
        .sidebar-item {
            margin-bottom: 1rem; /* Add spacing between sidebar items */
            padding: 0.5rem;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s, color 0.3s;
        }
        .sidebar-item:hover {
            background-color: #d4a017;
            color: white;
        }

        /* Adjust main content */
        .block-container {
            padding: 3rem;
            background-color: white;
            color: black;
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
            color: black;
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
            color: white;
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 12px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 6px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }

        /* Icon Styling */
        .material-icons {
            vertical-align: middle;
            margin-right: 8px;
        }
    </style>
    """
    return custom_css
