import os
import joblib
from together import Together

def load_models_and_encoders():
    # Determine the absolute path of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct absolute paths for the model and encoder files
    metro_model_path = os.path.join(base_dir, 'backend', 'metro', 'xgboost_metro_classifier.pkl')
    regressor_path = os.path.join(base_dir, 'backend', 'metro', 'xgboost_metro_regressor.pkl')
    encoder_path = os.path.join(base_dir, 'backend', 'metro', 'label_encoders.pkl')

    # Load the classifier, regressor, and label encoders
    classifier = joblib.load(metro_model_path)
    regressor = joblib.load(regressor_path)
    label_encoders = joblib.load(encoder_path)

    # Initialize the Together client with the API key from environment variables
    client = Together(api_key=os.getenv("TOGETHER_API_KEY1"))

    return classifier, regressor, label_encoders, client
