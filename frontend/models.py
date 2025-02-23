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

    # Check if the classifier file exists
    if not os.path.exists(metro_model_path):
        raise FileNotFoundError(f"Classifier file not found: {metro_model_path}")
    # Load the classifier
    classifier = joblib.load(metro_model_path)

    # Check if the regressor file exists
    if not os.path.exists(regressor_path):
        raise FileNotFoundError(f"Regressor file not found: {regressor_path}")
    # Load the regressor
    regressor = joblib.load(regressor_path)

    # Check if the encoder file exists
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder file not found: {encoder_path}")
    # Load the label encoders
    label_encoders = joblib.load(encoder_path)

    # Retrieve the Together API key from environment variables
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise EnvironmentError("TOGETHER_API_KEY environment variable is not set.")
    # Initialize the Together client with the API key
    client = Together(api_key=api_key)

    return classifier, regressor, label_encoders, client
