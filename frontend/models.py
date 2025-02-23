import os
import joblib
from xgboost import XGBClassifier
from together import Together

def load_models_and_encoders():
    # Determine the absolute path of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the backend directory
    backend_dir = os.path.abspath(os.path.join(base_dir, '..', 'backend', 'metro'))

    # Construct absolute paths for the model and encoder files
    metro_model_path = os.path.join(backend_dir, 'xgboost_metro_classifier.pkl')
    regressor_path = os.path.join(backend_dir, 'xgboost_metro_regressor.pkl')
    encoder_path = os.path.join(backend_dir, 'label_encoders.pkl')

    # Load classifier model
    if not os.path.exists(metro_model_path):
        raise FileNotFoundError(f"Classifier file not found: {metro_model_path}")
    classifier = joblib.load(metro_model_path)

    # Load regressor model
    if not os.path.exists(regressor_path):
        raise FileNotFoundError(f"Regressor file not found: {regressor_path}")
    regressor = joblib.load(regressor_path)

    # Load label encoders
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder file not found: {encoder_path}")
    label_encoders = joblib.load(encoder_path)

    # Initialize the Together API client
    api_key = os.getenv("TOGETHER_API_KEY1")
    if not api_key:
        raise EnvironmentError("TOGETHER_API_KEY1 environment variable not set.")
    client = Together(api_key=api_key)



    return classifier, regressor, label_encoders, client
