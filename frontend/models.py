import os
import joblib
from together import Together

def load_models_and_encoders():
    METRO_MODEL_PATH = "../backend/metro/xgboost_metro_classifier.pkl"
    ENCODER_PATH = "../backend/metro/label_encoders.pkl"
    classifier = joblib.load(METRO_MODEL_PATH)
    regressor = joblib.load("../backend/metro/xgboost_metro_regressor.pkl")
    label_encoders = joblib.load(ENCODER_PATH)
    client = Together(api_key=os.getenv("TOGETHER_API_KEY1"))
    return classifier, regressor, label_encoders, client
