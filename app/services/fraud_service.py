import pickle
import numpy as np
from app.utils.config import FRAUD_MODEL_PATH

# Load model once at startup
with open(FRAUD_MODEL_PATH, 'rb') as f:
    fraud_model = pickle.load(f)


def predict_fraud(
    plate_valid: int,
    match_score: float,
    num_detected: int,
    num_claimed: int,
    claim_amount: float,
    vehicle_age: int
) -> dict:
    """
    Predict fraud probability using XGBoost.
    Feature order must match training order exactly.
    """
    features = np.array([[
        plate_valid,
        match_score,
        num_detected,
        num_claimed,
        claim_amount,
        vehicle_age
    ]])

    fraud_probability = float(fraud_model.predict_proba(features)[0][1])

    return {
        "fraud_probability": round(fraud_probability, 4)
    }