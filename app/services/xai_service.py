import shap
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from app.utils.config import FRAUD_MODEL_PATH, OUTPUT_DIR

with open(FRAUD_MODEL_PATH, 'rb') as f:
    xai_model = pickle.load(f)

FEATURE_NAMES = ['plate_valid', 'match_score', 'num_detected',
                 'num_claimed', 'claim_amount', 'vehicle_age']

FEATURE_EXPLANATIONS = {
    'plate_valid':    ("Invalid license plate", "Valid license plate"),
    'match_score':    ("Low match between claim and detected damage", "High match between claim and detected damage"),
    'num_detected':   ("Few damages detected by AI", "Many damages detected by AI"),
    'num_claimed':    ("Few damages claimed", "Many damages claimed"),
    'claim_amount':   ("High claim amount", "Low claim amount"),
    'vehicle_age':    ("Old vehicle", "New vehicle")
}


def explain_prediction(features: dict) -> dict:
    """
    Generate SHAP explanation for a single prediction.
    Returns top 3 human readable reasons and saves SHAP chart.
    """
    feature_values = np.array([[
        features['plate_valid'],
        features['match_score'],
        features['num_detected'],
        features['num_claimed'],
        features['claim_amount'],
        features['vehicle_age']
    ]])

    explainer = shap.TreeExplainer(xai_model)
    shap_values = explainer.shap_values(feature_values)[0]

    # Sort features by absolute SHAP value
    sorted_idx = np.argsort(np.abs(shap_values))[::-1]
    top_reasons = []

    for idx in sorted_idx[:3]:
        feature = FEATURE_NAMES[idx]
        value = feature_values[0][idx]
        shap_val = shap_values[idx]
        low_msg, high_msg = FEATURE_EXPLANATIONS[feature]
        reason = low_msg if shap_val > 0 else high_msg
        top_reasons.append(reason)

    # Save SHAP waterfall chart
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(8, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value,
            data=feature_values[0],
            feature_names=FEATURE_NAMES
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_explanation.png'))
    plt.close()

    return {
        "top_reasons": top_reasons,
        "shap_values": shap_values.tolist()
    }