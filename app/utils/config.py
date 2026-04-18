import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo_damage.pt")
FRAUD_MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_xgb.pkl")

VEHICLE_DB_PATH = os.path.join(BASE_DIR, "data", "vehicle_db.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# YOLO class names (Vietnamese → English mapping)
CLASS_NAMES = {
    0: "missing_part",
    1: "crack",
    2: "dent",
    3: "scratch"
}

# Fraud decision thresholds
THRESHOLD_REJECT = 0.7
THRESHOLD_REVIEW = 0.4

# Claim keywords mapped to YOLO class names
CLAIM_KEYWORDS = {
    "missing": "missing_part",
    "missing part": "missing_part",
    "crack": "crack",
    "tear": "crack",
    "rip": "crack",
    "dent": "dent",
    "dented": "dent",
    "scratch": "scratch",
    "scratched": "scratch",
    "scrape": "scratch"
}