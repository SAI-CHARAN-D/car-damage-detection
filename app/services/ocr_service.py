import easyocr
import json
import re
from app.utils.config import VEHICLE_DB_PATH
from app.utils.image_utils import bytes_to_numpy

# Load OCR reader once at startup
reader = easyocr.Reader(['en'], gpu=False)

# Load vehicle database once at startup
with open(VEHICLE_DB_PATH, 'r') as f:
    vehicle_db = json.load(f)
VALID_PLATES = [plate.upper().replace(" ", "") for plate in vehicle_db.get("plates", [])]


def extract_plate(image_bytes: bytes) -> dict:
    """
    Extract license plate from image and validate against database.
    Returns plate text and validity flag.
    """
    img = bytes_to_numpy(image_bytes)
    results = reader.readtext(img)

    plate_text = ""
    for (_, text, confidence) in results:
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        if len(cleaned) >= 4:  # plates are at least 4 chars
            plate_text = cleaned
            break

    plate_valid = 1 if plate_text in VALID_PLATES else 0

    return {
        "plate_text": plate_text,
        "plate_valid": plate_valid
    }