import cv2
import numpy as np
import os
from app.utils.config import OUTPUT_DIR


def bytes_to_numpy(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes to numpy array for OpenCV/YOLO."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def resize_image(img: np.ndarray, max_size: int = 640) -> np.ndarray:
    """Resize image so longest side is max_size."""
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def draw_boxes(img: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw bounding boxes on image.
    detections: list of dicts with keys: label, confidence, box (x1,y1,x2,y2)
    """
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        conf = det["confidence"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(img, text, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img


def save_annotated_image(img: np.ndarray, filename: str = "annotated_image.jpg") -> str:
    """Save annotated image to outputs folder. Returns saved path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, img)
    return path