from ultralytics import YOLO
from app.utils.config import YOLO_MODEL_PATH, CLASS_NAMES
from app.utils.image_utils import bytes_to_numpy, resize_image, draw_boxes, save_annotated_image

# Load model once at startup
yolo_model = YOLO(YOLO_MODEL_PATH)


def run_yolo(image_bytes: bytes) -> dict:
    """
    Run YOLO inference on image bytes.
    Returns detected class names, count, confidence scores,
    and path to annotated image.
    """
    img = bytes_to_numpy(image_bytes)
    img = resize_image(img)

    results = yolo_model(img)[0]

    detections = []
    detected_classes = []

    for box in results.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = CLASS_NAMES.get(class_id, "unknown")

        detections.append({
            "label": label,
            "confidence": round(confidence, 3),
            "box": (x1, y1, x2, y2)
        })
        detected_classes.append(label)

    # Draw boxes and save
    annotated = draw_boxes(img.copy(), detections)
    annotated_path = save_annotated_image(annotated)

    return {
        "detections": detections,
        "detected_classes": list(set(detected_classes)),  # unique classes
        "num_detected": len(detections),
        "annotated_image_path": annotated_path
    }