from fastapi import APIRouter, UploadFile, File, Form
from app.services.yolo_service import run_yolo
from app.services.ocr_service import extract_plate
from app.services.claim_parser import parse_claim
from app.services.match_service import calculate_match
from app.services.fraud_service import predict_fraud
from app.services.decision_service import make_decision
from app.services.xai_service import explain_prediction

router = APIRouter()


@router.post("/claim")
async def process_claim(
    image: UploadFile = File(...),
    claim_text: str = Form(...),
    claim_amount: float = Form(...),
    vehicle_age: int = Form(...)
):
    image_bytes = await image.read()

    # Step 1 — YOLO damage detection
    yolo_result = run_yolo(image_bytes)

    # Step 2 — OCR plate validation
    ocr_result = extract_plate(image_bytes)

    # Step 3 — Parse claim text
    claim_result = parse_claim(claim_text)

    # Step 4 — Match score
    match_result = calculate_match(
        yolo_result["detected_classes"],
        claim_result["claimed_damages"]
    )

    # Step 5 — Fraud prediction
    fraud_result = predict_fraud(
        plate_valid=ocr_result["plate_valid"],
        match_score=match_result["match_score"],
        num_detected=yolo_result["num_detected"],
        num_claimed=claim_result["num_claimed"],
        claim_amount=claim_amount,
        vehicle_age=vehicle_age
    )

    # Step 6 — Decision
    decision_result = make_decision(fraud_result["fraud_probability"])

    # Step 7 — XAI explanation
    xai_result = explain_prediction({
        "plate_valid": ocr_result["plate_valid"],
        "match_score": match_result["match_score"],
        "num_detected": yolo_result["num_detected"],
        "num_claimed": claim_result["num_claimed"],
        "claim_amount": claim_amount,
        "vehicle_age": vehicle_age
    })

    return {
        "yolo": yolo_result,
        "ocr": ocr_result,
        "claim": claim_result,
        "match": match_result,
        "fraud": fraud_result,
        "decision": decision_result,
        "xai": xai_result
    }