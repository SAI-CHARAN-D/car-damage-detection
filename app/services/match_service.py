def calculate_match(detected_classes: list, claimed_damages: list) -> dict:
    """
    Compare YOLO detections with claimed damages.
    Returns match score between 0 and 1.
    """
    if not detected_classes or not claimed_damages:
        return {"match_score": 0.0}

    detected_set = set(detected_classes)
    claimed_set = set(claimed_damages)

    # Intersection over Union style scoring
    intersection = len(detected_set & claimed_set)
    union = len(detected_set | claimed_set)

    match_score = round(intersection / union, 3) if union > 0 else 0.0

    return {
        "match_score": match_score,
        "matched_damages": list(detected_set & claimed_set),
        "unmatched_claimed": list(claimed_set - detected_set),
        "unmatched_detected": list(detected_set - claimed_set)
    }