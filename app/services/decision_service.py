from app.utils.config import THRESHOLD_REJECT, THRESHOLD_REVIEW


def make_decision(fraud_probability: float) -> dict:
    """
    Apply threshold logic to fraud probability.
    Returns decision string and reason.
    """
    if fraud_probability >= THRESHOLD_REJECT:
        return {
            "decision": "REJECT",
            "reason": "High fraud probability detected",
            "color": "red"
        }
    elif fraud_probability >= THRESHOLD_REVIEW:
        return {
            "decision": "REVIEW",
            "reason": "Moderate suspicion — needs manual review",
            "color": "orange"
        }
    else:
        return {
            "decision": "APPROVE",
            "reason": "Claim appears legitimate",
            "color": "green"
        }