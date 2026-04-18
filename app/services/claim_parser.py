from app.utils.config import CLAIM_KEYWORDS


def parse_claim(claim_text: str) -> dict:
    """
    Parse claim text for damage keywords.
    Returns list of claimed damages and count.
    """
    text_lower = claim_text.lower()
    claimed_damages = set()

    # Check multi-word keywords first, then single word
    for keyword, damage_type in sorted(CLAIM_KEYWORDS.items(), key=lambda x: -len(x[0])):
        if keyword in text_lower:
            claimed_damages.add(damage_type)

    return {
        "claimed_damages": list(claimed_damages),
        "num_claimed": len(claimed_damages) if claimed_damages else 1
    }