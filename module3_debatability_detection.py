import os
import re
import requests

# ======================================================
# Hugging Face Inference API (FREE, STABLE MODEL)
# ======================================================
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # optional

API_URL = "https://api-inference.huggingface.co/models/typeform/distilbert-base-uncased-mnli"

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
} if HF_API_TOKEN else {}

# ======================================================
# LAYER 1: AUTHORITATIVE FACTUAL INDICATORS (HIGHEST)
# ======================================================
AUTHORITATIVE_SOURCES = [
    "official data",
    "government data",
    "according to official",
    "according to government",
    "world bank",
    "imf",
    "un report",
    "census",
    "statistics bureau",
    "ministry of",
]

def _is_authoritative_fact(text: str) -> bool:
    text = text.lower()

    has_number = bool(re.search(r"\d+(\.\d+)?%?", text))
    has_year = bool(re.search(r"\b(19|20)\d{2}\b", text))
    has_source = any(src in text for src in AUTHORITATIVE_SOURCES)

    return (has_number and has_year) or has_source

# ======================================================
# LAYER 2: ATTRIBUTION / STANCE
# ======================================================
ATTRIBUTION_MARKERS = [
    "argue", "argues", "argued",
    "claim", "claims",
    "believe", "believes",
    "warn", "warns", "warned",
    "critic", "critics",
    "supporter", "supporters",
    "experts say",
    "scientists say",
    "economists say",
    "analysts say",
]

# ======================================================
# LAYER 3: MODALITY / UNCERTAINTY
# ======================================================
MODAL_MARKERS = [
    "could", "may", "might", "likely", "unlikely",
    "potential", "risk", "threat",
    "expected to", "projected to",
    "forecast", "estimate",
    "continues to",
]

# ======================================================
# LAYER 4: ZERO-SHOT FALLBACK (API)
# ======================================================
def _zero_shot_debatable(claim: str) -> bool:
    payload = {
        "inputs": claim,
        "parameters": {
            "candidate_labels": [
                "pure factual statement",
                "claim that people can reasonably disagree about"
            ]
        }
    }

    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json=payload,
            timeout=20
        )

        if response.status_code != 200:
            return False

        result = response.json()
        labels = result.get("labels", [])

        return labels and "disagree" in labels[0].lower()

    except Exception:
        return False

# ======================================================
# FINAL DECISION FUNCTION
# ======================================================
def classify_claim_debatability(claim: str) -> str:
    text = claim.lower()

    # ✅ Layer 1: Authoritative facts override everything
    if _is_authoritative_fact(text):
        return "non-debatable"

    # ✅ Layer 2: Attribution → debatable
    if any(marker in text for marker in ATTRIBUTION_MARKERS):
        return "debatable"

    # ✅ Layer 3: Modality / uncertainty → debatable
    if any(marker in text for marker in MODAL_MARKERS):
        return "debatable"

    # ✅ Layer 4: Zero-shot semantic fallback
    if _zero_shot_debatable(claim):
        return "debatable"

    return "non-debatable"

# ======================================================
# MODULE INTERFACE
# ======================================================
def classify_debatability(claims: list[dict]) -> list[dict]:
    results = []

    for item in claims:
        claim_text = item.get("claim", "").strip()
        if not claim_text:
            continue

        label = classify_claim_debatability(claim_text)

        results.append({
            "claim_id": item.get("claim_id"),
            "claim": claim_text,
            "simplified_claim": item.get("simplified_claim"),
            "label": label
        })

    return results
