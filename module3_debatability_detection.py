import os
import re
import requests
from google import genai

# ======================================================
# GEMINI SETUP (PRIMARY CLASSIFIER)
# ======================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None


# ======================================================
# GEMINI CLASSIFIER (DOMAIN-ROBUST PROMPT)
# ======================================================
def _gemini_debatable(claim: str) -> str | None:
    """
    Returns:
        "debatable"
        "non-debatable"
        None if Gemini fails
    """

    if not client:
        print("⚠️ GEMINI_API_KEY not set.")
        return None

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"""
You are a strict logical classifier.

Your task is to classify the sentence as either:

- debatable
- non-debatable

DEFINITION OF NON-DEBATABLE:
A sentence is non-debatable if it:
- States a verifiable fact
- Describes a historical event
- Describes an ongoing mission, program, experiment, or activity
- Reports confirmed scientific findings
- Reports what an institution or authority said (without evaluating it)
- Describes exploration, investigation, research, or observation
- Contains uncertainty about outcome but does NOT express an opinion
- Is neutral and informational

DEFINITION OF DEBATABLE:
A sentence is debatable if it:
- Expresses an opinion or value judgment
- Makes a prediction about future impact
- Suggests something will significantly transform, revolutionize, solve, destroy, or harm
- Uses normative language (should, must, better, worse, necessary, harmful)
- Makes a claim about social, economic, ethical, or policy consequences
- Contains interpretation or evaluation beyond neutral reporting

IMPORTANT DISTINCTIONS:

1) Scientific uncertainty ≠ debatability.
   Example (non-debatable):
   "The rover is searching for signs of life."

2) Reporting research activity ≠ opinion.
   Example (non-debatable):
   "Scientists are studying quantum entanglement."

3) Predicting major impact = debatable.
   Example (debatable):
   "Quantum computers will revolutionize cybersecurity."

4) If the sentence only describes what is happening,
   even if the outcome is unknown, classify as non-debatable.

5) If ANY part evaluates impact,
   predicts consequences, or implies societal change,
   classify as debatable.

Final rule:
If the sentence can be verified as true or false
without requiring personal interpretation,
classify as non-debatable.

Respond with ONLY one word:
debatable
or
non-debatable

Sentence:
"{claim}"
"""
        )

        if not response.text:
            return None

        answer = response.text.strip().lower()

        if "non-debatable" in answer:
            return "non-debatable"
        if "debatable" in answer:
            return "debatable"

        return None

    except Exception as e:
        print("❌ Gemini exception:", e)
        return None


# ======================================================
# Hugging Face Zero-Shot Fallback
# ======================================================
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/typeform/distilbert-base-uncased-mnli"

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
} if HF_API_TOKEN else {}


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
            timeout=15
        )

        if response.status_code != 200:
            return False

        result = response.json()
        labels = result.get("labels", [])

        return labels and "disagree" in labels[0].lower()

    except Exception:
        return False


# ======================================================
# LAYER 1: AUTHORITATIVE FACTUAL INDICATORS
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
# LAYER 2: SCIENTIFIC / MISSION CONTEXT PROTECTION
# ======================================================
SCIENTIFIC_CONTEXT_KEYWORDS = [
    "rover",
    "nasa",
    "mission",
    "launched",
    "explore",
    "exploring",
    "observatory",
    "satellite",
    "collider",
    "experiment",
    "study",
    "researchers at",
    "ipcc",
    "cern",
    "laboratory",
    "telescope",
]


def _is_scientific_context(text: str) -> bool:
    text = text.lower()
    return any(keyword in text for keyword in SCIENTIFIC_CONTEXT_KEYWORDS)


# ======================================================
# LAYER 3: ATTRIBUTION / STANCE
# ======================================================
ATTRIBUTION_MARKERS = [
    "argue", "argues", "argued",
    "claim", "claims",
    "believe", "believes",
    "critic", "critics",
    "supporter", "supporters",
    "experts say",
    "scientists say",
    "economists say",
    "analysts say",
]


# ======================================================
# LAYER 4: MODALITY / UNCERTAINTY
# ======================================================
MODAL_MARKERS = [
    "could", "may", "might", "likely", "unlikely",
    "potential", "risk", "threat",
    "expected to", "projected to",
    "forecast", "estimate",
]


# ======================================================
# LAYER 5: IMPACT / TRANSFORMATION LANGUAGE
# ======================================================
IMPACT_MARKERS = [
    "revolutionize",
    "transform",
    "change society",
    "solve",
    "destroy",
    "reshape",
    "disrupt",
    "eliminate",
    "replace",
]


# ======================================================
# FINAL DECISION FUNCTION
# ======================================================
def classify_claim_debatability(claim: str) -> str:

    text = claim.lower()

    # 1️⃣ Hard factual override
    if _is_authoritative_fact(text):
        return "non-debatable"

    # 2️⃣ Scientific mission / reporting protection
    if _is_scientific_context(text) and not any(marker in text for marker in IMPACT_MARKERS):
        return "non-debatable"

    # 3️⃣ Strong impact or prediction markers
    if any(marker in text for marker in IMPACT_MARKERS):
        return "debatable"

    if any(marker in text for marker in MODAL_MARKERS):
        return "debatable"

    if any(marker in text for marker in ATTRIBUTION_MARKERS):
        return "debatable"

    # 4️⃣ Gemini primary reasoning
    gemini_result = _gemini_debatable(claim)
    if gemini_result:
        return gemini_result

    # 5️⃣ Zero-shot fallback
    if _zero_shot_debatable(claim):
        return "debatable"

    return "non-debatable"


# ======================================================
# MODULE INTERFACE (UI COMPATIBLE)
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
