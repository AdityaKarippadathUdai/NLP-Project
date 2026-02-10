import re
import torch
import nltk
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Ensure NLTK sentence tokenizer is available
nltk.download("punkt", quiet=True)

# ============================
# Device Selection
# ============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# Load Model & Tokenizer
# ============================
MODEL_NAME = "google/flan-t5-small"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

# ============================
# Text Preprocessing Function
# ============================
def _preprocess_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([.,;:!?])([A-Za-z])", r"\1 \2", text)
    return text

# ============================
# MODULE 1: CLAIM EXTRACTION
# ============================
def extract_claims(paragraph: str) -> list[dict]:
    if not paragraph or not paragraph.strip():
        return []

    paragraph = _preprocess_text(paragraph)
    sentences = nltk.sent_tokenize(paragraph)

    extracted_claims = []
    seen_claims = set()
    claim_id = 1

    for sentence in sentences:
        sentence = sentence.strip()

        if len(sentence) < 15:
            continue

        prompt = (
            "Extract factual claims from the following sentence. "
            "Return the claim text only:\n"
            f"{sentence}"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )

        decoded_output = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        candidate_claims = re.split(r"[.;]", decoded_output)

        for claim in candidate_claims:
            claim = claim.strip()

            if len(claim) < 10:
                continue

            # Require alphabetic content
            if len(re.findall(r"[A-Za-z]", claim)) < 5:
                continue

            # Drop numeric-only fragments
            if re.fullmatch(r"[\d\s.%]+", claim):
                continue

            # ✅ NUMERIC CONSISTENCY FIX
            orig_numbers = re.findall(r"\d+\.?\d*%?", sentence)
            claim_numbers = re.findall(r"\d+\.?\d*%?", claim)

            if orig_numbers and len(claim_numbers) < len(orig_numbers):
                claim = sentence

            normalized = claim.lower()
            if normalized in seen_claims:
                continue

            seen_claims.add(normalized)

            extracted_claims.append({
                "claim_id": claim_id,
                "claim": claim
            })

            claim_id += 1

    return extracted_claims

