

import torch
import re
import spacy
import wikipediaapi
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ============================
# Device Selection
# ============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# Load SpaCy NER Model
# ============================
nlp = spacy.load("en_core_web_sm")

# ============================
# Load Wikipedia API
# ============================
wiki = wikipediaapi.Wikipedia(
    user_agent="NLP-Claim-Simplification/1.0 (contact: aditya@example.com)",
    language="en",
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

# ============================
# Load Flan-T5 Model
# ============================
MODEL_NAME = "google/flan-t5-small"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

# ============================
# Wikipedia Summary Fetcher
# ============================
def _get_wikipedia_summary(entity_text: str) -> str | None:
    """
    Fetch a short Wikipedia definition for an entity.
    Returns 1–2 complete sentences.
    """

    page = wiki.page(entity_text)

    if not page.exists():
        return None

    summary = page.summary.strip()

    if not summary:
        return None

    # Normalize whitespace
    summary = re.sub(r"\s+", " ", summary)

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", summary)

    # Keep first 1 or 2 full sentences
    selected = sentences[:2]

    final_summary = " ".join(selected).strip()

    # Safety length check (optional)
    if len(final_summary) < 30:
        return None
    if len(final_summary) > 300:
        final_summary = final_summary[:300]
        final_summary = final_summary.rsplit(".", 1)[0] + "."

    return final_summary


# ============================
# Claim Simplification Function
# ============================
def simplify_claims(claims: list[dict]) -> list[dict]:
    """
    Simplify factual claims and enrich them with Wikipedia entity definitions.

    Args:
        claims (list[dict]): Output from Module 1
                             Each dict contains:
                             - "claim_id"
                             - "claim"

    Returns:
        list[dict]: Each dict contains:
                    - "claim_id"
                    - "original_claim"
                    - "simplified_claim"
    """

    simplified_results = []

    for item in claims:
        claim_id = item.get("claim_id")
        original_claim = item.get("claim", "").strip()

        if not original_claim:
            continue

        # ============================
        # Step 1: Flan-T5 Simplification
        # ============================
        prompt = (
            "Rewrite this claim as a clear, factual, readable sentence. "
            "Add definitions for important entities in brackets:\n"
            f"{original_claim}"
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

        simplified_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()

        # ============================
        # Step 2: Named Entity Detection
        # ============================
        doc = nlp(simplified_text)

        # Track replacements to avoid duplicates
        entity_replacements = {}

        for ent in doc.ents:
            if ent.label_ in {
                "PERSON",
                "ORG",
                "GPE",
                "LOC",
                "EVENT",
                "WORK_OF_ART"
            }:
                entity_text = ent.text

                if entity_text in entity_replacements:
                    continue

                definition = _get_wikipedia_summary(entity_text)

                if definition:
                    replacement = f"{entity_text} ({definition})"
                    entity_replacements[entity_text] = replacement

        # ============================
        # Step 3: Replace Entities
        # ============================
        for entity, replacement in entity_replacements.items():
            # Replace whole words only
            pattern = r"\b" + re.escape(entity) + r"\b"
            simplified_text = re.sub(pattern, replacement, simplified_text)

        simplified_results.append({
            "claim_id": claim_id,
            "original_claim": original_claim,
            "simplified_claim": simplified_text
        })

    return simplified_results
