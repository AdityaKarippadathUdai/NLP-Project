from llama_cpp import Llama
from typing import List, Dict, Generator
import re
import os

# ======================================================
# CONFIG (MISTRAL)
# ======================================================
MODEL_PATH = "/home/aditya/Project/models/quantized/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

N_CTX = 4096

# Bigger output (Mistral handles well)
MAX_GENERATION_TOKENS = 1800

TEMPERATURE = 0.7
TOP_P = 0.95
REPEAT_PENALTY = 1.1

RESERVED_OUTPUT_TOKENS = 1500
MAX_INPUT_TOKENS = N_CTX - RESERVED_OUTPUT_TOKENS


# ======================================================
# LOAD MODEL
# ======================================================
print("🔄 Loading Mistral model...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_threads=os.cpu_count(),
    n_gpu_layers=35  # adjust if no GPU
)

print("✅ Mistral loaded!")


# ======================================================
# CLEANING
# ======================================================
def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip() if text else ""


def _clean_for_llm(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    return _clean(text)


# ======================================================
# TOKEN ESTIMATION
# ======================================================
def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ======================================================
# SMART EVIDENCE SELECTION
# ======================================================
def _select_evidence(evidence: List[Dict], claim: str) -> str:

    selected = []
    total_tokens = _estimate_tokens(claim) + 150

    for ev in evidence:
        content = _clean_for_llm(ev.get("content", ""))[:350]

        if len(content) < 60:
            continue

        tokens = _estimate_tokens(content)

        if total_tokens + tokens > MAX_INPUT_TOKENS:
            break

        selected.append(content)
        total_tokens += tokens

    if not selected and evidence:
        selected.append(_clean_for_llm(evidence[0].get("content", ""))[:250])

    return "\n\n".join(f"- {txt}" for txt in selected)


# ======================================================
# MISTRAL PROMPT (CHAT FORMAT)
# ======================================================
def _build_prompt(claim: str, evidence_text: str) -> str:

    return f"""<s>[INST]
You are an expert debate analyst.

Analyze the claim deeply using provided evidence.

CLAIM:
{claim}

EVIDENCE:
{evidence_text}

INSTRUCTIONS:
- Provide a detailed debate
- Give 6-10 PRO arguments (deep reasoning)
- Give 6-10 AGAINST arguments (critical reasoning)
- Each point should be 2-4 lines
- Use evidence meaningfully (not generic)
- Avoid repetition

FORMAT:

PRO:
- ...

AGAINST:
- ...

CONCLUSION:
- 4-6 insightful sentences
[/INST]"""


# ======================================================
# STREAM GENERATION
# ======================================================
def _stream_generate(prompt: str) -> Generator[str, None, None]:

    stream = llm(
        prompt,
        max_tokens=MAX_GENERATION_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repeat_penalty=REPEAT_PENALTY,
        stream=True
    )

    full_text = ""

    for chunk in stream:
        token = chunk["choices"][0]["text"]

        if not token:
            continue

        full_text += token
        yield full_text


# ======================================================
# OUTPUT CLEANUP
# ======================================================
def _fix_output(text: str) -> str:

    text = re.sub(r"IMPLICATIONS.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"NOTE:.*", "", text, flags=re.IGNORECASE)

    if "CONCLUSION" not in text:
        text += "\n\nCONCLUSION:\n- The issue is complex with both strong advantages and drawbacks."

    return text.strip()


# ======================================================
# PARSER
# ======================================================
def _parse_output(text: str) -> Dict:

    result = {"pro": [], "against": [], "conclusion": ""}
    section = None

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            continue

        lower = line.lower()

        if lower.startswith("pro"):
            section = "pro"
            continue

        elif lower.startswith("against"):
            section = "against"
            continue

        elif lower.startswith("conclusion"):
            section = "conclusion"
            continue

        if section == "pro" and line.startswith("-"):
            result["pro"].append(line[1:].strip())

        elif section == "against" and line.startswith("-"):
            result["against"].append(line[1:].strip())

        elif section == "conclusion":
            result["conclusion"] += " " + line

    result["conclusion"] = result["conclusion"].strip()

    return result


# ======================================================
# MAIN STREAM FUNCTION
# ======================================================
def generate_debate_output_stream(filtered_results: List[Dict]):

    for item in filtered_results:

        claim = _clean(item.get("claim", ""))
        evidence = item.get("filtered_evidence", [])

        if not claim:
            continue

        evidence_text = _select_evidence(evidence, claim)
        prompt = _build_prompt(claim, evidence_text)

        full_output = ""

        try:
            # STREAM
            for partial in _stream_generate(prompt):
                full_output = partial

                yield {
                    "type": "stream",
                    "claim_id": item.get("claim_id"),
                    "claim": claim,
                    "text": full_output
                }

            # FINAL FIX
            full_output = _fix_output(full_output)
            parsed = _parse_output(full_output)

            yield {
                "type": "final",
                "claim_id": item.get("claim_id"),
                "claim": claim,
                "pro": parsed["pro"][:10],
                "against": parsed["against"][:10],
                "conclusion": parsed["conclusion"]
            }

        except Exception as e:
            yield {
                "type": "final",
                "claim_id": item.get("claim_id"),
                "claim": claim,
                "pro": [],
                "against": [],
                "conclusion": f"Error: {str(e)}"
            }