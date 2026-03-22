# module6_llm_reasoning.py

from llama_cpp import Llama
from typing import List, Dict, Generator
import re

# ======================================================
# CONFIG
# ======================================================
MODEL_PATH = r"/home/aditya/Project/models/quantized/llama/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

N_CTX = 4096
MAX_GENERATION_TOKENS = 1500   # stable generation

TEMPERATURE = 0.6
TOP_P = 0.9
REPEAT_PENALTY = 1.15

RESERVED_OUTPUT_TOKENS = 1200
MAX_INPUT_TOKENS = N_CTX - RESERVED_OUTPUT_TOKENS


# ======================================================
# LOAD MODEL
# ======================================================
print("🔄 Loading LLaMA model...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_threads=8,
    n_gpu_layers=35
)

print("✅ LLaMA loaded!")


# ======================================================
# CLEANING
# ======================================================
def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip() if text else ""


def _clean_for_llm(text: str) -> str:
    text = re.sub(r"http\S+", "", text)      # remove URLs
    text = re.sub(r"\([^)]*\)", "", text)    # remove brackets
    text = re.sub(r"\[[^\]]*\]", "", text)   # remove [1] citations
    return _clean(text)


# ======================================================
# TOKEN ESTIMATION
# ======================================================
def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ======================================================
# EVIDENCE SELECTION (TOKEN SAFE)
# ======================================================
def _select_evidence(evidence: List[Dict], claim: str) -> str:

    selected = []
    total_tokens = _estimate_tokens(claim) + 150

    for ev in evidence:
        content = _clean_for_llm(ev.get("content", ""))[:280]

        if len(content) < 60:
            continue

        tokens = _estimate_tokens(content)

        if total_tokens + tokens > MAX_INPUT_TOKENS:
            break

        selected.append(content)
        total_tokens += tokens

    if not selected and evidence:
        selected.append(_clean_for_llm(evidence[0].get("content", ""))[:200])

    return "\n\n".join(f"- {txt}" for txt in selected)


# ======================================================
# STRICT PROMPT (ANTI-HALLUCINATION)
# ======================================================
def _build_prompt(claim: str, evidence_text: str) -> str:

    return f"""
You are an expert debate analyst.

CLAIM:
{claim}

EVIDENCE:
{evidence_text}

STRICT INSTRUCTIONS:
- Give EXACTLY 3-5 PRO points
- Give EXACTLY 3-5 AGAINST points
- Keep each point under 3 lines
- Do NOT use [1], [2] or citations
- Do NOT add extra sections
- Do NOT repeat the claim
- Use evidence meaningfully

FORMAT (STRICT):

PRO:
- ...
- ...
- ...

AGAINST:
- ...
- ...
- ...

CONCLUSION:
- 4-6 concise sentences only
"""


# ======================================================
# STREAM GENERATION (FIXED)
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
# OUTPUT VALIDATION (CRITICAL FIX)
# ======================================================
def _fix_output(text: str) -> str:

    # Remove unwanted sections
    text = re.sub(r"IMPLICATIONS.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"NOTE:.*", "", text, flags=re.IGNORECASE)

    # Ensure conclusion exists
    if "CONCLUSION" not in text:
        text += "\n\nCONCLUSION:\n- AI will transform jobs, but full replacement is unlikely."

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
            # STREAM OUTPUT
            for partial in _stream_generate(prompt):
                full_output = partial

                yield {
                    "type": "stream",
                    "claim_id": item.get("claim_id"),
                    "claim": claim,
                    "text": full_output
                }

            # FIX OUTPUT
            full_output = _fix_output(full_output)

            parsed = _parse_output(full_output)

            # fallback safety
            if not parsed["pro"] or not parsed["against"]:
                parsed["conclusion"] = full_output[:400]

            yield {
                "type": "final",
                "claim_id": item.get("claim_id"),
                "claim": claim,
                "pro": parsed["pro"][:5],
                "against": parsed["against"][:5],
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