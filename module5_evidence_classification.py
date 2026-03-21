import re
import hashlib
from typing import List, Dict
from collections import defaultdict

# ======================================================
# CONFIG
# ======================================================
MIN_TEXT_LENGTH = 80
MAX_TEXT_LENGTH = 500
TOP_K = 10
MAX_PER_SOURCE = 2

MIN_RELEVANCE = 2
MIN_ARGUMENT_SCORE = 1


# ======================================================
# CLEAN
# ======================================================
def _clean(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


# ======================================================
# GENERIC FILTER
# ======================================================
GENERIC_PATTERNS = [
    "this article", "we explore", "this study",
    "in this article", "this chapter",
    "discussion", "learn more", "click here",
    "sign up", "newsletter", "advertisement",
    "the question of", "we examine"
]


def _is_generic(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in GENERIC_PATTERNS)


# ======================================================
# WEAK CONTENT FILTER 🔥
# ======================================================
def _is_weak(text: str) -> bool:
    t = text.lower()

    weak_patterns = [
        "experts say", "studies show",
        "it is believed", "it is expected",
        "many believe", "there is concern"
    ]

    if any(p in t for p in weak_patterns):
        has_number = bool(re.search(r"\d+", t))
        strong_words = [
            "replace", "eliminate", "create",
            "increase", "decrease", "loss",
            "growth", "automation"
        ]
        has_signal = any(w in t for w in strong_words)

        if not has_number and not has_signal:
            return True

    return False


# ======================================================
# TOKENIZE
# ======================================================
def _tokenize(text: str) -> set:
    return set(re.findall(r"\w+", text.lower()))


# ======================================================
# RELEVANCE
# ======================================================
def _relevance(claim: str, text: str) -> int:
    return len(_tokenize(claim).intersection(_tokenize(text)))


# ======================================================
# ARGUMENT SCORE
# ======================================================
ARG_KEYWORDS = [
    "job", "employment", "replace", "automation",
    "workers", "labor", "economy", "impact",
    "increase", "decrease", "growth", "loss",
    "risk", "benefit", "challenge", "disrupt"
]


def _arg_score(text: str) -> int:
    t = text.lower()
    return sum(1 for w in ARG_KEYWORDS if w in t)


# ======================================================
# STRONG SIGNAL SCORE 🔥
# ======================================================
def _strong_signal(text: str) -> int:
    t = text.lower()

    signals = [
        "replace", "eliminate", "displace",
        "job loss", "mass unemployment",
        "create jobs", "new jobs",
        "augment", "assist", "complement"
    ]

    return sum(1 for s in signals if s in t)


# ======================================================
# FACT DENSITY 🔥
# ======================================================
def _fact_density(text: str) -> float:
    words = text.split()
    if not words:
        return 0

    numbers = len(re.findall(r"\d+", text))
    return numbers / len(words)


# ======================================================
# HASH (DEDUP)
# ======================================================
def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ======================================================
# VALIDATION
# ======================================================
def _is_valid(text: str) -> bool:
    if not text:
        return False
    if len(text) < MIN_TEXT_LENGTH:
        return False
    if _is_generic(text):
        return False
    if _is_weak(text):
        return False
    return True


# ======================================================
# MAIN FUNCTION
# ======================================================
def filter_and_rank_evidence(retrieved_results: List[Dict]) -> List[Dict]:

    final_results = []

    for item in retrieved_results:

        if item.get("label") != "debatable":
            continue

        claim = _clean(item.get("claim", ""))
        chunks = item.get("evidence_chunks", [])

        scored = []
        seen = set()
        source_count = defaultdict(int)

        for ch in chunks:

            text = _clean(ch.get("content", ""))

            if not _is_valid(text):
                continue

            rel = _relevance(claim, text)
            arg = _arg_score(text)
            signal = _strong_signal(text)
            density = _fact_density(text)

            if rel < MIN_RELEVANCE or arg < MIN_ARGUMENT_SCORE:
                continue

            # 🔥 FINAL SCORE
            score = (
                (1.5 * rel) +
                (2.0 * arg) +
                (2.5 * signal) +
                (5.0 * density)
            )

            text = text[:MAX_TEXT_LENGTH]

            h = _hash_text(text)
            if h in seen:
                continue

            source = ch.get("source", "")
            if source_count[source] >= MAX_PER_SOURCE:
                continue

            seen.add(h)
            source_count[source] += 1

            scored.append({
                "source": source,
                "url": ch.get("url", ""),
                "content": text,
                "score": round(score, 3)
            })

        # =========================
        # SORT
        # =========================
        scored.sort(key=lambda x: x["score"], reverse=True)

        # =========================
        # SMART FALLBACK 🔥
        # =========================
        if not scored:
            fallback = []

            for ch in chunks:
                text = _clean(ch.get("content", ""))
                if len(text) > 80:
                    fallback.append({
                        "source": ch.get("source", ""),
                        "url": ch.get("url", ""),
                        "content": text[:MAX_TEXT_LENGTH],
                        "score": 0.1
                    })

                if len(fallback) >= TOP_K:
                    break

            scored = fallback

        final_results.append({
            "claim_id": item.get("claim_id"),
            "claim": claim,
            "filtered_evidence": scored[:TOP_K]
        })

    return final_results