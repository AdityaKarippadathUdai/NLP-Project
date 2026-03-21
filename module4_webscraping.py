import requests
import re
import time
import random
from bs4 import BeautifulSoup
from ddgs import DDGS
from urllib.parse import urlparse

# ======================================================
# CONFIG
# ======================================================
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

TIMEOUT = 10
MAX_RETRIES = 2
MAX_WEBSITES = 6

# ======================================================
# CLEAN TEXT
# ======================================================
def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ======================================================
# URL FILTERING
# ======================================================
BLOCKED_EXTENSIONS = [".pdf", ".ppt", ".doc"]
BLOCKED_DOMAINS = ["researchgate.net", "sciencedirect.com", "academia.edu"]

def _is_valid_url(url: str) -> bool:
    if not url:
        return False

    url_lower = url.lower()

    if "bing.com/aclick" in url_lower:
        return False

    if any(ext in url_lower for ext in BLOCKED_EXTENSIONS):
        return False

    if any(domain in url_lower for domain in BLOCKED_DOMAINS):
        return False

    return True


# ======================================================
# DOMAIN
# ======================================================
def _get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc
    except:
        return ""


# ======================================================
# ARGUMENT SIGNAL
# ======================================================
ARGUMENT_KEYWORDS = [
    "however","but","although","on the other hand",
    "benefit","advantage","improve","increase",
    "risk","problem","challenge","concern",
    "argue","claim","impact","affect",
    "replace","automation","jobs","employment"
]

def _arg_score(text: str) -> int:
    t = text.lower()
    return sum(1 for w in ARGUMENT_KEYWORDS if w in t)


# ======================================================
# RELEVANCE (RELAXED 🔥)
# ======================================================
def _relevance_score(claim: str, text: str) -> int:
    c = set(re.findall(r"\w+", claim.lower()))
    t = set(re.findall(r"\w+", text.lower()))
    return len(c.intersection(t))


# ======================================================
# BAD FILTER
# ======================================================
BAD_PATTERNS = [
    "sign up","learn more","advertisement",
    "click here","free trial","apply now"
]

def _is_bad_content(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in BAD_PATTERNS)


# ======================================================
# FETCH PAGE
# ======================================================
def _fetch_page(url: str):
    for _ in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.text
        except:
            time.sleep(1)
    return None


# ======================================================
# SMART MERGING 🔥
# ======================================================
def _merge_sentences(sentences, max_len=450):
    chunks = []
    current = ""

    for s in sentences:

        if len(current) == 0:
            current = s
            continue

        # merge only if related (keyword overlap)
        overlap = len(set(s.split()) & set(current.split()))

        if overlap > 2 and len(current) + len(s) < max_len:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s

    if current:
        chunks.append(current.strip())

    return chunks


# ======================================================
# CHUNK SCORING 🔥🔥🔥
# ======================================================
def _score_chunk(claim, text):
    rel = _relevance_score(claim, text)
    arg = _arg_score(text)
    length_bonus = min(len(text) / 200, 1)

    return rel * 0.5 + arg * 0.4 + length_bonus


# ======================================================
# EXTRACTION (IMPROVED)
# ======================================================
def _extract_chunks(url: str, claim: str, max_chunks: int = 6):

    html = _fetch_page(url)
    if not html:
        return []

    try:
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script","style","noscript"]):
            tag.decompose()

        paragraphs = soup.find_all("p")

        sentences = []

        for p in paragraphs:

            text = _clean_text(p.get_text())

            if len(text) < 80:
                continue

            if _is_bad_content(text):
                continue

            # relaxed filtering 🔥
            if _relevance_score(claim, text) < 1 and _arg_score(text) == 0:
                continue

            sents = re.split(r"(?<=[.!?])\s+", text)

            for s in sents:
                s = s.strip()

                if len(s) < 50:
                    continue

                sentences.append(s)

        # 🔥 merge intelligently
        merged = _merge_sentences(sentences)

        # 🔥 score + sort
        scored = [(c, _score_chunk(claim, c)) for c in merged]
        scored.sort(key=lambda x: x[1], reverse=True)

        # 🔥 deduplicate
        seen = set()
        final = []

        for c, _ in scored:
            key = c[:120]
            if key in seen:
                continue
            seen.add(key)
            final.append(c)

            if len(final) >= max_chunks:
                break

        return final

    except:
        return []


# ======================================================
# SEARCH
# ======================================================
def _search_web(query: str, max_results: int = 8):

    results = []

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title"),
                    "url": r.get("href")
                })
    except:
        pass

    return results


# ======================================================
# MAIN MODULE 4
# ======================================================
def retrieve_evidence_chunks(claims: list[dict]) -> list[dict]:

    enriched_results = []

    for item in claims:

        claim_text = item.get("simplified_claim") or item.get("claim", "")
        label = item.get("label", "")

        evidence_chunks = []
        seen_urls = set()

        if label == "debatable":

            queries = [
                f"{claim_text} pros cons",
                f"{claim_text} arguments for against",
                f"{claim_text} impact jobs",
                f"{claim_text} debate"
            ]

            search_results = []

            for q in queries:
                search_results.extend(_search_web(q, max_results=5))

            website_count = 0

            for result in search_results:

                if website_count >= MAX_WEBSITES:
                    break

                url = result.get("url")

                if not _is_valid_url(url):
                    continue

                if url in seen_urls:
                    continue

                seen_urls.add(url)
                website_count += 1

                chunks = _extract_chunks(url, claim_text)

                for c in chunks:
                    evidence_chunks.append({
                        "source": result.get("title"),
                        "url": url,
                        "content": c
                    })

                time.sleep(random.uniform(0.5, 1.0))

        enriched_results.append({
            "claim_id": item.get("claim_id"),
            "claim": claim_text,
            "label": label,
            "evidence_chunks": evidence_chunks
        })

    return enriched_results