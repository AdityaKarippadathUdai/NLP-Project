import requests
import re
from bs4 import BeautifulSoup
from ddgs import DDGS


# ======================================================
# Clean Text
# ======================================================
def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ======================================================
# Boilerplate Filtering
# ======================================================
BOILERPLATE_KEYWORDS = [
    "subscribe",
    "sign in",
    "privacy policy",
    "cookie policy",
    "advertisement",
    "all rights reserved",
    "newsletter",
    "terms of use",
]


def _is_boilerplate(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in BOILERPLATE_KEYWORDS)


# ======================================================
# URL Validation
# ======================================================
BLOCKED_EXTENSIONS = [".pdf"]
BLOCKED_DOMAINS = ["researchgate.net", "sciencedirect.com"]


def _is_valid_url(url: str) -> bool:
    if not url:
        return False

    url_lower = url.lower()

    if any(ext in url_lower for ext in BLOCKED_EXTENSIONS):
        return False

    if any(domain in url_lower for domain in BLOCKED_DOMAINS):
        return False

    return True


# ======================================================
# Extract Paragraph Chunks
# ======================================================
def _extract_paragraph_chunks(url: str, max_paragraphs: int = 12):

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        paragraphs = soup.find_all("p")

        chunks = []

        for p in paragraphs:
            text = _clean_text(p.get_text())

            if len(text) < 120:
                continue

            if _is_boilerplate(text):
                continue

            chunks.append(text)

            if len(chunks) >= max_paragraphs:
                break

        return chunks

    except Exception:
        return []


# ======================================================
# Web Search
# ======================================================
def _search_web(query: str, max_results: int = 5):

    results = []

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title"),
                    "url": r.get("href")
                })
    except Exception:
        pass

    return results


# ======================================================
# MODULE 4: PURE RETRIEVAL (5 PRO + 5 CON SEARCH)
# ======================================================
def retrieve_evidence_chunks(claims: list[dict]) -> list[dict]:

    enriched_results = []

    for item in claims:

        claim_text = item.get("claim", "")
        label = item.get("label", "")

        evidence_chunks = []
        seen_urls = set()

        if label == "debatable":

            # -------------------------
            # Pro-oriented queries
            # -------------------------
            pro_query = (
                claim_text +
                " benefits advantages positive impact supporting evidence research findings"
            )

            # -------------------------
            # Con-oriented queries
            # -------------------------
            con_query = (
                claim_text +
                " criticism risks negative impact opposing view counterargument concerns debate"
            )

            pro_results = _search_web(pro_query, max_results=5)
            con_results = _search_web(con_query, max_results=5)

            combined_results = pro_results + con_results

            for result in combined_results:

                url = result.get("url")

                if not _is_valid_url(url):
                    continue

                if url in seen_urls:
                    continue

                seen_urls.add(url)

                paragraphs = _extract_paragraph_chunks(url)

                for para in paragraphs:
                    evidence_chunks.append({
                        "source": result.get("title"),
                        "url": url,
                        "content": para
                    })

        enriched_results.append({
            "claim_id": item.get("claim_id"),
            "claim": claim_text,
            "label": label,
            "evidence_chunks": evidence_chunks
        })

    return enriched_results
