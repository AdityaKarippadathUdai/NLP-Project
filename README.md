#  🧠 Debate-Based Claim Analysis System 

A multi-stage NLP pipeline that:

Extracts factual claims from input text

Simplifies claims using LLM-based rewriting

Enriches entities with Wikipedia definitions

Classifies debatability using a hybrid AI + rule-based architecture

Provides an interactive Gradio UI

## 🚀 Project Overview

This project implements a claim-first NLP pipeline architecture designed for structured argument analysis.

Unlike simple text classifiers, this system:

Breaks a paragraph into atomic factual claims

Simplifies each claim independently

Detects whether each claim is debatable

Uses multiple reasoning layers for robustness

It is built for research, experimentation, and future expansion into argument mining or balanced debate systems.

## 🏗️ Architecture
```
Input Paragraph
      │
      ▼
Module 1: Claim Extraction (Flan-T5)
      │
      ▼
Module 2: Claim Simplification + Wikipedia Enrichment
      │
      ▼
Module 3: Debatability Classification
      │
      ├── Layer 1: Authoritative factual override
      ├── Layer 2: Gemini 2.5 Flash (semantic reasoning)
      ├── Layer 3: Rule-based fallback
      └── Layer 4: HuggingFace zero-shot fallback
      │
      ▼
Module 4: Evidence Retrieval
        │
        ├── Web Search (DDGS)
        ├── Scrape HTML (requests + BeautifulSoup)
        ├── Remove boilerplate
        ├── Extract meaningful paragraphs
        ├── Deduplicate URLs
        └── Return raw evidence chunks
        │
        ▼
Module 5 (Future): Stance Classification
Gradio UI Output
```
## 📦 Modules Implemented
### 🔹 Module 1 – Claim Extraction

File: module1_claim_extraction.py

Uses google/flan-t5-small

Splits paragraph into sentences (NLTK)

Extracts factual claims from each sentence

Applies:

Numeric consistency checks

Duplicate filtering

Minimum content validation

Output format:
```
[
  {
    "claim_id": 1,
    "claim": "India's GDP grew by 7.2% in 2022."
  }
]
```
### 🔹 Module 2 – Claim Simplification + Entity Enrichment

File: module2_claim_simplification.py

Features:

Rewrites claims using Flan-T5

Detects named entities via SpaCy

Fetches short Wikipedia summaries

Injects entity definitions inline

Example:
```
Original:
Barack Obama served as the 44th President of the United States.

Simplified:
Barack Obama (44th President of the United States from 2009–2017) served as the 44th President of the United States.
```
### 🔹 Module 3 – Debatability Classification

File: module3_debatability_detection.py

Hybrid layered architecture:

#### 🥇 Layer 1 – Authoritative Override

Automatically marks as non-debatable if:

Contains verified numerical data + year

Mentions official institutional sources

#### 🥈 Layer 2 – Gemini 2.5 Flash (Primary AI Layer)

Uses:

google-genai SDK
Model: gemini-2.5-flash


Determines if:

The claim is open to reasonable disagreement

Or purely factual

#### 🥉 Layer 3 – Rule-Based Fallback

Detects:

Attribution markers (experts argue, critics say)

Modality (could, might, expected to)

#### 🏁 Layer 4 – HuggingFace Zero-Shot Fallback

Model:

typeform/distilbert-base-uncased-mnli


Used only if previous layers fail.

### 🔹 Module 4 – Evidence Retrieval (Web Search + Scraping)

File: module4_webscraping.py

#### 🎯 Purpose

Module 4 performs pure evidence retrieval for debatable claims.

It does NOT:

Classify stance (that is Module 5’s job)

Generate arguments

Summarize content

It ONLY:

Searches the web

Scrapes relevant article content

Cleans and filters noise

Deduplicates sources

Returns structured evidence chunks

This separation keeps the pipeline modular and scalable.

#### 🌐 Retrieval Strategy

For each debatable claim:

🔎 Search 5 pro-leaning results

🔎 Search 5 opposing-leaning results

Scrape paragraphs from all 10 sources

Return them together (not separated)

⚠️ Module 4 does NOT label them as pro/anti.

### 🛠 Technologies Used

ddgs (DuckDuckGo search API)

requests

BeautifulSoup

Custom boilerplate filtering

URL validation (PDF & academic site blocking)

### 🧹 Cleaning Logic

Module 4 removes:

Script/style tags

Boilerplate phrases (subscribe, privacy policy, etc.)

Very short paragraphs (< 100 characters)

Duplicate URLs

Blocked domains (ResearchGate, ScienceDirect)

PDF links

This ensures high-quality evidence chunks for downstream stance modeling.

### ⚙️ Design Philosophy

Module 4 is intentionally:

🔹 Retrieval-only

🔹 Model-agnostic

🔹 Gemini-free

🔹 No stance bias

🔹 RAG-ready

This ensures:

Transparency

Scalability

Clean separation of concerns

Compatibility with local LLMs (Mistral, LLaMA, etc.)
### 📤 Output Format
```
[
  {
    "claim_id": 1,
    "claim": "Artificial intelligence will replace most human jobs within the next 20 years",
    "label": "debatable",
    "evidence_chunks": [
        {
            "source": "CNN – AI replace human workers",
            "url": "https://...",
            "content": "Paragraph text..."
        },
        {
            "source": "Brookings – AI and inequality",
            "url": "https://...",
            "content": "Paragraph text..."
        }
    ]
  }
]
```

### 🖥️ User Interface

File: interface.py

Built using:

Gradio Blocks


Displays:

Extracted Claims

Simplified Claims

Debatability Classification


Launch:
```
python interface.py
```
## 🔧 Installation Guide
#### 1️⃣ Create Virtual Environment
```
python -m venv gpuenv
```

Activate:

Windows (CMD):
```
gpuenv\Scripts\activate
```
#### 2️⃣ Install Dependencies
```
pip install torch transformers nltk spacy wikipedia-api requests gradio google-genai
python -m spacy download en_core_web_sm
```
#### 3️⃣ Set Gemini API Key

In CMD:
```
set GEMINI_API_KEY=YOUR_KEY_HERE
```

Verify:
```
echo %GEMINI_API_KEY%
```
#### 4️⃣ Verify Gemini API

Create verify_gemini.py:
```
from google import genai
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Reply with only hello"
)

print(response.text)

```
Run:
```
python verify_gemini.py
```

Expected output:
```
hello
```