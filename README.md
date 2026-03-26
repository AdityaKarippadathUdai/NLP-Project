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
[Module 1: Claim Extraction (Flan-T5)]
│
├── Input: Raw paragraph
├── Process:
│ • Prompt-based generation using FLAN-T5
│ • Splits paragraph into atomic, meaningful claims
│ • Removes redundant / overlapping statements
├── Output: List of structured claims
│
▼
[Module 2: Claim Simplification + Wikipedia Enrichment]
│
├── Input: Extracted claims
├── Process:
│ • Simplification using FLAN-T5 (text-to-text transformation)
│ • Named Entity Recognition using SpaCy
│ • Entity linking via Wikipedia API
│ • Fetch short definitions for detected entities
│ • Append contextual explanations inline
├── Output: Simplified + enriched claims
│
▼
[Module 3: Debatability Classification (Multi-Layer Hybrid)]
│
├── Input: Simplified claims
├── Layer 1: Authoritative Factual Override
│ • Detects universally accepted facts
│ • Uses predefined rules / keywords / patterns
│ • Immediately marks as NON-DEBATABLE
│
├── Layer 2: Gemini 2.5 Flash (Semantic Reasoning)
│ • Performs deep contextual understanding
│ • Identifies subjectivity, opinion, or ambiguity
│ • Returns reasoning-based label
│
├── Layer 3: Rule-Based Fallback
│ • Heuristic checks (modal verbs, opinion words)
│ • Ensures robustness if LLM response is unclear
│
├── Layer 4: HuggingFace Zero-Shot (DistilBERT)
│ • Uses zero-shot classification (no fine-tuning)
│ • Labels: "debatable" vs "non-debatable"
│ • Acts as final fallback layer
│
├── Output:
│ • Debatable claims → forwarded to next module
│ • Non-debatable claims → filtered or flagged as factual
│
▼
[Module 4: Evidence Retrieval]
│
├── Input: Debatable claims
├── Process:
│ • Query generation from claims
│ • Web search using DuckDuckGo Search (DDGS)
│ • Fetch top-k URLs
│ • Scrape HTML using requests + BeautifulSoup
│ • Remove boilerplate (scripts, ads, navigation elements)
│ • Extract meaningful paragraphs
│ • Chunk text into evidence snippets
│ • Deduplicate URLs and repeated content
├── Output: Raw evidence chunks per claim
│
▼
[Module 5: Evidence Filtering & Refinement]
│
├── Input: Raw evidence chunks
├── Process:
│ • Remove irrelevant or noisy text
│ • Rank evidence based on relevance to claim
│ • Filter contradictory / low-quality sources
│ • Keep top-k high-quality evidence snippets
│ • Ensure diversity of sources (optional)
├── Output: Clean, relevant, high-quality evidence
│
▼
[Module 6: Response Generation (Balanced Output)]
│
├── Input:
│ • Simplified claims
│ • Debatability labels
│ • Filtered evidence
├── Process:
│ • Generate supporting explanation using evidence
│ • Generate counterarguments for debatable claims
│ • Combine claim + explanation + counterpoints
│ • Maintain neutral, balanced tone
│ • Structure output for readability
├── Output: Final explainable, balanced response
│
▼
[Gradio UI]
│
├── Displays:
│ • Simplified claims
│ • Debatability status
│ • Supporting evidence
│ • Counterarguments
│
└── Interactive user interface
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

### 🔹 Module 5 – Evidence Filtering & Ranking (Semantic + Heuristic)

File: module5_evidence_classification.py

#### 🎯 Purpose

Module 5 processes raw evidence from Module 4 and selects the most relevant, high-quality evidence for each debatable claim.

It does NOT:

Classify stance (pro / anti)  

Generate responses  

Perform web scraping  

It ONLY:

Filters noisy or weak evidence  

Ranks evidence using hybrid scoring  

Uses semantic similarity + heuristics  

Returns top-k high-quality evidence  

This improves downstream generation quality and reduces hallucination.

---

#### 🧠 Processing Strategy

For each debatable claim:

🔹 Input: Raw evidence chunks from Module 4  

🔹 Step 1: Cleaning  
- Normalize whitespace  
- Trim text  

🔹 Step 2: Hard Filtering  
- Remove short text (< 80 chars)  
- Remove generic content (e.g., “this article”, “click here”)  
- Remove weak statements (e.g., “experts say” without numbers)  

🔹 Step 3: Relevance Scoring  
- Token overlap between claim and evidence  
- Ensures topical alignment  

🔹 Step 4: Argument Strength Scoring  
- Detects domain-relevant keywords (e.g., jobs, economy, automation)  
- Filters non-informative text  

🔹 Step 5: Deduplication  
- Uses MD5 hashing  
- Removes repeated or near-identical chunks  

🔹 Step 6: Source Control  
- Limits max chunks per source (prevents bias)  

🔹 Step 7: Semantic Similarity (MiniLM 🔥)  
- Uses `sentence-transformers/all-MiniLM-L6-v2`  
- Mean pooling over transformer embeddings  
- Computes cosine similarity with claim  

🔹 Step 8: Final Scoring Formula  

Score is computed as:

- Relevance score × 1.5  
- Argument score × 2.0  
- Semantic similarity × 3.0  

Higher weight is given to semantic similarity.

🔹 Step 9: Ranking  
- Sort all candidates by final score  
- Select top-k results  

🔹 Step 10: Fallback Mechanism  
- If no candidates survive filtering:  
  • Select top raw chunks  
  • Assign minimal default score  

---

### 🛠 Technologies Used

PyTorch (GPU/CPU support)  

HuggingFace Transformers  

MiniLM (sentence-transformers/all-MiniLM-L6-v2)  

NumPy  

Regex + Hashing  

---

### 🧹 Filtering Logic

Module 5 removes:

Short or incomplete text  

Generic boilerplate phrases  

Weak claims without supporting detail  

Duplicate content  

Low relevance evidence  

Over-represented sources  

---

### ⚙️ Design Philosophy

Module 5 is designed to be:

🔹 Retrieval refinement focused  

🔹 Fully local (no API calls / no Gemini)  

🔹 Efficient (MiniLM lightweight model)  

🔹 Hybrid (rules + semantic embeddings)  

🔹 Scalable and model-independent  

This ensures:

High precision evidence  

Reduced noise for generation  

Better factual grounding  

Compatibility with local LLM pipelines  

---

### 📤 Output Format
```
[
  {
    "claim_id": 1,
    "claim": "Artificial intelligence will replace most human jobs within the next 20 years",
    "filtered_evidence": [
      {
      "source": "CNN – AI replace human workers",
      "url": "https://...",
      "content": "Relevant paragraph text...",
      "score": 8.73,
      "semantic": 0.81
      }
    ]
  }
]
```

### 🔹 Module 6 – Debate Response Generation (LLaMA + Evidence Grounding)

File: module6_llm_reasoning.py

#### 🎯 Purpose

Module 6 generates the final balanced debate output using filtered evidence from Module 5.

It does NOT:

Perform retrieval or filtering  

Classify debatability  

Modify claims  

It ONLY:

Generates structured debate responses  

Uses evidence-grounded reasoning  

Produces pro vs against arguments  

Streams output in real-time  

This is the final reasoning and presentation layer of the pipeline.

---

#### 🧠 Processing Strategy

For each claim:

🔹 Input:  
- Claim  
- Filtered evidence (from Module 5)  

---

🔹 Step 1: Cleaning for LLM  
- Removes URLs, citations, brackets  
- Normalizes whitespace  
- Prevents noisy tokens entering prompt  

---

🔹 Step 2: Token-Aware Evidence Selection  
- Estimates token usage (`len(text)//4`)  
- Ensures total input stays within context window  
- Selects top evidence chunks safely  
- Truncates long text (~280 chars per chunk)  

---

🔹 Step 3: Prompt Construction (Strict 🔒)  

The model is guided using a highly controlled prompt:

- Exactly 3–5 PRO points  
- Exactly 3–5 AGAINST points  
- Each point ≤ 2 lines  
- No citations ([1], etc.)  
- No hallucinated sections  
- Must use evidence  

This ensures structured and reliable output.

---

🔹 Step 4: LLaMA Inference (Local 🔥)  
- Model: Meta-Llama-3-8B-Instruct (GGUF quantized)  
- Runs via `llama.cpp`  
- GPU + CPU hybrid execution  
- Config:
  • Temperature = 0.6  
  • Top-p = 0.9  
  • Repeat penalty = 1.15  

---

🔹 Step 5: Streaming Output  
- Generates tokens incrementally  
- Yields partial responses in real-time  
- Improves UI responsiveness (Gradio streaming)  

---

🔹 Step 6: Output Cleaning & Safety Fix  
- Removes unwanted sections (e.g., “IMPLICATIONS”)  
- Ensures conclusion exists  
- Adds fallback conclusion if missing  

---

🔹 Step 7: Structured Parsing  
- Extracts:
  • PRO arguments  
  • AGAINST arguments  
  • CONCLUSION  

- Uses rule-based parsing from generated text  

---

🔹 Step 8: Fallback Handling  
- If parsing fails:
  • Returns raw truncated output  
  • Prevents pipeline crash  

---

### 🛠 Technologies Used

llama.cpp (via llama-cpp-python)  

Meta LLaMA 3 8B Instruct (quantized GGUF)  

Regex-based cleaning & parsing  

Streaming generation  

---

### ⚙️ Design Philosophy

Module 6 is designed to be:

🔹 Fully local (no API dependency)  

🔹 Evidence-grounded (reduces hallucination)  

🔹 Streaming-first (real-time UX)  

🔹 Strictly structured (controlled output format)  

🔹 Token-aware (efficient context usage)  

This ensures:

Reliable argument generation  

Balanced reasoning  

Low latency with large models  

Compatibility with offline systems  

---

### 📤 Output Format
```
{
  "type": "final",
  "claim_id": 1,
  "claim": "Artificial intelligence will replace most human jobs within the next 20 years",
  "pro": [
    "AI can automate repetitive tasks, increasing efficiency",
    "Companies can reduce labor costs significantly"
  ],
  "against": [
    "AI cannot replace creative and emotional human roles",
    "New jobs will emerge alongside automation"
  ],
    "conclusion": "AI will significantly transform jobs, but complete replacement is unlikely."
}
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