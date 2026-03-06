# Automated Resume Screening

> Citation-Grounded Evaluation System for Fair & Transparent Hiring

Solves the "ATS Black Hole" problem — where qualified candidates get rejected due to terminology mismatches — using a **citation-grounded evaluation approach** that automates the screening PROCESS, not the DECISION.

## Key Philosophy

**LLMs are used ONLY where human-like judgment is genuinely required** (interpreting unstructured text, semantic equivalence, natural language synthesis). All other steps are **deterministic, reproducible, and verifiable**.

## Architecture: 6-Node Citation-Grounded Pipeline

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Node 0     │──▶│   Node 1     │──▶│   Node 2     │──▶│   Node 3     │──▶│   Node 4     │──▶│   Node 5     │
│  ResuShield  │   │  Criteria    │   │  RAG-Based   │   │  Citation    │   │  Summary     │   │   Report     │
│  Security    │   │  Generation  │   │  Evaluation  │   │  Validator   │   │  Generator   │   │  Generator   │
│  + OCR       │   │              │   │              │   │              │   │              │   │              │
│ Deterministic│   │  Cloud LLM   │   │  Cloud LLM   │   │ Deterministic│   │  Cloud LLM   │   │ Deterministic│
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

### LLM vs Deterministic Split

| Node | Task | Method | Why |
|------|------|--------|-----|
| 0 | Security scan + OCR | Deterministic | Text comparison, pattern matching |
| 1 | Extract criteria from JD | Cloud LLM (Groq) | Requires understanding unstructured text |
| 2 | Evaluate resume evidence | Cloud LLM (Groq) | Requires semantic reasoning with citations |
| 3 | Validate citations | Sentence-Transformer Embeddings | Deterministic similarity check |
| 4 | Generate summary | Cloud LLM (Groq) | Natural language synthesis with citations |
| 5 | Rank candidates | Python sort | Pure math, no LLM |

## Quick Start

### Prerequisites

- Python 3.10+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (optional, for ResuShield OCR)
- [Poppler](https://poppler.freedesktop.org/) (optional, for pdf2image)
- Groq API key(s) — free tier available at [console.groq.com](https://console.groq.com)

### Installation

```bash
# Clone repository
git clone https://github.com/Sampath-2211/Automated-Resume-Screening.git
cd automated-resume-screening

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install system dependencies (macOS)
brew install tesseract poppler

# Install system dependencies (Ubuntu/Debian)
# sudo apt install tesseract-ocr poppler-utils

# Setup environment
cp .env.example .env
# Edit .env with your Groq API key(s)

# Run the app
streamlit run app.py
```

### Environment Setup

Create a `.env` file:

```env
# Required — comma-separated for rotation (recommended: 3+ keys)
GROQ_API_KEYS=gsk_key1,gsk_key2,gsk_key3

# Model (all nodes use Groq cloud)
QUESTION_GEN_MODEL=llama-3.3-70b-versatile
EVAL_MODEL=llama-3.3-70b-versatile
RESPONSE_GEN_MODEL=llama-3.3-70b-versatile

# Thresholds
CITATION_THRESHOLD=0.55
FALLBACK_THRESHOLD=0.45
EMBEDDING_MODEL=all-MiniLM-L6-v2

# RAG settings
RAG_CHUNK_SIZE=100
RAG_CHUNK_OVERLAP=20
RAG_TOP_K=3
```

## Key Innovations

### 1. Citation-Grounded Evaluation
Every score ≥ 3 **must** include a `<cite>exact quote</cite>` from the resume. No citation = score reduced. This forces the LLM to ground every claim in verifiable evidence.

### 2. Anti-Hallucination Validation (Node 3)
Citations are verified against the resume using a multi-strategy approach:
1. Normalized exact match
2. Fuzzy sliding window
3. Keyword overlap
4. Semantic similarity (sentence-transformers)

Invalid citations trigger a semantic fallback search; if no alternative evidence exists, the score is reduced.

### 3. RAG-Based Evaluation (Node 2)
The full resume is **never** sent to the LLM. Instead, it's chunked into 100-word segments with 20-word overlap, and only the top-3 most relevant chunks per criterion are sent. This prevents hallucination from context overload.

### 4. Semantic Interpretation Rules
The LLM is explicitly instructed to use semantic understanding rather than exact keyword matching — e.g., B.Tech/B.E./B.S. in CS or CSE all qualify as "Bachelor's degree in Computer Science."

### 5. ResuShield Security (Node 0)
- **OCR extraction:** Only human-visible text enters the pipeline (defeats hidden white-text attacks)
- **Visual-semantic detection:** Compares raw PDF text vs OCR text to detect invisible keyword stuffing
- **Prompt injection shield:** Pattern-based detection of LLM manipulation attempts

### 6. Click-to-Verify Citations
Every citation in the UI is clickable → opens a modal showing the PDF page with a green highlight box at the citation's bounding box, plus the similarity score and verification status.

### 7. API Key Rotation
Thread-safe round-robin rotation across multiple Groq API keys with automatic per-key cooldown on 429 rate-limit responses.

## UI: 5 Tabs

| Tab | Purpose |
|-----|---------|
| **Rankings** | Candidate scores, rankings, per-criterion breakdown with clickable citations |
| **Validation Log** | All citations checked with valid/invalid status and similarity % |
| **Comparison Mode** | Before (raw LLM) vs After (citation-grounded) scores — shows hallucination impact |
| **Pipeline Log** | Node-by-node execution times, failed candidates, OCR warnings |
| **How It Works** | 6-node pipeline diagram, LLM vs deterministic breakdown |

## Project Structure

```
automated-resume-screening/
├── app.py                  # Streamlit Dashboard (5 tabs)
├── core.py                 # 6-Node Pipeline Engine
├── citation_validator.py   # Citation Validation + Bounding Box
├── pdf_highlighter.py      # PDF Rendering + Green Highlights
├── summary_generator.py    # Adaptive Summary Generation
├── visual_detector.py      # ResuShield Security (OCR + Injection Shield)
├── diagnostic.py           # Pre-run diagnostic checks
├── requirements.txt        # Python Dependencies
├── .env.example            # Environment Variable Template
├── .env                    # API Keys (not in git)
└── README.md
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEYS` | — | Comma-separated Groq API keys for rotation |
| `QUESTION_GEN_MODEL` | `llama-3.3-70b-versatile` | Model for criteria generation |
| `EVAL_MODEL` | `llama-3.3-70b-versatile` | Model for resume evaluation |
| `RESPONSE_GEN_MODEL` | `llama-3.3-70b-versatile` | Model for summary generation |
| `CITATION_THRESHOLD` | `0.55` | Min similarity for valid citation |
| `FALLBACK_THRESHOLD` | `0.45` | Min similarity for semantic fallback |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `RAG_CHUNK_SIZE` | `100` | Words per RAG chunk |
| `RAG_CHUNK_OVERLAP` | `20` | Word overlap between RAG chunks |
| `RAG_TOP_K` | `3` | Chunks retrieved per criterion |
| `VALIDATION_CHUNK_SIZE` | `30` | Words per validation chunk |
| `LLM_TIMEOUT` | `60` | Seconds before LLM timeout |

## How It Addresses Common ATS Problems

**"System is just giving resumes to an LLM"**
— LLM receives only RAG-retrieved chunks (top 3 × 100 words), NOT the full resume. 3 of 6 pipeline nodes are fully deterministic. Ranking is pure Python sort — no LLM in the final decision.

**"Can't verify what the LLM claims"**
— Every score has clickable citation verification with PDF highlight. Validation log shows all similarity checks with exact scores. Deterministic nodes produce identical results every run.

**"Keyword matching rejects qualified candidates"**
— Semantic interpretation rules ensure that equivalent credentials (e.g., B.Tech CSE = Bachelor's in Computer Science) are recognized as matches, not rejected for terminology differences.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM (all nodes) | Groq Cloud — `llama-3.3-70b-versatile` |
| Embeddings / Citation Validation | `sentence-transformers` (all-MiniLM-L6-v2) |
| PDF Processing | PyMuPDF (primary), Tesseract OCR (fallback) |
| UI | Streamlit |
| Environment | Python 3.10+, any OS |
