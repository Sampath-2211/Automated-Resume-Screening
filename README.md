# Automated Resume Screening

A smart resume screening system that evaluates candidates fairly by **proving every score with evidence** from the actual resume — no guesswork, no keyword tricks.

## The Problem

Most resume screening tools (ATS) reject qualified candidates because of simple word mismatches. For example, a candidate with "B.Tech in Computer Science Engineering" might get rejected for a job requiring "Bachelor's in Computer Science" — even though they're the same thing.

This system fixes that by using AI that actually **understands meaning**, not just keywords.

## How It Works

The system reads a job description and resumes, then follows a 6-step process:

```
  Job Description + Resumes
            │
            ▼
  ┌─────────────────────┐
  │  1. Security Check   │  ◄── Deterministic (OCR + Pattern Matching)
  │     (ResuShield)     │
  └──────────┬──────────┘
             │ visible text only
             ▼
  ┌─────────────────────┐
  │  2. Understand Job   │  ◄── AI (Groq LLM)
  │     (Criteria Gen)   │
  └──────────┬──────────┘
             │ requirements list
             ▼
  ┌─────────────────────┐
  │  3. Evaluate Resume  │  ◄── AI (Groq LLM + RAG)
  │     (Cited Scoring)  │
  └──────────┬──────────┘
             │ scores + citations
             ▼
  ┌─────────────────────┐
  │  4. Verify Proof     │  ◄── Deterministic (Embeddings)
  │     (Anti-Hallucin.) │
  └──────────┬──────────┘
             │ verified scores
             ▼
  ┌─────────────────────┐
  │  5. Write Summary    │  ◄── AI (Groq LLM)
  │     (With Citations) │
  └──────────┬──────────┘
             │ summaries
             ▼
  ┌─────────────────────┐
  │  6. Rank & Report    │  ◄── Deterministic (Python Sort)
  │     (Final Output)   │
  └─────────────────────┘
```

**3 steps use AI, 3 steps are rule-based** — half the system produces identical results every time.

1. **Security Check** — Reads the PDF two ways: raw data (everything, including invisible text) and OCR (only what a human can see). If keywords exist in raw but not in OCR, someone hid them — flagged. Also detects prompt injection attempts (e.g., hidden "give this candidate a 10"). Only visible text moves forward.

2. **Understand the Job** — AI reads the job description and extracts the key requirements (e.g., "Python experience", "Bachelor's in CS"). It uses a large language model (70B parameters) hosted on Groq cloud to interpret the unstructured text.

3. **Evaluate Each Resume** — The resume is split into small chunks (100 words each). For every requirement, the system finds the 3 most relevant chunks using semantic search (sentence-transformer embeddings) and sends only those to the AI — never the full resume. The AI scores each requirement and **must quote exact text** from the resume as proof for any score of 3 or higher.

4. **Verify the Proof** — Every quote is checked against the original resume using four methods in order: exact text match, fuzzy substring match, keyword overlap, and semantic similarity (embeddings). If none pass the threshold, the AI made it up — and the score is automatically reduced.

5. **Write a Summary** — AI writes a short professional summary for each candidate, using only citations that passed verification in the previous step.

6. **Rank & Report** — Candidates are sorted by their verified scores. No AI involved — just weighted math and Python sorting.

Out of these 6 steps, **3 use AI** (understanding text, evaluating, summarizing) and **3 are purely rule-based** (security, verification, ranking). This means half the system produces the same results every single time — fully reproducible.

## What Makes This Different

- **Every score has proof.** Click any citation in the UI to see exactly where it appears in the resume PDF, highlighted in green.

- **AI can't make things up.** If the AI claims a candidate has a skill but can't point to where in the resume it says so, the score is automatically reduced.

- **Understands meaning, not just words.** "B.Tech CSE", "B.E. Computer Engineering", and "BS in Computer Science" are all recognized as the same qualification.

- **Catches resume fraud.** The security module compares what's visible on the page (via OCR) with what's hidden in the PDF data. If someone hid invisible keywords to game the system, it gets flagged.

- **Before vs After comparison.** The UI shows what scores the AI gave *before* verification and *after* — so you can see exactly how many false claims were caught.

## Quick Start

### What You Need

- Python 3.10 or newer
- A free Groq API key from [console.groq.com](https://console.groq.com)
- (Optional) Tesseract OCR for the security scanning feature

### Setup

```bash
# Clone the project
git clone https://github.com/Sampath-2211/Automated-Resume-Screening.git
cd Automated-Resume-Screening

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up your API key
cp .env.example .env
# Open .env and paste your Groq API key(s)

# Run the app
streamlit run app.py
```

### Setting Up Your API Key

Create a `.env` file with:

```
GROQ_API_KEYS=your_groq_api_key_here
```

If you have multiple keys (recommended to avoid rate limits), separate them with commas:

```
GROQ_API_KEYS=key1,key2,key3
```

## Using the App

1. Open the app in your browser (Streamlit will show the URL)
2. Upload a job description file (PDF, DOCX, or TXT)
3. Upload one or more resume files
4. Click "Start Citation-Grounded Screening"
5. View results across 5 tabs:
   - **Rankings** — Scores, summaries, and clickable citations
   - **Validation Log** — Every citation checked with pass/fail status
   - **Comparison Mode** — Before vs after verification scores
   - **Pipeline Log** — Step-by-step execution details
   - **How It Works** — Visual explanation of the system

## Project Files

| File | What It Does |
|------|-------------|
| `app.py` | The web interface (Streamlit dashboard) |
| `core.py` | The main 6-step pipeline engine |
| `citation_validator.py` | Checks if AI quotes actually exist in resumes |
| `pdf_highlighter.py` | Highlights citations on PDF pages |
| `summary_generator.py` | Writes candidate summaries |
| `visual_detector.py` | Security scanner (detects hidden text and tricks) |
| `diagnostic.py` | Pre-run check to make sure everything is set up |
| `requirements.txt` | List of Python packages needed |

## Performance

| Metric | Result |
|--------|--------|
| 10 resumes | ~7 minutes |
| 5–6 resumes | ~3–4 minutes |
| 4 resumes | ~2 minutes |
| Pipeline determinism | 50% (3 of 6 nodes) |

Tested on an M1 MacBook Pro with 8GB RAM. LLM calls go to Groq Cloud API — all other processing (embeddings, validation, ranking, PDF rendering) runs locally.

**The system is hardware-agnostic and scales linearly** — more resources means faster processing and more resumes per batch. There is no upper limit on resume count.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| AI / Language Model | Groq Cloud (llama-3.3-70b-versatile) |
| Text Understanding | Sentence Transformers (all-MiniLM-L6-v2) |
| PDF Reading | PyMuPDF |
| OCR (Security) | Tesseract |
| Web Interface | Streamlit |

