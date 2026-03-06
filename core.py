"""
Automated Resume Screening - Core Pipeline Engine
6-Node Citation-Grounded Architecture

All LLM calls use Groq cloud with API key rotation to avoid rate limits.

Nodes:
  0: ResuShield Pre-Screening (Deterministic + OCR)
  1: Question Generation (Groq LLM)
  2: Resume Evaluation with RAG (Groq LLM)
  3: Citation Validator (Deterministic - Embeddings)
  4: Response Generator (Groq LLM)
  5: Report Generator (Deterministic - NO LLM)

Author: Sampath Krishna Tekumalla
"""
import json
import re
import logging
import os
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Callable
from datetime import datetime
from pathlib import Path
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

# =============================================================================
# LOGGING
# =============================================================================
class CleanFormatter(logging.Formatter):
    def format(self, record):
        return record.getMessage()

def setup_logger():
    logger = logging.getLogger("ResumeScreening")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CleanFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger

logger = setup_logger()

# =============================================================================
# API KEY ROTATOR
# =============================================================================
class APIKeyRotator:
    """Thread-safe round-robin API key rotation.
    Loads comma-separated keys from GROQ_API_KEYS env var.
    Automatically skips rate-limited keys for a cooldown period."""

    def __init__(self):
        keys_str = os.getenv('GROQ_API_KEYS', '')
        self._keys = [k.strip() for k in keys_str.split(',') if k.strip()]

        # Fallback: try individual key env vars
        if not self._keys:
            single = os.getenv('QUESTION_GEN_API_KEY', os.getenv('GROQ_API_KEY', ''))
            if single:
                self._keys = [single]

        if not self._keys:
            logger.warning("No Groq API keys found! Set GROQ_API_KEYS in .env")

        self._index = 0
        self._lock = threading.Lock()
        self._cooldowns: Dict[int, float] = {}  # key_index -> cooldown_until timestamp
        logger.info(f"API Key Rotator initialized with {len(self._keys)} key(s)")

    @property
    def key_count(self) -> int:
        return len(self._keys)

    def get_key(self) -> str:
        """Get next available API key, skipping cooled-down keys."""
        if not self._keys:
            raise RuntimeError("No API keys configured. Set GROQ_API_KEYS in .env")

        with self._lock:
            now = time.time()
            # Try each key once
            for _ in range(len(self._keys)):
                idx = self._index % len(self._keys)
                self._index += 1

                # Check if this key is in cooldown
                cooldown_until = self._cooldowns.get(idx, 0)
                if now >= cooldown_until:
                    return self._keys[idx]

            # All keys in cooldown — wait for the shortest cooldown
            min_wait = min(self._cooldowns.values()) - now
            if min_wait > 0:
                logger.warning(f"  All {len(self._keys)} keys rate-limited. Waiting {min_wait:.0f}s...")
                time.sleep(min_wait + 1)

            # Return next key after waiting
            idx = self._index % len(self._keys)
            self._index += 1
            self._cooldowns.pop(idx, None)
            return self._keys[idx]

    def mark_rate_limited(self, key: str, cooldown_seconds: int = 62):
        """Mark a key as rate-limited for cooldown_seconds."""
        with self._lock:
            try:
                idx = self._keys.index(key)
                self._cooldowns[idx] = time.time() + cooldown_seconds
                remaining = sum(1 for i in range(len(self._keys))
                              if time.time() >= self._cooldowns.get(i, 0))
                logger.warning(f"  Key #{idx+1} rate-limited for {cooldown_seconds}s. "
                             f"{remaining}/{len(self._keys)} keys available.")
            except ValueError:
                pass

# Global rotator instance
_key_rotator: Optional[APIKeyRotator] = None

def get_key_rotator() -> APIKeyRotator:
    global _key_rotator
    if _key_rotator is None:
        _key_rotator = APIKeyRotator()
    return _key_rotator

# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class PipelineConfig:
    question_gen_url: str = field(default_factory=lambda: os.getenv('QUESTION_GEN_URL', 'https://api.groq.com/openai/v1'))
    question_gen_model: str = field(default_factory=lambda: os.getenv('QUESTION_GEN_MODEL', 'llama-3.3-70b-versatile'))

    eval_url: str = field(default_factory=lambda: os.getenv('EVAL_URL', 'https://api.groq.com/openai/v1'))
    eval_model: str = field(default_factory=lambda: os.getenv('EVAL_MODEL', 'llama-3.3-70b-versatile'))

    response_gen_url: str = field(default_factory=lambda: os.getenv('RESPONSE_GEN_URL', 'https://api.groq.com/openai/v1'))
    response_gen_model: str = field(default_factory=lambda: os.getenv('RESPONSE_GEN_MODEL', 'llama-3.3-70b-versatile'))

    citation_threshold: float = field(default_factory=lambda: float(os.getenv('CITATION_THRESHOLD', '0.55')))
    citation_required_score: int = field(default_factory=lambda: int(os.getenv('CITATION_REQUIRED_SCORE', '3')))
    fallback_threshold: float = field(default_factory=lambda: float(os.getenv('FALLBACK_THRESHOLD', '0.45')))
    embedding_model: str = field(default_factory=lambda: os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'))

    rag_chunk_size: int = field(default_factory=lambda: int(os.getenv('RAG_CHUNK_SIZE', '100')))
    rag_chunk_overlap: int = field(default_factory=lambda: int(os.getenv('RAG_CHUNK_OVERLAP', '20')))
    rag_top_k: int = field(default_factory=lambda: int(os.getenv('RAG_TOP_K', '3')))

    validation_chunk_size: int = field(default_factory=lambda: int(os.getenv('VALIDATION_CHUNK_SIZE', '30')))
    validation_chunk_overlap: int = field(default_factory=lambda: int(os.getenv('VALIDATION_CHUNK_OVERLAP', '10')))

    llm_timeout: int = field(default_factory=lambda: int(os.getenv('LLM_TIMEOUT', '60')))
    cloud_llm_timeout: int = field(default_factory=lambda: int(os.getenv('CLOUD_LLM_TIMEOUT', '60')))
    max_reasoning_tokens: int = field(default_factory=lambda: int(os.getenv('MAX_REASONING_TOKENS', '500')))


def load_config() -> PipelineConfig:
    return PipelineConfig()

# =============================================================================
# UNIFIED LLM CLIENT - WITH KEY ROTATION & RATE LIMIT HANDLING
# =============================================================================
class RateLimitError(Exception):
    pass

class UnifiedLLMClient:
    """LLM client with automatic API key rotation for Groq.
    On 429, marks the current key as rate-limited and rotates to next."""

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: int = 60):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.provider = self._detect_provider()
        self._static_key = api_key  # Fallback for non-rotating use
        self._rotator = get_key_rotator() if self.provider in ('groq', 'openai_compatible', 'openai') else None

    def _detect_provider(self) -> str:
        if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            return "ollama"
        elif "groq" in self.base_url:
            return "groq"
        elif "openai" in self.base_url:
            return "openai"
        else:
            return "openai_compatible"

    def _get_api_key(self) -> str:
        if self._rotator and self._rotator.key_count > 0:
            return self._rotator.get_key()
        return self._static_key

    def generate(self, prompt: str, max_tokens: int = 500, retries: int = 5) -> str:
        last_error = None
        for attempt in range(retries + 1):
            api_key = self._get_api_key()
            try:
                if self.provider == "ollama":
                    return self._generate_ollama(prompt, max_tokens)
                else:
                    return self._generate_cloud(prompt, max_tokens, api_key)
            except RateLimitError:
                # Mark this key and immediately try next key
                if self._rotator:
                    self._rotator.mark_rate_limited(api_key, cooldown_seconds=62)
                last_error = RateLimitError(f"Key rotation attempt {attempt+1}")
                # Don't sleep — just rotate to next key immediately
                continue
            except Exception as e:
                last_error = e
                if attempt < retries:
                    time.sleep(1)
        raise RuntimeError(f"LLM failed after {retries+1} attempts: {last_error}")

    def _generate_ollama(self, prompt: str, max_tokens: int) -> str:
        r = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.1, "num_predict": max_tokens, "num_ctx": 4096}},
            timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:200]}")
        return r.json().get('response', '')

    def _generate_cloud(self, prompt: str, max_tokens: int, api_key: str) -> str:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        r = requests.post(
            f"{self.base_url}/chat/completions", headers=headers,
            json={"model": self.model, "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.1, "max_tokens": max_tokens},
            timeout=self.timeout)
        if r.status_code == 429:
            raise RateLimitError(f"429 from {self.provider}")
        if r.status_code != 200:
            raise RuntimeError(f"API error {r.status_code}: {r.text[:200]}")
        return r.json()['choices'][0]['message']['content']

# =============================================================================
# RAG RETRIEVER
# =============================================================================
class RAGRetriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self._model = None
        self._model_name = model_name

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def chunk_text(self, text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks, step = [], max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            chunks.append(" ".join(words[i:i + chunk_size]))
            if i + chunk_size >= len(words):
                break
        return chunks

    def get_relevant_chunks(self, query: str, chunks: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        if not chunks:
            return []
        query_emb = self.model.encode([query])[0]
        chunk_embs = self.model.encode(chunks, show_progress_bar=False)
        scores = cosine_similarity([query_emb], chunk_embs)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(chunks[i], float(scores[i])) for i in top_indices]

# =============================================================================
# NODE 0: RESUSHIELD
# =============================================================================
class Node0_ResuShield:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def process(self, pdf_bytes: bytes, filename: str, raw_text: str) -> Dict[str, Any]:
        start = datetime.now()
        threats, trust_score, ocr_text = [], 1.0, raw_text
        ocr_used, ocr_warning = False, None
        try:
            from visual_detector import ResuShield as _RS
            shield = _RS(self.config.embedding_model)
            scan = shield.analyze(raw_text, pdf_bytes if pdf_bytes and len(pdf_bytes) > 0 else None,
                                  perform_ocr=bool(pdf_bytes and len(pdf_bytes) > 0))
            trust_score, threats = scan.overall_trust, scan.threats_detected
            if scan.ocr_text and len(scan.ocr_text) > 100:
                ocr_text, ocr_used = scan.ocr_text, True
            elif pdf_bytes and not scan.ocr_text:
                ocr_warning = "OCR returned no text. Ensure Tesseract/Poppler installed."
        except ImportError as e:
            ocr_warning = f"OCR unavailable ({e}). Using raw PDF text."
            if self._check_injection(raw_text):
                threats.append({'type': 'prompt_injection', 'severity': 'critical'})
                trust_score = 0.1
        except Exception as e:
            ocr_warning = f"ResuShield error ({e})."
            if self._check_injection(raw_text):
                threats.append({'type': 'prompt_injection', 'severity': 'critical'})
                trust_score = 0.1
        return {'node': 'ResuShield', 'status': 'completed',
                'time_ms': (datetime.now() - start).total_seconds() * 1000,
                'visible_text': ocr_text, 'is_safe': len(threats) == 0,
                'trust_score': round(trust_score, 2), 'threats': threats,
                'ocr_used': ocr_used, 'ocr_warning': ocr_warning}

    @staticmethod
    def _check_injection(text: str) -> bool:
        patterns = [r'ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|rules?)',
                    r'(give|assign|set)\s+(me|this|candidate)\s+(a\s+)?(score|rating)\s+\d+',
                    r'you\s+are\s+(now|actually)\s+(a|an)']
        return any(re.search(p, text.lower()) for p in patterns)

# =============================================================================
# NODE 1: QUESTION GENERATION
# =============================================================================
class Node1_QuestionGeneration:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.llm = UnifiedLLMClient(config.question_gen_url, config.question_gen_model, timeout=config.cloud_llm_timeout)

    def process(self, job_description: str, num_criteria: int = 5) -> Dict[str, Any]:
        start = datetime.now()
        prompt = f"""Analyze this JOB DESCRIPTION and extract exactly {num_criteria} evaluation criteria.
Use ONLY requirements explicitly stated in the job description.

Job Description:
---
{job_description[:4000]}
---

Extract {num_criteria} specific skills/requirements. Order by importance (first 3 are CRITICAL).
Return ONLY valid JSON:
{{"job_title": "title from JD", "criteria": ["criterion 1", "criterion 2", ...]}}
JSON:"""
        response = self.llm.generate(prompt, max_tokens=500)
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if not match:
            raise RuntimeError("Node 1 failed: No JSON in LLM response")
        data = json.loads(match.group())
        criteria_list = data.get('criteria', [])[:num_criteria]
        job_title = data.get('job_title', 'Position')
        criteria = [{"id": i+1, "criterion": c, "weight": 5 if i < 3 else 3, "critical": i < 3}
                   for i, c in enumerate(criteria_list)]
        return {'node': 'QuestionGeneration', 'status': 'completed',
                'time_ms': (datetime.now() - start).total_seconds() * 1000,
                'job_title': job_title, 'criteria': criteria, 'criteria_count': len(criteria)}

# =============================================================================
# NODE 2: RESUME EVALUATION WITH RAG (70B model — proper citations)
# =============================================================================
class Node2_ResumeEvaluation:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.llm = UnifiedLLMClient(config.eval_url, config.eval_model, timeout=config.llm_timeout)
        self.rag = RAGRetriever(config.embedding_model)

    def process(self, candidate_name: str, resume_text: str, criteria: List[Dict]) -> Dict[str, Any]:
        start = datetime.now()
        chunks = self.rag.chunk_text(resume_text, self.config.rag_chunk_size, self.config.rag_chunk_overlap)
        scores = []
        for idx, crit in enumerate(criteria):
            criterion = crit['criterion']
            relevant = self.rag.get_relevant_chunks(criterion, chunks, self.config.rag_top_k)
            chunks_text = "\n---\n".join([c[0] for c in relevant])
            logger.info(f"  Evaluating: {criterion[:50]}...")
            # Small delay between calls to stay under per-minute limits
            if idx > 0:
                time.sleep(1.5)
            result = self._evaluate_criterion(criterion, chunks_text)
            scores.append({
                'criterion_id': crit['id'], 'criterion': criterion,
                'weight': crit.get('weight', 3), 'critical': crit.get('critical', False),
                'raw_score': result['score'], 'naive_score': result['score'],
                'reasoning': result['reasoning'], 'citations': result['citations'],
                'chunks_used': [c[0][:100] for c in relevant]})
            logger.info(f"    Score={result['score']}, Citations={len(result['citations'])}")
        return {'node': 'ResumeEvaluation', 'status': 'completed',
                'time_ms': (datetime.now() - start).total_seconds() * 1000,
                'candidate_name': candidate_name, 'scores': scores}

    def _evaluate_criterion(self, criterion: str, chunks_text: str) -> Dict[str, Any]:
        prompt = f"""You are evaluating a resume for this criterion: "{criterion}"

Resume excerpts (retrieved via semantic search):
---
{chunks_text}
---

Score this criterion from 0-5:
5 = Expert level (years of professional experience directly matching this criterion)
4 = Strong (project or job experience clearly relevant to this criterion)
3 = Moderate (meaningful project, coursework, or education matching this criterion)
2 = Basic (certification, listed skill, or indirect relevance)
1 = Tangential mention only
0 = Truly no relevant evidence in the excerpts

IMPORTANT INTERPRETATION RULES:
- Use SEMANTIC understanding, not exact keyword matching
- "Bachelor's degree in Computer Science" matches B.Tech/B.E./B.S. in CS, CSE, Computer Engineering, etc.
- "Python" matches if Python is listed in skills, used in projects, or mentioned in coursework
- "LangChain" matches if LangChain is explicitly mentioned anywhere in the excerpts
- A skill listed in a "Skills" section counts as score 2 minimum
- A degree in a closely related field (e.g., "Computer Science Engineering" for "Computer Science") is a MATCH
- If the excerpts mention other unrelated skills but NOT this criterion at all, score 0

CITATION RULES:
- If score >= 3, include ONE short <cite>exact verbatim quote from the excerpts</cite>
- The citation must be copied EXACTLY from the text above — do NOT paraphrase or summarize
- Keep citations under 15 words
- Do NOT cite your own analysis — only quote the resume text

Respond in JSON:
{{"score": <0-5>, "reasoning": "Your explanation. <cite>exact verbatim quote</cite> if score >= 3"}}

JSON:"""
        try:
            response = self.llm.generate(prompt, max_tokens=self.config.max_reasoning_tokens)
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                score = min(5, max(0, int(data.get('score', 0))))
                reasoning = data.get('reasoning', '')
                citations = [c.strip() for c in re.findall(r'<cite>(.*?)</cite>', reasoning, re.DOTALL | re.IGNORECASE) if c.strip()]
                # Trim long citations
                clean = []
                for c in citations:
                    words = c.split()
                    clean.append(' '.join(words[:20]) if len(words) > 20 else c)
                return {'score': score, 'reasoning': reasoning, 'citations': clean}
        except json.JSONDecodeError:
            sm = re.search(r'"score"\s*:\s*(\d)', response or '')
            if sm:
                return {'score': int(sm.group(1)), 'reasoning': response,
                        'citations': re.findall(r'<cite>(.*?)</cite>', response, re.DOTALL | re.IGNORECASE)}
        except Exception as e:
            logger.warning(f"    Evaluation error: {e}")
        return {'score': 0, 'reasoning': 'Evaluation failed', 'citations': []}

# =============================================================================
# NODE 3: CITATION VALIDATOR (DETERMINISTIC)
# =============================================================================
class Node3_CitationValidator:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.config.embedding_model)
        return self._model

    def process(self, evaluation_result: Dict, resume_text: str, pdf_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        start = datetime.now()
        candidate_name = evaluation_result.get('candidate_name', 'Candidate')
        scores = evaluation_result.get('scores', [])
        val_chunks = self._chunk_for_validation(resume_text)
        val_embs = self.model.encode(val_chunks, show_progress_bar=False) if val_chunks else np.array([])

        validated_scores, validation_log = [], []

        for se in scores:
            criterion, raw_score = se['criterion'], se['raw_score']
            naive_score = se.get('naive_score', raw_score)
            citations = se.get('citations', [])
            citation_results, any_valid = [], False

            for cite in citations:
                if not cite or len(cite.strip()) < 5:
                    continue
                er = self._validate_citation(cite, val_chunks, val_embs, resume_text)
                # Relevance check
                rel_ok = self._check_relevance(cite, criterion) if er['valid'] else True
                final_valid = er['valid'] and rel_ok
                if not rel_ok:
                    logger.info(f"    IRRELEVANT to '{criterion[:25]}': {cite[:40]}...")
                citation_results.append({
                    'citation': cite, 'valid': final_valid, 'similarity': er['similarity'],
                    'matched_chunk': er.get('matched_chunk'), 'match_type': er.get('match_type', ''),
                    'is_fallback': False})
                validation_log.append({
                    'candidate': candidate_name, 'criterion': criterion,
                    'citation': cite[:80] + '...' if len(cite) > 80 else cite,
                    'valid': final_valid, 'similarity': er['similarity']})
                if final_valid:
                    any_valid = True

            # Score adjustment
            if raw_score >= self.config.citation_required_score:
                if not citations:
                    vs, notes = max(0, raw_score - 2), "No citation provided — score reduced"
                elif not any_valid:
                    fb = self._semantic_fallback(criterion, val_chunks, val_embs)
                    if fb['found']:
                        vs, notes = raw_score, f"Fallback evidence found (sim={fb['similarity']:.0%})"
                        citation_results.append({
                            'citation': fb['evidence'], 'valid': True, 'similarity': fb['similarity'],
                            'matched_chunk': fb['evidence'], 'match_type': 'semantic_fallback', 'is_fallback': True})
                    else:
                        vs, notes = max(0, raw_score - 2), "Citation not verified — score reduced"
                else:
                    vs, notes = raw_score, "Verified"
            else:
                vs, notes = raw_score, "No citation required"

            # Bounding boxes
            if pdf_bytes:
                for cr in citation_results:
                    if cr['valid']:
                        bb = self._find_bbox(pdf_bytes, cr['citation'])
                        cr['page_num'], cr['bbox'] = bb.get('page'), bb.get('bbox')
                    else:
                        cr['page_num'] = cr['bbox'] = None

            validated_scores.append({
                'criterion_id': se.get('criterion_id'), 'criterion': criterion,
                'weight': se.get('weight', 3), 'critical': se.get('critical', False),
                'raw_score': raw_score, 'naive_score': naive_score,
                'validated_score': vs, 'reasoning': se.get('reasoning', ''),
                'validation_notes': notes, 'citation_results': citation_results})

        return {'node': 'CitationValidator', 'status': 'completed',
                'time_ms': (datetime.now() - start).total_seconds() * 1000,
                'candidate_name': candidate_name,
                'validated_scores': validated_scores, 'validation_log': validation_log}

    def _chunk_for_validation(self, text: str) -> List[str]:
        words = text.split()
        chunks, sz, ov = [], self.config.validation_chunk_size, self.config.validation_chunk_overlap
        step = max(1, sz - ov)
        for i in range(0, len(words), step):
            chunks.append(" ".join(words[i:i + sz]))
            if i + sz >= len(words):
                break
        return chunks

    def _validate_citation(self, citation: str, chunks: List[str], chunk_embs: np.ndarray, resume_text: str) -> Dict[str, Any]:
        if not citation or not resume_text:
            return {'valid': False, 'similarity': 0.0, 'match_type': 'empty'}
        threshold = self.config.citation_threshold

        def norm(t):
            t = t.lower().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            return ' '.join(re.sub(r'[:\;\,\.\-\_\/\|\(\)\[\]]+', ' ', t).split())

        cn, rn = norm(citation), norm(resume_text)

        # Exact
        if cn in rn:
            return {'valid': True, 'similarity': 1.0, 'matched_chunk': citation, 'match_type': 'exact'}

        # Fuzzy substring
        cw = cn.split()
        if len(cw) >= 3:
            for ws in range(min(10, len(cw)), 2, -1):
                for s in range(len(cw) - ws + 1):
                    w = ' '.join(cw[s:s+ws])
                    if w in rn:
                        sim = 0.7 + (ws / len(cw)) * 0.3
                        if sim >= threshold:
                            return {'valid': True, 'similarity': round(sim, 2), 'matched_chunk': w, 'match_type': 'fuzzy'}

        # Keyword overlap
        def kw(t):
            stops = {'this','that','with','have','from','they','been','were','said','each','the','and','for'}
            return {w for w in norm(t).split() if len(w) > 3 and w not in stops}
        ck, rk = kw(citation), kw(resume_text)
        if ck:
            ov = len(ck & rk) / len(ck)
            if ov >= 0.5:
                sim = 0.5 + ov * 0.5
                if sim >= threshold:
                    return {'valid': True, 'similarity': round(sim, 2), 'matched_chunk': ', '.join(list(ck & rk)[:5]), 'match_type': 'keyword'}

        # Semantic
        if len(chunks) > 0 and len(chunk_embs) > 0:
            ce = self.model.encode([citation])[0]
            sims = cosine_similarity([ce], chunk_embs)[0]
            bi = np.argmax(sims)
            bs = float(sims[bi])
            if bs >= threshold:
                return {'valid': True, 'similarity': round(bs, 4), 'matched_chunk': chunks[bi], 'match_type': 'semantic'}
            return {'valid': False, 'similarity': round(bs, 4), 'matched_chunk': None, 'match_type': 'below_threshold'}
        return {'valid': False, 'similarity': 0.0, 'match_type': 'no_match'}

    def _check_relevance(self, citation: str, criterion: str) -> bool:
        ce = self.model.encode([citation])[0]
        cr = self.model.encode([criterion])[0]
        return float(cosine_similarity([ce], [cr])[0][0]) >= 0.20

    def _semantic_fallback(self, criterion: str, chunks: List[str], chunk_embs: np.ndarray) -> Dict[str, Any]:
        if not chunks or len(chunk_embs) == 0:
            return {'found': False}
        ce = self.model.encode([criterion])[0]
        sims = cosine_similarity([ce], chunk_embs)[0]
        bi, bs = np.argmax(sims), float(np.max(cosine_similarity([self.model.encode([criterion])[0]], chunk_embs)[0]))
        if bs >= self.config.fallback_threshold:
            return {'found': True, 'evidence': chunks[bi], 'similarity': round(bs, 4)}
        return {'found': False}

    @staticmethod
    def _find_bbox(pdf_bytes: bytes, search_text: str) -> Dict[str, Any]:
        try:
            import fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            clean = ' '.join(search_text.split())
            words = clean.split()
            attempts = [clean[:120]]
            for n in [10, 7, 5, 4, 3]:
                if len(words) >= n:
                    attempts.append(' '.join(words[:n]))
            if len(words) > 6:
                attempts += [' '.join(words[1:6]), ' '.join(words[2:7])]
            for att in attempts:
                att = att.strip()
                if not att or len(att) < 6:
                    continue
                for pn in range(len(doc)):
                    inst = doc[pn].search_for(att)
                    if inst:
                        r = inst[0]; doc.close()
                        return {'found': True, 'page': pn, 'bbox': (r.x0, r.y0, r.x1, r.y1)}
            doc.close()
        except Exception:
            pass
        return {'found': False, 'page': None, 'bbox': None}

# =============================================================================
# NODE 4: RESPONSE GENERATOR
# =============================================================================
class Node4_ResponseGenerator:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.llm = UnifiedLLMClient(config.response_gen_url, config.response_gen_model, timeout=config.llm_timeout)

    def process(self, validated_result: Dict, job_title: str) -> Dict[str, Any]:
        start = datetime.now()
        name = validated_result.get('candidate_name', 'Candidate')
        scores = validated_result.get('validated_scores', [])

        tw = sum(s['validated_score'] * s['weight'] for s in scores)
        mw = sum(5 * s['weight'] for s in scores)
        final_score = round(tw / mw * 100) if mw else 0

        ct = sum(1 for s in scores if s.get('critical'))
        cm = sum(1 for s in scores if s.get('critical') and s['validated_score'] >= 3)

        if final_score >= 75 and cm == ct:
            tone, rec = 'excellent', 'strongly_recommend'
        elif final_score >= 55 and cm >= ct - 1:
            tone, rec = 'strong', 'recommend'
        elif final_score >= 35 or cm >= 1:
            tone, rec = 'average', 'consider'
        else:
            tone, rec = 'below', 'do_not_recommend'

        vcites = [cr['citation'] for s in scores for cr in s.get('citation_results', [])
                  if cr.get('valid') and cr.get('citation') and not cr.get('is_fallback')]

        fs = "\n".join([f"- {s['criterion']}: {s['validated_score']}/5 (critical={s.get('critical')})" for s in scores])
        cl = "\n".join([f'  - <cite>{c}</cite>' for c in vcites[:6]])

        # Small delay before summary call
        time.sleep(1.5)

        prompt = f"""Write a 2-4 sentence professional summary for this candidate.

Candidate: {name}
Position: {job_title}
Score: {final_score}/100
Critical Met: {cm}/{ct}

Scores:
{fs}

Verified citations (use ONLY these exact quotes with <cite> tags):
{cl if cl else "  (none)"}

Start with "{name}". Do NOT invent job titles or companies not in the data above.
Summary:"""

        try:
            raw = self.llm.generate(prompt, max_tokens=300)
            summary = self._clean(raw, name)
            if not summary or len(summary) < 20:
                raise ValueError("Too short")
        except Exception as e:
            logger.warning(f"  Summary failed ({e}), using fallback")
            summary = self._fallback(name, job_title, final_score, scores, tone, cm, ct, vcites)

        return {'node': 'ResponseGenerator', 'status': 'completed',
                'time_ms': (datetime.now() - start).total_seconds() * 1000,
                'candidate_name': name, 'summary': summary, 'tone': tone,
                'final_score': final_score, 'critical_met': cm, 'critical_total': ct,
                'recommendation': rec}

    @staticmethod
    def _clean(raw, name):
        t = raw.strip().strip('"\'')
        t = re.sub(r'```[a-z]*\n?', '', t).replace('```', '')
        m = re.search(r'"summary"\s*:\s*"(.*?)"', t, re.DOTALL)
        if m: t = m.group(1)
        t = re.sub(r'^(Here\s+(is|are)\s+(the|a)\s+)?summary:?\s*', '', t, flags=re.IGNORECASE)
        t = re.sub(r'<\s*cite\s*>', '<cite>', t, flags=re.IGNORECASE)
        t = re.sub(r'<\s*/\s*cite\s*>', '</cite>', t, flags=re.IGNORECASE)
        if t.lower().count('<cite>') != t.lower().count('</cite>'):
            t = re.sub(r'</?cite>', '', t, flags=re.IGNORECASE)
        t = re.sub(r'\n+', ' ', t)
        t = re.sub(r'\s{2,}', ' ', t).strip()
        if len(t) < 20: return ""
        fw = name.lower().split()[0] if name else ""
        if fw and not t.lower().startswith(fw):
            t = f"{name} — {t}"
        return t

    @staticmethod
    def _fallback(name, title, score, scores, tone, cm, ct, cites):
        op = {'excellent': "is an excellent candidate", 'strong': "is a strong candidate",
              'average': "shows potential", 'below': "does not meet key requirements"}
        p = [f"{name} {op.get(tone, '')} for {title} (Score: {score}/100)."]
        st = [s for s in scores if s['validated_score'] >= 4]
        ga = [s for s in scores if s['validated_score'] <= 1 and s.get('critical')]
        if st:
            p.append(f"Strengths: {', '.join(s['criterion'] for s in st[:2])}.")
            if cites: p.append(f"Evidence: <cite>{cites[0][:80]}</cite>.")
        if ga: p.append(f"Gaps: {', '.join(g['criterion'] for g in ga[:2])}.")
        p.append(f"Met {cm}/{ct} critical criteria.")
        return " ".join(p)

# =============================================================================
# NODE 5: REPORT GENERATOR
# =============================================================================
class Node5_ReportGenerator:
    def process(self, all_candidates: List[Dict], job_title: str, criteria: List[Dict]) -> Dict[str, Any]:
        start = datetime.now()
        sc = sorted(all_candidates, key=lambda x: x['final_score'], reverse=True)
        results = [{
            'rank': i+1, 'name': c['candidate_name'], 'score': c['final_score'],
            'critical_met': c['critical_met'], 'critical_total': c['critical_total'],
            'recommendation': c['recommendation'], 'summary': c['summary'],
            'tone': c['tone'], 'scores_detail': c.get('validated_scores', []),
            'filename': c.get('filename'), 'security_status': c.get('security_status', 'safe'),
            'ocr_used': c.get('ocr_used', False), 'ocr_warning': c.get('ocr_warning')
        } for i, c in enumerate(sc)]
        return {'node': 'ReportGenerator', 'status': 'completed',
                'time_ms': (datetime.now() - start).total_seconds() * 1000,
                'job_title': job_title, 'total_candidates': len(results),
                'qualified_count': sum(1 for r in results if r['score'] >= 55),
                'criteria': criteria, 'results': results}

# =============================================================================
# PIPELINE
# =============================================================================
class ResumeScreeningPipeline:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.node0 = Node0_ResuShield(self.config)
        self.node1 = Node1_QuestionGeneration(self.config)
        self.node2 = Node2_ResumeEvaluation(self.config)
        self.node3 = Node3_CitationValidator(self.config)
        self.node4 = Node4_ResponseGenerator(self.config)
        self.node5 = Node5_ReportGenerator()

    def run(self, job_description: str, resumes: List[Dict], pipeline_config: Dict = None,
            progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        start_time = datetime.now()
        cfg = pipeline_config or {}
        num_criteria = cfg.get('eval_criteria_count', 5)
        pl, vl, ow = [], [], []

        def prg(p, m):
            if progress_callback: progress_callback(p, m)

        try:
            prg(10, "Node 1: Generating criteria...")
            logger.info("=" * 50 + "\nNode 1: Generating criteria...")
            n1 = self.node1.process(job_description, num_criteria)
            pl.append({'node': 'Node 1: Question Generation', 'status': n1['status'], 'time_ms': n1['time_ms']})
            criteria, job_title = n1['criteria'], n1['job_title']
            logger.info(f"  Job: {job_title}, Criteria: {[c['criterion'][:30] for c in criteria]}")

            ac, fc = [], []
            total = len(resumes)
            for i, res in enumerate(resumes):
                nm = res.get('name', f'Candidate {i+1}')
                ct, pb, fn = res.get('content', ''), res.get('pdf_bytes'), res.get('filename', '')
                prg(20 + int(70*i/max(total,1)), f"Processing [{i+1}/{total}]: {nm}...")
                logger.info(f"{'='*50}\nProcessing [{i+1}/{total}]: {nm}")
                try:
                    n0 = self.node0.process(pb or b'', fn, ct)
                    vt = n0['visible_text']
                    ss = 'safe' if n0['is_safe'] else 'flagged'
                    if n0.get('ocr_warning'): ow.append({'candidate': nm, 'warning': n0['ocr_warning']})
                    pl.append({'node': f'Node 0 ({nm})', 'status': 'completed', 'time_ms': n0['time_ms']})
                    if not vt or len(vt.strip()) < 50:
                        raise RuntimeError("Text too short")

                    n2 = self.node2.process(nm, vt, criteria)
                    pl.append({'node': f'Node 2 ({nm})', 'status': n2['status'], 'time_ms': n2['time_ms']})

                    n3 = self.node3.process(n2, vt, pb)
                    vl.extend(n3.get('validation_log', []))
                    pl.append({'node': f'Node 3 ({nm})', 'status': n3['status'], 'time_ms': n3['time_ms']})

                    time.sleep(1.5)  # Pace before summary
                    n4 = self.node4.process(n3, job_title)
                    pl.append({'node': f'Node 4 ({nm})', 'status': n4['status'], 'time_ms': n4['time_ms']})
                    logger.info(f"  Score: {n4['final_score']}/100")

                    ac.append({**n4, 'validated_scores': n3['validated_scores'],
                              'filename': fn, 'security_status': ss,
                              'ocr_used': n0.get('ocr_used', False), 'ocr_warning': n0.get('ocr_warning')})
                except Exception as e:
                    logger.error(f"  FAILED: {e}")
                    fc.append({'name': nm, 'error': str(e)})
                    pl.append({'node': f'Pipeline ({nm})', 'status': 'failed', 'time_ms': 0})

            if not ac:
                return {'success': False, 'error': "All failed", 'failed_candidates': fc, 'pipeline_log': pl}

            prg(92, "Node 5: Final report...")
            n5 = self.node5.process(ac, job_title, criteria)
            pl.append({'node': 'Node 5: Report', 'status': n5['status'], 'time_ms': n5['time_ms']})
            prg(100, "Complete!")
            return {
                'success': True, 'processing_time_seconds': round((datetime.now()-start_time).total_seconds(), 1),
                'job_title': job_title, 'evaluation_criteria': criteria,
                'results': n5['results'], 'qualified_count': n5['qualified_count'],
                'total_count': n5['total_candidates'], 'failed_count': len(fc),
                'failed_candidates': fc, 'validation_log': vl, 'pipeline_log': pl, 'ocr_warnings': ow}
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            import traceback; traceback.print_exc()
            return {'success': False, 'error': str(e), 'pipeline_log': pl}

# =============================================================================
# UTILITIES
# =============================================================================
def extract_text_from_bytes(file_bytes: bytes, filename: str) -> Tuple[str, Dict]:
    suffix = Path(filename).suffix.lower()
    if suffix == '.pdf':
        try:
            import fitz
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                text = " ".join(page.get_text() for page in doc)
            return re.sub(r'\s+', ' ', text).strip(), {'method': 'pymupdf'}
        except Exception as e: return "", {'method': 'failed', 'error': str(e)}
    elif suffix == '.docx':
        try:
            from docx import Document; import io as _io
            doc = Document(_io.BytesIO(file_bytes))
            return re.sub(r'\s+', ' ', " ".join(p.text for p in doc.paragraphs)).strip(), {'method': 'docx'}
        except Exception as e: return "", {'method': 'failed', 'error': str(e)}
    elif suffix == '.txt':
        return file_bytes.decode('utf-8', errors='ignore').strip(), {'method': 'txt'}
    return "", {'method': 'unsupported'}

def is_jd_file(fn: str) -> bool:
    return any(x in fn.lower() for x in ['jd', 'job', 'description', 'position', 'role'])

def extract_citations(text: str) -> List[str]:
    if not text: return []
    return [c.strip() for c in re.findall(r'<cite>(.*?)</cite>', text, re.DOTALL|re.IGNORECASE) if c.strip() and len(c.strip()) > 5]

__all__ = ['PipelineConfig', 'load_config', 'ResumeScreeningPipeline',
           'extract_text_from_bytes', 'is_jd_file', 'extract_citations',
           'RAGRetriever', 'UnifiedLLMClient', 'RateLimitError', 'APIKeyRotator', 'get_key_rotator']