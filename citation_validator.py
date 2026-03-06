"""
Automated Resume Screening - Citation Validator
Citation extraction, multi-strategy validation, bounding box extraction.

Threshold: 0.80 semantic similarity (configurable via CITATION_THRESHOLD).

Author: Sampath Krishna Tekumalla
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("ResumeScreening")

# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class CitationMatch:
    citation_text: str
    valid: bool
    similarity: float
    matched_chunk: Optional[str]
    page_num: Optional[int]
    bbox: Optional[Tuple[float, float, float, float]]
    extraction_method: str
    is_fallback: bool = False

@dataclass
class ValidationResult:
    candidate_name: str
    criterion: str
    original_score: int
    validated_score: int
    citations: List[CitationMatch]
    score_changed: bool
    reason: str
    fallback_used: bool = False

# =============================================================================
# CITATION EXTRACTOR
# =============================================================================
class CitationExtractor:
    """Extracts and parses <cite> tags from LLM output."""
    CITE_PATTERN = re.compile(r'<cite>(.*?)</cite>', re.DOTALL | re.IGNORECASE)

    @staticmethod
    def extract(text: str) -> List[str]:
        if not text:
            return []
        return [m.strip() for m in CitationExtractor.CITE_PATTERN.findall(text) if m.strip()]

    @staticmethod
    def remove_tags(text: str) -> str:
        return CitationExtractor.CITE_PATTERN.sub(r'\1', text)

    @staticmethod
    def highlight_citations(text: str) -> str:
        def replacer(match):
            content = match.group(1)
            escaped = content.replace('"', '&quot;').replace("'", "&#39;")
            return f'<span class="citation-link" data-citation="{escaped}">{content}</span>'
        return CitationExtractor.CITE_PATTERN.sub(replacer, text)

    @staticmethod
    def wrap_citation(text: str) -> str:
        return f'<cite>{text}</cite>'

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'[:\;\,\.\-\_\/\|\(\)\[\]]+', ' ', text)
    return ' '.join(text.split())

def extract_key_terms(text: str) -> set:
    normalized = normalize_text(text)
    stopwords = {'this', 'that', 'with', 'have', 'from', 'they', 'been', 'were', 'said', 'each', 'the', 'and', 'for'}
    return {w for w in normalized.split() if len(w) > 3 and w not in stopwords}

def keyword_overlap_score(text1: str, text2: str) -> float:
    t1 = extract_key_terms(text1)
    t2 = extract_key_terms(text2)
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1)

def fuzzy_substring_match(citation: str, resume: str, min_words: int = 3) -> Tuple[bool, float]:
    cite_norm = normalize_text(citation)
    resume_norm = normalize_text(resume)
    cite_words = cite_norm.split()

    if len(cite_words) < min_words:
        all_found = all(w in resume_norm for w in cite_words)
        return all_found, 0.9 if all_found else 0.0

    for window_size in range(min(10, len(cite_words)), min_words - 1, -1):
        for start in range(len(cite_words) - window_size + 1):
            window = ' '.join(cite_words[start:start + window_size])
            if window in resume_norm:
                ratio = window_size / len(cite_words)
                return True, 0.7 + (0.3 * ratio)
    return False, 0.0

# =============================================================================
# SEMANTIC VALIDATOR
# =============================================================================
class SemanticValidator:
    """Validates citations using sentence-transformer embeddings."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', threshold: float = 0.80):
        self._model = None
        self._model_name = model_name
        self.threshold = threshold
        self.fallback_threshold = 0.60

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def chunk_text(self, text: str, chunk_size: int = 30, overlap: int = 10) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks = []
        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            chunks.append(" ".join(words[i:i + chunk_size]))
            if i + chunk_size >= len(words):
                break
        return chunks

    def validate_citation(self, citation: str, resume_text: str) -> Dict[str, Any]:
        if not citation or not resume_text:
            return {"valid": False, "similarity": 0.0, "matched_chunk": None}
        chunks = self.chunk_text(resume_text)
        if not chunks:
            return {"valid": False, "similarity": 0.0, "matched_chunk": None}
        cite_emb = self.model.encode([citation])[0]
        chunk_embs = self.model.encode(chunks, show_progress_bar=False)
        sims = cosine_similarity([cite_emb], chunk_embs)[0]
        best_idx = np.argmax(sims)
        best_sim = float(sims[best_idx])
        return {
            "valid": best_sim >= self.threshold,
            "similarity": round(best_sim, 4),
            "matched_chunk": chunks[best_idx] if best_sim >= self.threshold else None
        }

    def semantic_fallback(self, criterion: str, resume_text: str) -> Dict[str, Any]:
        chunks = self.chunk_text(resume_text, chunk_size=40, overlap=15)
        if not chunks:
            return {"found": False}
        crit_emb = self.model.encode([criterion])[0]
        chunk_embs = self.model.encode(chunks, show_progress_bar=False)
        sims = cosine_similarity([crit_emb], chunk_embs)[0]
        best_idx = np.argmax(sims)
        best_sim = float(sims[best_idx])
        if best_sim >= self.fallback_threshold:
            return {"found": True, "evidence": chunks[best_idx], "similarity": round(best_sim, 4)}
        return {"found": False}

# =============================================================================
# BOUNDING BOX EXTRACTOR
# =============================================================================
class BoundingBoxExtractor:
    """Finds citation location in PDF for click-to-highlight."""

    @staticmethod
    def find_in_pdf_bytes(pdf_bytes: bytes, search_text: str) -> Dict[str, Any]:
        try:
            import fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            strategies = [
                ' '.join(search_text.split())[:100],
                ' '.join(search_text.split()[:8]),
                ' '.join(search_text.split()[:5]),
            ]
            for strat in strategies:
                if not strat.strip():
                    continue
                for page_num in range(len(doc)):
                    instances = doc[page_num].search_for(strat)
                    if instances:
                        rect = instances[0]
                        doc.close()
                        return {"found": True, "page": page_num, "bbox": (rect.x0, rect.y0, rect.x1, rect.y1)}
            doc.close()
            return {"found": False, "page": None, "bbox": None}
        except ImportError:
            return {"found": False, "page": None, "bbox": None}
        except Exception:
            return {"found": False, "page": None, "bbox": None}

# =============================================================================
# MAIN CITATION VALIDATOR
# =============================================================================
class CitationValidator:
    """
    Multi-strategy validator:
    1. Normalized exact match
    2. Fuzzy substring match
    3. Keyword overlap
    4. Semantic similarity (threshold 0.80)
    """

    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2', threshold: float = 0.80):
        self.semantic_validator = SemanticValidator(embedding_model, threshold)
        self.bbox_extractor = BoundingBoxExtractor()
        self.threshold = threshold

    def validate_citation(self, citation: str, resume_text: str) -> Dict[str, Any]:
        if not citation or not resume_text:
            return {"valid": False, "similarity": 0.0, "method": "empty"}

        cite_norm = normalize_text(citation)
        resume_norm = normalize_text(resume_text)

        # 1. Exact normalized
        if cite_norm in resume_norm:
            return {"valid": True, "similarity": 1.0, "method": "exact"}

        # 2. Fuzzy substring
        matched, score = fuzzy_substring_match(citation, resume_text)
        if matched and score >= self.threshold:
            return {"valid": True, "similarity": score, "method": "fuzzy"}

        # 3. Keyword overlap
        kw = keyword_overlap_score(citation, resume_text)
        if kw >= 0.7:
            sim = 0.5 + kw * 0.5
            if sim >= self.threshold:
                return {"valid": True, "similarity": sim, "method": "keyword"}

        # 4. Semantic
        sem = self.semantic_validator.validate_citation(citation, resume_text)
        if sem["valid"]:
            return {"valid": True, "similarity": sem["similarity"], "method": "semantic"}

        return {"valid": False, "similarity": sem["similarity"], "method": "none"}

    def validate_candidate_citations(
        self, candidate_name: str, scores_with_citations: List[Dict],
        resume_text: str, pdf_bytes: Optional[bytes] = None
    ) -> Tuple[List[ValidationResult], List[Dict]]:
        results = []
        validation_log = []

        for score_entry in scores_with_citations:
            criterion = score_entry.get('criterion', 'Unknown')
            original_score = score_entry.get('raw_score', 0)
            reasoning = score_entry.get('reasoning', '')
            citations = CitationExtractor.extract(reasoning)

            citation_matches = []
            all_valid = True

            for cite_text in citations:
                result = self.validate_citation(cite_text, resume_text)
                bbox_result = {"found": False, "page": None, "bbox": None}
                if pdf_bytes and result["valid"]:
                    bbox_result = self.bbox_extractor.find_in_pdf_bytes(pdf_bytes, cite_text)

                citation_matches.append(CitationMatch(
                    citation_text=cite_text,
                    valid=result["valid"],
                    similarity=result["similarity"],
                    matched_chunk=None,
                    page_num=bbox_result.get("page"),
                    bbox=bbox_result.get("bbox"),
                    extraction_method=result.get("method", "unknown")
                ))

                if not result["valid"]:
                    all_valid = False

                validation_log.append({
                    "candidate": candidate_name,
                    "criterion": criterion,
                    "citation": cite_text[:80] + "..." if len(cite_text) > 80 else cite_text,
                    "valid": result["valid"],
                    "similarity": result["similarity"]
                })

            # Determine validated score
            if original_score >= 3 and citations and not all_valid:
                # Try fallback
                fallback = self.semantic_validator.semantic_fallback(criterion, resume_text)
                if fallback.get("found"):
                    validated_score = min(original_score, 3)
                    reason = "Fallback evidence found"
                else:
                    validated_score = 0
                    reason = "Citation not verified, no alternative"
            elif original_score >= 3 and not citations:
                validated_score = max(0, original_score - 2)
                reason = "No citation provided"
            else:
                validated_score = original_score
                reason = "Verified" if citations else "No citation required"

            results.append(ValidationResult(
                candidate_name=candidate_name,
                criterion=criterion,
                original_score=original_score,
                validated_score=validated_score,
                citations=citation_matches,
                score_changed=original_score != validated_score,
                reason=reason
            ))

        return results, validation_log

    def get_citation_bbox(self, pdf_bytes: bytes, citation_text: str) -> Optional[Dict]:
        return self.bbox_extractor.find_in_pdf_bytes(pdf_bytes, citation_text)

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================
def validate_and_adjust_scores(
    scores_with_citations: List[Dict], resume_text: str,
    pdf_bytes: Optional[bytes] = None, threshold: float = 0.80
) -> Tuple[List[Dict], List[Dict]]:
    validator = CitationValidator(threshold=threshold)
    validation_log = []

    for entry in scores_with_citations:
        reasoning = entry.get('reasoning', '')
        citations = CitationExtractor.extract(reasoning)
        original = entry.get('raw_score', 0)
        criterion = entry.get('criterion', '')

        all_valid = True
        citation_results = []

        for cite in citations:
            result = validator.validate_citation(cite, resume_text)
            citation_results.append({
                "citation": cite, "valid": result["valid"], "similarity": result["similarity"]
            })
            validation_log.append({
                "criterion": criterion,
                "citation": cite[:60] + "..." if len(cite) > 60 else cite,
                "valid": result["valid"], "similarity": result["similarity"]
            })
            if not result["valid"]:
                all_valid = False

        if original >= 3 and citations and not all_valid:
            entry['validated_score'] = 0
            entry['validation_status'] = 'invalidated'
        elif original >= 3 and not citations:
            entry['validated_score'] = max(0, original - 2)
            entry['validation_status'] = 'no_citation'
        else:
            entry['validated_score'] = original
            entry['validation_status'] = 'verified' if citations else 'no_claim'

        entry['citation_results'] = citation_results

    return scores_with_citations, validation_log

__all__ = [
    'CitationExtractor', 'SemanticValidator', 'BoundingBoxExtractor',
    'CitationValidator', 'CitationMatch', 'ValidationResult',
    'validate_and_adjust_scores', 'normalize_text',
    'fuzzy_substring_match', 'keyword_overlap_score'
]