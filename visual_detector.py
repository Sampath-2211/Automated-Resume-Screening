"""
Automated Resume Screening - ResuShield Security Module
Visual-Semantic Discrepancy Detection, OCR Extraction & Prompt Injection Shield

Node 0 of the pipeline - processes BEFORE any LLM evaluation.
CRITICAL: get_visible_text() provides text to Nodes 1-4.

Author: Sampath Krishna Tekumalla
"""
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("ResumeScreening")

# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class DiscrepancyResult:
    trust_score: float
    discrepancy_score: float
    hidden_keywords: List[str]
    injection_detected: bool
    injection_type: Optional[str]
    semantic_distance: float
    evidence: Dict[str, Any]
    recommendation: str  # 'trust', 'review', 'reject'

@dataclass
class SecurityScanResult:
    overall_trust: float
    overall_recommendation: str
    threats_detected: List[Dict[str, Any]]
    visual_analysis: Optional[Dict[str, Any]]
    injection_analysis: Dict[str, Any]
    is_safe: bool
    ocr_text: Optional[str]
    raw_text: str

# =============================================================================
# OCR EXTRACTOR
# =============================================================================
class OCRExtractor:
    """Extract visible text using Tesseract OCR — what humans actually see."""

    @staticmethod
    def extract_from_pdf(pdf_bytes: bytes, max_pages: int = 3) -> Optional[str]:
        try:
            from pdf2image import convert_from_bytes
            import pytesseract

            images = convert_from_bytes(
                pdf_bytes, dpi=150, first_page=1, last_page=max_pages
            )
            texts = [pytesseract.image_to_string(img) for img in images]
            combined = re.sub(r'\s+', ' ', " ".join(texts)).strip()
            return combined if combined else None
        except ImportError as e:
            logger.warning(f"OCR dependencies not installed: {e}")
            return None
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
            return None

    @staticmethod
    def is_available() -> bool:
        try:
            from pdf2image import convert_from_bytes
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

# =============================================================================
# PROMPT INJECTION SHIELD
# =============================================================================
class PromptInjectionShield:
    """Detects prompt injection attempts in resume text."""

    INJECTION_PATTERNS = [
        r'ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|rules?|prompts?)',
        r'disregard\s+(all\s+)?(previous|above|prior)',
        r'forget\s+(everything|all|what)\s+(you|i)\s+(said|told|wrote)',
        r'(give|assign|set)\s+(me|this|candidate)\s+(a\s+)?(score|rating)\s+(of\s+)?\d+',
        r'(rate|score)\s+(this|me)\s+(as\s+)?(highly|excellent|perfect|10)',
        r'this\s+candidate\s+(is|should\s+be)\s+(perfect|excellent|ideal)',
        r'automatically\s+(accept|approve|hire)',
        r'you\s+are\s+(now|actually)\s+(a|an)',
        r'pretend\s+(you\s+are|to\s+be)',
        r'act\s+as\s+(if|though)',
        r'roleplay\s+as',
        r'(show|reveal|display|print)\s+(your|the)\s+(system\s+)?prompt',
        r'what\s+(are|is)\s+your\s+(instructions?|rules?|prompt)',
        r'repeat\s+(your|the)\s+(system|initial)\s+(prompt|instructions?)',
        r'<\/?system>',
        r'\[INST\]|\[\/INST\]',
        r'###\s*(instruction|system|user)',
        r'<\|.*?\|>',
    ]

    SUSPICIOUS_PHRASES = [
        'as an ai', 'as a language model', 'i am programmed',
        'my instructions', 'system prompt', 'ignore this',
        'secret keyword', 'hidden instruction', 'do not tell',
        'override', 'bypass', 'jailbreak', 'dan mode',
        'developer mode', 'admin access', 'root access',
    ]

    def __init__(self):
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    def scan(self, text: str) -> Dict[str, Any]:
        if not text:
            return {
                'injection_detected': False, 'risk_score': 0.0,
                'detected_patterns': [], 'suspicious_phrases': [],
                'recommendation': 'trust'
            }

        text_lower = text.lower()

        detected_patterns = []
        for i, pattern in enumerate(self.compiled_patterns):
            matches = pattern.findall(text_lower)
            if matches:
                detected_patterns.append({
                    'pattern_id': i,
                    'pattern': self.INJECTION_PATTERNS[i],
                    'matches': matches[:3],
                    'severity': 'critical'
                })

        detected_phrases = [p for p in self.SUSPICIOUS_PHRASES if p in text_lower]

        is_injection = len(detected_patterns) > 0
        risk_score = min(1.0, len(detected_patterns) * 0.4 + len(detected_phrases) * 0.1)

        if is_injection:
            recommendation = 'reject'
        elif risk_score > 0.3:
            recommendation = 'review'
        else:
            recommendation = 'trust'

        return {
            'injection_detected': is_injection,
            'risk_score': round(risk_score, 3),
            'detected_patterns': detected_patterns,
            'suspicious_phrases': detected_phrases,
            'recommendation': recommendation
        }

# =============================================================================
# VISUAL-SEMANTIC DETECTOR
# =============================================================================
class VisualSemanticDetector:
    """Compares OCR text (human-visible) vs raw PDF text to detect hidden content."""

    TECH_KEYWORDS = [
        'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue',
        'aws', 'azure', 'gcp', 'kubernetes', 'docker', 'terraform', 'ansible',
        'postgresql', 'mongodb', 'redis', 'kafka', 'elasticsearch',
        'machine learning', 'deep learning', 'tensorflow', 'pytorch',
        'agile', 'scrum', 'devops', 'ci/cd', 'jenkins', 'github actions',
        'microservices', 'rest api', 'graphql', 'grpc',
        'langchain', 'langgraph', 'llm', 'gpt', 'openai', 'anthropic',
        'leadership', 'management', 'strategic', 'innovative',
        'senior', 'principal', 'staff', 'director', 'vp',
    ]

    SEMANTIC_THRESHOLD = 0.15
    KEYWORD_THRESHOLD = 5

    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self._model = None
        self._model_name = embedding_model

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def compare(self, raw_text: str, ocr_text: str) -> DiscrepancyResult:
        if not raw_text or not ocr_text:
            return DiscrepancyResult(
                trust_score=0.5, discrepancy_score=0.5, hidden_keywords=[],
                injection_detected=False, injection_type=None, semantic_distance=0.0,
                evidence={'error': 'Missing text'}, recommendation='review'
            )

        raw_emb = self.model.encode(raw_text[:5000])
        ocr_emb = self.model.encode(ocr_text[:5000])
        similarity = cosine_similarity([raw_emb], [ocr_emb])[0][0]
        semantic_distance = 1 - similarity

        raw_words = set(self._extract_keywords(raw_text))
        ocr_words = set(self._extract_keywords(ocr_text))
        hidden_words = list(raw_words - ocr_words)
        hidden_tech = [w for w in hidden_words if w.lower() in self.TECH_KEYWORDS]

        keyword_injection = len(hidden_tech) > self.KEYWORD_THRESHOLD
        semantic_injection = semantic_distance > self.SEMANTIC_THRESHOLD
        injection_detected = keyword_injection or semantic_injection

        injection_type = None
        if keyword_injection and semantic_injection:
            injection_type = "severe_hidden_text"
        elif keyword_injection:
            injection_type = "keyword_stuffing"
        elif semantic_injection:
            injection_type = "semantic_manipulation"

        if injection_detected:
            trust_score = max(0.1, 1 - semantic_distance - len(hidden_tech) * 0.05)
        else:
            trust_score = min(1.0, similarity + 0.1)

        discrepancy_score = semantic_distance + len(hidden_tech) * 0.02

        if trust_score < 0.5:
            recommendation = "reject"
        elif trust_score < 0.8:
            recommendation = "review"
        else:
            recommendation = "trust"

        return DiscrepancyResult(
            trust_score=round(trust_score, 2),
            discrepancy_score=round(min(1.0, discrepancy_score), 2),
            hidden_keywords=hidden_tech[:20],
            injection_detected=injection_detected,
            injection_type=injection_type,
            semantic_distance=round(semantic_distance, 4),
            evidence={
                'raw_word_count': len(raw_words),
                'ocr_word_count': len(ocr_words),
                'hidden_word_count': len(hidden_words),
                'hidden_tech_count': len(hidden_tech),
                'similarity': round(similarity, 4),
                'sample_hidden': hidden_words[:10]
            },
            recommendation=recommendation
        )

    def _extract_keywords(self, text: str) -> List[str]:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
            'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
            'have', 'been', 'were', 'being', 'with', 'they', 'this',
            'from', 'that', 'which', 'their', 'will', 'would', 'there',
            'what', 'about', 'into', 'than', 'them', 'these', 'then'
        }
        return [w for w in words if w not in stopwords]

# =============================================================================
# RESUSHIELD - MAIN SECURITY SCANNER
# =============================================================================
class ResuShield:
    """
    Main security scanner combining OCR, visual-semantic detection, and injection shield.
    Node 0 of the pipeline — runs BEFORE any LLM processing.
    """

    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.ocr_extractor = OCRExtractor()
        self.visual_detector = VisualSemanticDetector(embedding_model)
        self.injection_shield = PromptInjectionShield()

    def analyze(self, raw_text: str, pdf_bytes: Optional[bytes] = None,
                perform_ocr: bool = True) -> SecurityScanResult:
        threats_detected = []
        overall_trust = 1.0
        ocr_text = None
        visual_analysis = None

        # 1. Prompt injection scan
        injection_result = self.injection_shield.scan(raw_text)
        if injection_result['injection_detected']:
            threats_detected.append({
                'type': 'prompt_injection', 'severity': 'critical',
                'details': {
                    'patterns': injection_result['detected_patterns'],
                    'phrases': injection_result['suspicious_phrases']
                }
            })
            overall_trust = 0.1

        # 2. OCR + visual-semantic comparison
        if perform_ocr and pdf_bytes:
            ocr_text = self.ocr_extractor.extract_from_pdf(pdf_bytes)
            if ocr_text:
                visual_result = self.visual_detector.compare(raw_text, ocr_text)
                visual_analysis = {
                    'trust_score': visual_result.trust_score,
                    'discrepancy_score': visual_result.discrepancy_score,
                    'semantic_distance': visual_result.semantic_distance,
                    'hidden_keywords': visual_result.hidden_keywords,
                    'injection_detected': visual_result.injection_detected,
                    'injection_type': visual_result.injection_type,
                    'recommendation': visual_result.recommendation
                }
                if visual_result.injection_detected:
                    threats_detected.append({
                        'type': visual_result.injection_type or 'hidden_text',
                        'severity': 'high' if visual_result.trust_score < 0.5 else 'medium',
                        'details': {
                            'hidden_keywords': visual_result.hidden_keywords[:10],
                            'semantic_distance': visual_result.semantic_distance
                        }
                    })
                    overall_trust = min(overall_trust, visual_result.trust_score)

        if overall_trust < 0.5:
            overall_recommendation = 'reject'
        elif overall_trust < 0.8 or len(threats_detected) > 0:
            overall_recommendation = 'review'
        else:
            overall_recommendation = 'trust'

        return SecurityScanResult(
            overall_trust=round(overall_trust, 2),
            overall_recommendation=overall_recommendation,
            threats_detected=threats_detected,
            visual_analysis=visual_analysis,
            injection_analysis=injection_result,
            is_safe=len(threats_detected) == 0,
            ocr_text=ocr_text,
            raw_text=raw_text
        )

    def get_visible_text(self, raw_text: str, pdf_bytes: Optional[bytes] = None) -> str:
        """Returns ONLY visible text (what humans see). This feeds Nodes 1-4."""
        if pdf_bytes:
            ocr_text = self.ocr_extractor.extract_from_pdf(pdf_bytes)
            if ocr_text and len(ocr_text) > 100:
                return ocr_text
        return raw_text

    def get_security_report(self, result: SecurityScanResult) -> str:
        lines = ["=" * 50, "RESUSHIELD SECURITY REPORT", "=" * 50, ""]
        lines.append(f"Overall Trust Score: {result.overall_trust:.0%}")
        lines.append(f"Recommendation: {result.overall_recommendation.upper()}")
        lines.append(f"Safe: {'Yes' if result.is_safe else 'No'}")
        lines.append("")
        if result.threats_detected:
            lines.append(f"{len(result.threats_detected)} THREAT(S) DETECTED:")
            for t in result.threats_detected:
                lines.append(f"  [{t['severity'].upper()}] {t['type']}")
                if 'hidden_keywords' in t.get('details', {}):
                    lines.append(f"    Hidden: {', '.join(t['details']['hidden_keywords'][:5])}")
        else:
            lines.append("No threats detected")
        return "\n".join(lines)

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================
def quick_security_check(raw_text: str, pdf_bytes: Optional[bytes] = None) -> Tuple[float, str, List[str]]:
    shield = ResuShield()
    result = shield.analyze(raw_text, pdf_bytes)
    return (result.overall_trust, result.overall_recommendation, [t['type'] for t in result.threats_detected])

def scan_resume_security(raw_text: str, pdf_bytes: Optional[bytes] = None, perform_ocr: bool = True) -> Dict[str, Any]:
    shield = ResuShield()
    result = shield.analyze(raw_text, pdf_bytes, perform_ocr)
    return {
        'is_safe': result.is_safe,
        'trust_score': result.overall_trust,
        'recommendation': result.overall_recommendation,
        'threats': result.threats_detected,
        'visible_text': result.ocr_text or result.raw_text,
        'report': shield.get_security_report(result)
    }

def get_visible_text_only(raw_text: str, pdf_bytes: Optional[bytes] = None) -> str:
    shield = ResuShield()
    return shield.get_visible_text(raw_text, pdf_bytes)

__all__ = [
    'OCRExtractor', 'PromptInjectionShield', 'VisualSemanticDetector',
    'ResuShield', 'DiscrepancyResult', 'SecurityScanResult',
    'quick_security_check', 'scan_resume_security', 'get_visible_text_only'
]