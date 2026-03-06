"""
FastScreen AI - Pure Cloud Engine
Multi-provider support via .env configuration
Supports: Ollama (local), Groq, Gemini, OpenRouter, OpenAI
"""
import json
import re
import hashlib
import logging
import zipfile
import os
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration loaded from .env file"""
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).parent)
    UPLOAD_DIR: Path = None
    CACHE_DIR: Path = None
    
    # LLM Settings (loaded from .env)
    LLM_PROVIDER: str = None
    LLM_TIMEOUT: int = 120
    
    # Provider-specific settings
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma2:2b"
    
    GROQ_API_KEY: str = None
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    
    GEMINI_API_KEY: str = None
    GEMINI_MODEL: str = "gemini-2.0-flash-lite"
    
    OPENROUTER_API_KEY: str = None
    OPENROUTER_MODEL: str = "meta-llama/llama-3.1-8b-instruct:free"
    
    OPENAI_API_KEY: str = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    
    # Embedding model
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    def __post_init__(self):
        self.UPLOAD_DIR = self.BASE_DIR / "uploads"
        self.CACHE_DIR = self.BASE_DIR / ".cache"
        self.UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
        self.CACHE_DIR.mkdir(exist_ok=True, parents=True)
        
        # Load from environment
        self.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
        self.LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))
        
        # Ollama
        self.OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
        
        # Groq
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        
        # Gemini
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
        
        # OpenRouter
        self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        self.OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
        
        # OpenAI
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # Embedding
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    @property
    def provider(self) -> str:
        return self.LLM_PROVIDER
    
    @property
    def is_configured(self) -> bool:
        """Check if the selected provider is properly configured"""
        if self.LLM_PROVIDER == "ollama":
            return True  # Ollama just needs to be running
        elif self.LLM_PROVIDER == "groq":
            return bool(self.GROQ_API_KEY)
        elif self.LLM_PROVIDER == "gemini":
            return bool(self.GEMINI_API_KEY)
        elif self.LLM_PROVIDER == "openrouter":
            return bool(self.OPENROUTER_API_KEY)
        elif self.LLM_PROVIDER == "openai":
            return bool(self.OPENAI_API_KEY)
        return False
    
    @property
    def current_model(self) -> str:
        """Get the model name for the current provider"""
        models = {
            "ollama": self.OLLAMA_MODEL,
            "groq": self.GROQ_MODEL,
            "gemini": self.GEMINI_MODEL,
            "openrouter": self.OPENROUTER_MODEL,
            "openai": self.OPENAI_MODEL
        }
        return models.get(self.LLM_PROVIDER, "unknown")


config = Config()


class RateLimiter:
    """Simple rate limiter to avoid hitting API limits"""
    
    def __init__(self, calls_per_minute: int = 25, tokens_per_minute: int = 5000):
        self.calls_per_minute = calls_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.call_times = []
        self.token_counts = []
    
    def wait_if_needed(self, estimated_tokens: int = 500):
        now = time.time()
        minute_ago = now - 60
        
        self.call_times = [t for t in self.call_times if t > minute_ago]
        self.token_counts = [(t, c) for t, c in self.token_counts if t > minute_ago]
        
        if len(self.call_times) >= self.calls_per_minute:
            wait_time = self.call_times[0] - minute_ago + 1
            logger.info(f"Rate limit: waiting {wait_time:.1f}s")
            time.sleep(wait_time)
        
        recent_tokens = sum(c for _, c in self.token_counts)
        if recent_tokens + estimated_tokens > self.tokens_per_minute:
            if self.token_counts:
                wait_time = min(t for t, _ in self.token_counts) - minute_ago + 1
                logger.info(f"Rate limit: waiting {wait_time:.1f}s (tokens)")
                time.sleep(max(1, wait_time))
        
        self.call_times.append(time.time())
    
    def record_tokens(self, tokens: int):
        self.token_counts.append((time.time(), tokens))


class DocumentProcessor:
    """Document text extraction"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()
    
    @classmethod
    def extract_text(cls, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.txt':
                return cls.clean_text(file_path.read_text(encoding='utf-8', errors='ignore'))
            
            elif suffix == '.pdf':
                import fitz
                text_parts = []
                with fitz.open(str(file_path)) as doc:
                    for page in doc:
                        text_parts.append(page.get_text())
                return cls.clean_text(" ".join(text_parts))
            
            elif suffix == '.docx':
                from docx import Document
                doc = Document(str(file_path))
                return cls.clean_text(" ".join([p.text for p in doc.paragraphs if p.text.strip()]))
        
        except Exception as e:
            logger.error(f"Extract error {file_path}: {e}")
        
        return ""
    
    @staticmethod
    def is_jd_file(filename: str) -> bool:
        return any(x in filename.lower() for x in ['jd', 'job', 'description', 'position', 'role', 'requirement'])


class MultiProviderLLM:
    """LLM client supporting multiple providers via .env configuration"""
    
    def __init__(self):
        self.cache_dir = config.CACHE_DIR
        self.rate_limiter = RateLimiter(calls_per_minute=25, tokens_per_minute=5000)
        self.consecutive_errors = 0
    
    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.json"
    
    # =========================================================================
    # Provider-specific API calls
    # =========================================================================
    
    def _call_ollama(self, prompt: str, max_tokens: int = 4096) -> str:
        """Call local Ollama instance"""
        try:
            response = requests.post(
                f"{config.OLLAMA_HOST}/api/generate",
                json={
                    "model": config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_ctx": 4096,
                        "num_predict": max_tokens,
                        "seed": 42
                    }
                },
                timeout=config.LLM_TIMEOUT
            )
            
            if response.status_code == 200:
                self.consecutive_errors = 0
                return response.json().get("response", "")
            else:
                logger.error(f"Ollama error {response.status_code}: {response.text[:200]}")
                self.consecutive_errors += 1
                return ""
        except requests.ConnectionError:
            logger.error(f"Cannot connect to Ollama at {config.OLLAMA_HOST}. Run: ollama serve")
            self.consecutive_errors += 1
            return ""
        except Exception as e:
            logger.error(f"Ollama exception: {e}")
            self.consecutive_errors += 1
            return ""
    
    def _call_groq(self, prompt: str, max_tokens: int = 4096) -> str:
        """Call Groq API"""
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {config.GROQ_API_KEY}"
                },
                json={
                    "model": config.GROQ_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": max_tokens
                },
                timeout=config.LLM_TIMEOUT
            )
            
            if response.status_code == 200:
                self.consecutive_errors = 0
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 429:
                logger.warning("Groq rate limited (429)")
                self.consecutive_errors += 1
                return ""
            else:
                logger.error(f"Groq error {response.status_code}: {response.text[:200]}")
                self.consecutive_errors += 1
                return ""
        except Exception as e:
            logger.error(f"Groq exception: {e}")
            self.consecutive_errors += 1
            return ""
    
    def _call_gemini(self, prompt: str, max_tokens: int = 4096) -> str:
        """Call Gemini API"""
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent",
                headers={"Content-Type": "application/json"},
                params={"key": config.GEMINI_API_KEY},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.1, "maxOutputTokens": max_tokens}
                },
                timeout=config.LLM_TIMEOUT
            )
            
            if response.status_code == 200:
                self.consecutive_errors = 0
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
            elif response.status_code == 429:
                logger.warning("Gemini rate limited (429)")
                self.consecutive_errors += 1
                return ""
            else:
                logger.error(f"Gemini error {response.status_code}: {response.text[:200]}")
                self.consecutive_errors += 1
                return ""
        except Exception as e:
            logger.error(f"Gemini exception: {e}")
            self.consecutive_errors += 1
            return ""
    
    def _call_openrouter(self, prompt: str, max_tokens: int = 4096) -> str:
        """Call OpenRouter API"""
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                    "HTTP-Referer": "https://fastscreen.ai",
                    "X-Title": "FastScreen AI"
                },
                json={
                    "model": config.OPENROUTER_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": max_tokens
                },
                timeout=config.LLM_TIMEOUT
            )
            
            if response.status_code == 200:
                self.consecutive_errors = 0
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 429:
                logger.warning("OpenRouter rate limited (429)")
                self.consecutive_errors += 1
                return ""
            else:
                logger.error(f"OpenRouter error {response.status_code}: {response.text[:200]}")
                self.consecutive_errors += 1
                return ""
        except Exception as e:
            logger.error(f"OpenRouter exception: {e}")
            self.consecutive_errors += 1
            return ""
    
    def _call_openai(self, prompt: str, max_tokens: int = 4096) -> str:
        """Call OpenAI API"""
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {config.OPENAI_API_KEY}"
                },
                json={
                    "model": config.OPENAI_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": max_tokens
                },
                timeout=config.LLM_TIMEOUT
            )
            
            if response.status_code == 200:
                self.consecutive_errors = 0
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 429:
                logger.warning("OpenAI rate limited (429)")
                self.consecutive_errors += 1
                return ""
            else:
                logger.error(f"OpenAI error {response.status_code}: {response.text[:200]}")
                self.consecutive_errors += 1
                return ""
        except Exception as e:
            logger.error(f"OpenAI exception: {e}")
            self.consecutive_errors += 1
            return ""
    
    # =========================================================================
    # Main generate methods
    # =========================================================================
    
    def generate(self, prompt: str, cache_key: str = "", max_tokens: int = 4096) -> str:
        """Generate response using configured provider"""
        
        # Check cache first
        if cache_key:
            cache_path = self._cache_path(cache_key)
            if cache_path.exists():
                try:
                    logger.info("Cache hit")
                    return json.loads(cache_path.read_text()).get("response", "")
                except:
                    pass
        
        # Rate limiting (skip for local Ollama)
        if config.LLM_PROVIDER != "ollama":
            estimated_tokens = len(prompt) // 3 + max_tokens // 2
            self.rate_limiter.wait_if_needed(estimated_tokens)
        
        # Call the appropriate provider
        provider = config.LLM_PROVIDER
        logger.info(f"Calling {provider} ({config.current_model})...")
        
        if provider == "ollama":
            response = self._call_ollama(prompt, max_tokens)
        elif provider == "groq":
            response = self._call_groq(prompt, max_tokens)
        elif provider == "gemini":
            response = self._call_gemini(prompt, max_tokens)
        elif provider == "openrouter":
            response = self._call_openrouter(prompt, max_tokens)
        elif provider == "openai":
            response = self._call_openai(prompt, max_tokens)
        else:
            raise RuntimeError(f"Unknown provider: {provider}")
        
        # Record tokens for rate limiting
        if response and provider != "ollama":
            actual_tokens = len(prompt) // 4 + len(response) // 4
            self.rate_limiter.record_tokens(actual_tokens)
        
        # Cache response
        if cache_key and response:
            cache_path = self._cache_path(cache_key)
            cache_path.write_text(json.dumps({"response": response, "provider": provider}))
        
        logger.info(f"Response: {len(response)} chars")
        return response
    
    def generate_json(self, prompt: str, cache_key: str = "") -> Dict:
        """Generate and parse JSON response"""
        raw = self.generate(prompt, cache_key)
        if not raw:
            return {}
        
        # Try direct parse
        try:
            return json.loads(raw)
        except:
            pass
        
        # Try extracting JSON from response
        try:
            start = raw.find('{')
            end = raw.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = raw[start:end+1]
                
                # Clean common issues
                json_str = re.sub(r'//[^\n]*', '', json_str)
                json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
                json_str = re.sub(r',\s*]', ']', json_str)
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
                
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error: {e}")
        except Exception as e:
            logger.error(f"JSON extraction error: {e}")
        
        # Last resort: extract scores
        scores_match = re.search(r'"scores"\s*:\s*\[([\d,\s]+)\]', raw)
        if scores_match:
            try:
                scores = [int(s.strip()) for s in scores_match.group(1).split(',') if s.strip()]
                return {"scores": scores, "percentage_score": 0, "summary": "Partial parse"}
            except:
                pass
        
        return {}
    
    def check_connection(self) -> Dict[str, Any]:
        """Verify provider connection"""
        if not config.is_configured:
            return {"connected": False, "error": f"Provider '{config.LLM_PROVIDER}' not configured"}
        
        test_prompt = "Reply with exactly: {\"status\": \"ok\"}"
        
        try:
            response = self.generate(test_prompt)
            if response and "ok" in response.lower():
                return {"connected": True, "provider": config.provider, "model": config.current_model}
        except Exception as e:
            return {"connected": False, "error": str(e)}
        
        return {"connected": False, "error": "Invalid response"}


class SemanticFilter:
    """Phase 1: Semantic similarity screening"""
    
    def __init__(self):
        logger.info(f"Loading embeddings model ({config.EMBEDDING_MODEL})...")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("Embeddings ready")
    
    def filter(self, resumes: List[Dict], jd_text: str, llm: MultiProviderLLM) -> List[Dict]:
        if not resumes:
            return []
        
        logger.info(f"Phase 1: Filtering {len(resumes)} resumes...")
        
        jd_embedding = self.model.encode(jd_text[:2000])
        resume_embeddings = self.model.encode([r['content'][:1500] for r in resumes], show_progress_bar=False)
        
        similarities = cosine_similarity([jd_embedding], resume_embeddings)[0]
        
        scored_resumes = []
        for resume, sim in zip(resumes, similarities):
            scored_resumes.append({**resume, 'semantic_score': round(float(sim) * 100, 1)})
        
        scored_resumes.sort(key=lambda x: x['semantic_score'], reverse=True)
        
        scores_summary = [f"{r['name']}: {r['semantic_score']}%" for r in scored_resumes[:20]]
        
        prompt = f"""You are screening resumes for a job. Here are semantic similarity scores:

{chr(10).join(scores_summary)}

Job Description Summary:
{jd_text[:1000]}

How many candidates should proceed to detailed evaluation?
Be INCLUSIVE - it's better to evaluate more candidates than miss good ones.

Reply with ONLY valid JSON:
{{"pass_count": 10, "reasoning": "explanation"}}

JSON:"""

        result = llm.generate_json(prompt)
        pass_count = result.get("pass_count", len(scored_resumes))
        
        min_pass = max(5, int(len(scored_resumes) * 0.8))
        pass_count = max(min_pass, min(pass_count, len(scored_resumes)))
        
        logger.info(f"Phase 1: {pass_count} candidates passed")
        
        passed = scored_resumes[:pass_count]
        for i, r in enumerate(passed):
            r['phase1_rank'] = i + 1
        
        return passed


class LLMEvaluator:
    """Phase 2: Pure LLM evaluation"""
    
    def __init__(self, llm: MultiProviderLLM):
        self.llm = llm
    
    def analyze_job(self, jd_text: str) -> Dict:
        logger.info("Phase 2: Analyzing job description...")
        
        prompt = f"""Analyze this job description.

JOB DESCRIPTION:
{jd_text}

Return ONLY valid JSON:
{{"job_title": "title", "seniority_level": "senior", "job_type": "engineering", "must_have_skills": [{{"skill": "Python", "importance": "critical", "years_preferred": 3}}], "important_skills": [{{"skill": "AWS", "importance": "high"}}], "nice_to_have": [{{"skill": "Go", "importance": "bonus"}}], "evaluation_criteria": [{{"criterion": "Question?", "weight": 3, "category": "technical", "what_to_look_for": "evidence"}}], "red_flags": ["flag1"], "green_flags": ["flag1"], "evaluation_notes": "notes"}}

Create 12-20 evaluation_criteria based on actual job requirements.
weight: 1-5 (5=critical), category: technical/experience/domain/education/soft_skills

JSON:"""

        cache_key = f"job_analysis_{hashlib.md5(jd_text[:500].encode()).hexdigest()}"
        return self.llm.generate_json(prompt, cache_key)
    
    def evaluate_candidate(self, candidate: Dict, job_analysis: Dict, jd_text: str) -> Dict:
        logger.info(f"  Evaluating: {candidate['name']}...")
        
        criteria = job_analysis.get("evaluation_criteria", [])
        criteria_text = "\n".join([
            f"{i+1}. [W{c.get('weight', 1)}] {c.get('criterion', '')} - Look for: {c.get('what_to_look_for', '')}"
            for i, c in enumerate(criteria)
        ])
        
        red_flags = job_analysis.get("red_flags", [])
        green_flags = job_analysis.get("green_flags", [])
        
        prompt = f"""Evaluate candidate for: {job_analysis.get('job_title', 'this position')}

CRITERIA:
{criteria_text}

RED FLAGS: {', '.join(red_flags) if red_flags else 'None'}
GREEN FLAGS: {', '.join(green_flags) if green_flags else 'None'}

RESUME:
{candidate['content'][:4500]}

Score 0-5 per criterion:
5=Exceptional, 4=Strong, 3=Good, 2=Weak/vague, 1=Minimal, 0=None

Rules: Listed without context = max 2. "familiar with" = max 2.

Return ONLY valid JSON:
{{"scores": [3,4,2,5,3,4,2,3,4,3,2,4], "total_weighted_score": 45, "max_possible_score": 60, "percentage_score": 75, "strengths": ["s1", "s2"], "weaknesses": ["w1", "w2"], "red_flags_found": [], "green_flags_found": [], "experience_match": "strong", "skills_match": "moderate", "overall_fit": "good", "summary": "summary", "recommendation": "recommend"}}

JSON:"""

        cache_key = f"eval_{candidate['name']}_{hashlib.md5(jd_text[:100].encode()).hexdigest()}"
        result = self.llm.generate_json(prompt, cache_key)
        
        if not result or not result.get('scores'):
            logger.warning(f"    Retrying {candidate['name']}...")
            cache_path = self.llm._cache_path(cache_key)
            if cache_path.exists():
                cache_path.unlink()
            result = self.llm.generate_json(prompt, "")
        
        if not result:
            return {**candidate, 'final_score': 0, 'recommendation': 'do_not_recommend',
                    'overall_fit': 'not_qualified', 'summary': 'Evaluation failed'}
        
        criteria_list = job_analysis.get("evaluation_criteria", [])
        scores = result.get("scores", [])
        score_breakdown = {}
        
        for i, criterion in enumerate(criteria_list):
            score = scores[i] if i < len(scores) else 0
            score_breakdown[criterion.get("criterion", f"Criterion {i+1}")] = {
                "score": score,
                "weight": criterion.get("weight", 1),
                "category": criterion.get("category", "general")
            }
        
        final_score = result.get("percentage_score", 0)
        logger.info(f"    {candidate['name']}: {final_score}/100 - {result.get('recommendation', 'N/A')}")
        
        return {
            **candidate,
            'final_score': final_score,
            'total_weighted': result.get("total_weighted_score", 0),
            'max_weighted': result.get("max_possible_score", 0),
            'score_breakdown': score_breakdown,
            'strengths': result.get("strengths", []),
            'weaknesses': result.get("weaknesses", []),
            'red_flags_found': result.get("red_flags_found", []),
            'green_flags_found': result.get("green_flags_found", []),
            'experience_match': result.get("experience_match", ""),
            'skills_match': result.get("skills_match", ""),
            'overall_fit': result.get("overall_fit", ""),
            'summary': result.get("summary", ""),
            'recommendation': result.get("recommendation", "")
        }
    
    def rank_candidates(self, evaluated: List[Dict], job_analysis: Dict):
        logger.info("Phase 2: Final ranking...")
        
        candidates_summary = [{
            "name": c['name'],
            "score": c.get('final_score', 0),
            "recommendation": c.get('recommendation', ''),
            "overall_fit": c.get('overall_fit', ''),
            "strengths": c.get('strengths', [])[:2],
            "weaknesses": c.get('weaknesses', [])[:2]
        } for c in evaluated]
        
        prompt = f"""Rank these candidates for: {job_analysis.get('job_title', 'position')}

Candidates:
{json.dumps(candidates_summary, indent=2)}

Return ONLY valid JSON:
{{"rankings": [{{"rank": 1, "name": "name", "final_score": 85, "ranking_reason": "reason"}}], "top_candidate_summary": "why best", "hiring_recommendation": "recommendation"}}

Include ALL {len(candidates_summary)} candidates.

JSON:"""

        result = self.llm.generate_json(prompt)
        rankings = result.get("rankings", [])
        
        rank_map = {r["name"]: r["rank"] for r in rankings}
        
        for candidate in evaluated:
            candidate['final_rank'] = rank_map.get(candidate['name'], len(evaluated))
            for r in rankings:
                if r.get("name") == candidate['name']:
                    candidate['ranking_reason'] = r.get("ranking_reason", "")
                    break
        
        evaluated.sort(key=lambda x: x.get('final_rank', 999))
        
        return evaluated, result.get("top_candidate_summary", ""), result.get("hiring_recommendation", "")


class FastScreenAI:
    """Main Application - Multi-Provider Support"""
    
    def __init__(self):
        if not config.is_configured:
            raise RuntimeError(f"Provider '{config.LLM_PROVIDER}' not configured. Check your .env file.")
        
        logger.info(f"Initializing FastScreen AI ({config.provider}: {config.current_model})...")
        self.llm = MultiProviderLLM()
        self.semantic_filter = SemanticFilter()
        self.evaluator = LLMEvaluator(self.llm)
        logger.info("FastScreen AI ready")
    
    def clear_cache(self):
        if config.CACHE_DIR.exists():
            count = 0
            for f in config.CACHE_DIR.glob("*.json"):
                f.unlink()
                count += 1
            logger.info(f"Cleared {count} cached files")
    
    def check_system(self) -> Dict[str, Any]:
        llm_status = self.llm.check_connection()
        return {
            'provider': config.provider,
            'model': config.current_model,
            'llm_connected': llm_status.get('connected', False),
            'llm_error': llm_status.get('error'),
            'embeddings_ready': self.semantic_filter.model is not None
        }
    
    def process_zip(self, zip_path: Path) -> Dict[str, Any]:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        extract_dir = config.UPLOAD_DIR / f"batch_{timestamp}"
        extract_dir.mkdir(exist_ok=True, parents=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
            
            jd_text = ""
            resumes = []
            
            for file_path in extract_dir.rglob("*"):
                if file_path.name.startswith('.') or file_path.name.startswith('_'):
                    continue
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in {'.txt', '.pdf', '.docx'}:
                    continue
                
                text = DocumentProcessor.extract_text(file_path)
                if not text:
                    continue
                
                if DocumentProcessor.is_jd_file(file_path.name):
                    jd_text = text
                    logger.info(f"Found JD: {file_path.name}")
                else:
                    resumes.append({
                        'name': file_path.stem,
                        'filename': file_path.name,
                        'content': text
                    })
                    logger.info(f"Found resume: {file_path.name}")
            
            if not jd_text:
                return {'error': 'No job description found.'}
            if not resumes:
                return {'error': 'No resumes found.'}
            
            return self.screen(resumes, jd_text)
            
        except zipfile.BadZipFile:
            return {'error': 'Invalid ZIP file.'}
        except Exception as e:
            logger.error(f"Process error: {e}")
            return {'error': str(e)}
    
    def screen(self, resumes: List[Dict], jd_text: str) -> Dict[str, Any]:
        start_time = datetime.now()
        
        phase1_results = self.semantic_filter.filter(resumes, jd_text, self.llm)
        
        if not phase1_results:
            return {'error': 'No candidates passed initial screening.'}
        
        job_analysis = self.evaluator.analyze_job(jd_text)
        
        if not job_analysis:
            return {'error': 'Failed to analyze job description.'}
        
        evaluated = []
        for candidate in phase1_results:
            result = self.evaluator.evaluate_candidate(candidate, job_analysis, jd_text)
            evaluated.append(result)
        
        ranked, top_summary, hiring_rec = self.evaluator.rank_candidates(evaluated, job_analysis)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'processing_time_seconds': round(duration, 1),
            'provider': config.provider,
            'model': config.current_model,
            'total_candidates': len(resumes),
            'phase1_passed': len(phase1_results),
            'evaluated': len(ranked),
            'job_analysis': job_analysis,
            'results': ranked,
            'top_candidate_summary': top_summary,
            'hiring_recommendation': hiring_rec
        }


if __name__ == "__main__":
    print("=" * 60)
    print("FastScreen AI - Multi-Provider Configuration")
    print("=" * 60)
    print(f"\nProvider: {config.provider}")
    print(f"Model: {config.current_model}")
    print(f"Configured: {config.is_configured}")
    
    if config.is_configured:
        screener = FastScreenAI()
        status = screener.check_system()
        print(f"\n✓ Provider: {status['provider']}")
        print(f"✓ Model: {status['model']}")
        print(f"✓ LLM Connected: {status['llm_connected']}")
        print(f"✓ Embeddings: {'Ready' if status['embeddings_ready'] else 'Not Ready'}")
    else:
        print(f"\n✗ Provider '{config.provider}' not configured!")
        print("\nCheck your .env file and ensure the required API key is set.")