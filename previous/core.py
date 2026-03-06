"""
FastScreen AI - Agentic Node Architecture
Pure LLM evaluation system
"""
import json
import re
import hashlib
import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum
from pathlib import Path

import time
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()


class CleanFormatter(logging.Formatter):
    def format(self, record):
        return record.getMessage()

def setup_logger():
    logger = logging.getLogger("FastScreen")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CleanFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    
    for name in ['sentence_transformers', 'transformers', 'urllib3', 'httpx']:
        logging.getLogger(name).setLevel(logging.WARNING)
    
    return logger

logger = setup_logger()


EVALUATION_CRITERIA_COUNT = int(os.environ.get('EVAL_CRITERIA_COUNT', '6'))

@dataclass(frozen=True)
class Config:
    base_dir: Path
    upload_dir: Path
    cache_dir: Path
    llm_provider: str
    api_key: str
    model_name: str
    llm_timeout: int
    embedding_model: str
    eval_criteria_count: int

    @classmethod
    def from_env(cls) -> 'Config':
        base_dir = Path(__file__).parent
        provider = os.environ.get('LLM_PROVIDER', 'groq').lower()
        
        provider_configs = {
            'groq': ('GROQ_API_KEY', 'GROQ_MODEL', 'llama-3.1-8b-instant'),
            'gemini': ('GEMINI_API_KEY', 'GEMINI_MODEL', 'gemini-2.0-flash-lite'),
            'openai': ('OPENAI_API_KEY', 'OPENAI_MODEL', 'gpt-4o-mini'),
            'ollama': (None, 'OLLAMA_MODEL', 'llama3.1:8b'),
            'openrouter': ('OPENROUTER_API_KEY', 'OPENROUTER_MODEL', 'meta-llama/llama-3.1-8b-instruct:free'),
        }
        
        if provider not in provider_configs:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        key_env, model_env, default_model = provider_configs[provider]
        api_key = 'not-needed' if provider == 'ollama' else os.environ.get(key_env)
        model_name = os.environ.get(model_env, default_model)
        
        if not api_key:
            raise ValueError(f"API key not found for provider '{provider}'. Set {key_env} environment variable.")
        
        upload_dir = base_dir / "uploads"
        cache_dir = base_dir / ".cache"
        upload_dir.mkdir(exist_ok=True, parents=True)
        cache_dir.mkdir(exist_ok=True, parents=True)
        
        return cls(
            base_dir=base_dir,
            upload_dir=upload_dir,
            cache_dir=cache_dir,
            llm_provider=provider,
            api_key=api_key,
            model_name=model_name,
            llm_timeout=int(os.environ.get('LLM_TIMEOUT', '120')),
            embedding_model=os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
            eval_criteria_count=EVALUATION_CRITERIA_COUNT
        )


def load_config() -> Config:
    return Config.from_env()


def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()


def extract_text_from_pdf_bytes(pdf_bytes: bytes, filename: str = "document.pdf") -> str:
    import fitz
    text_parts = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            page_text = page.get_text("text")
            if page_text.strip():
                text_parts.append(page_text)
    if not text_parts:
        raise ValueError(f"No text content extracted from PDF: {filename}")
    return clean_text(" ".join(text_parts))


def extract_text_from_docx_bytes(docx_bytes: bytes, filename: str = "document.docx") -> str:
    from docx import Document
    import io
    doc = Document(io.BytesIO(docx_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        raise ValueError(f"No text content extracted from DOCX: {filename}")
    return clean_text(" ".join(paragraphs))


def extract_text_from_txt_bytes(txt_bytes: bytes, filename: str = "document.txt") -> str:
    text = txt_bytes.decode('utf-8', errors='ignore')
    if not text.strip():
        raise ValueError(f"No text content in TXT file: {filename}")
    return clean_text(text)


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    extractors = {
        '.pdf': extract_text_from_pdf_bytes,
        '.docx': extract_text_from_docx_bytes,
        '.txt': extract_text_from_txt_bytes,
    }
    extractor = extractors.get(suffix)
    if not extractor:
        raise ValueError(f"Unsupported file type: {suffix}")
    return extractor(file_bytes, filename)


def is_jd_file(filename: str) -> bool:
    jd_indicators = ['jd', 'job', 'description', 'position', 'role', 'requirement', 'posting']
    filename_lower = filename.lower()
    return any(indicator in filename_lower for indicator in jd_indicators)


class LLMClient:
    MAX_RETRIES = 10
    BASE_DELAY = 10.0
    PROVIDER_INTERVALS = {'groq': 12.0, 'gemini': 5.0, 'openai': 1.0}
    
    def __init__(self, config: Config):
        self.config = config
        self.cache_dir = config.cache_dir
        self._call_fn = self._get_provider_fn()
        self._last_call_time = 0
        default_interval = self.PROVIDER_INTERVALS.get(config.llm_provider, 2.0)
        self._min_call_interval = float(os.environ.get('LLM_CALL_INTERVAL', str(default_interval)))
    
    def _get_provider_fn(self) -> Callable[[str, int], str]:
        providers = {
            'groq': self._call_groq,
            'gemini': self._call_gemini,
            'openai': self._call_openai,
            'ollama': self._call_ollama,
            'openrouter': self._call_openrouter,
        }
        fn = providers.get(self.config.llm_provider)
        if not fn:
            raise ValueError(f"Unsupported provider: {self.config.llm_provider}")
        return fn
    
    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.json"
    
    def _wait_for_rate_limit(self) -> None:
        elapsed = time.time() - self._last_call_time
        if elapsed < self._min_call_interval:
            time.sleep(self._min_call_interval - elapsed)
    
    def _call_with_retry(self, call_fn: Callable[[], requests.Response], provider: str) -> str:
        for attempt in range(self.MAX_RETRIES):
            self._wait_for_rate_limit()
            try:
                response = call_fn()
                self._last_call_time = time.time()
                
                if response.status_code == 200:
                    return self._extract_response(response, provider)
                elif response.status_code == 429:
                    wait_time = 15.0 + (attempt * 5) if provider.lower() == 'groq' else self.BASE_DELAY * (1.5 ** attempt)
                    wait_time = min(wait_time, 60.0)
                    time.sleep(wait_time)
                    continue
                else:
                    try:
                        error_detail = response.json().get('error', {}).get('message', response.text)
                    except:
                        error_detail = response.text[:500]
                    raise RuntimeError(f"{provider} API error ({response.status_code}): {error_detail}")
            except requests.exceptions.Timeout:
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.BASE_DELAY * (2 ** attempt))
                    continue
                raise RuntimeError(f"{provider} API timeout after {self.MAX_RETRIES} attempts")
        raise RuntimeError(f"{provider} API rate limit exceeded after {self.MAX_RETRIES} retries.")
    
    def _extract_response(self, response: requests.Response, provider: str) -> str:
        data = response.json()
        if provider.lower() == 'gemini':
            if 'error' in data:
                raise RuntimeError(f"Gemini error: {data['error'].get('message', data['error'])}")
            candidates = data.get('candidates', [])
            if not candidates:
                raise RuntimeError("Gemini returned no candidates")
            parts = candidates[0].get('content', {}).get('parts', [])
            if not parts:
                raise RuntimeError("Gemini returned empty response")
            return parts[0].get('text', '')
        else:
            if 'error' in data:
                raise RuntimeError(f"{provider} error: {data['error'].get('message', data['error'])}")
            choices = data.get('choices', [])
            if not choices:
                raise RuntimeError(f"{provider} returned no choices")
            return choices[0].get('message', {}).get('content', '')
    
    def _call_groq(self, prompt: str, max_tokens: int) -> str:
        def make_request():
            return requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.config.api_key}"},
                json={"model": self.config.model_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": max_tokens},
                timeout=self.config.llm_timeout
            )
        return self._call_with_retry(make_request, "groq")
    
    def _call_gemini(self, prompt: str, max_tokens: int) -> str:
        def make_request():
            return requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.model_name}:generateContent",
                headers={"Content-Type": "application/json"},
                params={"key": self.config.api_key},
                json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.1, "maxOutputTokens": max_tokens}},
                timeout=self.config.llm_timeout
            )
        return self._call_with_retry(make_request, "gemini")
    
    def _call_openai(self, prompt: str, max_tokens: int) -> str:
        def make_request():
            return requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.config.api_key}"},
                json={"model": self.config.model_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": max_tokens},
                timeout=self.config.llm_timeout
            )
        return self._call_with_retry(make_request, "openai")
    
    def _call_ollama(self, prompt: str, max_tokens: int) -> str:
        ollama_host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
        self._wait_for_rate_limit()
        response = requests.post(
            f"{ollama_host}/api/generate",
            json={"model": self.config.model_name, "prompt": prompt, "stream": False, "options": {"temperature": 0.1, "num_predict": max_tokens}},
            timeout=self.config.llm_timeout
        )
        self._last_call_time = time.time()
        if response.status_code == 200:
            return response.json().get('response', '')
        raise RuntimeError(f"Ollama error ({response.status_code}): {response.text[:500]}")
    
    def _call_openrouter(self, prompt: str, max_tokens: int) -> str:
        def make_request():
            return requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.config.api_key}", "HTTP-Referer": "https://github.com/fastscreen-ai", "X-Title": "FastScreen AI"},
                json={"model": self.config.model_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": max_tokens},
                timeout=self.config.llm_timeout
            )
        return self._call_with_retry(make_request, "openrouter")
    
    def generate(self, prompt: str, cache_key: str = "", max_tokens: int = 2048) -> str:
        if cache_key:
            cache_path = self._cache_path(cache_key)
            if cache_path.exists():
                return json.loads(cache_path.read_text()).get("response", "")
        
        response = self._call_fn(prompt, max_tokens)
        
        if cache_key and response:
            self._cache_path(cache_key).write_text(json.dumps({"response": response}))
        return response
    
    def generate_json(self, prompt: str, cache_key: str = "", max_tokens: int = 2048) -> Dict[str, Any]:
        raw = self.generate(prompt, cache_key, max_tokens)
        if not raw:
            raise RuntimeError("Empty response from LLM")
        
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        
        start, end = raw.find('{'), raw.rfind('}')
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(f"No valid JSON found in response: {raw[:200]}...")
        
        json_str = raw[start:end + 1]
        json_str = re.sub(r'(\d)\s*,?\s*#[^\n\]]*', r'\1,', json_str)
        json_str = re.sub(r'^\s*#[^\n]*\n', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'//[^\n]*', '', json_str)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        json_str = re.sub(r',\s*,+', ',', json_str)
        json_str = re.sub(r'\[\s*,', '[', json_str)
        json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            open_braces = json_str.count('{') - json_str.count('}')
            open_brackets = json_str.count('[') - json_str.count(']')
            if open_braces > 0 or open_brackets > 0:
                json_str = json_str.rstrip().rstrip(',')
                json_str += ']' * open_brackets + '}' * open_braces
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            raise RuntimeError(f"Failed to parse JSON from response")
    
    def check_connection(self) -> Dict[str, Any]:
        try:
            response = self.generate('Reply with only: ok', max_tokens=50)
            if response:
                return {"connected": True, "provider": self.config.llm_provider, "model": self.config.model_name}
            return {"connected": False, "error": "Empty response"}
        except Exception as e:
            return {"connected": False, "error": str(e)}


class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class NodeResult:
    node_name: str
    status: NodeStatus
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    execution_time_ms: float = 0
    
    @property
    def success(self) -> bool:
        return self.status == NodeStatus.COMPLETED


@dataclass
class PipelineContext:
    job_description: str
    resumes: List[Dict[str, Any]]
    node_results: Dict[str, NodeResult] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    
    def get_result(self, node_name: str) -> Optional[NodeResult]:
        return self.node_results.get(node_name)
    
    def set_result(self, result: NodeResult) -> None:
        self.node_results[result.node_name] = result


class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        pass


class Node(ABC):
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self._tools: Dict[str, Tool] = {}
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    def tools(self) -> List[str]:
        return list(self._tools.keys())
    
    def register_tool(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
    
    def use_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not registered")
        return self._tools[tool_name].execute(**kwargs)
    
    @abstractmethod
    def process(self, context: PipelineContext) -> NodeResult:
        pass
    
    def run(self, context: PipelineContext) -> NodeResult:
        start = datetime.now()
        try:
            result = self.process(context)
            result.execution_time_ms = (datetime.now() - start).total_seconds() * 1000
            return result
        except Exception as e:
            return NodeResult(
                node_name=self.name,
                status=NodeStatus.FAILED,
                data={},
                errors=[str(e)],
                execution_time_ms=(datetime.now() - start).total_seconds() * 1000
            )


class SemanticSearchTool(Tool):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self._model = None
        self._model_name = model_name
    
    @property
    def name(self) -> str:
        return "semantic_search"
    
    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model
    
    def execute(self, query: str, documents: List[Dict], top_k: int = 50) -> Dict[str, Any]:
        if not documents or not query:
            raise ValueError("Query and documents required")
        
        query_embedding = self.model.encode(query[:2000])
        doc_texts = [d.get('content', '')[:1500] for d in documents]
        doc_embeddings = self.model.encode(doc_texts, show_progress_bar=False)
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        scored = [{**doc, 'semantic_score': round(float(sim) * 100, 2)} for doc, sim in zip(documents, similarities)]
        scored.sort(key=lambda x: x['semantic_score'], reverse=True)
        return {"results": scored[:top_k], "total_processed": len(documents)}


class KeywordMatchingTool(Tool):
    @property
    def name(self) -> str:
        return "keyword_matching"
    
    def execute(self, keywords: List[str], documents: List[Dict], weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        if not documents or not keywords:
            raise ValueError("Keywords and documents required")
        
        weights = weights or {k: 1.0 for k in keywords}
        keywords_lower = {k.lower(): w for k, w in weights.items()}
        max_possible = sum(keywords_lower.values())
        
        scored = []
        for doc in documents:
            content_lower = doc.get('content', '').lower()
            matches, total_score = {}, 0
            for keyword, weight in keywords_lower.items():
                if keyword in content_lower:
                    count = content_lower.count(keyword)
                    match_score = min(1.0, 0.5 + (count * 0.1)) * weight
                    matches[keyword] = {"count": count, "score": round(match_score, 3)}
                    total_score += match_score
            keyword_score = (total_score / max_possible * 100) if max_possible > 0 else 0
            scored.append({**doc, 'keyword_score': round(keyword_score, 2), 'keyword_matches': matches, 'keywords_found': len(matches)})
        
        scored.sort(key=lambda x: x['keyword_score'], reverse=True)
        return {"results": scored, "keywords_used": list(keywords_lower.keys())}


class HybridFilteringNode(Node):
    def __init__(self, llm: LLMClient, embedding_model: str = 'all-MiniLM-L6-v2'):
        super().__init__(llm)
        self.register_tool(SemanticSearchTool(embedding_model))
        self.register_tool(KeywordMatchingTool())
    
    @property
    def name(self) -> str:
        return "hybrid_filtering"
    
    def process(self, context: PipelineContext) -> NodeResult:
        jd = context.job_description
        resumes = context.resumes
        
        prompt = f"""Extract keywords from this job description for resume screening.

JOB DESCRIPTION:
{jd[:3000]}

Return ONLY valid JSON (no comments, no trailing commas):
{{
  "must_have": ["4-5 critical required skills"],
  "important": ["3-4 preferred skills"],
  "nice_to_have": ["2-3 bonus skills"]
}}

JSON:"""
        
        cache_key = f"kw_{hashlib.md5(jd[:500].encode()).hexdigest()}"
        keywords_data = self.llm.generate_json(prompt, cache_key, max_tokens=512)
        
        all_keywords = keywords_data.get('must_have', []) + keywords_data.get('important', []) + keywords_data.get('nice_to_have', [])
        if not all_keywords:
            raise RuntimeError("LLM failed to extract keywords from job description")
        
        weighted = {}
        for kw in keywords_data.get('must_have', []): weighted[kw] = 3.0
        for kw in keywords_data.get('important', []): weighted[kw] = 2.0
        for kw in keywords_data.get('nice_to_have', []): weighted[kw] = 1.0
        
        semantic_results = self.use_tool("semantic_search", query=jd, documents=resumes, top_k=len(resumes))
        keyword_results = self.use_tool("keyword_matching", keywords=list(weighted.keys()), documents=resumes, weights=weighted)
        
        semantic_map = {r['name']: r for r in semantic_results['results']}
        keyword_map = {r['name']: r for r in keyword_results['results']}
        
        merged = []
        for resume in resumes:
            name = resume['name']
            sem_data = semantic_map.get(name, {})
            kw_data = keyword_map.get(name, {})
            merged.append({
                **resume,
                'semantic_score': sem_data.get('semantic_score', 0),
                'keyword_score': kw_data.get('keyword_score', 0),
                'keyword_matches': kw_data.get('keyword_matches', {}),
                'keywords_found': kw_data.get('keywords_found', 0)
            })
        
        return NodeResult(
            node_name=self.name, status=NodeStatus.COMPLETED,
            data={"candidates": merged, "keywords_extracted": keywords_data, "weighted_keywords": weighted},
            metadata={"total_candidates": len(resumes), "keywords_count": len(weighted)}
        )


class RankFusionNode(Node):
    RRF_K = 60
    
    @property
    def name(self) -> str:
        return "rank_fusion"
    
    def process(self, context: PipelineContext) -> NodeResult:
        prev_result = context.get_result("hybrid_filtering")
        if not prev_result or not prev_result.success:
            raise RuntimeError("Hybrid filtering result required")
        
        candidates = prev_result.data["candidates"]
        
        by_semantic = sorted(candidates, key=lambda x: x['semantic_score'], reverse=True)
        semantic_ranks = {c['name']: i + 1 for i, c in enumerate(by_semantic)}
        
        by_keyword = sorted(candidates, key=lambda x: x['keyword_score'], reverse=True)
        keyword_ranks = {c['name']: i + 1 for i, c in enumerate(by_keyword)}
        
        fused = []
        for c in candidates:
            sem_rank = semantic_ranks[c['name']]
            kw_rank = keyword_ranks[c['name']]
            rrf_score = (1 / (self.RRF_K + sem_rank)) + (1 / (self.RRF_K + kw_rank))
            fused.append({**c, 'semantic_rank': sem_rank, 'keyword_rank': kw_rank, 'rrf_score': round(rrf_score, 6)})
        
        fused.sort(key=lambda x: x['rrf_score'], reverse=True)
        for i, c in enumerate(fused):
            c['phase1_rank'] = i + 1
        
        return NodeResult(
            node_name=self.name, status=NodeStatus.COMPLETED,
            data={"all_candidates": fused, "passed_candidates": fused},
            metadata={"total": len(fused), "passed": len(fused)}
        )


class QuestionGenerationNode(Node):
    @property
    def name(self) -> str:
        return "question_generation"
    
    def process(self, context: PipelineContext) -> NodeResult:
        jd = context.job_description
        criteria_count = context.config.get('eval_criteria_count', EVALUATION_CRITERIA_COUNT)
        
        prompt = f"""Analyze this job description and create exactly {criteria_count} evaluation criteria.

JOB DESCRIPTION:
{jd[:3500]}

Return ONLY valid JSON (no comments, no trailing commas):
{{
  "job_title": "title from JD",
  "seniority": "entry/mid/senior",
  "evaluation_criteria": [
    {{
      "id": 1,
      "criterion": "Clear, specific skill or requirement",
      "category": "critical|important|preferred",
      "weight": 5,
      "what_counts_as_5": "Expert level description",
      "what_counts_as_0": "No evidence"
    }}
  ]
}}

IMPORTANT:
- Create exactly {criteria_count} criteria
- Criterion text should be clear skill descriptions WITHOUT any weight numbers
- Weight values: 5=critical, 3=important, 1=preferred
- Focus on specific, measurable skills from the JD

JSON:"""

        cache_key = f"rubric_{criteria_count}_{hashlib.md5(jd[:500].encode()).hexdigest()}"
        result = self.llm.generate_json(prompt, cache_key, max_tokens=1024)
        
        criteria = result.get("evaluation_criteria", [])
        if len(criteria) < 3:
            raise RuntimeError(f"LLM generated insufficient criteria: {len(criteria)} (minimum 3 required)")
        
        job_title = result.get("job_title")
        if not job_title:
            raise RuntimeError("LLM failed to extract job title from job description")
        
        seniority = result.get("seniority")
        if not seniority:
            raise RuntimeError("LLM failed to determine seniority level from job description")
        
        return NodeResult(
            node_name=self.name, status=NodeStatus.COMPLETED,
            data={
                "job_title": job_title,
                "seniority": seniority,
                "evaluation_criteria": criteria[:criteria_count]
            },
            metadata={"criteria_count": len(criteria)}
        )


class ResumeRankingNode(Node):
    @property
    def name(self) -> str:
        return "resume_ranking"
    
    def process(self, context: PipelineContext) -> NodeResult:
        fusion_result = context.get_result("rank_fusion")
        questions_result = context.get_result("question_generation")
        
        if not fusion_result or not fusion_result.success:
            raise RuntimeError("Rank fusion result required")
        if not questions_result or not questions_result.success:
            raise RuntimeError("Question generation result required")
        
        candidates = fusion_result.data["passed_candidates"]
        criteria = questions_result.data["evaluation_criteria"]
        job_title = questions_result.data["job_title"]
        
        evaluated = []
        
        for candidate in candidates:
            criteria_text = "\n".join([f"{i+1}. {c['criterion']}" for i, c in enumerate(criteria)])
            
            prompt = f"""Evaluate candidate "{candidate['name']}" for: {job_title}

EVALUATION CRITERIA (score each 0-5):
{criteria_text}

CANDIDATE RESUME ({candidate['name']}):
{candidate['content'][:4000]}

SCORING GUIDE:
- 5: Expert level, proven production/work experience
- 4: Strong experience with projects
- 3: Moderate experience, academic projects
- 2: Basic familiarity, some exposure
- 1: Minimal mention or tangential
- 0: No evidence at all

IMPORTANT: 
- Score EACH criterion independently based on evidence in THIS specific resume
- Look for transferable skills and related experience
- This is {candidate['name']}'s resume - evaluate THEIR specific experience

Return ONLY valid JSON (no comments, no trailing commas):
{{
  "scores": [{len(criteria)} integers from 0-5, one for each criterion in order],
  "strengths": ["specific strength from resume", "another strength"],
  "weaknesses": ["gap or missing skill"],
  "overall_fit": "excellent|good|moderate|poor",
  "summary": "2 sentence assessment of {candidate['name']}'s fit for this role"
}}

JSON:"""

            result = self.llm.generate_json(prompt, max_tokens=768)
            
            scores = result.get("scores")
            if scores is None:
                raise RuntimeError(f"LLM failed to generate scores for candidate: {candidate['name']}")
            
            if len(scores) != len(criteria):
                raise RuntimeError(f"LLM returned {len(scores)} scores but {len(criteria)} criteria exist for candidate: {candidate['name']}")
            
            for i, s in enumerate(scores):
                if not isinstance(s, (int, float)) or s < 0 or s > 5:
                    raise RuntimeError(f"Invalid score '{s}' at position {i} for candidate: {candidate['name']}")
            
            scores = [max(0, min(5, int(s))) for s in scores]
            
            strengths = result.get("strengths")
            if strengths is None:
                raise RuntimeError(f"LLM failed to identify strengths for candidate: {candidate['name']}")
            
            weaknesses = result.get("weaknesses")
            if weaknesses is None:
                raise RuntimeError(f"LLM failed to identify weaknesses for candidate: {candidate['name']}")
            
            overall_fit = result.get("overall_fit")
            if overall_fit is None:
                raise RuntimeError(f"LLM failed to determine overall fit for candidate: {candidate['name']}")
            
            summary = result.get("summary")
            if summary is None:
                raise RuntimeError(f"LLM failed to generate summary for candidate: {candidate['name']}")
            
            total_weighted = sum(s * c.get("weight", 1) for s, c in zip(scores, criteria))
            max_weighted = sum(c.get("weight", 1) * 5 for c in criteria)
            final_score = round((total_weighted / max_weighted) * 100) if max_weighted > 0 else 0
            
            breakdown = {c["criterion"]: {"score": s, "weight": c.get("weight", 1)} for s, c in zip(scores, criteria)}
            
            evaluated.append({
                **candidate,
                "raw_scores": scores,
                "score_breakdown": breakdown,
                "final_score": final_score,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "red_flags": result.get("red_flags", []),
                "overall_fit": overall_fit,
                "summary": summary
            })
        
        evaluated.sort(key=lambda x: x['final_score'], reverse=True)
        for i, c in enumerate(evaluated): c['eval_rank'] = i + 1
        
        return NodeResult(
            node_name=self.name, status=NodeStatus.COMPLETED,
            data={"evaluated_candidates": evaluated},
            metadata={"evaluated_count": len(evaluated)}
        )


class ReportGenerationNode(Node):
    @property
    def name(self) -> str:
        return "report_generation"
    
    def process(self, context: PipelineContext) -> NodeResult:
        ranking_result = context.get_result("resume_ranking")
        questions_result = context.get_result("question_generation")
        
        if not ranking_result or not ranking_result.success:
            raise RuntimeError("Resume ranking result required")
        
        evaluated = ranking_result.data["evaluated_candidates"]
        job_title = questions_result.data["job_title"]
        
        summary_data = [{"name": c['name'], "score": c['final_score'], "fit": c.get('overall_fit', ''), "strengths": c.get('strengths', [])[:2]} for c in evaluated[:10]]
        
        prompt = f"""Create final rankings for: {job_title}

CANDIDATES (with their automated scores):
{json.dumps(summary_data, indent=2)}

YOUR TASK:
1. Review each candidate's score and strengths
2. Rank them from best to worst fit for the role
3. Candidates with higher scores should generally rank higher
4. Provide specific reasons for each ranking

Return ONLY valid JSON (no comments, no trailing commas):
{{
  "final_rankings": [
    {{"rank": 1, "name": "exact_candidate_name", "recommendation": "strongly_recommend|recommend|consider|do_not_recommend", "ranking_reason": "specific reason based on their skills"}}
  ],
  "top_candidate_summary": "Why the #1 candidate is best fit",
  "hiring_recommendation": "Overall hiring advice"
}}

IMPORTANT: 
- Include ALL {len(evaluated)} candidates in final_rankings
- Use exact candidate names as shown above
- Rank 1 = best candidate, Rank {len(evaluated)} = weakest
- Every candidate MUST have a rank, name, recommendation, and ranking_reason

JSON:"""

        result = self.llm.generate_json(prompt, max_tokens=1024)
        
        rankings = result.get("final_rankings")
        if rankings is None:
            raise RuntimeError("LLM failed to generate final rankings")
        
        if len(rankings) != len(evaluated):
            raise RuntimeError(f"LLM ranked {len(rankings)} candidates but {len(evaluated)} were evaluated")
        
        top_candidate_summary = result.get("top_candidate_summary")
        if top_candidate_summary is None:
            raise RuntimeError("LLM failed to generate top candidate summary")
        
        hiring_recommendation = result.get("hiring_recommendation")
        if hiring_recommendation is None:
            raise RuntimeError("LLM failed to generate hiring recommendation")
        
        rank_map = {r["name"]: r for r in rankings}
        
        final_results = []
        for candidate in evaluated:
            rank_info = rank_map.get(candidate['name'])
            if rank_info is None:
                raise RuntimeError(f"LLM did not provide ranking for candidate: {candidate['name']}")
            
            if rank_info.get('rank') is None:
                raise RuntimeError(f"LLM did not provide rank number for candidate: {candidate['name']}")
            
            if rank_info.get('recommendation') is None:
                raise RuntimeError(f"LLM did not provide recommendation for candidate: {candidate['name']}")
            
            if rank_info.get('ranking_reason') is None:
                raise RuntimeError(f"LLM did not provide ranking reason for candidate: {candidate['name']}")
            
            final_results.append({
                **candidate,
                'final_rank': rank_info['rank'],
                'recommendation': rank_info['recommendation'],
                'ranking_reason': rank_info['ranking_reason']
            })
        
        final_results.sort(key=lambda x: x['final_rank'])
        
        return NodeResult(
            node_name=self.name, status=NodeStatus.COMPLETED,
            data={
                "final_results": final_results,
                "top_candidate_summary": top_candidate_summary,
                "hiring_recommendation": hiring_recommendation
            },
            metadata={"total_ranked": len(final_results)}
        )


class FastScreenPipeline:
    def __init__(self, llm: LLMClient, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.llm = llm
        self.nodes: List[Node] = [
            HybridFilteringNode(llm, embedding_model),
            RankFusionNode(llm),
            QuestionGenerationNode(llm),
            ResumeRankingNode(llm),
            ReportGenerationNode(llm)
        ]
    
    def run(self, job_description: str, resumes: List[Dict[str, Any]], pipeline_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not job_description:
            raise ValueError("Job description is required")
        if not resumes:
            raise ValueError("At least one resume is required")
        
        config = pipeline_config or {}
        config.setdefault('eval_criteria_count', EVALUATION_CRITERIA_COUNT)
        
        context = PipelineContext(job_description=job_description, resumes=resumes, config=config)
        
        start_time = datetime.now()
        logger.info(f"\n{'='*50}")
        logger.info(f"FastScreen Pipeline Started")
        logger.info(f"Resumes: {len(resumes)} | Criteria: {config['eval_criteria_count']}")
        logger.info(f"{'='*50}\n")
        
        for node in self.nodes:
            logger.info(f"[{node.name}] Running...")
            result = node.run(context)
            context.set_result(result)
            
            if result.success:
                logger.info(f"[{node.name}] Done ({result.execution_time_ms:.0f}ms)")
            else:
                logger.info(f"[{node.name}] FAILED: {'; '.join(result.errors)}")
                return {
                    "success": False,
                    "error": f"Pipeline failed at {node.name}: {'; '.join(result.errors)}",
                    "failed_node": node.name
                }
        
        report = context.get_result("report_generation")
        criteria = context.get_result("question_generation")
        
        total_time = round((datetime.now() - start_time).total_seconds(), 1)
        logger.info(f"\n{'='*50}")
        logger.info(f"Pipeline Complete ({total_time}s)")
        logger.info(f"{'='*50}\n")
        
        return {
            "success": True,
            "processing_time_seconds": total_time,
            "total_candidates": len(resumes),
            "evaluated_count": len(report.data["final_results"]),
            "job_title": criteria.data["job_title"],
            "seniority": criteria.data["seniority"],
            "evaluation_criteria": criteria.data["evaluation_criteria"],
            "results": report.data["final_results"],
            "top_candidate_summary": report.data["top_candidate_summary"],
            "hiring_recommendation": report.data["hiring_recommendation"],
            "node_timings": {name: res.execution_time_ms for name, res in context.node_results.items()}
        }