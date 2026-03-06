"""
Automated Resume Screening - Diagnostic Script
Run this to verify your setup before the demo/viva.

Usage: python diagnostic.py

Author: Sampath Krishna Tekumalla
"""
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")

def print_pass(text):
    print(f"  {Colors.GREEN}✓ PASS:{Colors.RESET} {text}")

def print_fail(text):
    print(f"  {Colors.RED}✗ FAIL:{Colors.RESET} {text}")

def print_warn(text):
    print(f"  {Colors.YELLOW}⚠ WARN:{Colors.RESET} {text}")

def print_info(text):
    print(f"  {Colors.CYAN}ℹ INFO:{Colors.RESET} {text}")

# =============================================================================
# TEST 1: Environment Variables
# =============================================================================
def test_environment():
    print_header("TEST 1: Environment Variables")
    
    required_vars = [
        ('QUESTION_GEN_API_KEY', 'Cloud LLM API key (or GROQ_API_KEY)'),
        ('EVAL_URL', 'Evaluation LLM URL'),
        ('EVAL_MODEL', 'Evaluation model name'),
    ]
    
    optional_vars = [
        ('CITATION_THRESHOLD', '0.80'),
        ('RAG_CHUNK_SIZE', '100'),
        ('RAG_TOP_K', '3'),
    ]
    
    all_pass = True
    
    # Check for API key (either QUESTION_GEN_API_KEY or GROQ_API_KEY)
    api_key = os.getenv('QUESTION_GEN_API_KEY') or os.getenv('GROQ_API_KEY')
    if api_key:
        masked = api_key[:10] + "..." + api_key[-4:] if len(api_key) > 14 else "***"
        print_pass(f"API Key found: {masked}")
    else:
        print_fail("No API key found! Set QUESTION_GEN_API_KEY or GROQ_API_KEY")
        all_pass = False
    
    for var, desc in required_vars[1:]:
        val = os.getenv(var)
        if val:
            print_pass(f"{var} = {val}")
        else:
            print_warn(f"{var} not set (using default)")
    
    print()
    print_info("Optional settings:")
    for var, default in optional_vars:
        val = os.getenv(var, f"NOT SET (default: {default})")
        print_info(f"  {var} = {val}")
    
    return all_pass

# =============================================================================
# TEST 2: Python Dependencies
# =============================================================================
def test_dependencies():
    print_header("TEST 2: Python Dependencies")
    
    dependencies = [
        ('streamlit', 'Web framework'),
        ('sentence_transformers', 'Embeddings'),
        ('sklearn', 'Machine learning'),
        ('numpy', 'Numerical computing'),
        ('fitz', 'PDF processing (PyMuPDF)'),
        ('docx', 'DOCX processing'),
        ('requests', 'HTTP client'),
        ('pandas', 'Data processing'),
        ('dotenv', 'Environment variables'),
    ]
    
    all_pass = True
    
    for module, desc in dependencies:
        try:
            __import__(module)
            print_pass(f"{module} - {desc}")
        except ImportError:
            print_fail(f"{module} - {desc} (NOT INSTALLED)")
            all_pass = False
    
    # Optional dependencies
    print()
    print_info("Optional dependencies (for OCR):")
    
    try:
        from pdf2image import convert_from_bytes
        print_pass("pdf2image - PDF to image conversion")
    except ImportError:
        print_warn("pdf2image not installed (OCR will be disabled)")
    
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        print_pass("pytesseract - OCR text extraction")
    except:
        print_warn("Tesseract not available (OCR will be disabled)")
    
    return all_pass

# =============================================================================
# TEST 3: Ollama Connection
# =============================================================================
def test_ollama():
    print_header("TEST 3: Local LLM (Ollama) Connection")
    
    import requests
    
    host = os.getenv('EVAL_URL', 'http://localhost:11434')
    model = os.getenv('EVAL_MODEL', 'qwen2.5:3b')
    
    # Test connection
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        if r.status_code == 200:
            print_pass(f"Ollama is running at {host}")
            
            # Check if model exists
            models = r.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            print_info(f"Available models: {[m.split(':')[0] for m in model_names]}")
            
            # Check for our model
            model_base = model.split(':')[0]
            if any(model_base in m for m in model_names):
                print_pass(f"Model '{model}' is available")
            else:
                print_fail(f"Model '{model}' not found!")
                print_info(f"Run: ollama pull {model}")
                return False
        else:
            print_fail(f"Ollama returned status {r.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_fail(f"Cannot connect to Ollama at {host}")
        print_info("Run: ollama serve")
        return False
    except Exception as e:
        print_fail(f"Error: {e}")
        return False
    
    # Test generation
    print_info("Testing LLM generation...")
    try:
        r = requests.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": "Reply with exactly: OK",
                "stream": False,
                "options": {"num_predict": 10}
            },
            timeout=60
        )
        if r.status_code == 200:
            response = r.json().get('response', '').strip()
            print_pass(f"LLM responded: '{response[:30]}'")
            return True
        else:
            print_fail(f"Generation failed: {r.status_code}")
            return False
    except Exception as e:
        print_fail(f"Generation error: {e}")
        return False

# =============================================================================
# TEST 4: Embedding Model
# =============================================================================
def test_embeddings():
    print_header("TEST 4: Embedding Model")
    
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        
        print_info(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)
        print_pass("Model loaded successfully")
        
        # Test encoding
        test_texts = ["Python programming experience", "Machine learning engineer"]
        embeddings = model.encode(test_texts)
        print_pass(f"Generated embeddings shape: {embeddings.shape}")
        
        # Test similarity
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        print_info(f"Test similarity: {sim:.4f}")
        
        return True
    except Exception as e:
        print_fail(f"Embedding error: {e}")
        return False

# =============================================================================
# TEST 5: RAG Pipeline
# =============================================================================
def test_rag():
    print_header("TEST 5: RAG Retrieval")
    
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Sample resume text
        resume = """
        Sampath Krishna Tekumalla - AI Engineer
        Experience with LangChain and LangGraph for building agentic workflows.
        Built an AI-powered application at ISRO using Mistral 7B and Phi-3.
        Skills: Python, Docker, SQL, Machine Learning, WebSockets.
        """
        
        # Chunk the text
        words = resume.split()
        chunk_size, overlap = 20, 5
        step = chunk_size - overlap
        chunks = []
        for i in range(0, len(words), step):
            chunks.append(" ".join(words[i:i + chunk_size]))
        
        print_pass(f"Created {len(chunks)} chunks")
        
        # Test retrieval
        query = "Experience with LangGraph"
        query_emb = model.encode([query])[0]
        chunk_embs = model.encode(chunks)
        
        sims = cosine_similarity([query_emb], chunk_embs)[0]
        best_idx = np.argmax(sims)
        
        print_pass(f"Best match similarity: {sims[best_idx]:.4f}")
        print_info(f"Retrieved: '{chunks[best_idx][:60]}...'")
        
        return True
    except Exception as e:
        print_fail(f"RAG error: {e}")
        return False

# =============================================================================
# TEST 6: Citation Validation
# =============================================================================
def test_citation_validation():
    print_header("TEST 6: Citation Validation")
    
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Use longer, more realistic resume text
        resume = """
        Sampath Krishna Tekumalla is an AI Engineer with experience at ISRO.
        He integrated LangChain for function-calling capabilities and architected 
        a modular workflow with LangGraph for building agentic AI systems.
        Built machine learning models using Python and TensorFlow for geospatial analysis.
        Experience with Docker containerization and WebSocket-based real-time communication.
        Deployed Mistral 7B and Phi-3 Mini for natural language query processing.
        """
        
        threshold = float(os.getenv('CITATION_THRESHOLD', '0.65'))
        
        # Test cases with realistic citations
        test_cases = [
            ("integrated LangChain for function-calling capabilities", True, "Exact quote"),
            ("architected a modular workflow with LangGraph", True, "Exact quote"),
            ("experience with Docker containerization", True, "Partial match"),
            ("Java Spring Boot microservices development", False, "Unrelated tech"),
            ("10 years of management experience at Google", False, "Fabricated claim"),
        ]
        
        print_info(f"Using threshold: {threshold}")
        
        # Create fine-grained chunks (similar to actual validation)
        words = resume.split()
        chunks = []
        chunk_size, overlap = 30, 10
        step = chunk_size - overlap
        for i in range(0, len(words), step):
            chunks.append(" ".join(words[i:i + chunk_size]))
        
        print_info(f"Created {len(chunks)} validation chunks")
        chunk_embs = model.encode(chunks, show_progress_bar=False)
        
        all_pass = True
        for citation, should_match, desc in test_cases:
            cite_emb = model.encode([citation])[0]
            sims = cosine_similarity([cite_emb], chunk_embs)[0]
            best_sim = max(sims)
            is_valid = best_sim >= threshold
            
            status = "✓" if is_valid == should_match else "✗"
            
            if is_valid == should_match:
                print_pass(f"{desc}: sim={best_sim:.2f} {'(valid)' if is_valid else '(invalid)'}")
            else:
                print_fail(f"{desc}: sim={best_sim:.2f} (expected {'valid' if should_match else 'invalid'})")
                all_pass = False
        
        # Show threshold recommendation
        print()
        if all_pass:
            print_info(f"Threshold {threshold} is working correctly")
        else:
            print_warn(f"Consider adjusting CITATION_THRESHOLD in .env")
            print_info(f"Recommended range: 0.55 - 0.70 for typical citations")
        
        return all_pass
    except Exception as e:
        print_fail(f"Validation error: {e}")
        return False

# =============================================================================
# TEST 7: Cloud API (Groq)
# =============================================================================
def test_cloud_api():
    print_header("TEST 7: Cloud API (Groq)")
    
    import requests
    
    api_key = os.getenv('QUESTION_GEN_API_KEY') or os.getenv('GROQ_API_KEY')
    
    if not api_key:
        print_warn("No API key configured - skipping cloud API test")
        return True
    
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": os.getenv('QUESTION_GEN_MODEL', 'llama-3.3-70b-versatile'),
                "messages": [{"role": "user", "content": "Reply with: OK"}],
                "max_tokens": 10
            },
            timeout=30
        )
        
        if r.status_code == 200:
            response = r.json()['choices'][0]['message']['content']
            print_pass(f"Groq API responded: '{response.strip()}'")
            return True
        else:
            print_fail(f"Groq API error: {r.status_code}")
            print_info(r.text[:200])
            return False
    except Exception as e:
        print_fail(f"Cloud API error: {e}")
        return False

# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}  AUTOMATED RESUME SCREENING - DIAGNOSTIC TOOL{Colors.RESET}")
    print(f"{Colors.BOLD}  Sampath Krishna Tekumalla{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
    
    results = {}
    
    # Run all tests
    results['environment'] = test_environment()
    results['dependencies'] = test_dependencies()
    results['ollama'] = test_ollama()
    results['embeddings'] = test_embeddings()
    results['rag'] = test_rag()
    results['citation'] = test_citation_validation()
    results['cloud_api'] = test_cloud_api()
    
    # Summary
    print_header("DIAGNOSTIC SUMMARY")
    
    all_pass = True
    for test_name, passed in results.items():
        if passed:
            print_pass(f"{test_name.upper()}")
        else:
            print_fail(f"{test_name.upper()}")
            all_pass = False
    
    print()
    if all_pass:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ All tests passed! System is ready for demo.{Colors.RESET}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ Some tests failed. Check errors above.{Colors.RESET}")
        print(f"\n{Colors.YELLOW}Common fixes:{Colors.RESET}")
        print("  1. Start Ollama: ollama serve")
        print("  2. Pull model: ollama pull qwen2.5:3b")
        print("  3. Check .env file has GROQ_API_KEY")
        print("  4. Install dependencies: pip install -r requirements.txt")
    
    return all_pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)