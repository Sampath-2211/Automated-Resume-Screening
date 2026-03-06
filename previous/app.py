"""
FastScreen AI - Streamlit Application
Clean, minimal UI without sidebar
"""
import streamlit as st
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
import zipfile
import io
from typing import Dict, List, Any, Optional

st.set_page_config(
    page_title="Automated Resume Screener",
    page_icon="F",
    layout="wide",
    initial_sidebar_state="collapsed"
)

try:
    from core import (
        load_config,
        Config,
        LLMClient,
        FastScreenPipeline,
        extract_text_from_bytes,
        is_jd_file,
        EVALUATION_CRITERIA_COUNT,
    )
    CORE_AVAILABLE = True
    INIT_ERROR = None
except Exception as e:
    CORE_AVAILABLE = False
    INIT_ERROR = str(e)


st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; max-width: 1000px; }
    #MainMenu, footer, .stDeployButton { display: none; }
    [data-testid="stSidebar"] { display: none; }
    
    .status-box {
        padding: 8px 12px;
        border: 1px solid #ddd;
        border-radius: 4px;
        margin: 4px 0;
        font-family: monospace;
        font-size: 14px;
    }
    .status-done { border-left: 3px solid #28a745; }
    .status-fail { border-left: 3px solid #dc3545; }
    .status-wait { border-left: 3px solid #6c757d; }
    
    .candidate-card {
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 16px;
        margin: 12px 0;
        background: #f8f9fa;
        color: #333;
    }
    .candidate-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #ddd;
        color: #333;
    }
    .candidate-header strong, .candidate-header span {
        color: #333;
    }
    .score-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }
    .score-table th, .score-table td {
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid #eee;
    }
    .score-table th { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


def process_uploaded_files(jd_file: Optional[Any], resume_files: List[Any]) -> Dict[str, Any]:
    """Process uploaded job description and resume files."""
    if jd_file is None:
        raise ValueError("Job description file is required")
    
    jd_bytes = jd_file.getvalue()
    jd_text = extract_text_from_bytes(jd_bytes, jd_file.name)
    
    if not jd_text.strip():
        raise ValueError(f"Could not extract text from: {jd_file.name}")
    
    if not resume_files:
        raise ValueError("At least one resume file is required")
    
    resumes, errors = [], []
    
    for resume_file in resume_files:
        try:
            resume_bytes = resume_file.getvalue()
            resume_text = extract_text_from_bytes(resume_bytes, resume_file.name)
            if resume_text.strip():
                name = Path(resume_file.name).stem.replace('_', ' ').replace('-', ' ')
                resumes.append({'name': name, 'filename': resume_file.name, 'content': resume_text})
        except Exception as e:
            errors.append(f"{resume_file.name}: {str(e)}")
    
    if not resumes:
        raise ValueError(f"No valid resumes extracted. Errors: {'; '.join(errors)}")
    
    return {"jd": jd_text, "jd_filename": jd_file.name, "resumes": resumes, "extraction_errors": errors}


def process_zip_file(zip_file: Any) -> Dict[str, Any]:
    """Process a ZIP file containing JD and resumes."""
    zip_bytes = zip_file.getvalue()
    jd_text, jd_filename = "", ""
    resumes, errors = [], []
    
    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
        for file_info in zf.infolist():
            if file_info.is_dir() or file_info.filename.startswith('.') or file_info.filename.startswith('__'):
                continue
            
            filename = Path(file_info.filename).name
            suffix = Path(filename).suffix.lower()
            
            if suffix not in {'.pdf', '.docx', '.txt'}:
                continue
            
            try:
                file_bytes = zf.read(file_info.filename)
                text = extract_text_from_bytes(file_bytes, filename)
                
                if not text.strip():
                    errors.append(f"{filename}: Empty content")
                    continue
                
                if is_jd_file(filename):
                    jd_text, jd_filename = text, filename
                else:
                    name = Path(filename).stem.replace('_', ' ').replace('-', ' ')
                    resumes.append({'name': name, 'filename': filename, 'content': text})
            except Exception as e:
                errors.append(f"{filename}: {str(e)}")
    
    if not jd_text:
        raise ValueError("No job description found in ZIP. File name must contain: jd, job, description, position, or role")
    if not resumes:
        raise ValueError(f"No valid resumes found in ZIP. Errors: {'; '.join(errors[:5])}")
    
    return {"jd": jd_text, "jd_filename": jd_filename, "resumes": resumes, "extraction_errors": errors}


def display_candidate(candidate: Dict[str, Any]):
    """Display a single candidate's results."""
    rank = candidate['final_rank']
    name = candidate['name'].replace('_', ' ')
    score = candidate['final_score']
    recommendation = candidate['recommendation'].replace('_', ' ').title()
    
    st.markdown(f"""
    <div class="candidate-card">
        <div class="candidate-header">
            <span><strong>#{rank} {name}</strong></span>
            <span>Score: {score}/100 | {recommendation}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write(f"**Reason:** {candidate['ranking_reason']}")
    st.write(f"**Summary:** {candidate['summary']}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Strengths:**")
        for s in candidate['strengths']:
            st.write(f"- {s}")
    with col2:
        st.write("**Weaknesses:**")
        for w in candidate['weaknesses']:
            st.write(f"- {w}")
    
    for rf in candidate.get('red_flags', []):
        st.warning(f"Flag: {rf}")
    
    with st.expander("Score Breakdown"):
        breakdown = candidate['score_breakdown']
        table_html = '<table class="score-table"><tr><th>Criterion</th><th>Weight</th><th>Score</th></tr>'
        for criterion, data in breakdown.items():
            score_val = data['score']
            weight = data['weight']
            table_html += f'<tr><td>{criterion}</td><td>{weight}</td><td>{score_val}/5</td></tr>'
        table_html += '</table>'
        st.markdown(table_html, unsafe_allow_html=True)
    
    with st.expander("Phase 1 Scores"):
        phase1_rank = candidate.get('phase1_rank', '-')
        sem_rank = candidate.get('semantic_rank', '-')
        kw_rank = candidate.get('keyword_rank', '-')
        
        st.write(f"### Phase 1 Rank: #{phase1_rank}")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Semantic Rank", f"#{sem_rank}", f"{candidate['semantic_score']:.1f}%")
        with col2:
            st.metric("Keyword Rank", f"#{kw_rank}", f"{candidate['keyword_score']:.1f}%")
        
        st.write(f"**RRF Score:** {candidate['rrf_score']:.6f}")
        st.caption(f"Formula: RRF = 1/(60+{sem_rank}) + 1/(60+{kw_rank}) = {candidate['rrf_score']:.6f}")


def main():
    for key in ['config', 'llm', 'pipeline', 'results', 'processing', 'error', 'connected']:
        if key not in st.session_state:
            st.session_state[key] = None if key not in ['processing', 'connected'] else False
    
    st.title("Automated Resume Screener")
    st.caption("Resume Screening Pipeline")
    
    if not CORE_AVAILABLE:
        st.error(f"Failed to load core module: {INIT_ERROR}")
        st.code("Required: LLM_PROVIDER and API key environment variables")
        return
    
    if st.session_state.config is None:
        try:
            st.session_state.config = load_config()
            st.session_state.llm = LLMClient(st.session_state.config)
            st.session_state.pipeline = FastScreenPipeline(
                st.session_state.llm,
                st.session_state.config.embedding_model
            )
        except Exception as e:
            st.error(f"Configuration error: {e}")
            return
    
    config = st.session_state.config
    llm = st.session_state.llm
    pipeline = st.session_state.pipeline
    
    st.markdown("---")
    st.subheader("Configuration")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Provider:** {config.llm_provider.upper()}")
    with col2:
        st.write(f"**Model:** {config.model_name}")
    with col3:
        criteria_count = st.number_input(
            "Evaluation Criteria", 
            min_value=3, 
            max_value=15, 
            value=config.eval_criteria_count,
            key="criteria_count"
        )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Test Connection"):
            with st.spinner("Testing..."):
                status = llm.check_connection()
                st.session_state.connected = status.get('connected', False)
                if st.session_state.connected:
                    st.success("Connected")
                else:
                    st.error(f"Connection failed: {status.get('error')}")
    
    st.markdown("---")
    st.subheader("Upload Files")
    
    upload_mode = st.radio("Mode", ["Individual Files", "ZIP Archive"], horizontal=True, label_visibility="collapsed")
    
    jd_file, resume_files, zip_file = None, [], None
    
    if upload_mode == "Individual Files":
        col1, col2 = st.columns(2)
        with col1:
            jd_file = st.file_uploader("Job Description", type=['pdf', 'docx', 'txt'])
        with col2:
            resume_files = st.file_uploader("Resumes", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
        
        if jd_file:
            st.caption(f"JD: {jd_file.name}")
        if resume_files:
            st.caption(f"Resumes: {len(resume_files)} files")
    else:
        zip_file = st.file_uploader("ZIP Archive (JD + Resumes)", type=['zip'])
        if zip_file:
            st.caption(f"ZIP: {zip_file.name}")
    
    can_run = (jd_file and resume_files) or zip_file
    
    if st.button("Run Pipeline", type="primary", disabled=not can_run):
        st.session_state.processing = True
        st.session_state.error = None
        st.session_state.results = None
    
    if st.session_state.processing:
        progress_container = st.container()
        
        with progress_container:
            st.markdown("---")
            st.subheader("Pipeline Execution")
            
            try:
                st.markdown('<div class="status-box status-wait">Extracting documents...</div>', unsafe_allow_html=True)
                
                if zip_file:
                    extracted = process_zip_file(zip_file)
                else:
                    extracted = process_uploaded_files(jd_file, resume_files)
                
                st.markdown(f'<div class="status-box status-done">Extracted: {extracted["jd_filename"]} + {len(extracted["resumes"])} resumes</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="status-box status-wait">Running pipeline...</div>', unsafe_allow_html=True)
                
                results = pipeline.run(
                    job_description=extracted['jd'],
                    resumes=extracted['resumes'],
                    pipeline_config={'eval_criteria_count': criteria_count}
                )
                
                if results['success']:
                    st.markdown(f'<div class="status-box status-done">Complete ({results["processing_time_seconds"]}s)</div>', unsafe_allow_html=True)
                    st.session_state.results = results
                else:
                    st.markdown(f'<div class="status-box status-fail">Failed: {results["error"]}</div>', unsafe_allow_html=True)
                    st.session_state.error = results['error']
                    
            except Exception as e:
                st.markdown(f'<div class="status-box status-fail">Error: {str(e)}</div>', unsafe_allow_html=True)
                st.session_state.error = str(e)
            finally:
                st.session_state.processing = False
    
    if st.session_state.error:
        st.error(st.session_state.error)
    
    if st.session_state.results:
        results = st.session_state.results
        
        st.markdown("---")
        st.subheader("Results")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Candidates", results['total_candidates'])
        col2.metric("Evaluated", results['evaluated_count'])
        col3.metric("Criteria", len(results['evaluation_criteria']))
        col4.metric("Time", f"{results['processing_time_seconds']}s")
        
        st.write(f"**Position:** {results['job_title']} ({results['seniority']})")
        st.info(f"**Recommendation:** {results['hiring_recommendation']}")
        st.write(f"**Top Candidate:** {results['top_candidate_summary']}")
        
        with st.expander(f"Evaluation Criteria ({len(results['evaluation_criteria'])})"):
            for c in results['evaluation_criteria']:
                st.write(f"**{c['criterion']}** (Weight: {c.get('weight', 3)}, {c.get('category', 'important')})")
                what_5 = c.get('what_counts_as_5', 'Expert level evidence')
                what_0 = c.get('what_counts_as_0', 'No evidence')
                st.caption(f"5 = {what_5} | 0 = {what_0}")
                st.markdown("---")
        
        st.markdown("---")
        st.subheader("Candidate Rankings")
        
        for candidate in results['results']:
            display_candidate(candidate)
            st.markdown("---")
        
        st.subheader("Export")
        
        col1, col2 = st.columns(2)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with col1:
            st.download_button(
                "Download JSON",
                json.dumps(results, indent=2, default=str),
                f"fastscreen_{timestamp}.json",
                "application/json"
            )
        
        with col2:
            df = pd.DataFrame([
                {
                    'Rank': c['final_rank'],
                    'Name': c['name'],
                    'Score': c['final_score'],
                    'Recommendation': c['recommendation'],
                    'Summary': c['summary']
                }
                for c in results['results']
            ])
            st.download_button("Download CSV", df.to_csv(index=False), f"fastscreen_{timestamp}.csv", "text/csv")
    
    elif not st.session_state.error and not st.session_state.processing:
        st.markdown("---")
        st.markdown("""
        ### Pipeline Nodes
        
        | Node | Purpose |
        |------|---------|
        | 1. Hybrid Filtering | Semantic + Keyword matching |
        | 2. Rank Fusion | RRF algorithm |
        | 3. Question Generation | Create evaluation rubric |
        | 4. Resume Ranking | Score candidates |
        | 5. Report Generation | Final rankings |
        
        ### Upload Options
        
        **Individual Files:** Upload JD and resume files separately.
        
        **ZIP Archive:** Single ZIP with all files. JD filename should contain: jd, job, description, position, role.
        
        ### Environment Variables
        LLM_PROVIDER=groq|gemini|openai
        GROQ_API_KEY=your_key
        EVAL_CRITERIA_COUNT=6
        """)

if __name__ == "__main__":
    main()