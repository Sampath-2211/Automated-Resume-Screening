"""
FastScreen AI - Streamlit Application
Clean, Professional UI with Page-based Navigation
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
    page_title="FastScreen AI",
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
    .main .block-container { 
        padding-top: 1rem; 
        max-width: 1100px; 
    }
    #MainMenu, footer, .stDeployButton { display: none; }
    [data-testid="stSidebar"] { display: none; }
    
    .status-box {
        padding: 10px 14px;
        border-radius: 6px;
        margin: 6px 0;
        font-family: 'SF Mono', 'Consolas', monospace;
        font-size: 13px;
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        color: #212529;
    }
    .status-done { 
        border-left: 4px solid #28a745; 
        background: #f1f8f4;
    }
    .status-fail { 
        border-left: 4px solid #dc3545; 
        background: #fdf2f2;
    }
    .status-wait { 
        border-left: 4px solid #6c757d;
        background: #f8f9fa;
    }
    
    .candidate-card {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 20px;
        margin: 16px 0;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .candidate-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 12px;
        border-bottom: 1px solid #e9ecef;
    }
    .candidate-header strong {
        color: #212529;
        font-size: 18px;
    }
    .candidate-header span {
        color: #495057;
        font-size: 14px;
    }
    
    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
    }
    .score-high { background: #d4edda; color: #155724; }
    .score-medium { background: #fff3cd; color: #856404; }
    .score-low { background: #f8d7da; color: #721c24; }
    
    .score-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
        margin-top: 10px;
    }
    .score-table th {
        padding: 12px 10px;
        text-align: left;
        border-bottom: 2px solid #dee2e6;
        color: #495057;
        font-weight: 600;
        background: #f8f9fa;
    }
    .score-table td {
        padding: 10px;
        text-align: left;
        border-bottom: 1px solid #e9ecef;
        color: #212529;
    }
    .score-table tr:hover {
        background: #f8f9fa;
    }
    
    .alert-box {
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0;
        font-size: 14px;
    }
    .alert-warning {
        background: #fff3cd;
        border: 1px solid #ffc107;
        color: #856404;
    }
    .alert-danger {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    
    .section-header {
        font-size: 20px;
        font-weight: 600;
        color: #212529;
        margin: 24px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #e9ecef;
    }
    
    .info-text {
        color: #495057;
        font-size: 14px;
        line-height: 1.6;
    }
    
    .recommendation-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 13px;
    }
    .rec-strong { background: #28a745; color: white; }
    .rec-recommend { background: #5cb85c; color: white; }
    .rec-consider { background: #f0ad4e; color: white; }
    .rec-not { background: #d9534f; color: white; }
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


def get_score_class(score: int) -> str:
    if score >= 70:
        return "score-high"
    elif score >= 50:
        return "score-medium"
    return "score-low"


def get_recommendation_class(rec: str) -> str:
    rec_lower = rec.lower().replace(' ', '_')
    if 'strongly' in rec_lower:
        return "rec-strong"
    elif 'recommend' in rec_lower and 'not' not in rec_lower:
        return "rec-recommend"
    elif 'consider' in rec_lower:
        return "rec-consider"
    return "rec-not"


def display_candidate(candidate: Dict[str, Any]):
    """Display a single candidate's results."""
    import pandas as pd  # Ensure pandas is available
    
    rank = candidate['final_rank']
    name = candidate['name'].replace('_', ' ')
    score = candidate['final_score']
    recommendation = candidate['recommendation'].replace('_', ' ').title()
    
    gating_applied = candidate.get('gating_applied', False)
    critical_failures = candidate.get('critical_failures', [])
    raw_score = candidate.get('raw_score_before_gating', score)
    
    score_class = get_score_class(score)
    rec_class = get_recommendation_class(recommendation)
    
    st.markdown(f"""
    <div class="candidate-card">
        <div class="candidate-header">
            <strong>#{rank} {name}</strong>
            <span>
                <span class="score-badge {score_class}">{score}/100</span>
                <span class="recommendation-badge {rec_class}" style="margin-left: 8px;">{recommendation}</span>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if gating_applied:
        severity = "alert-danger" if len(critical_failures) >= 2 else "alert-warning"
        failures_text = ", ".join([f"{cf['criterion']} ({cf['score']}/5)" for cf in critical_failures])
        
        st.markdown(f"""
        <div class="alert-box {severity}">
            <strong>Score Adjusted:</strong> Original score was {raw_score}%, adjusted to {score}% due to missing critical requirements.<br>
            <strong>Missing:</strong> {failures_text}
        </div>
        """, unsafe_allow_html=True)
        
        rec_before = candidate.get('recommendation_before_gating', '').replace('_', ' ').title()
        if rec_before and rec_before != recommendation:
            st.caption(f"Recommendation changed from {rec_before} to {recommendation}")
    
    st.write(f"**Reason:** {candidate['ranking_reason']}")
    st.write(f"**Summary:** {candidate['summary']}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Strengths**")
        for s in candidate['strengths']:
            st.write(f"- {s}")
    with col2:
        st.write("**Weaknesses**")
        for w in candidate['weaknesses']:
            st.write(f"- {w}")
    
    red_flags = candidate.get('red_flags', [])
    if red_flags:
        st.write("**Flags**")
        for rf in red_flags:
            st.warning(rf)
    
    skill_matches = candidate.get('skill_match_details', [])
    if skill_matches:
        with st.expander(f"Skills Detected ({len(skill_matches)})"):
            for sm in skill_matches:
                match_type = sm.get('type', 'unknown')
                confidence = sm.get('confidence', 0)
                found_as = ', '.join(sm.get('found', []))
                
                if match_type == "exact":
                    st.write(f"**{sm['skill']}**: {found_as} (exact match)")
                elif match_type == "related":
                    st.write(f"**{sm['skill']}**: {found_as} (related skill)")
                else:
                    st.write(f"**{sm['skill']}**: {found_as} ({confidence:.0%} match)")
    
    with st.expander("Score Breakdown"):
        breakdown = candidate['score_breakdown']
        
        # Use a simple table without complex styling
        breakdown_data = []
        for criterion, data in breakdown.items():
            score_val = data['score']
            weight = data['weight']
            category = data.get('category', 'important').title()
            breakdown_data.append({
                "Criterion": criterion,
                "Category": category,
                "Weight": weight,
                "Score": f"{score_val}/5"
            })
        
        df_breakdown = pd.DataFrame(breakdown_data)
        st.dataframe(df_breakdown, use_container_width=True, hide_index=True)
    
    with st.expander("Initial Screening Scores"):
        phase1_rank = candidate.get('phase1_rank', '-')
        sem_rank = candidate.get('semantic_rank', '-')
        kw_rank = candidate.get('keyword_rank', '-')
        
        st.write(f"**Initial Rank:** #{phase1_rank}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Semantic Rank", f"#{sem_rank}", f"{candidate['semantic_score']:.1f}%")
        with col2:
            st.metric("Keyword Rank", f"#{kw_rank}", f"{candidate['keyword_score']:.1f}%")
        
        st.write(f"**Combined Score:** {candidate['rrf_score']:.6f}")


def render_upload_page(config, llm, pipeline):
    """Render the upload and configuration page."""
    st.title("FastScreen AI")
    st.caption("Automated Resume Screening System")
    
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
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Test Connection"):
            with st.spinner("Testing..."):
                status = llm.check_connection()
                if status.get('connected', False):
                    st.success("Connected")
                else:
                    st.error(f"Failed: {status.get('error')}")
    
    st.markdown("---")
    st.subheader("Upload Files")
    
    upload_mode = st.radio("Upload Mode", ["Individual Files", "ZIP Archive"], horizontal=True, label_visibility="collapsed")
    
    jd_file, resume_files, zip_file = None, [], None
    
    if upload_mode == "Individual Files":
        col1, col2 = st.columns(2)
        with col1:
            jd_file = st.file_uploader("Job Description", type=['pdf', 'docx', 'txt'])
        with col2:
            resume_files = st.file_uploader("Resumes", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
        
        if jd_file:
            st.caption(f"Job Description: {jd_file.name}")
        if resume_files:
            st.caption(f"Resumes: {len(resume_files)} file(s)")
    else:
        zip_file = st.file_uploader("ZIP Archive (JD + Resumes)", type=['zip'])
        if zip_file:
            st.caption(f"Archive: {zip_file.name}")
    
    can_run = (jd_file and resume_files) or zip_file
    
    st.markdown("---")
    
    if st.button("Run Screening", type="primary", disabled=not can_run, use_container_width=True):
        with st.container():
            st.subheader("Processing")
            
            try:
                st.markdown('<div class="status-box status-wait">Extracting documents...</div>', unsafe_allow_html=True)
                
                if zip_file:
                    extracted = process_zip_file(zip_file)
                else:
                    extracted = process_uploaded_files(jd_file, resume_files)
                
                st.markdown(f'<div class="status-box status-done">Extracted: {extracted["jd_filename"]} + {len(extracted["resumes"])} resumes</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="status-box status-wait">Running analysis pipeline...</div>', unsafe_allow_html=True)
                
                results = pipeline.run(
                    job_description=extracted['jd'],
                    resumes=extracted['resumes'],
                    pipeline_config={'eval_criteria_count': criteria_count}
                )
                
                if results['success']:
                    st.markdown(f'<div class="status-box status-done">Complete - {results["processing_time_seconds"]} seconds</div>', unsafe_allow_html=True)
                    st.session_state.results = results
                    st.rerun()
                else:
                    st.error(f"Pipeline failed: {results['error']}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Show instructions when no files uploaded
    if not can_run:
        st.markdown("---")
        
        with st.container(border=True):
            st.subheader("How It Works")
            st.write("""
            This system evaluates resumes against job descriptions using a multi-stage pipeline:
            
            1. **Document Extraction** - Parses PDF, DOCX, and TXT files
            2. **Initial Screening** - Semantic analysis and keyword matching
            3. **Criteria Generation** - Creates evaluation rubric from job description
            4. **Detailed Evaluation** - Scores each candidate against criteria
            5. **Final Ranking** - Produces recommendations and rankings
            """)
            
            st.subheader("Upload Options")
            st.write("""
            **Individual Files:** Upload job description and resume files separately.
            
            **ZIP Archive:** Single ZIP containing all files. Job description filename should contain: jd, job, description, position, or role.
            """)
            
            st.subheader("Supported Formats")
            st.write("PDF, DOCX, TXT")


def render_results_page(results):
    """Render the results dashboard page."""
    
    # Header with New Screening button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Screening Results")
        st.caption(f"{results['job_title']} ({results['seniority']})")
    with col2:
        st.write("")  # Spacing
        if st.button("New Screening", type="secondary", use_container_width=True):
            st.session_state.results = None
            st.rerun()
    
    st.markdown("---")
    
    # Build dashboard data
    candidates = results['results']
    total = len(candidates)
    
    # Recommendation distribution
    rec_counts = {"Strongly Recommend": 0, "Recommend": 0, "Consider": 0, "Do Not Recommend": 0}
    for c in candidates:
        rec = c['recommendation'].replace('_', ' ').title()
        if rec in rec_counts:
            rec_counts[rec] += 1
    
    # Score distribution
    score_ranges = {"Excellent (80+)": 0, "Good (60-79)": 0, "Fair (40-59)": 0, "Poor (<40)": 0}
    for c in candidates:
        score = c['final_score']
        if score >= 80:
            score_ranges["Excellent (80+)"] += 1
        elif score >= 60:
            score_ranges["Good (60-79)"] += 1
        elif score >= 40:
            score_ranges["Fair (40-59)"] += 1
        else:
            score_ranges["Poor (<40)"] += 1
    
    # Top 3 candidates
    top_3 = candidates[:3] if len(candidates) >= 3 else candidates
    
    # === DASHBOARD ===
    
    # Row 1: Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Candidates", total)
    with col2:
        shortlist_count = rec_counts["Strongly Recommend"] + rec_counts["Recommend"]
        st.metric("Shortlisted", shortlist_count)
    with col3:
        avg_score = sum(c['final_score'] for c in candidates) / total if total else 0
        st.metric("Avg Score", f"{avg_score:.0f}")
    with col4:
        st.metric("Process Time", f"{results['processing_time_seconds']}s")
    
    st.markdown("")
    
    # Row 2: Distribution Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Recommendation Distribution**")
        for label, count in rec_counts.items():
            pct = (count / total * 100) if total > 0 else 0
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.progress(pct / 100, text=label)
            with col_b:
                st.write(f"**{count}**")
    
    with col2:
        st.write("**Score Distribution**")
        for label, count in score_ranges.items():
            pct = (count / total * 100) if total > 0 else 0
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.progress(pct / 100, text=label)
            with col_b:
                st.write(f"**{count}**")
    
    st.markdown("")
    
    # Row 3: Top Candidates
    st.write("**Top Candidates**")
    
    cols = st.columns(min(3, len(top_3)))
    for i, candidate in enumerate(top_3):
        with cols[i]:
            score = candidate['final_score']
            name = candidate['name'].replace('_', ' ')
            rec = candidate['recommendation'].replace('_', ' ').title()
            strengths = candidate.get('strengths', [])[:2]
            
            with st.container(border=True):
                st.markdown(f"### #{i+1}")
                st.metric(label=name, value=f"{score}/100", delta=rec)
                if strengths:
                    st.caption("Key Strengths:")
                    for s in strengths:
                        display_text = f"{s[:50]}..." if len(s) > 50 else s
                        st.caption(f"- {display_text}")
    
    st.markdown("")
    
    # Row 4: Overview Table
    st.write("**All Candidates Overview**")
    
    table_data = []
    for candidate in candidates:
        rec = candidate['recommendation'].replace('_', ' ').title()
        strengths = candidate.get('strengths', ['N/A'])
        key_strength = strengths[0][:40] + "..." if len(strengths[0]) > 40 else strengths[0] if strengths else "N/A"
        red_flags = candidate.get('red_flags', [])
        
        table_data.append({
            "Rank": f"#{candidate['final_rank']}",
            "Candidate": candidate['name'].replace('_', ' '),
            "Score": candidate['final_score'],
            "Recommendation": rec,
            "Key Strength": key_strength,
            "Flags": len(red_flags)
        })
    
    df_display = pd.DataFrame(table_data)
    
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.TextColumn("Rank", width="small"),
            "Candidate": st.column_config.TextColumn("Candidate", width="medium"),
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
            "Recommendation": st.column_config.TextColumn("Recommendation", width="medium"),
            "Key Strength": st.column_config.TextColumn("Key Strength", width="large"),
            "Flags": st.column_config.NumberColumn("Flags", width="small")
        }
    )
    
    st.markdown("")
    
    # Hiring Recommendation
    st.info(f"**Hiring Recommendation:** {results['hiring_recommendation']}")
    
    st.markdown("---")
    
    # === DETAILED RESULTS ===
    st.subheader("Detailed Results")
    
    # Evaluation errors warning
    eval_errors = sum(1 for c in candidates if c.get('evaluation_error', False))
    if eval_errors > 0:
        st.warning(f"Note: {eval_errors} candidate(s) could not be fully evaluated due to processing limits.")
    
    # Evaluation Criteria
    with st.expander(f"Evaluation Criteria ({len(results['evaluation_criteria'])})"):
        for c in results['evaluation_criteria']:
            weight = c.get('weight', 3)
            category = c.get('category', 'important').title()
            
            st.write(f"**{c['criterion']}**")
            st.caption(f"Category: {category} | Weight: {weight}")
            
            what_5 = c.get('what_counts_as_5', 'Expert level evidence')
            what_0 = c.get('what_counts_as_0', 'No evidence')
            st.caption(f"Score 5: {what_5}")
            st.caption(f"Score 0: {what_0}")
            st.markdown("---")
    
    # Candidate Selector
    st.write("**Select Candidate for Details**")
    
    candidate_options = [f"#{c['final_rank']} - {c['name'].replace('_', ' ')} ({c['final_score']}/100)" for c in candidates]
    selected_idx = st.selectbox("Choose a candidate", range(len(candidate_options)), format_func=lambda x: candidate_options[x], label_visibility="collapsed")
    
    if selected_idx is not None:
        display_candidate(candidates[selected_idx])
    
    st.markdown("---")
    
    # === EXPORT ===
    st.subheader("Export Results")
    
    col1, col2 = st.columns(2)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with col1:
        st.download_button(
            "Download JSON Report",
            json.dumps(results, indent=2, default=str),
            f"screening_report_{timestamp}.json",
            "application/json",
            use_container_width=True
        )
    
    with col2:
        df_export = pd.DataFrame([
            {
                'Rank': c['final_rank'],
                'Name': c['name'],
                'Score': c['final_score'],
                'Recommendation': c['recommendation'],
                'Summary': c['summary'],
                'Skills Matched': c.get('keywords_found', 0)
            }
            for c in candidates
        ])
        st.download_button(
            "Download CSV Report", 
            df_export.to_csv(index=False), 
            f"screening_report_{timestamp}.csv", 
            "text/csv",
            use_container_width=True
        )


def main():
    # Initialize session state
    if 'config' not in st.session_state:
        st.session_state.config = None
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Check core availability
    if not CORE_AVAILABLE:
        st.error(f"Failed to load core module: {INIT_ERROR}")
        st.code("Required: LLM_PROVIDER and API key environment variables")
        return
    
    # Initialize config, llm, pipeline
    if st.session_state.config is None:
        try:
            st.session_state.config = load_config()
            st.session_state.llm = LLMClient(st.session_state.config)
            st.session_state.pipeline = FastScreenPipeline(
                st.session_state.llm,
                st.session_state.config.embedding_model,
                st.session_state.config.skill_similarity_threshold
            )
        except Exception as e:
            st.error(f"Configuration error: {e}")
            return
    
    # Route to appropriate page
    if st.session_state.results is not None:
        render_results_page(st.session_state.results)
    else:
        render_upload_page(
            st.session_state.config,
            st.session_state.llm,
            st.session_state.pipeline
        )


if __name__ == "__main__":
    main()