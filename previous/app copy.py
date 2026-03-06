"""
FastScreen AI - Streamlit UI
Multi-provider support via .env configuration
"""
import streamlit as st
from pathlib import Path
import tempfile
import json
import pandas as pd
from datetime import datetime
import os

st.set_page_config(page_title="FastScreen AI", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")

try:
    from core import FastScreenAI, config
    CORE_AVAILABLE = True
    CONFIG_ERROR = None
except RuntimeError as e:
    CORE_AVAILABLE = False
    CONFIG_ERROR = str(e)
except ImportError as e:
    CORE_AVAILABLE = False
    CONFIG_ERROR = f"Import error: {e}"

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1200px; }
    #MainMenu, footer, .stDeployButton { display: none; }
    .provider-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 8px;
    }
    .provider-ollama { background: #10b981; color: white; }
    .provider-groq { background: #f97316; color: white; }
    .provider-gemini { background: #3b82f6; color: white; }
    .provider-openrouter { background: #8b5cf6; color: white; }
    .provider-openai { background: #000; color: white; }
</style>
""", unsafe_allow_html=True)


def get_provider_badge(provider: str) -> str:
    """Get HTML badge for provider"""
    return f'<span class="provider-badge provider-{provider}">{provider.upper()}</span>'


def get_score_color(score: int) -> str:
    if score >= 4: return "#27ae60"
    if score >= 3: return "#3498db"
    if score >= 2: return "#f39c12"
    return "#e74c3c"


def display_candidate(candidate: dict, job_analysis: dict):
    """Display a candidate card"""
    rank = candidate.get('final_rank', 0)
    name = candidate['name'].replace('_', ' ')
    score = candidate.get('final_score', 0)
    rec = candidate.get('recommendation', 'consider')
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"### #{rank} {name}")
        rec_display = rec.replace('_', ' ').title()
        if rec == 'strongly_recommend':
            st.success(f"**{rec_display}**")
        elif rec == 'recommend':
            st.info(f"**{rec_display}**")
        elif rec == 'consider':
            st.warning(f"**{rec_display}**")
        else:
            st.error(f"**{rec_display}**")
    with col2:
        st.metric("Score", f"{score}/100")
    
    ranking_reason = candidate.get('ranking_reason', '')
    summary = candidate.get('summary', '')
    
    explanation = ""
    if ranking_reason:
        explanation += ranking_reason
    if summary:
        if explanation:
            explanation += " " + summary
        else:
            explanation = summary
    
    if explanation:
        st.markdown(f"**Why this rank:** {explanation}")
    
    with st.expander("📊 Detailed Score Breakdown", expanded=False):
        breakdown = candidate.get('score_breakdown', {})
        
        if breakdown:
            scores = [v.get('score', 0) if isinstance(v, dict) else v for v in breakdown.values()]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total", f"{candidate.get('total_weighted', 0)}/{candidate.get('max_weighted', 0)}")
            c2.metric("Strong (4-5)", sum(1 for s in scores if s >= 4))
            c3.metric("Moderate (2-3)", sum(1 for s in scores if 2 <= s < 4))
            c4.metric("Weak (0-1)", sum(1 for s in scores if s < 2))
            
            st.markdown("---")
            
            for criterion, data in breakdown.items():
                if isinstance(data, dict):
                    score_val = data.get('score', 0)
                    weight = data.get('weight', 1)
                else:
                    score_val = data
                    weight = 1
                
                bar_color = get_score_color(score_val)
                bar_width = (score_val / 5) * 100
                
                st.markdown(f"""
                <div style="margin: 8px 0; padding: 10px; background: #f5f5f5; border-radius: 6px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                        <span style="font-size: 0.9rem; color: #333;">{criterion}</span>
                        <span style="font-weight: 600;">
                            <span style="color: #888; font-size: 0.8rem;">W{weight}</span>
                            <span style="background: {bar_color}; color: white; padding: 2px 8px; border-radius: 4px; margin-left: 8px;">{score_val}/5</span>
                        </span>
                    </div>
                    <div style="background: #ddd; border-radius: 4px; height: 6px;">
                        <div style="background: {bar_color}; width: {bar_width}%; height: 100%; border-radius: 4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No score breakdown available")
    
    st.markdown("---")


def display_setup_instructions():
    """Show setup instructions"""
    st.markdown("""
    ## 🚀 Setup Required
    
    FastScreen AI needs an LLM provider configured in your `.env` file.
    
    ### Recommended: Ollama (FREE & UNLIMITED)
    
    ```bash
    # Install Ollama
    curl -fsSL https://ollama.com/install.sh | sh
    
    # Pull a model
    ollama pull gemma2:2b
    
    # Start server
    ollama serve
    ```
    
    Then in `.env`:
    ```
    LLM_PROVIDER=ollama
    OLLAMA_MODEL=gemma2:2b
    ```
    
    ### Alternative Cloud Providers
    
    | Provider | Free Tier | Setup |
    |----------|-----------|-------|
    | **Groq** | 100K tokens/day | `GROQ_API_KEY=...` |
    | **Gemini** | 20 req/day | `GEMINI_API_KEY=...` |
    | **OpenRouter** | Some free models | `OPENROUTER_API_KEY=...` |
    | **OpenAI** | Paid only | `OPENAI_API_KEY=...` |
    
    ### Then restart:
    ```bash
    streamlit run app.py
    ```
    """)


def main():
    for k, v in {'screener': None, 'results': None, 'processing': False, 'error': None}.items():
        if k not in st.session_state:
            st.session_state[k] = v
    
    # Header with provider badge
    provider = config.provider if CORE_AVAILABLE else "not configured"
    provider_badge = get_provider_badge(provider) if CORE_AVAILABLE else ""
    
    st.markdown(f"""
        <h1>🚀 FastScreen AI {provider_badge}</h1>
        <p style="color: #666;">Multi-Provider Resume Screening • Model: <b>{config.current_model if CORE_AVAILABLE else 'N/A'}</b></p>
    """, unsafe_allow_html=True)
    
    if not CORE_AVAILABLE:
        display_setup_instructions()
        if CONFIG_ERROR:
            st.error(CONFIG_ERROR)
        return
    
    # Sidebar
    with st.sidebar:
        st.header("System")
        
        if st.session_state.screener is None:
            with st.spinner("Initializing..."):
                try:
                    st.session_state.screener = FastScreenAI()
                except Exception as e:
                    st.error(f"Failed: {e}")
                    return
        
        status = st.session_state.screener.check_system()
        
        # Provider info
        st.markdown(f"**Provider:** {status.get('provider', 'N/A').upper()}")
        st.markdown(f"**Model:** `{status.get('model', 'N/A')}`")
        
        llm_ok = status.get('llm_connected', False)
        if llm_ok:
            st.success("✓ LLM Connected")
        else:
            st.error(f"✗ LLM: {status.get('llm_error', 'Not connected')}")
            if status.get('provider') == 'ollama':
                st.code("ollama serve", language="bash")
        
        emb_ok = status.get('embeddings_ready', False)
        if emb_ok:
            st.success("✓ Embeddings Ready")
        else:
            st.error("✗ Embeddings Not Ready")
        
        st.divider()
        
        # Upload
        st.subheader("Upload")
        uploaded = st.file_uploader("ZIP (JD + Resumes)", type=['zip'])
        
        if uploaded and llm_ok and emb_ok:
            if st.button("🚀 Analyze", type="primary", use_container_width=True):
                st.session_state.processing = True
                st.session_state.error = None
                st.session_state.results = None
        
        st.divider()
        
        if st.button("Clear Cache", use_container_width=True):
            st.session_state.screener.clear_cache()
            st.success("Cleared!")
        
        st.divider()
        st.caption(f"Provider: {config.provider}")
        st.caption(f"Model: {config.current_model}")
    
    # Processing
    if st.session_state.processing and uploaded:
        with st.status("Processing...", expanded=True) as proc_status:
            try:
                st.write("📦 Extracting files...")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = Path(tmp.name)
                
                st.write("🔍 Phase 1: Semantic filtering...")
                st.write("🧠 Phase 2: LLM analysis...")
                
                results = st.session_state.screener.process_zip(tmp_path)
                os.unlink(tmp_path)
                
                if 'error' in results:
                    proc_status.update(label="Failed", state="error")
                    st.session_state.error = results['error']
                else:
                    proc_status.update(label=f"Done ({results.get('processing_time_seconds', 0)}s)", state="complete")
                    st.session_state.results = results
                
            except Exception as e:
                proc_status.update(label="Error", state="error")
                st.session_state.error = str(e)
            finally:
                st.session_state.processing = False
                st.rerun()
    
    if st.session_state.error:
        st.error(st.session_state.error)
    
    # Results
    if st.session_state.results:
        results = st.session_state.results
        job_analysis = results.get('job_analysis', {})
        
        # Provider used
        st.caption(f"Processed with: **{results.get('provider', 'N/A').upper()}** ({results.get('model', 'N/A')})")
        
        st.subheader(f"📋 {job_analysis.get('job_title', 'Position Analysis')}")
        st.caption(f"Seniority: {job_analysis.get('seniority_level', 'N/A')} | {len(job_analysis.get('evaluation_criteria', []))} criteria")
        
        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Applicants", results.get('total_candidates', 0))
        c2.metric("Screened", results.get('phase1_passed', 0))
        c3.metric("Evaluated", results.get('evaluated', 0))
        c4.metric("Time", f"{results.get('processing_time_seconds', 0)}s")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        hiring_rec = results.get('hiring_recommendation', '')
        if hiring_rec:
            st.success(f"**🎯 Hiring Recommendation:** {hiring_rec}")
        
        top_summary = results.get('top_candidate_summary', '')
        if top_summary:
            st.info(f"**🏆 Top Candidate:** {top_summary}")
        
        # Job analysis
        with st.expander("📋 Full Job Analysis"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Must Have Skills:**")
                for skill in job_analysis.get('must_have_skills', []):
                    st.markdown(f"• {skill.get('skill', '')} ({skill.get('importance', '')})")
                
                st.markdown("**Important Skills:**")
                for skill in job_analysis.get('important_skills', []):
                    st.markdown(f"• {skill.get('skill', '')}")
            
            with col2:
                st.markdown("**Nice to Have:**")
                for skill in job_analysis.get('nice_to_have', []):
                    st.markdown(f"• {skill.get('skill', '')}")
                
                st.markdown("**Red Flags:**")
                for rf in job_analysis.get('red_flags', []):
                    st.markdown(f"• ⚠️ {rf}")
            
            st.markdown("---")
            st.markdown("**Evaluation Criteria:**")
            for i, c in enumerate(job_analysis.get('evaluation_criteria', [])):
                st.markdown(f"{i+1}. **[W{c.get('weight', 1)}]** {c.get('criterion', '')}")
        
        st.markdown("---")
        
        # Rankings
        st.subheader("🏆 Candidate Rankings")
        
        for candidate in results.get('results', []):
            display_candidate(candidate, job_analysis)
        
        # Export
        st.subheader("📥 Export")
        c1, c2 = st.columns(2)
        
        with c1:
            st.download_button(
                "JSON Report",
                json.dumps(results, indent=2, default=str),
                f"fastscreen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                use_container_width=True
            )
        
        with c2:
            if results.get('results'):
                df = pd.DataFrame([{
                    'Rank': c.get('final_rank'),
                    'Name': c['name'],
                    'Score': c.get('final_score'),
                    'Recommendation': c.get('recommendation'),
                    'Summary': c.get('summary', '')
                } for c in results['results']])
                
                st.download_button(
                    "CSV Export",
                    df.to_csv(index=False),
                    f"fastscreen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
    
    elif not st.session_state.error:
        st.markdown("""
        ### How It Works
        
        | Phase | Description |
        |-------|-------------|
        | **1. Semantic Filter** | Embeddings filter irrelevant resumes |
        | **2. LLM Analysis** | AI analyzes JD → creates criteria → evaluates candidates → ranks |
        
        ### Upload
        
        Create a ZIP with:
        - Job description (filename contains "JD", "job", or "description")
        - Resume files (PDF, DOCX, or TXT)
        """)


if __name__ == "__main__":
    main()