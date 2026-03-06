"""
Automated Resume Screening - Streamlit Dashboard
5-Tab UI: Main Screening, Results, Validation Log, Comparison Mode, How It Works
Citation click -> st.dialog modal with green PDF highlight.

Author: Sampath Krishna Tekumalla
"""
import streamlit as st
from pathlib import Path
import pandas as pd
from datetime import datetime
import zipfile
import io
from typing import Dict, List, Any
import re

st.set_page_config(
    page_title="Automated Resume Screening",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# IMPORTS
# =============================================================================
try:
    from core import (
        load_config, PipelineConfig, ResumeScreeningPipeline,
        extract_text_from_bytes, is_jd_file, extract_citations
    )
    from citation_validator import CitationExtractor, CitationValidator
    from pdf_highlighter import (
        PDFHighlighter, CitationHighlightModal, get_highlighted_page_html
    )
    from summary_generator import AdaptiveSummaryGenerator, generate_summary_for_candidate
    CORE_AVAILABLE = True
    INIT_ERROR = None
except Exception as e:
    CORE_AVAILABLE = False
    INIT_ERROR = str(e)
    import traceback
    traceback.print_exc()

# =============================================================================
# STYLES
# =============================================================================
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }
    [data-testid="stSidebar"] { display: none; }
    .citation-link {
        background: #fef3c7; padding: 2px 6px; border-radius: 4px;
        cursor: pointer; border-bottom: 2px solid #f59e0b;
        transition: all 0.2s; color: #1e293b !important;
    }
    .citation-link:hover { background: #fde68a; }
    .summary-box {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 16px; line-height: 1.6; color: #1e293b !important;
    }
    .summary-box * { color: #1e293b !important; }
    .reasoning-box {
        background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 6px;
        padding: 12px; margin: 8px 0; color: #334155 !important;
        font-size: 0.9rem; line-height: 1.5; max-height: 300px; overflow-y: auto;
    }
    .reasoning-box * { color: #334155 !important; }
    .criterion-card {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 12px; margin: 8px 0;
    }
    .score-high { background: #dcfce7; color: #166534; }
    .score-mid { background: #fef3c7; color: #92400e; }
    .score-low { background: #fee2e2; color: #991b1b; }
    .ocr-warning {
        background: #fff7ed; border: 1px solid #fed7aa; color: #9a3412;
        padding: 10px 14px; border-radius: 6px; margin: 6px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPERS
# =============================================================================
def alert_error(msg):
    st.markdown(
        f'<div style="background:#fef2f2;border:1px solid #fecaca;color:#991b1b;'
        f'padding:12px 16px;border-radius:6px;">{msg}</div>',
        unsafe_allow_html=True
    )

def alert_success(msg):
    st.markdown(
        f'<div style="background:#dcfce7;border:1px solid #bbf7d0;color:#166534;'
        f'padding:12px 16px;border-radius:6px;">{msg}</div>',
        unsafe_allow_html=True
    )

def alert_warning(msg):
    st.markdown(
        f'<div style="background:#fffbeb;border:1px solid #fde68a;color:#92400e;'
        f'padding:12px 16px;border-radius:6px;margin:8px 0;">{msg}</div>',
        unsafe_allow_html=True
    )

def format_citation_display(summary: str) -> str:
    """Convert <cite> tags to styled spans with explicit dark text color."""
    return re.sub(
        r'<cite>(.*?)</cite>',
        r'<span style="background:#fef3c7;padding:2px 6px;border-radius:4px;'
        r'cursor:pointer;border-bottom:2px solid #f59e0b;color:#1e293b !important;">\1</span>',
        summary
    )

def process_files(jd_file, resume_files) -> Dict[str, Any]:
    jd_text, _ = extract_text_from_bytes(jd_file.getvalue(), jd_file.name)
    resumes = []
    for rf in resume_files:
        content = rf.getvalue()
        text, _ = extract_text_from_bytes(content, rf.name)
        if text.strip():
            resumes.append({
                'name': Path(rf.name).stem.replace('_', ' ').replace('-', ' ').title(),
                'filename': rf.name,
                'content': text,
                'pdf_bytes': content if rf.name.lower().endswith('.pdf') else None
            })
    return {"jd": jd_text, "resumes": resumes}

def process_zip(zip_file) -> Dict[str, Any]:
    jd_text, resumes = "", []
    with zipfile.ZipFile(io.BytesIO(zip_file.getvalue()), 'r') as zf:
        for fi in zf.infolist():
            if fi.is_dir() or fi.filename.startswith('.'):
                continue
            fn = Path(fi.filename).name
            if Path(fn).suffix.lower() not in {'.pdf', '.docx', '.txt'}:
                continue
            content = zf.read(fi.filename)
            text, _ = extract_text_from_bytes(content, fn)
            if not text.strip():
                continue
            if is_jd_file(fn):
                jd_text = text
            else:
                resumes.append({
                    'name': Path(fn).stem.replace('_', ' ').replace('-', ' ').title(),
                    'filename': fn,
                    'content': text,
                    'pdf_bytes': content if fn.lower().endswith('.pdf') else None
                })
    return {"jd": jd_text, "resumes": resumes}

# =============================================================================
# CITATION MODAL (st.dialog)
# =============================================================================
@st.dialog("Citation Verification", width="large")
def show_citation_modal(pdf_bytes, citation_text, similarity, is_valid):
    """Modal showing PDF page with green highlight at citation bounding box."""
    if pdf_bytes is None:
        st.warning("PDF not available for highlighting")
        st.markdown(f'**Citation:** "{citation_text}"')
        st.markdown(f"**Similarity:** {similarity:.1%}")
        st.markdown(f"**Status:** {'✅ VERIFIED' if is_valid else '❌ NOT VERIFIED'}")
        return

    try:
        modal = CitationHighlightModal()
        data = modal.prepare_modal_data(pdf_bytes, citation_text, similarity, is_valid)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(
                f"data:image/png;base64,{data['image_base64']}",
                caption=f"Page {data['page_num']} — Green highlight shows citation location"
            )
        with col2:
            if is_valid:
                st.success(f"✅ VERIFIED ({similarity:.1%})")
            else:
                st.error(f"❌ NOT VERIFIED ({similarity:.1%})")

            st.markdown(f'**Citation:** "{citation_text[:200]}"')
            st.markdown(f"**Page:** {data['page_num']}")
            st.markdown(f"**Similarity:** {data['similarity_percent']}")
            st.markdown(f"**Threshold:** 80%")
            st.markdown(f"**Found in PDF:** {'Yes' if data['found_in_pdf'] else 'No'}")
    except Exception as e:
        st.error(f"Could not render PDF: {e}")

# =============================================================================
# UPLOAD PAGE (before results exist)
# =============================================================================
def render_upload_page(config, pipeline):
    st.title("📄 Automated Resume Screening")
    st.caption("Citation-Grounded Evaluation System | By Sampath Krishna Tekumalla")

    tab_screen, tab_how = st.tabs(["🚀 Screen Resumes", "📚 How It Works"])

    with tab_screen:
        render_screening_tab(config, pipeline)
    with tab_how:
        render_how_it_works_content()

# =============================================================================
# TAB: MAIN SCREENING
# =============================================================================
def render_screening_tab(config, pipeline):
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        upload_mode = st.radio(
            "Input Method", ["Individual Files", "ZIP Archive"],
            horizontal=True, label_visibility="collapsed"
        )
    with col2:
        criteria_count = st.number_input(
            "Evaluation Criteria Count", min_value=3, max_value=10, value=5
        )

    jd_file, resume_files, zip_file = None, [], None
    if upload_mode == "Individual Files":
        c1, c2 = st.columns(2)
        with c1:
            jd_file = st.file_uploader("📋 Job Description", type=['pdf', 'docx', 'txt'])
        with c2:
            resume_files = st.file_uploader(
                "📄 Resumes", type=['pdf', 'docx', 'txt'], accept_multiple_files=True
            )
    else:
        zip_file = st.file_uploader(
            "📦 ZIP Archive", type=['zip'],
            help="Include JD file with 'jd' or 'job' in filename"
        )

    ready = (jd_file and resume_files) or zip_file

    if st.button("🔍 Start Citation-Grounded Screening", type="primary",
                 disabled=not ready, use_container_width=True):
        run_pipeline(pipeline, jd_file, resume_files, zip_file, criteria_count)

def run_pipeline(pipeline, jd_file, resume_files, zip_file, criteria_count):
    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_cb(pct, msg):
        progress_bar.progress(min(pct, 100))
        status_text.text(msg)

    try:
        progress_cb(5, "Extracting documents...")
        if zip_file:
            extracted = process_zip(zip_file)
        else:
            extracted = process_files(jd_file, resume_files)

        if not extracted['jd']:
            alert_error("No job description found!")
            return
        if not extracted['resumes']:
            alert_error("No resumes found!")
            return

        progress_cb(10, f"Found {len(extracted['resumes'])} resumes. Starting pipeline...")

        results = pipeline.run(
            extracted['jd'],
            extracted['resumes'],
            pipeline_config={'eval_criteria_count': criteria_count},
            progress_callback=progress_cb
        )

        progress_bar.progress(100)
        status_text.text("✅ Screening complete!")

        if results['success']:
            pdf_map = {r['filename']: r.get('pdf_bytes') for r in extracted['resumes']}
            results['pdf_bytes_map'] = pdf_map
            st.session_state.results = results
            st.rerun()
        else:
            alert_error(f"Pipeline failed: {results.get('error', 'Unknown error')}")
            if results.get('failed_candidates'):
                for fc in results['failed_candidates']:
                    st.text(f"  - {fc['name']}: {fc['error']}")
    except Exception as e:
        alert_error(f"Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# =============================================================================
# RESULTS PAGE (after screening)
# =============================================================================
def render_results_page(results: Dict):
    st.title("📊 Screening Results")
    st.caption(f"Position: {results.get('job_title', 'Unknown')}")

    col_back, _, _ = st.columns([1, 1, 1])
    with col_back:
        if st.button("← New Screening"):
            st.session_state.results = None
            st.rerun()

    # Show OCR warnings prominently
    ocr_warnings = results.get('ocr_warnings', [])
    if ocr_warnings:
        for w in ocr_warnings:
            st.markdown(
                f'<div class="ocr-warning">⚠️ <strong>{w["candidate"]}</strong>: '
                f'{w["warning"]}</div>',
                unsafe_allow_html=True
            )

    if results.get('failed_count', 0) > 0:
        alert_warning(
            f"⚠️ {results['failed_count']} candidate(s) failed processing. "
            f"Check Pipeline Log for details."
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Rankings",
        "✅ Validation Log",
        "🔬 Comparison Mode",
        "⚙️ Pipeline Log",
        "📚 How It Works"
    ])

    with tab1:
        render_rankings_tab(results)
    with tab2:
        render_validation_log_tab(results)
    with tab3:
        render_comparison_tab(results)
    with tab4:
        render_pipeline_log_tab(results)
    with tab5:
        render_how_it_works_content()

# =============================================================================
# TAB: RANKINGS
# =============================================================================
def render_rankings_tab(results: Dict):
    candidates = results.get('results', [])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Candidates", len(candidates))
    c2.metric("Qualified (≥55)", results.get('qualified_count', 0))
    c3.metric("Processing Time", f"{results.get('processing_time_seconds', 0)}s")
    c4.metric("Criteria Used", len(results.get('evaluation_criteria', [])))

    st.divider()
    st.subheader("🏆 Candidate Rankings")

    rec_emoji = {
        "strongly_recommend": "🟢", "recommend": "🟡",
        "consider": "🟠", "do_not_recommend": "🔴"
    }
    table_data = []
    for c in candidates:
        table_data.append({
            "Rank": c['rank'],
            "Name": c['name'],
            "Score": f"{c['score']}/100",
            "Critical Met": f"{c.get('critical_met', 0)}/{c.get('critical_total', 0)}",
            "Status": (
                f"{rec_emoji.get(c['recommendation'], '⚪')} "
                f"{c['recommendation'].replace('_', ' ').title()}"
            )
        })

    st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("📝 Detailed Analysis")

    if candidates:
        selected_idx = st.selectbox(
            "Select Candidate",
            options=range(len(candidates)),
            format_func=lambda i: (
                f"#{candidates[i]['rank']} - {candidates[i]['name']} "
                f"({candidates[i]['score']}/100)"
            )
        )
        if selected_idx is not None:
            render_candidate_detail(candidates[selected_idx], results, selected_idx)

def render_candidate_detail(candidate: Dict, results: Dict, candidate_idx: int):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"### {candidate['name']}")

        # OCR warning for this candidate
        ocr_warn = candidate.get('ocr_warning')
        if ocr_warn:
            st.markdown(
                f'<div class="ocr-warning">⚠️ OCR: {ocr_warn}</div>',
                unsafe_allow_html=True
            )

        # Summary box
        summary = candidate.get('summary', '')
        if summary:
            display_summary = format_citation_display(summary)
            st.markdown(
                f'<div class="summary-box">{display_summary}</div>',
                unsafe_allow_html=True
            )

        st.markdown("#### Criteria Scores")
        scores_detail = candidate.get('scores_detail', [])

        if not scores_detail:
            st.warning("No scoring details available")

        for score_idx, score in enumerate(scores_detail):
            criterion = score.get('criterion', 'Unknown')
            validated_score = score.get('validated_score', 0)
            raw_score = score.get('raw_score', 0)
            is_critical = score.get('critical', False)
            reasoning = score.get('reasoning', '')

            if validated_score >= 4:
                score_class = "score-high"
            elif validated_score >= 2:
                score_class = "score-mid"
            else:
                score_class = "score-low"

            critical_badge = " 🔴 CRITICAL" if is_critical else ""
            score_change = f" (raw: {raw_score})" if raw_score != validated_score else ""

            st.markdown(f"""
            <div class="criterion-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-weight:600;color:#1e293b;">
                        {criterion}{critical_badge}
                    </span>
                    <span class="{score_class}" style="font-weight:700;padding:4px 8px;border-radius:4px;">
                        {validated_score}/5{score_change}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Reasoning expander
            if reasoning:
                with st.expander("📝 View Reasoning", expanded=False):
                    st.markdown(
                        f'<div class="reasoning-box">{format_citation_display(reasoning)}</div>',
                        unsafe_allow_html=True
                    )

            # Validation notes
            notes = score.get('validation_notes', '')
            if notes and notes not in ('Verified', 'No citation required (score < 3)'):
                st.caption(f"⚠️ {notes}")

            # Citations
            citations = score.get('citation_results', [])
            if citations:
                for cite_idx, cite in enumerate(citations):
                    valid = cite.get('valid', False)
                    sim = cite.get('similarity', 0)
                    cite_text = cite.get('citation', '')
                    is_fallback = cite.get('is_fallback', False)

                    if not cite_text:
                        continue

                    status_icon = "✅" if valid else "❌"
                    fallback_tag = " [FALLBACK]" if is_fallback else ""
                    display_text = cite_text[:80] + "..." if len(cite_text) > 80 else cite_text
                    bg_color = "#dcfce7" if valid else "#fee2e2"
                    border_color = "#22c55e" if valid else "#ef4444"

                    st.markdown(f"""
                    <div style="background:{bg_color};border-left:3px solid {border_color};
                                padding:8px 12px;margin:4px 0;border-radius:4px;">
                        <span style="color:#1e293b;">
                            {status_icon} "{display_text}"{fallback_tag}
                        </span>
                        <span style="color:#64748b;font-size:0.85em;margin-left:8px;">
                            Similarity: {sim:.0%}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    # View in PDF button
                    uid = hash(f"{candidate['name']}_{criterion}_{cite_text[:20]}") % 10000
                    btn_key = f"cite_{candidate_idx}_{score_idx}_{cite_idx}_{uid}"

                    pdf_bytes = results.get('pdf_bytes_map', {}).get(candidate.get('filename'))
                    if st.button("🔍 View in PDF", key=btn_key):
                        show_citation_modal(pdf_bytes, cite_text, sim, valid)
            else:
                if validated_score >= 3:
                    st.caption("⚠️ No citation provided (score may be reduced)")
                else:
                    st.caption("ℹ️ No citation required for scores < 3")

    with col2:
        st.markdown("#### Assessment")

        score = candidate['score']
        if score >= 75:
            st.success(f"Score: {score}/100")
        elif score >= 55:
            st.info(f"Score: {score}/100")
        elif score >= 35:
            st.warning(f"Score: {score}/100")
        else:
            st.error(f"Score: {score}/100")

        st.markdown(f"""
**Critical Criteria:** {candidate.get('critical_met', 0)}/{candidate.get('critical_total', 0)} met

**Recommendation:** {candidate.get('recommendation', 'N/A').replace('_', ' ').title()}

**Tone:** {candidate.get('tone', 'N/A').title()}
        """)

        security = candidate.get('security_status', 'safe')
        if security == 'safe':
            st.success("🛡️ Security: Passed")
        elif security == 'flagged':
            st.warning("⚠️ Security: Flagged")
        else:
            st.info("🔍 Security: Not scanned")

        # OCR status
        if candidate.get('ocr_used'):
            st.success("👁️ OCR: Active (visible text only)")
        else:
            st.warning("👁️ OCR: Not used (raw PDF text)")

# =============================================================================
# TAB: VALIDATION LOG
# =============================================================================
def render_validation_log_tab(results: Dict):
    st.subheader("✅ Citation Validation Log")
    st.caption(
        "Every citation is verified against the original resume "
        "using semantic similarity (threshold: 80%)"
    )

    log = results.get('validation_log', [])

    if not log:
        st.info("No validation data available. Run a screening to see citation validation results.")
        return

    valid_count = sum(1 for l in log if l.get('valid'))
    invalid_count = len(log) - valid_count

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Citations", len(log))
    c2.metric("Verified ✅", valid_count)
    c3.metric("Invalidated ❌", invalid_count)

    st.divider()

    filter_option = st.radio("Filter", ["All", "Valid Only", "Invalid Only"], horizontal=True)

    if filter_option == "Valid Only":
        filtered_log = [l for l in log if l.get('valid')]
    elif filter_option == "Invalid Only":
        filtered_log = [l for l in log if not l.get('valid')]
    else:
        filtered_log = log

    if filtered_log:
        df = pd.DataFrame(filtered_log)
        df['Status'] = df['valid'].apply(lambda x: "✅ Verified" if x else "❌ Invalid")
        df['Similarity'] = df['similarity'].apply(lambda x: f"{x:.1%}")
        display_cols = ['candidate', 'criterion', 'citation', 'Status', 'Similarity']
        available_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available_cols], hide_index=True, use_container_width=True)
    else:
        st.info("No citations match the selected filter.")

# =============================================================================
# TAB: COMPARISON MODE - BEFORE vs AFTER VALIDATION
# =============================================================================
def render_comparison_tab(results: Dict):
    st.subheader("🔬 Before vs After Citation Validation")

    st.info(
        "**Before Validation (Raw LLM Score):** The score the LLM assigned "
        "based on resume evidence, before any citation checking.\n\n"
        "**After Validation (Grounded Score):** The score after verifying that "
        "every cited claim actually exists in the resume. Invalid citations "
        "cause score reductions or zeroing — this is the anti-hallucination layer."
    )

    candidates = results.get('results', [])
    if not candidates:
        st.warning("No candidates to compare.")
        return

    # Summary comparison table
    st.markdown("#### Score Comparison")

    comparison_data = []
    for c in candidates:
        scores = c.get('scores_detail', [])

        # Naive = raw LLM scores (before validation)
        naive_total = sum(s.get('naive_score', s.get('raw_score', 0)) * s.get('weight', 1) for s in scores)
        # Validated = after citation checking
        validated_total = sum(s.get('validated_score', 0) * s.get('weight', 1) for s in scores)
        max_score = sum(5 * s.get('weight', 1) for s in scores)

        naive_pct = round(naive_total / max_score * 100) if max_score else 0
        validated_pct = round(validated_total / max_score * 100) if max_score else 0
        diff = naive_pct - validated_pct

        changed_criteria = [
            s for s in scores
            if s.get('naive_score', s.get('raw_score', 0)) != s.get('validated_score', 0)
        ]

        comparison_data.append({
            "Candidate": c['name'],
            "Before Validation": f"{naive_pct}/100",
            "After Validation": f"{validated_pct}/100",
            "Score Change": f"−{diff}" if diff > 0 else "+0" if diff == 0 else f"+{abs(diff)}",
            "Hallucinations Caught": len(changed_criteria),
            "Impact": "🔴 High" if diff >= 15 else "🟡 Moderate" if diff >= 5 else "🟢 None/Low"
        })

    df = pd.DataFrame(comparison_data)
    st.dataframe(df, hide_index=True, use_container_width=True)

    # Highlight what this proves
    total_hallucinations = sum(d["Hallucinations Caught"] for d in comparison_data)
    if total_hallucinations > 0:
        st.markdown(
            f"**Citation validation caught {total_hallucinations} hallucinated or unverifiable claim(s)** "
            f"across {len(candidates)} candidates. Without this layer, candidates would have received "
            f"inflated scores based on evidence that doesn't exist in their resumes."
        )
    else:
        st.markdown(
            "All LLM citations were verified successfully. This demonstrates the model produced "
            "well-grounded evaluations for this batch."
        )

    # Per-candidate criterion-level detail
    st.divider()
    st.markdown("#### Per-Criterion Breakdown")

    selected = st.selectbox(
        "Select candidate for detailed comparison",
        options=range(len(candidates)),
        format_func=lambda i: candidates[i]['name'],
        key="comparison_select"
    )

    if selected is not None:
        cand = candidates[selected]
        scores = cand.get('scores_detail', [])
        detail_rows = []
        for s in scores:
            naive = s.get('naive_score', s.get('raw_score', 0))
            val = s.get('validated_score', 0)
            changed = naive != val
            detail_rows.append({
                "Criterion": s.get('criterion', ''),
                "Critical": "🔴 Yes" if s.get('critical') else "No",
                "Before (Raw)": f"{naive}/5",
                "After (Validated)": f"{val}/5",
                "Changed": "⚠️ Yes" if changed else "✓ No",
                "Reason": s.get('validation_notes', '')
            })
        st.dataframe(pd.DataFrame(detail_rows), hide_index=True, use_container_width=True)

# =============================================================================
# TAB: PIPELINE LOG
# =============================================================================
def render_pipeline_log_tab(results: Dict):
    st.subheader("⚙️ Pipeline Execution Log")

    # OCR warnings
    ocr_warnings = results.get('ocr_warnings', [])
    if ocr_warnings:
        st.markdown("##### ⚠️ OCR Warnings")
        for w in ocr_warnings:
            st.markdown(
                f'<div class="ocr-warning">⚠️ <strong>{w["candidate"]}</strong>: {w["warning"]}</div>',
                unsafe_allow_html=True
            )
        st.divider()

    # Failed candidates
    failed = results.get('failed_candidates', [])
    if failed:
        st.error(f"❌ {len(failed)} candidate(s) failed processing:")
        for fc in failed:
            st.markdown(f"- **{fc['name']}**: {fc['error']}")
        st.divider()

    log = results.get('pipeline_log', [])
    if log:
        for entry in log:
            node = entry.get('node', 'Unknown')
            status = entry.get('status', 'unknown')
            time_ms = entry.get('time_ms', 0)

            status_color = {
                'completed': '#22c55e', 'failed': '#ef4444', 'fallback': '#f59e0b'
            }.get(status, '#94a3b8')

            st.markdown(f"""
            <div style="display:flex;align-items:center;padding:12px;background:#f8fafc;
                        border-radius:6px;margin:4px 0;border-left:4px solid {status_color};">
                <span style="flex:1;font-weight:600;color:#1e293b;">{node}</span>
                <span style="color:{status_color};margin-right:16px;">{status.upper()}</span>
                <span style="color:#64748b;">{time_ms:.0f}ms</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Pipeline log will show after running a screening.")

# =============================================================================
# TAB: HOW IT WORKS
# =============================================================================
def render_how_it_works_content():
    st.subheader("🔧 6-Node Pipeline Architecture")

    st.markdown(
        "This system uses LLMs **only where human judgment is required** "
        "(interpreting unstructured text). All other steps are "
        "**deterministic, reproducible, and verifiable**."
    )

    st.markdown("### Pipeline Flow")

    cols = st.columns(6)
    nodes = [
        ("0️⃣", "ResuShield", "Security+OCR", "#ef4444", "Deterministic"),
        ("1️⃣", "Question Gen", "Cloud LLM", "#3b82f6", "LLM"),
        ("2️⃣", "RAG Eval", "Local LLM", "#8b5cf6", "LLM"),
        ("3️⃣", "Citation Val", "Embeddings", "#22c55e", "Deterministic"),
        ("4️⃣", "Summary Gen", "Local LLM", "#f59e0b", "LLM"),
        ("5️⃣", "Report Gen", "Sorting", "#64748b", "Deterministic"),
    ]
    for col, (num, name, tech, color, llm_type) in zip(cols, nodes):
        with col:
            badge_color = "#dc2626" if llm_type == "LLM" else "#16a34a"
            st.markdown(f"""
            <div style="background:{color}20;border:2px solid {color};border-radius:8px;
                        padding:8px;text-align:center;min-height:110px;">
                <div style="font-size:1.2rem;">{num}</div>
                <div style="font-weight:600;color:{color};font-size:0.8rem;">{name}</div>
                <div style="font-size:0.7rem;color:#64748b;">{tech}</div>
                <div style="margin-top:4px;">
                    <span style="background:{badge_color};color:white;padding:2px 6px;
                                border-radius:8px;font-size:0.6rem;">{llm_type}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🎯 Key Innovations")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**1. RAG-Based Evaluation (Node 2)**
- Resume chunked into 100-word segments with 20-word overlap
- Only top-3 relevant chunks sent to LLM per criterion
- LLM never sees the full resume — prevents hallucination from context overload

**2. Citation-Grounded Scoring**
- Score ≥ 3 requires `<cite>exact quote</cite>` from resume
- No citation = score reduced or zeroed
- Forces LLM to ground every claim in evidence
        """)

    with col2:
        st.markdown("""
**3. Anti-Hallucination Validation (Node 3)**
- Embeddings verify citations exist in resume text
- Threshold: 80% semantic similarity
- Multi-strategy: exact match → fuzzy substring → keyword overlap → semantic
- Semantic fallback searches for alternative evidence before zeroing score

**4. ResuShield Security (Node 0)**
- OCR extracts only human-visible text (defeats hidden white text)
- Visual-semantic comparison detects keyword stuffing
- Prompt injection shield blocks LLM manipulation attempts
        """)

    st.divider()
    st.markdown("### 📊 LLM vs Deterministic Split")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**🤖 LLM Tasks (3 nodes)**
- Node 1: Extract criteria from JD (Cloud — Groq)
- Node 2: Evaluate evidence with citations (Local — Ollama)
- Node 4: Generate natural language summary (Local — Ollama)
        """)
    with col2:
        st.markdown("""
**⚙️ Deterministic Tasks (3 nodes)**
- Node 0: Security scan + OCR text extraction
- Node 3: Citation validation via embeddings (threshold 80%)
- Node 5: Score calculation + ranking (pure Python sort)
        """)

    st.divider()
    st.markdown("### 🔑 Addressing Reviewer Concerns")

    st.markdown("""
> **"System is just giving resumes to LLM"**
> LLM receives only RAG-retrieved chunks (top 3 of 100-word segments), NOT full resume. 
> 3 of 6 pipeline nodes are fully deterministic with zero LLM involvement.

> **"No real innovation"**
> Citation-grounding with anti-hallucination validation, ResuShield OCR security module, 
> and the "Before vs After Validation" comparison mode demonstrating measurable impact.

> **"Can't prove accuracy"**
> Every score has clickable citation verification. Validation log shows all similarity checks.
> Deterministic nodes produce identical results every run = reproducible.
    """)

# =============================================================================
# MAIN
# =============================================================================
def main():
    if 'results' not in st.session_state:
        st.session_state.results = None

    if not CORE_AVAILABLE:
        st.error("⚠️ System Initialization Failed")
        alert_error(f"Error: {INIT_ERROR}")
        st.markdown("""
**Troubleshooting:**
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Check that Ollama is running: `ollama serve`
3. Verify the model is available: `ollama pull qwen2.5:3b`
4. Check your `.env` file has correct API keys
        """)
        st.stop()

    if 'config' not in st.session_state:
        st.session_state.config = load_config()
        st.session_state.pipeline = ResumeScreeningPipeline(st.session_state.config)

    if st.session_state.results:
        render_results_page(st.session_state.results)
    else:
        render_upload_page(st.session_state.config, st.session_state.pipeline)


if __name__ == "__main__":
    main()