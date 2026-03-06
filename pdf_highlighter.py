"""
Automated Resume Screening - PDF Highlighter
Renders PDF pages with GREEN highlight boxes for citation verification modals.

Author: Sampath Krishna Tekumalla
"""
import base64
import tempfile
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger("ResumeScreening")

TEMP_PDF_DIR = Path(tempfile.gettempdir()) / "resume_screening_pdfs"
TEMP_PDF_DIR.mkdir(exist_ok=True)

# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class HighlightRegion:
    page_num: int
    x0: float
    y0: float
    x1: float
    y1: float
    color: Tuple[float, float, float] = (0.2, 0.8, 0.2)  # GREEN
    alpha: float = 0.4

@dataclass
class PDFPageImage:
    page_num: int
    image_base64: str
    width: int
    height: int
    has_highlight: bool

# =============================================================================
# PDF STORAGE
# =============================================================================
class PDFStorage:
    @staticmethod
    def save_pdf(pdf_bytes: bytes, filename: str) -> str:
        safe_name = "".join(c if c.isalnum() or c in '._-' else '_' for c in filename)
        path = TEMP_PDF_DIR / safe_name
        with open(path, 'wb') as f:
            f.write(pdf_bytes)
        return str(path)

    @staticmethod
    def get_pdf(filename: str) -> Optional[bytes]:
        safe_name = "".join(c if c.isalnum() or c in '._-' else '_' for c in filename)
        path = TEMP_PDF_DIR / safe_name
        if path.exists():
            with open(path, 'rb') as f:
                return f.read()
        return None

    @staticmethod
    def cleanup_old_files(max_age_hours: int = 24):
        import time
        cutoff = time.time() - (max_age_hours * 3600)
        for f in TEMP_PDF_DIR.iterdir():
            if f.stat().st_mtime < cutoff:
                try:
                    f.unlink()
                except Exception:
                    pass

# =============================================================================
# PDF HIGHLIGHTER
# =============================================================================
class PDFHighlighter:
    DEFAULT_VALID_COLOR = (0.2, 0.8, 0.2)    # Green
    DEFAULT_INVALID_COLOR = (1.0, 0.3, 0.3)  # Red

    def __init__(self, dpi: int = 150):
        self.dpi = dpi
        self.zoom = dpi / 72

    def render_page_with_highlight(self, pdf_bytes: bytes, page_num: int,
                                    highlight: Optional[HighlightRegion] = None) -> PDFPageImage:
        try:
            import fitz
        except ImportError:
            raise ImportError("PyMuPDF (fitz) is required for PDF rendering")

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if page_num >= len(doc):
            page_num = 0

        page = doc[page_num]

        if highlight:
            rect = fitz.Rect(highlight.x0, highlight.y0, highlight.x1, highlight.y1)
            annot = page.add_highlight_annot(rect)
            annot.set_colors(stroke=highlight.color)
            annot.set_opacity(highlight.alpha)
            annot.update()

            shape = page.new_shape()
            shape.draw_rect(rect)
            shape.finish(color=highlight.color, width=2, fill=None)
            shape.commit()

        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        b64 = base64.b64encode(png_bytes).decode('utf-8')

        result = PDFPageImage(
            page_num=page_num,
            image_base64=b64,
            width=pix.width,
            height=pix.height,
            has_highlight=highlight is not None
        )
        doc.close()
        return result

    def find_and_highlight(self, pdf_bytes: bytes, search_text: str,
                           is_valid: bool = True) -> Tuple[Optional[PDFPageImage], Dict[str, Any]]:
        try:
            import fitz
        except ImportError:
            return None, {"found": False, "error": "PyMuPDF not installed"}

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        search_clean = ' '.join(search_text.split())
        words = search_clean.split()

        # Build search strategies: full text, then progressively shorter prefixes,
        # then individual distinctive phrases (3-word windows from the start)
        strategies = [search_clean[:120]]
        for n in [10, 7, 5, 4, 3]:
            if len(words) >= n:
                strategies.append(' '.join(words[:n]))
        # Also try middle portion (sometimes citations start mid-sentence)
        if len(words) > 6:
            strategies.append(' '.join(words[1:6]))
            strategies.append(' '.join(words[2:7]))

        for strategy in strategies:
            strategy = strategy.strip()
            if not strategy or len(strategy) < 6:
                continue
            for page_num in range(len(doc)):
                instances = doc[page_num].search_for(strategy)
                if instances:
                    rect = instances[0]
                    doc.close()
                    color = self.DEFAULT_VALID_COLOR if is_valid else self.DEFAULT_INVALID_COLOR
                    highlight = HighlightRegion(
                        page_num=page_num,
                        x0=rect.x0 - 2, y0=rect.y0 - 2,
                        x1=rect.x1 + 2, y1=rect.y1 + 2,
                        color=color
                    )
                    page_image = self.render_page_with_highlight(pdf_bytes, page_num, highlight)
                    return page_image, {
                        "found": True, "page": page_num,
                        "bbox": (rect.x0, rect.y0, rect.x1, rect.y1),
                        "strategy_used": strategy[:30] + "..." if len(strategy) > 30 else strategy
                    }

        doc.close()
        page_image = self.render_page_with_highlight(pdf_bytes, 0, None)
        return page_image, {"found": False, "page": 0, "bbox": None, "reason": "Text not found in PDF"}

    def get_page_count(self, pdf_bytes: bytes) -> int:
        try:
            import fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            count = len(doc)
            doc.close()
            return count
        except Exception:
            return 0

    def render_page(self, pdf_bytes: bytes, page_num: int) -> PDFPageImage:
        return self.render_page_with_highlight(pdf_bytes, page_num, None)

# =============================================================================
# CITATION HIGHLIGHT MODAL HELPER
# =============================================================================
class CitationHighlightModal:
    def __init__(self, dpi: int = 150):
        self.highlighter = PDFHighlighter(dpi=dpi)

    def prepare_modal_data(self, pdf_bytes: bytes, citation_text: str,
                           similarity: float, is_valid: bool) -> Dict[str, Any]:
        page_image, search_result = self.highlighter.find_and_highlight(
            pdf_bytes, citation_text, is_valid=is_valid
        )
        return {
            "citation": citation_text,
            "similarity": similarity,
            "similarity_percent": f"{similarity * 100:.1f}%",
            "is_valid": is_valid,
            "status": "VERIFIED" if is_valid else "NOT VERIFIED",
            "found_in_pdf": search_result["found"],
            "page_num": search_result.get("page", 0) + 1,
            "image_base64": page_image.image_base64 if page_image else "",
            "image_width": page_image.width if page_image else 0,
            "image_height": page_image.height if page_image else 0,
            "bbox": search_result.get("bbox")
        }

# =============================================================================
# HTML GENERATION
# =============================================================================
def get_highlighted_page_html(modal_data: Dict[str, Any], max_width: int = 700) -> str:
    status_color = "#22c55e" if modal_data["is_valid"] else "#ef4444"
    status_bg = "#dcfce7" if modal_data["is_valid"] else "#fee2e2"
    border_color = "#22c55e" if modal_data["is_valid"] else "#ef4444"

    if modal_data["image_width"] > 0:
        scale = min(1.0, max_width / modal_data["image_width"])
        display_width = int(modal_data["image_width"] * scale)
    else:
        display_width = max_width

    return f"""
    <div style="background:white;border-radius:8px;padding:16px;border:2px solid {border_color};">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <span style="font-weight:600;color:#1e293b;">Page {modal_data['page_num']}</span>
            <span style="background:{status_bg};color:{status_color};padding:4px 12px;border-radius:12px;font-size:14px;font-weight:600;">
                {modal_data['status']}
            </span>
        </div>
        <div style="border:1px solid #e2e8f0;border-radius:4px;overflow:hidden;margin-bottom:12px;">
            <img src="data:image/png;base64,{modal_data['image_base64']}"
                 style="width:{display_width}px;display:block;" alt="PDF page"/>
        </div>
        <div style="background:#f8fafc;padding:12px;border-radius:6px;border-left:4px solid {status_color};">
            <div style="font-size:12px;color:#64748b;margin-bottom:4px;">Citation:</div>
            <div style="color:#1e293b;font-style:italic;">"{modal_data['citation'][:200]}{'...' if len(modal_data['citation']) > 200 else ''}"</div>
        </div>
        <div style="display:flex;gap:16px;margin-top:12px;font-size:14px;color:#64748b;">
            <span>Similarity: <strong style="color:#1e293b;">{modal_data['similarity_percent']}</strong></span>
            <span>Threshold: <strong style="color:#1e293b;">80%</strong></span>
            <span>Found in PDF: <strong style="color:#1e293b;">{'Yes' if modal_data['found_in_pdf'] else 'No'}</strong></span>
        </div>
    </div>
    """

def create_citation_popup_data(pdf_bytes: bytes, citations_with_results: List[Dict]) -> List[Dict]:
    modal_helper = CitationHighlightModal()
    popup_data = []
    for cr in citations_with_results:
        citation = cr.get("citation", "")
        if not citation:
            continue
        data = modal_helper.prepare_modal_data(
            pdf_bytes, citation, cr.get("similarity", 0), cr.get("valid", False)
        )
        popup_data.append(data)
    return popup_data

__all__ = [
    'HighlightRegion', 'PDFPageImage', 'PDFStorage', 'PDFHighlighter',
    'CitationHighlightModal', 'get_highlighted_page_html', 'create_citation_popup_data'
]