from typing import List, Optional, Tuple
from PIL import Image, ImageDraw
import fitz  # PyMuPDF


def render_pdf_page_to_image(
    doc: fitz.Document,
    page_num: int,
    dpi: int = 150
) -> Optional[Image.Image]:
    """Renders a 1-indexed page of a PyMuPDF document to a PIL Image."""
    if page_num < 1 or page_num > len(doc):
        return None
    page = doc[page_num - 1]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def get_pdf_page_count(doc: fitz.Document) -> int:
    """Return total pages in PDF."""
    return len(doc)


def search_and_highlight_in_image(
    base_image: Image.Image,
    doc: fitz.Document,
    page_num: int,
    query: str,
    dpi: int = 150
) -> Tuple[Image.Image, int]:
    """
    Overlays visual search highlight boxes onto a rendered page image.
    Returns (annotated_image, count_found).
    """
    if not query.strip() or page_num < 1 or page_num > len(doc):
        return base_image, 0

    page = doc[page_num - 1]
    rects = page.search_for(query.strip())
    if not rects:
        return base_image, 0

    annotated = base_image.copy()
    draw = ImageDraw.Draw(annotated)
    scale = dpi / 72.0

    for r in rects:
        # Scale PDF coordinates (points) to pixel coordinates
        x0 = int(r.x0 * scale)
        y0 = int(r.y0 * scale)
        x1 = int(r.x1 * scale)
        y1 = int(r.y1 * scale)
        draw.rectangle([x0, y0, x1, y1], outline=(255, 60, 60), width=3)

    return annotated, len(rects)
