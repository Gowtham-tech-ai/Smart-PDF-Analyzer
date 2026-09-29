import io
from typing import Dict, Any, List, Tuple
from PIL import Image
import fitz  # PyMuPDF
from .ocr_engine import extract_text_from_pdf_page_ocr


def is_scanned_page(page: fitz.Page, min_char_count: int = 40) -> bool:
    """Detect if a page is likely a scanned document by inspecting native text length."""
    text = page.get_text().strip()
    return len(text) < min_char_count


def extract_direct_text(doc: fitz.Document) -> Dict[str, Any]:
    """
    Extracts direct selectable text from a PDF document.
    Returns:
        dict containing 'full_text', 'page_texts', and 'metadata'.
    """
    page_texts = []
    blocks_by_page = []

    for page_idx, page in enumerate(doc):
        p_text = page.get_text("text")
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, "text", block_no, block_type)
        page_texts.append(p_text)
        blocks_by_page.append([
            {
                "bbox": (b[0], b[1], b[2], b[3]),
                "text": b[4],
                "block_no": b[5],
                "type": "text" if b[6] == 0 else "image"
            }
            for b in blocks
        ])

    return {
        "full_text": "\n\n--- Page Break ---\n\n".join(page_texts),
        "page_texts": page_texts,
        "blocks_by_page": blocks_by_page,
        "page_count": len(doc),
        "metadata": doc.metadata
    }


def extract_embedded_images(doc: fitz.Document) -> List[Dict[str, Any]]:
    """
    Extracts all raster images embedded inside the PDF pages.
    Returns:
        list of dicts containing page_num, xref, pil_image, format, width, height.
    """
    images_info = []
    seen_xrefs = set()

    for page_idx, page in enumerate(doc):
        image_list = page.get_images(full=True)
        for img_tuple in image_list:
            xref = img_tuple[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            try:
                pil_img = Image.open(io.BytesIO(image_bytes))
                images_info.append({
                    "page_num": page_idx + 1,
                    "xref": xref,
                    "image": pil_img,
                    "format": image_ext,
                    "width": pil_img.width,
                    "height": pil_img.height,
                    "size_bytes": len(image_bytes)
                })
            except Exception:
                continue

    return images_info


def extract_document_hybrid(
    doc: fitz.Document,
    ocr_if_scanned: bool = True,
    ocr_lang: str = "eng"
) -> Dict[str, Any]:
    """
    Extract text using direct extraction if digital, or fallback/supplement with OCR if scanned.
    """
    pages_output = []
    for idx, page in enumerate(doc):
        direct = page.get_text("text").strip()
        scanned = is_scanned_page(page)

        if scanned and ocr_if_scanned:
            ocr_text = extract_text_from_pdf_page_ocr(page, lang=ocr_lang)
            text_used = ocr_text
            method = "OCR"
        else:
            text_used = direct
            method = "Direct"

        pages_output.append({
            "page_num": idx + 1,
            "text": text_used,
            "method": method,
            "is_scanned": scanned
        })

    combined = "\n\n--- Page Break ---\n\n".join(
        f"Page {p['page_num']} ({p['method']}):\n{p['text']}" for p in pages_output
    )

    return {
        "pages": pages_output,
        "combined_text": combined,
        "page_count": len(doc)
    }
