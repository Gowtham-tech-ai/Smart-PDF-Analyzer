import io
import re
from typing import Dict, Any, List, Optional, Tuple, Literal
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
from .image_processor import pil_to_cv, cv_to_pil


def apply_blur_to_region(
    cv_img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    blur_type: Literal["gaussian", "pixelate"] = "gaussian",
    ksize: int = 25
) -> np.ndarray:
    """
    Applies Gaussian blur or pixelation (mosaic) to a specific bounding box (x, y, w, h).
    """
    x, y, w, h = bbox
    img_h, img_w = cv_img.shape[:2]

    # Ensure box is within bounds
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))

    roi = cv_img[y : y + h, x : x + w]
    if roi.size == 0:
        return cv_img

    if blur_type == "gaussian":
        # Ensure ksize is odd
        k = ksize if ksize % 2 == 1 else ksize + 1
        blurred_roi = cv2.GaussianBlur(roi, (k, k), 0)
    else:  # pixelate
        pixel_size = max(4, ksize // 2)
        small_w = max(1, w // pixel_size)
        small_h = max(1, h // pixel_size)
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        blurred_roi = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    cv_img[y : y + h, x : x + w] = blurred_roi
    return cv_img


def blur_image_regions(
    image: Image.Image | np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    blur_type: Literal["gaussian", "pixelate"] = "gaussian",
    blur_strength: int = 25
) -> Image.Image:
    """Blur multiple rectangular regions on an image."""
    cv_img = pil_to_cv(image) if isinstance(image, Image.Image) else image.copy()
    for box in bboxes:
        cv_img = apply_blur_to_region(cv_img, box, blur_type=blur_type, ksize=blur_strength)
    return cv_to_pil(cv_img)


# Common PII Regex Patterns
PII_PATTERNS = {
    "Email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b",
    "Phone Number": r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b",
    "ID / SSN": r"\b\d{3}-\d{2}-\d{4}\b|\bID:?\s*[A-Z0-9]{5,12}\b",
    "Credit Card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
}


def redact_text_in_pdf(
    doc: fitz.Document,
    patterns_or_keywords: List[str],
    blur_effect: bool = True,
    fill_color: Tuple[float, float, float] = (0.9, 0.9, 0.9)
) -> int:
    """
    Finds sensitive text matching patterns or keywords across PDF pages
    and applies privacy redactions/blur overlays.
    Returns the total count of redacted regions.
    """
    redaction_count = 0

    for page in doc:
        page_text = page.get_text("text")
        page_rects = []

        for pat in patterns_or_keywords:
            if not pat.strip():
                continue
            try:
                rx = re.compile(pat, re.IGNORECASE)
                # Find all matching text spans in the page text
                matches = rx.findall(page_text)
                for m in matches:
                    m_str = m[0] if isinstance(m, tuple) else m
                    m_str = m_str.strip()
                    if m_str:
                        rects = page.search_for(m_str)
                        page_rects.extend(rects)
            except re.error:
                pass

            # Also search as literal phrase
            literal_rects = page.search_for(pat.strip())
            page_rects.extend(literal_rects)

        # Deduplicate rects by rounding coordinates slightly
        unique_rects = []
        seen_coords = set()
        for r in page_rects:
            key = (round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1))
            if key not in seen_coords:
                seen_coords.add(key)
                unique_rects.append(r)

        for rect in unique_rects:
            if blur_effect:
                try:
                    pix = page.get_pixmap(clip=rect, dpi=150)
                    if pix.width > 0 and pix.height > 0:
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        cv_img = pil_to_cv(img)
                        k = max(11, min(pix.width, pix.height) // 2)
                        k = k if k % 2 == 1 else k + 1
                        blurred = cv2.GaussianBlur(cv_img, (k, k), 0)
                        blurred_pil = cv_to_pil(blurred)
                        
                        buf = io.BytesIO()
                        blurred_pil.save(buf, format="PNG")
                        page.insert_image(rect, stream=buf.getvalue(), overlay=True)
                except Exception:
                    pass

            page.add_redact_annot(rect, fill=fill_color if not blur_effect else None)
            redaction_count += 1

        page.apply_redactions()

    return redaction_count


def blur_custom_coordinates(
    doc: fitz.Document,
    page_num: int,
    rect: Tuple[float, float, float, float],
    blur_strength: int = 25
) -> bool:
    """
    Blurs an arbitrary bounding box (x0, y0, x1, y1) on a given 1-indexed page.
    """
    if page_num < 1 or page_num > len(doc):
        return False
    
    page = doc[page_num - 1]
    frect = fitz.Rect(*rect)
    
    try:
        pix = page.get_pixmap(clip=frect, dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        cv_img = pil_to_cv(img)
        k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        blurred = cv2.GaussianBlur(cv_img, (k, k), 0)
        blurred_pil = cv_to_pil(blurred)

        buf = io.BytesIO()
        blurred_pil.save(buf, format="PNG")
        page.insert_image(frect, stream=buf.getvalue(), overlay=True)
        page.add_redact_annot(frect)
        page.apply_redactions()
        return True
    except Exception:
        return False
