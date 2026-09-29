import os
import shutil
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from PIL import Image
import pytesseract
import fitz  # PyMuPDF

# Detect and set Tesseract executable path on Windows if needed
def configure_tesseract():
    common_win_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    if shutil.which("tesseract") is None:
        for path in common_win_paths:
            if os.path.isfile(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break

configure_tesseract()


def get_available_languages() -> List[str]:
    """Return available Tesseract languages installed on the system."""
    try:
        langs = pytesseract.get_languages(config="")
        return langs if langs else ["eng"]
    except Exception:
        return ["eng"]


def preprocess_image_for_ocr(cv_image: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """
    Enhance document image before OCR using OpenCV techniques:
    - Grayscale conversion
    - Noise filtering
    - Adaptive / Otsu binarization
    """
    if len(cv_image.shape) == 3:
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv_image.copy()

    # Denoise
    denoised = cv2.medianBlur(gray, 3)

    if method == "otsu":
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return binary
    return denoised


def extract_text_from_image(
    image: Image.Image | np.ndarray,
    lang: str = "eng",
    preprocess: bool = True
) -> str:
    """Extract text from a PIL Image or OpenCV array using Tesseract OCR."""
    if isinstance(image, Image.Image):
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        cv_img = image

    if preprocess:
        proc_img = preprocess_image_for_ocr(cv_img)
    else:
        proc_img = cv_img

    try:
        text = pytesseract.image_to_string(proc_img, lang=lang)
        return text.strip()
    except Exception as e:
        return f"[OCR Error: {e}]"


def extract_ocr_data_with_bboxes(
    image: Image.Image | np.ndarray,
    lang: str = "eng"
) -> List[Dict[str, Any]]:
    """Return detailed word-level OCR boxes, confidence scores, and text."""
    if isinstance(image, Image.Image):
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        cv_img = image

    proc_img = preprocess_image_for_ocr(cv_img)
    data = pytesseract.image_to_data(proc_img, lang=lang, output_type=pytesseract.Output.DICT)

    results = []
    n_boxes = len(data["text"])
    for i in range(n_boxes):
        word = data["text"][i].strip()
        conf = int(data["conf"][i]) if str(data["conf"][i]).lstrip("-").isdigit() else 0
        if word and conf > 20:
            results.append({
                "text": word,
                "confidence": conf,
                "box": (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
            })
    return results


def extract_text_from_pdf_page_ocr(
    page: fitz.Page,
    lang: str = "eng",
    dpi: int = 200
) -> str:
    """Renders a PDF page to image and performs OCR extraction."""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return extract_text_from_image(img, lang=lang, preprocess=True)
