import io
from typing import Dict, Any, List, Optional, Tuple, Literal
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF


def pil_to_cv(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR numpy array."""
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv_to_pil(cv_img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR or Grayscale numpy array to PIL Image."""
    if len(cv_img.shape) == 2:
        return Image.fromarray(cv_img)
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def apply_grayscale(image: np.ndarray | Image.Image) -> np.ndarray:
    """Convert image to grayscale."""
    cv_img = pil_to_cv(image) if isinstance(image, Image.Image) else image
    if len(cv_img.shape) == 3:
        return cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    return cv_img.copy()


def apply_sharpening(image: np.ndarray | Image.Image, strength: float = 1.0) -> np.ndarray:
    """Apply unsharp masking or high-pass sharpening kernel."""
    cv_img = pil_to_cv(image) if isinstance(image, Image.Image) else image
    kernel = np.array([
        [0, -1 * strength, 0],
        [-1 * strength, 4 * strength + 1, -1 * strength],
        [0, -1 * strength, 0]
    ], dtype=np.float32)
    sharpened = cv2.filter2D(cv_img, -1, kernel)
    return sharpened


def apply_edge_detection(
    image: np.ndarray | Image.Image,
    method: Literal["canny", "sobel"] = "canny",
    low_threshold: int = 50,
    high_threshold: int = 150
) -> np.ndarray:
    """Apply Canny or Sobel edge detection."""
    cv_img = pil_to_cv(image) if isinstance(image, Image.Image) else image
    gray = apply_grayscale(cv_img)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    if method == "canny":
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        return edges
    else:  # Sobel
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobelx, sobely)
        magnitude = np.uint8(np.clip(magnitude, 0, 255))
        return magnitude


def apply_contrast_enhancement(
    image: np.ndarray | Image.Image,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    cv_img = pil_to_cv(image) if isinstance(image, Image.Image) else image
    if len(cv_img.shape) == 3:
        # Convert to LAB color space and apply CLAHE to L-channel
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(cv_img)


def apply_binarization(image: np.ndarray | Image.Image) -> np.ndarray:
    """Apply Otsu's thresholding for high-contrast B&W document cleanup."""
    gray = apply_grayscale(image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def process_image(
    image: Image.Image | np.ndarray,
    techniques: List[str],
    params: Optional[Dict[str, Any]] = None
) -> Image.Image:
    """
    Apply an ordered sequence of image processing techniques.
    Techniques: 'grayscale', 'sharpen', 'edge_canny', 'edge_sobel', 'contrast', 'binarize'
    """
    params = params or {}
    cv_img = pil_to_cv(image) if isinstance(image, Image.Image) else image.copy()

    for tech in techniques:
        if tech == "grayscale":
            cv_img = apply_grayscale(cv_img)
        elif tech == "sharpen":
            strength = params.get("sharpen_strength", 1.0)
            cv_img = apply_sharpening(cv_img, strength=strength)
        elif tech == "edge_canny":
            low = params.get("canny_low", 50)
            high = params.get("canny_high", 150)
            cv_img = apply_edge_detection(cv_img, method="canny", low_threshold=low, high_threshold=high)
        elif tech == "edge_sobel":
            cv_img = apply_edge_detection(cv_img, method="sobel")
        elif tech == "contrast":
            clip = params.get("clahe_clip", 2.0)
            cv_img = apply_contrast_enhancement(cv_img, clip_limit=clip)
        elif tech == "binarize":
            cv_img = apply_binarization(cv_img)

    return cv_to_pil(cv_img)


def process_and_replace_pdf_images(
    doc: fitz.Document,
    techniques: List[str],
    params: Optional[Dict[str, Any]] = None
) -> int:
    """
    Finds raster images embedded in the PDF, applies the selected processing
    techniques to each, and updates the image streams directly in the PDF.
    Returns the count of modified images.
    """
    count = 0
    seen_xrefs = set()

    for page in doc:
        image_list = page.get_images(full=True)
        for img_info in image_list:
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            base_img = doc.extract_image(xref)
            img_bytes = base_img["image"]
            try:
                pil_img = Image.open(io.BytesIO(img_bytes))
                processed = process_image(pil_img, techniques, params)
                
                # Convert processed back to PNG bytes
                buf = io.BytesIO()
                processed.save(buf, format="PNG")
                new_bytes = buf.getvalue()

                # Replace image stream in PDF using PyMuPDF
                doc.update_stream(xref, new_bytes)
                count += 1
            except Exception:
                continue

    return count
