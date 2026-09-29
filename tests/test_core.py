import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import fitz
from PIL import Image

from core.text_extractor import extract_direct_text, extract_embedded_images, extract_document_hybrid
from core.ocr_engine import extract_text_from_image, get_available_languages
from core.image_processor import process_image, apply_grayscale, apply_edge_detection, apply_sharpening
from core.redact_blur import redact_text_in_pdf, blur_image_regions
from core.keyword_highlighter import highlight_keywords_in_pdf, extract_keywords_nlp
from core.pdf_linker import find_and_create_links_in_pdf
from core.batch_processor import process_single_pdf, process_batch
from utils.sample_generator import (
    generate_sample_patient_report,
    generate_sample_renewable_energy_article,
    generate_sample_scanned_ocr_pdf,
    create_synthetic_test_image
)


def test_sample_generation():
    print("[1/7] Testing Sample PDF Generators...")
    pdf1 = generate_sample_patient_report()
    assert len(pdf1) > 0, "Patient report generation failed"

    pdf2 = generate_sample_renewable_energy_article()
    assert len(pdf2) > 0, "Renewable energy report generation failed"

    pdf3 = generate_sample_scanned_ocr_pdf()
    assert len(pdf3) > 0, "Scanned OCR PDF generation failed"
    print("  -> Passed: All synthetic sample PDFs created successfully.")


def test_text_and_ocr_extraction():
    print("[2/7] Testing Direct Text & OCR Extraction...")
    pdf_bytes = generate_sample_renewable_energy_article()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    direct_res = extract_direct_text(doc)
    assert "renewable energy" in direct_res["full_text"].lower()
    assert direct_res["page_count"] == 2

    # Test OCR on synthetic scanned PDF
    scanned_pdf = generate_sample_scanned_ocr_pdf()
    doc_scanned = fitz.open(stream=scanned_pdf, filetype="pdf")
    hybrid = extract_document_hybrid(doc_scanned, ocr_if_scanned=True)
    assert hybrid["pages"][0]["is_scanned"] is True
    assert len(hybrid["pages"][0]["text"]) > 10
    print("  -> Passed: Direct text and Tesseract OCR successfully extracted.")


def test_image_processing_filters():
    print("[3/7] Testing Image Processing Filters...")
    img = create_synthetic_test_image("Filter Test")
    
    gray = apply_grayscale(img)
    assert len(gray.shape) == 2

    edges = apply_edge_detection(img, method="canny")
    assert edges.shape == gray.shape

    sharpened = apply_sharpening(img)
    assert sharpened.shape == (300, 600, 3)

    multi = process_image(img, ["contrast", "grayscale", "sharpen"])
    assert isinstance(multi, Image.Image)
    print("  -> Passed: Grayscale, Canny edge detection, sharpening, and contrast pipelines passed.")


def test_blurring_and_redaction():
    print("[4/7] Testing Image Blurring and PII Redaction...")
    pdf_bytes = generate_sample_patient_report()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # Redact patient email, phone, and name
    targets = [r"johndoe@email.com", r"\+91 98765 43210", r"John Doe"]
    count = redact_text_in_pdf(doc, targets, blur_effect=True)
    assert count >= 3, f"Expected at least 3 redactions, got {count}"
    
    # Verify text is purged
    page_text = doc[0].get_text()
    assert "johndoe@email.com" not in page_text
    print(f"  -> Passed: {count} sensitive PII targets redacted and blurred successfully.")


def test_keyword_highlighter():
    print("[5/7] Testing NLP Keyword Detection & Highlighting...")
    pdf_bytes = generate_sample_renewable_energy_article()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    full_text = doc[0].get_text() + " " + doc[1].get_text()
    auto_kws = extract_keywords_nlp(full_text, top_n=5)
    assert len(auto_kws) > 0

    manual_kws = ["renewable energy", "solar energy", "wind energy", "hydropower", "pollution"]
    summary = highlight_keywords_in_pdf(doc, manual_kws, color_name="Yellow")
    assert summary["total_highlights"] > 0
    assert summary["keyword_counts"]["renewable energy"] > 0
    print(f"  -> Passed: {summary['total_highlights']} keyword instances highlighted across pages.")


def test_pdf_linker():
    print("[6/7] Testing Smart PDF Linker...")
    pdf_bytes = generate_sample_renewable_energy_article()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    link_res = find_and_create_links_in_pdf(doc, link_urls=True, link_emails=True, link_page_refs=True)
    assert link_res["total_links"] >= 3, f"Expected at least 3 links, got {link_res['total_links']}"
    
    types = [l["type"] for l in link_res["links"]]
    assert "URL" in types
    assert "Email" in types
    assert "Internal Page Jump" in types
    print(f"  -> Passed: {link_res['total_links']} clickable links (URLs, emails, and page jumps) inserted.")


def test_batch_processor():
    print("[7/7] Testing Full Pipeline & Batch Processing...")
    pdf1 = generate_sample_patient_report()
    pdf2 = generate_sample_renewable_energy_article()

    files = {
        "report.pdf": pdf1,
        "energy.pdf": pdf2
    }

    opts = {
        "do_text_extraction": True,
        "do_blur_redaction": True,
        "blur_pii_auto": True,
        "image_techniques": ["contrast"],
        "do_highlight": True,
        "keywords": ["energy", "patient"],
        "auto_keywords": True,
        "do_links": True
    }

    zip_buf, summaries = process_batch(files, opts)
    assert len(summaries) == 2
    assert zip_buf.getbuffer().nbytes > 1000
    print("  -> Passed: Batch runner processed files and created valid ZIP archive.")


if __name__ == "__main__":
    test_sample_generation()
    test_text_and_ocr_extraction()
    test_image_processing_filters()
    test_blurring_and_redaction()
    test_keyword_highlighter()
    test_pdf_linker()
    test_batch_processor()
    print("\nALL 7 TEST SUITES PASSED FLAWLESSLY!")
