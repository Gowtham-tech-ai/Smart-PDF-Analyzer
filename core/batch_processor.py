import io
import os
import zipfile
from typing import Dict, Any, List, Optional, Callable, Tuple
import fitz  # PyMuPDF

from .text_extractor import extract_document_hybrid
from .image_processor import process_and_replace_pdf_images
from .redact_blur import redact_text_in_pdf, PII_PATTERNS
from .keyword_highlighter import highlight_keywords_in_pdf, extract_keywords_nlp
from .pdf_linker import find_and_create_links_in_pdf


def process_single_pdf(
    pdf_bytes: bytes,
    options: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Applies the full user-configured processing pipeline to a PDF document.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    results: Dict[str, Any] = {
        "text_extraction": None,
        "image_count_processed": 0,
        "redactions_count": 0,
        "highlights_summary": None,
        "links_summary": None,
        "processed_pdf_bytes": None
    }

    # 1. Text Extraction / Hybrid OCR
    if options.get("do_text_extraction", False):
        ocr_lang = options.get("ocr_lang", "eng")
        ocr_if_scanned = options.get("ocr_if_scanned", True)
        results["text_extraction"] = extract_document_hybrid(
            doc, ocr_if_scanned=ocr_if_scanned, ocr_lang=ocr_lang
        )

    # 2. Image Processing Techniques
    img_techniques = options.get("image_techniques", [])
    if img_techniques:
        params = options.get("image_params", {})
        count = process_and_replace_pdf_images(doc, img_techniques, params)
        results["image_count_processed"] = count

    # 3. Image Blurring & PII Redaction
    if options.get("do_blur_redaction", False):
        redact_targets = []
        if options.get("blur_pii_auto", True):
            for pat in PII_PATTERNS.values():
                redact_targets.append(pat)
        custom_redact = options.get("custom_redact_words", [])
        redact_targets.extend(custom_redact)

        blur_effect = options.get("blur_effect", True)
        results["redactions_count"] = redact_text_in_pdf(
            doc, redact_targets, blur_effect=blur_effect
        )

    # 4. Keyword Highlighting
    if options.get("do_highlight", False):
        keywords = list(options.get("keywords", []))
        if options.get("auto_keywords", False):
            # Extract full text to run NLP
            full_txt = ""
            for p in doc:
                full_txt += p.get_text() + " "
            custom_sw = set(options.get("custom_stopwords", []))
            nlp_kws = extract_keywords_nlp(full_txt, top_n=options.get("top_kws", 8), custom_stopwords=custom_sw)
            keywords.extend(nlp_kws)
            keywords = list(dict.fromkeys(keywords))  # Deduplicate

        color = options.get("highlight_color", "Yellow")
        results["highlights_summary"] = highlight_keywords_in_pdf(
            doc, keywords, color_name=color
        )

    # 5. Smart Linker
    if options.get("do_links", False):
        link_urls = options.get("link_urls", True)
        link_emails = options.get("link_emails", True)
        link_page_refs = options.get("link_page_refs", True)
        custom_links = options.get("custom_links", {})
        results["links_summary"] = find_and_create_links_in_pdf(
            doc,
            link_urls=link_urls,
            link_emails=link_emails,
            link_page_refs=link_page_refs,
            custom_links=custom_links
        )

    # Save output to bytes
    out_buf = io.BytesIO()
    doc.save(out_buf, garbage=4, deflate=True)
    doc.close()
    results["processed_pdf_bytes"] = out_buf.getvalue()

    return results


def process_batch(
    files_dict: Dict[str, bytes],
    options: Dict[str, Any],
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Tuple[io.BytesIO, Dict[str, Any]]:
    """
    Processes multiple PDF files in batch and packs all results and extracted text
    into an in-memory ZIP archive.
    """
    zip_buffer = io.BytesIO()
    batch_summaries = {}
    total_files = len(files_dict)

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for idx, (filename, pdf_bytes) in enumerate(files_dict.items()):
            if progress_callback:
                progress_callback(idx + 1, total_files, filename)

            res = process_single_pdf(pdf_bytes, options)
            batch_summaries[filename] = {
                "redactions": res.get("redactions_count", 0),
                "images_processed": res.get("image_count_processed", 0),
                "highlights": res.get("highlights_summary", {}).get("total_highlights", 0) if res.get("highlights_summary") else 0,
                "links": res.get("links_summary", {}).get("total_links", 0) if res.get("links_summary") else 0,
            }

            # Add processed PDF to zip
            base_name, _ = os.path.splitext(filename)
            zip_file.writestr(f"processed/{base_name}_processed.pdf", res["processed_pdf_bytes"])

            # Add extracted text if available
            if res.get("text_extraction"):
                txt_content = res["text_extraction"]["combined_text"]
                zip_file.writestr(f"text_extractions/{base_name}_extracted.txt", txt_content)

    zip_buffer.seek(0)
    return zip_buffer, batch_summaries
