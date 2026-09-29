# core/__init__.py
"""
Core processing modules for Smart PDF Analyzer:
- text_extractor: Direct text and layout extraction from PDFs
- ocr_engine: OCR text extraction with OpenCV preprocessing for scanned pages/images
- image_processor: Image filters (Grayscale, Sharpen, Edge detection, Contrast CLAHE)
- redact_blur: Image blurring and digital PDF redaction
- keyword_highlighter: NLP keyword detection and PDF highlight annotations
- pdf_linker: Automatic URL, email, and internal page reference linking
- batch_processor: Pipeline orchestration for single and batch processing
"""
