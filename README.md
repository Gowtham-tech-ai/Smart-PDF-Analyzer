# 📄 Smart PDF Analyzer

> **One Tool. Many Powerful Features.**
> Image & Text Extraction, Image Blurring, Image Processing, Keyword Highlighter & Linker for PDFs.

---

## 🌟 Key Features

1. **Image & Text Extraction**
   - Direct digital extraction of text, blocks, fonts, and layouts using PyMuPDF.
   - Intelligent fallback/detection of scanned pages with Tesseract OCR & OpenCV preprocessing.
   - Multi-language OCR support (`eng`, `spa`, `deu`, `fra`, etc.).
   - Extraction of embedded raster images.

2. **Image Blurring & Redaction**
   - Automatically detects and blurs PII (emails, phone numbers, patient IDs, SSNs, credit cards).
   - Supports custom keyword redaction and Gaussian blur patches.
   - Applies permanent PDF redaction annotations to purge underlying digital text.

3. **Image Processing Techniques**
   - Grayscale conversion (`cv2.cvtColor`).
   - Sharpening using convolution kernels.
   - Edge detection (Canny / Sobel).
   - Contrast enhancement using CLAHE.
   - Otsu binarization / thresholding.
   - Re-embeds enhanced images directly into the PDF.

4. **Keyword Highlighter**
   - NLP-driven automated topical keyword discovery using NLTK and TF-IDF n-grams.
   - Custom keyword list matching.
   - Stop words filtering with custom include/exclude words.
   - Visual color palette selector (Yellow, Green, Cyan, Magenta, Orange).

5. **Smart Linker for PDFs**
   - Automatically detects web URLs (`http`, `https`, `www`) and creates clickable links.
   - Automatically detects email addresses and creates `mailto:` links.
   - Automatically detects cross-references (e.g. *"See page 2"*) and creates internal PDF page jump links.
   - Generates an interactive index of all links.

6. **Additional Capabilities**
   - **Batch Processing**: Queue multiple PDFs and export processed results in a `.zip` archive.
   - **In-PDF Search**: Interactive text search with occurrences count and visual bounding box overlays.
   - **Pre-configured Blueprints**: 1-click loading of demo Patient Reports, Renewable Energy Articles, and Scanned Invoices.
   - **Export Options**: Download processed PDFs, extracted text `.txt`, and enhanced images.

---

## 🏗 Directory Architecture

```
smart_pdf_analyzer/
├── app.py                      # Interactive Streamlit Web Application
├── requirements.txt            # Dependency manifest
├── README.md                   # Documentation & guide
├── core/
│   ├── __init__.py
│   ├── text_extractor.py       # Direct text extraction & layout preservation
│   ├── ocr_engine.py           # OpenCV preprocessing & Tesseract multi-language OCR
│   ├── image_processor.py      # Grayscale, sharpening, edge detection, CLAHE, binarization
│   ├── redact_blur.py          # Bounding box & PII blurring / redaction engine
│   ├── keyword_highlighter.py  # NLTK keyword discovery & PyMuPDF highlight annotator
│   ├── pdf_linker.py           # URL, email, and internal page reference linker
│   └── batch_processor.py      # Single and batch pipeline orchestrator
├── utils/
│   ├── __init__.py
│   ├── sample_generator.py     # Synthetic demo PDFs generator (Patient report, energy, invoice)
│   └── visualizer.py           # PDF-to-image renderers and visual search highlighter
└── tests/
    └── test_core.py            # Complete 7-suite unit and integration test
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure Python 3.10+ and Tesseract OCR are installed on your machine.
- Tesseract default Windows path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Unit Tests
```bash
python tests/test_core.py
```

### 4. Launch the Streamlit Web Application
```bash
streamlit run app.py
```

Once launched, your browser will open `http://localhost:8501`. You can upload your own PDFs or select **"Load Sample Demo"** in the sidebar to try all features instantly!
