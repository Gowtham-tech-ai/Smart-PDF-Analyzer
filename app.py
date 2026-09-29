import io
import os
import json
import streamlit as st
from PIL import Image
import fitz  # PyMuPDF

from core.text_extractor import extract_direct_text, extract_embedded_images, extract_document_hybrid
from core.ocr_engine import get_available_languages, extract_text_from_image
from core.image_processor import process_image, pil_to_cv, cv_to_pil
from core.redact_blur import PII_PATTERNS
from core.keyword_highlighter import COLOR_PALETTE
from core.batch_processor import process_single_pdf, process_batch
from utils.sample_generator import (
    generate_sample_patient_report,
    generate_sample_renewable_energy_article,
    generate_sample_scanned_ocr_pdf
)
from utils.visualizer import render_pdf_page_to_image, search_and_highlight_in_image


# Page Configuration
st.set_page_config(
    page_title="Smart PDF Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling matching the project infographic
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #0F172A 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: #FFFFFF !important;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #93C5FD !important;
        font-size: 1.05rem;
        margin-bottom: 12px;
    }
    .badge-container {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 10px;
    }
    .badge {
        background: rgba(255, 255, 255, 0.15);
        color: #F8FAFC;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .metric-card {
        background: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


# Render Header
st.markdown("""
<div class="main-header">
    <h1>SMART PDF ANALYZER</h1>
    <p>Image & Text Extraction, Image Blurring, Image Processing, Keyword Highlighter & Linker for PDFs</p>
    <div class="badge-container">
        <span class="badge">✔ Enhance PDF readability</span>
        <span class="badge">✔ Extract & analyze content</span>
        <span class="badge">✔ Protect sensitive information</span>
        <span class="badge">✔ Navigate smarter with links</span>
    </div>
</div>
""", unsafe_allow_html=True)


# Sidebar - Workflow & Controls
with st.sidebar:
    st.header("⚙ System Workflow")
    
    input_mode = st.radio(
        "Source Input Mode",
        ["Single Document", "Batch Processing", "Load Sample Demo"],
        help="Process an uploaded PDF or explore pre-configured blueprints"
    )

    pdf_files: dict[str, bytes] = {}

    if input_mode == "Load Sample Demo":
        sample_choice = st.selectbox(
            "Select Pre-configured Blueprint:",
            [
                "Patient Report (Blurring & Medical Scan)",
                "Renewable Energy (Keywords & Smart Links)",
                "Scanned Invoice (Image & OCR Extraction)"
            ]
        )
        if sample_choice.startswith("Patient Report"):
            pdf_files["patient_report.pdf"] = generate_sample_patient_report()
            st.info("Loaded Patient Report: test PII blurring, patient info redaction, and scan enhancement.")
        elif sample_choice.startswith("Renewable Energy"):
            pdf_files["renewable_energy.pdf"] = generate_sample_renewable_energy_article()
            st.info("Loaded Renewable Energy Article: test keyword highlighting, URLs, emails, and page jumps.")
        else:
            pdf_files["scanned_invoice.pdf"] = generate_sample_scanned_ocr_pdf()
            st.info("Loaded Scanned Invoice: test pure Tesseract OCR and binarization.")

    elif input_mode == "Single Document":
        uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
        if uploaded_file is not None:
            pdf_files[uploaded_file.name] = uploaded_file.read()

    else:  # Batch Processing
        uploaded_files = st.file_uploader("Upload Multiple PDFs", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            for uf in uploaded_files:
                pdf_files[uf.name] = uf.read()

    st.markdown("---")
    st.subheader("🛠 Pipeline Operations")

    # Operation 1: Text & OCR
    enable_text = st.checkbox("1. Image & Text Extraction", value=True)
    ocr_lang = "eng"
    ocr_scanned = True
    if enable_text:
        avail_langs = get_available_languages()
        ocr_lang = st.selectbox("OCR Language", avail_langs, index=avail_langs.index("eng") if "eng" in avail_langs else 0)
        ocr_scanned = st.checkbox("Auto-detect & OCR Scanned Pages", value=True)

    # Operation 2: Image Blurring
    enable_blur = st.checkbox("2. Image Blurring & Redaction", value=True)
    blur_pii = True
    custom_redact_str = ""
    blur_effect = True
    if enable_blur:
        blur_pii = st.checkbox("Auto-detect & Blur PII (Emails, Phones, IDs)", value=True)
        custom_redact_str = st.text_input("Custom Redact Words (comma-separated)", "John Doe, Acute Respiratory")
        blur_effect = st.radio("Redaction Style", ["Gaussian Blur Patch", "Solid Blackout"], index=0) == "Gaussian Blur Patch"

    # Operation 3: Image Processing
    enable_img_proc = st.checkbox("3. Image Processing Techniques", value=True)
    img_techniques = []
    if enable_img_proc:
        tech_options = st.multiselect(
            "Select Filters for Embedded Images:",
            ["contrast", "sharpen", "grayscale", "edge_canny", "binarize"],
            default=["contrast"]
        )
        img_techniques = tech_options

    # Operation 4: Keyword Highlighter
    enable_highlight = st.checkbox("4. Keyword Highlighter", value=True)
    auto_kws = True
    custom_kws_str = ""
    custom_stopwords_str = ""
    highlight_color = "Yellow"
    if enable_highlight:
        auto_kws = st.checkbox("Auto-detect Keywords via NLP", value=True)
        custom_kws_str = st.text_input(
            "Manual Keywords (comma-separated)",
            "renewable energy, solar energy, wind energy, hydropower, pollution"
        )
        custom_stopwords_str = st.text_input(
            "Custom Stop Words to Exclude",
            "future, planet, sources"
        )
        highlight_color = st.selectbox("Highlight Color", list(COLOR_PALETTE.keys()), index=0)

    # Operation 5: PDF Linker
    enable_links = st.checkbox("5. Linker for PDFs", value=True)
    link_urls = True
    link_emails = True
    link_page_refs = True
    if enable_links:
        link_urls = st.checkbox("Auto-link Web URLs", value=True)
        link_emails = st.checkbox("Auto-link Email Addresses (mailto:)", value=True)
        link_page_refs = st.checkbox("Auto-link 'See page X' references", value=True)

    st.markdown("---")
    run_btn = st.button("🚀 Process Document(s)", type="primary", use_container_width=True)


# Build Pipeline Options Dict
options = {
    "do_text_extraction": enable_text,
    "ocr_lang": ocr_lang,
    "ocr_if_scanned": ocr_scanned,
    "do_blur_redaction": enable_blur,
    "blur_pii_auto": blur_pii,
    "custom_redact_words": [w.strip() for w in custom_redact_str.split(",") if w.strip()],
    "blur_effect": blur_effect,
    "image_techniques": img_techniques,
    "image_params": {"clahe_clip": 2.5, "sharpen_strength": 1.2},
    "do_highlight": enable_highlight,
    "auto_keywords": auto_kws,
    "keywords": [k.strip() for k in custom_kws_str.split(",") if k.strip()],
    "custom_stopwords": [s.strip().lower() for s in custom_stopwords_str.split(",") if s.strip()],
    "highlight_color": highlight_color,
    "do_links": enable_links,
    "link_urls": link_urls,
    "link_emails": link_emails,
    "link_page_refs": link_page_refs,
}


# Main Content Area
if not pdf_files:
    st.info("👋 Welcome! Please upload a PDF file using the sidebar, or select **'Load Sample Demo'** to explore pre-built blueprints instantly.")
    
    # Feature Showcase Grid
    st.subheader("Explore Included Modules:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔍 1. Text & OCR Extraction")
        st.write("Extract direct digital text or leverage OpenCV preprocessing + Tesseract OCR for scanned documents.")
        st.markdown("### 🕶 2. Image Blurring")
        st.write("Detect and blur sensitive personal details, medical reports, or custom rectangular regions.")
    with col2:
        st.markdown("### 🎨 3. Image Processing")
        st.write("Enhance embedded photos or scans using Grayscale, Sharpening, Canny Edge Detection, and CLAHE.")
        st.markdown("### 🖍 4. Keyword Highlighter")
        st.write("Identify salient topics via NLP TF-IDF and highlight occurrences throughout the PDF.")
    with col3:
        st.markdown("### 🔗 5. Smart PDF Linker")
        st.write("Convert URLs, emails, and cross-page references (e.g. 'See page 5') into clickable annotations.")
        st.markdown("### 📦 6. Batch & Search")
        st.write("Process entire document folders in batch and execute in-document interactive searches.")

else:
    # State tracking
    if run_btn or "processed_results" not in st.session_state:
        with st.spinner("Processing document pipeline..."):
            if len(pdf_files) == 1:
                filename = list(pdf_files.keys())[0]
                res = process_single_pdf(pdf_files[filename], options)
                st.session_state["processed_results"] = {filename: res}
                st.session_state["is_batch"] = False
            else:
                zip_buf, summaries = process_batch(pdf_files, options)
                st.session_state["batch_zip"] = zip_buf
                st.session_state["batch_summaries"] = summaries
                st.session_state["is_batch"] = True

    # Display results
    if st.session_state.get("is_batch", False):
        st.success(f"Batch processing completed for {len(pdf_files)} documents!")
        st.download_button(
            label="⬇ Download All Processed Files (ZIP)",
            data=st.session_state["batch_zip"].getvalue(),
            file_name="smart_pdf_analyzer_batch.zip",
            mime="application/zip",
            type="primary"
        )
        st.subheader("Batch Summary Report")
        st.dataframe(st.session_state["batch_summaries"])

    else:
        results_dict = st.session_state.get("processed_results", {})
        active_filename = list(pdf_files.keys())[0]
        res = results_dict.get(active_filename, {})
        processed_pdf_bytes = res.get("processed_pdf_bytes")

        # Top Metric Bar
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Images Processed", res.get("image_count_processed", 0))
        m2.metric("Redactions Applied", res.get("redactions_count", 0))
        m3.metric("Keywords Highlighted", res.get("highlights_summary", {}).get("total_highlights", 0) if res.get("highlights_summary") else 0)
        m4.metric("Clickable Links Created", res.get("links_summary", {}).get("total_links", 0) if res.get("links_summary") else 0)

        # Tabs matching the infographic architecture
        tab_preview, tab_text, tab_images, tab_links, tab_export = st.tabs([
            "📖 PDF Interactive Preview & Search",
            "📝 Extracted Text & OCR",
            "🖼 Processed Images",
            "🔗 Smart Linker Index",
            "💾 Download Center"
        ])

        # TAB 1: PDF Viewer & Search
        with tab_preview:
            col_view, col_search = st.columns([3, 1])

            with col_search:
                st.subheader("🔎 Search in PDF")
                search_query = st.text_input("Search keyword or phrase:", "")
                st.caption("Matches will be outlined in red on the rendered page.")

                # Open doc for page navigation
                orig_doc = fitz.open(stream=pdf_files[active_filename], filetype="pdf")
                proc_doc = fitz.open(stream=processed_pdf_bytes, filetype="pdf") if processed_pdf_bytes else orig_doc
                total_pages = len(proc_doc)

                page_selected = st.number_input("Go to Page:", min_value=1, max_value=total_pages, value=1)
                st.info(f"Viewing Page **{page_selected}** of **{total_pages}**")

                if search_query.strip():
                    total_matches = 0
                    for p_i in range(total_pages):
                        matches_p = len(proc_doc[p_i].search_for(search_query.strip()))
                        if matches_p > 0:
                            st.write(f"• Page {p_i + 1}: **{matches_p}** occurrence(s)")
                            total_matches += matches_p
                    st.metric("Total Occurrences", total_matches)

            with col_view:
                st.subheader(f"Processed PDF Page {page_selected}")
                rendered_page = render_pdf_page_to_image(proc_doc, page_selected, dpi=160)
                
                if rendered_page:
                    if search_query.strip():
                        rendered_page, _ = search_and_highlight_in_image(
                            rendered_page, proc_doc, page_selected, search_query.strip(), dpi=160
                        )
                    st.image(rendered_page, use_container_width=True)

        # TAB 2: Text Extraction & OCR
        with tab_text:
            st.subheader("Extracted Text Stream")
            extraction_data = res.get("text_extraction")
            if extraction_data:
                for p in extraction_data["pages"]:
                    with st.expander(f"📄 Page {p['page_num']} — Extraction Method: {p['method']}", expanded=True):
                        st.text_area(f"Page {p['page_num']} Text", p["text"], height=200)
                
                st.download_button(
                    label="⬇ Download All Extracted Text (.txt)",
                    data=extraction_data["combined_text"],
                    file_name=f"{active_filename}_text.txt",
                    mime="text/plain"
                )
            else:
                st.info("Text extraction was not enabled for this run.")

        # TAB 3: Embedded & Processed Images
        with tab_images:
            st.subheader("Extracted & Processed Image Assets")
            orig_doc = fitz.open(stream=pdf_files[active_filename], filetype="pdf")
            extracted_imgs = extract_embedded_images(orig_doc)

            if extracted_imgs:
                for idx, img_info in enumerate(extracted_imgs):
                    st.write(f"**Image #{idx + 1}** — Page: {img_info['page_num']} | Resolution: {img_info['width']}x{img_info['height']} | Format: {img_info['format'].upper()}")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption("Original Embedded Image:")
                        st.image(img_info["image"], use_container_width=True)
                    
                    with c2:
                        st.caption(f"Processed with {img_techniques or ['original']}:")
                        filtered_img = process_image(img_info["image"], img_techniques, {"clahe_clip": 2.5, "sharpen_strength": 1.2})
                        st.image(filtered_img, use_container_width=True)
            else:
                st.info("No embedded raster images detected in this document.")

        # TAB 4: Linker Index
        with tab_links:
            st.subheader("Smart PDF Linker Index")
            link_summary = res.get("links_summary")
            if link_summary and link_summary.get("links"):
                st.write(f"Detected and created **{link_summary['total_links']}** clickable navigational links:")
                for lk in link_summary["links"]:
                    with st.container():
                        st.markdown(f"- **[Page {lk['page']}] {lk['type']}**: `{lk['text']}` ➔ [{lk['target']}]({lk['target']})")
            else:
                st.info("No external URLs, emails, or internal page references were detected to link.")

        # TAB 5: Download Center
        with tab_export:
            st.subheader("Export Options")
            st.write("Download your processed documents, annotations, and extracted data:")
            
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                if processed_pdf_bytes:
                    st.download_button(
                        label="⬇ Download Processed PDF (.pdf)",
                        data=processed_pdf_bytes,
                        file_name=f"processed_{active_filename}",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )
            with dcol2:
                if res.get("text_extraction"):
                    st.download_button(
                        label="⬇ Download Extracted Content (.txt)",
                        data=res["text_extraction"]["combined_text"],
                        file_name=f"extracted_{active_filename}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
