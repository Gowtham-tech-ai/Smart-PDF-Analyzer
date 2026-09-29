import io
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors


def create_synthetic_test_image(text_overlay: str = "Sample Scanned Medical Record") -> Image.Image:
    """Generates an image of a document or landscape with text to test image processing and OCR."""
    img = Image.new("RGB", (600, 300), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)

    # Draw simulated mountain / landscape shapes
    draw.polygon([(0, 300), (150, 100), (300, 300)], fill=(80, 120, 180))
    draw.polygon([(200, 300), (380, 70), (550, 300)], fill=(60, 90, 150))
    draw.polygon([(400, 300), (500, 150), (600, 300)], fill=(100, 140, 200))
    
    # Sun / circle
    draw.ellipse((450, 30, 520, 100), fill=(255, 200, 50))

    # Water reflection
    draw.rectangle([0, 240, 600, 300], fill=(40, 80, 120))

    # Text overlay
    draw.text((20, 20), text_overlay, fill=(20, 20, 20))
    return img


def generate_sample_patient_report() -> bytes:
    """
    Creates a synthetic Patient Report matching Feature Preview 2 from the blueprint.
    Contains sensitive PII (Name, ID, Email, Phone, Diagnosis) and an embedded test image.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.setFillColor(colors.HexColor("#1A365D"))
    c.drawString(50, height - 60, "Patient Clinical Report")
    
    # Divider line
    c.setStrokeColor(colors.HexColor("#CBD5E1"))
    c.setLineWidth(1)
    c.line(50, height - 70, width - 50, height - 70)

    # Demographic data
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.black)
    c.drawString(50, height - 100, "Patient Information:")
    
    c.setFont("Helvetica", 11)
    c.drawString(70, height - 125, "Name: John Doe")
    c.drawString(70, height - 145, "ID: 123457890")
    c.drawString(70, height - 165, "Email: johndoe@email.com")
    c.drawString(70, height - 185, "Phone: +91 98765 43210")
    c.drawString(70, height - 205, "Diagnosis: Acute Respiratory Viral Syndrome")
    c.drawString(70, height - 225, "Treatment: Prescribed Amoxicillin 500mg, Rest, Hydration")

    # Add embedded image for image processing / blurring
    test_img = create_synthetic_test_image("Chest Radiograph Scanned Plate #942")
    img_buf = io.BytesIO()
    test_img.save(img_buf, format="PNG")
    img_buf.seek(0)
    
    # Save temp image for reportlab canvas
    from reportlab.lib.utils import ImageReader
    img_reader = ImageReader(img_buf)
    c.drawString(50, height - 270, "Clinical Scan Imaging Asset:")
    c.drawImage(img_reader, 50, height - 480, width=500, height=190)

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 40, "Confidential Medical Document - For Authorized Personnel Only")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def generate_sample_renewable_energy_article() -> bytes:
    """
    Creates a multi-page PDF matching Feature Previews 4 & 5 (Keyword Highlighter & Linker).
    Contains:
    - Keywords: renewable energy, solar energy, wind energy, hydropower, pollution
    - URLs: www.example.com, https://en.wikipedia.org
    - Emails: support@example.com
    - Page References: "See page 2 for more details"
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # --- Page 1 ---
    c.setFont("Helvetica-Bold", 22)
    c.setFillColor(colors.HexColor("#0F766E"))
    c.drawString(50, height - 60, "Renewable Energy & Sustainable Future")

    c.setFont("Helvetica", 11)
    c.setFillColor(colors.HexColor("#1F2937"))
    
    text_lines = [
        "Renewable energy is the future of our planet. As traditional fossil fuels deplete and global",
        "temperatures rise, transitioning to clean alternatives is essential.",
        "",
        "Solar energy and wind energy have become the fastest-growing power sources in modern times.",
        "Hydropower provides stable baseline power, while reducing carbon pollution across major cities.",
        "Investing in renewable energy reduces pollution and creates a better future for coming generations.",
        "",
        "Economic incentives and battery storage advancements have accelerated renewable energy adoption.",
        "Both solar energy and wind energy can be deployed locally, democratizing clean electricity.",
        "",
        "For more details, visit our research portal at website: www.example.com",
        "For direct scientific inquiries, contact us: support@example.com",
        "",
        "Please see page 2 for detailed energy consumption metrics and environmental statistics."
    ]

    y = height - 100
    for line in text_lines:
        c.drawString(50, y, line)
        y -= 20

    # Embedded graphic
    test_img = create_synthetic_test_image("Solar and Wind Farm Satellite Monitoring")
    img_buf = io.BytesIO()
    test_img.save(img_buf, format="PNG")
    img_buf.seek(0)
    from reportlab.lib.utils import ImageReader
    c.drawImage(ImageReader(img_buf), 50, y - 220, width=500, height=200)

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 40, "Page 1 of 2 - Global Energy Review")
    c.showPage()

    # --- Page 2 ---
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(colors.HexColor("#0F766E"))
    c.drawString(50, height - 60, "Detailed Energy Metrics & Environmental Impact")

    c.setFont("Helvetica", 11)
    c.setFillColor(colors.HexColor("#1F2937"))
    p2_lines = [
        "This is Page 2 of the report referenced on page 1.",
        "",
        "Key Metrics:",
        "1. Solar energy installation capacity grew by 24% year-over-year.",
        "2. Wind energy offshore efficiency reached new records in Northern Europe.",
        "3. Hydropower reservoirs retained stable output despite changing seasonal weather patterns.",
        "4. Urban air pollution indices dropped by 18% in regions adopting clean power grids.",
        "",
        "Continued investment in renewable energy is critical to eradicate toxic atmospheric pollution.",
        "To contribute to this initiative or review raw data, visit https://www.example.com/energy"
    ]
    y = height - 100
    for line in p2_lines:
        c.drawString(50, y, line)
        y -= 22

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 40, "Page 2 of 2 - Global Energy Review")
    c.showPage()

    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def generate_sample_scanned_ocr_pdf() -> bytes:
    """
    Creates a PDF where the page is a pure bitmap image (no digital text),
    simulating a scanned document for testing OCR.
    """
    # Create PIL image of a scanned letter
    img = Image.new("RGB", (800, 1100), color=(250, 248, 245))
    draw = ImageDraw.Draw(img)

    # Scanned header
    draw.rectangle([50, 50, 750, 55], fill=(70, 70, 70))
    draw.text((60, 70), "SCANNED INVOICE & ACCOUNTS SUMMARY", fill=(20, 20, 20))
    draw.text((60, 110), "Invoice Number: INV-98741", fill=(40, 40, 40))
    draw.text((60, 140), "Customer: Acme Global Logistics Ltd.", fill=(40, 40, 40))
    draw.text((60, 170), "Contact: accounts@acmeglobal.org", fill=(40, 40, 40))
    draw.text((60, 200), "Billing Address: 42 Silicon Parkway, Tech District", fill=(40, 40, 40))

    # Table items
    y = 260
    draw.line([(50, y), (750, y)], fill=(120, 120, 120), width=2)
    y += 20
    draw.text((60, y), "Item Description                     Qty    Rate      Total", fill=(20, 20, 20))
    y += 25
    draw.line([(50, y), (750, y)], fill=(120, 120, 120), width=1)
    y += 20
    items = [
        ("Cloud Infrastructure Hosting        12     $450.00   $5,400.00"),
        ("AI Document Processing Services      1     $1,200.00  $1,200.00"),
        ("Security Audit & Vulnerability Scan  1     $2,500.00  $2,500.00"),
    ]
    for it in items:
        draw.text((60, y), it, fill=(30, 30, 30))
        y += 30

    y += 30
    draw.line([(50, y), (750, y)], fill=(120, 120, 120), width=2)
    y += 20
    draw.text((450, y), "Grand Total:  $9,100.00", fill=(10, 10, 10))

    # Add slight simulated noise
    cv_img = np.array(img)
    noise = np.random.normal(0, 4, cv_img.shape).astype(np.uint8)
    noisy_img = cv2.add(cv_img, noise)
    final_pil = Image.fromarray(noisy_img)

    # Convert to PDF
    pdf_buf = io.BytesIO()
    final_pil.save(pdf_buf, format="PDF")
    pdf_buf.seek(0)
    return pdf_buf.getvalue()
