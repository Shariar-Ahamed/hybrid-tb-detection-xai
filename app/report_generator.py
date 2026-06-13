import os
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib import colors
from PIL import Image as PILImage
import numpy as np

def generate_pdf_report(patient_name, patient_id, patient_age, patient_gender, symptoms, prediction, confidence, insights, original_image, xai_overlay, xai_method_name="Grad-CAM"):
    """
    Generates a print-ready clinical diagnostic PDF report using ReportLab.
    
    Args:
        patient_name (str): Name of the patient.
        patient_id (str): Unique Reference ID.
        patient_age (str): Age of the patient.
        patient_gender (str): Gender of the patient.
        symptoms (str): Clinical details/symptoms.
        prediction (bool): True if Tuberculosis signs are detected, False otherwise.
        confidence (float): Confidence score of the prediction.
        insights (str): Markdown-styled insights/bullet points to print.
        original_image (PIL.Image): Original uploaded Chest X-Ray image.
        xai_overlay (PIL.Image or numpy.ndarray): Selected XAI visualization overlay.
        xai_method_name (str): Selected XAI method name.
        
    Returns:
        bytes: Binary PDF file data.
    """
    # Save original image to in-memory buffer
    orig_buffer = io.BytesIO()
    original_image.save(orig_buffer, format="PNG")
    orig_buffer.seek(0)
    
    # Save XAI heatmap/saliency overlay to in-memory buffer (fallback to original image if overlay generation failed)
    grad_buffer = io.BytesIO()
    if xai_overlay is None:
        original_image.save(grad_buffer, format="PNG")
    elif isinstance(xai_overlay, np.ndarray):
        # Convert floats to uint8 if necessary
        if xai_overlay.dtype == np.float32 or xai_overlay.dtype == np.float64:
            xai_overlay = (xai_overlay * 255).astype(np.uint8)
        gradcam_pil = PILImage.fromarray(xai_overlay)
        gradcam_pil.save(grad_buffer, format="PNG")
    else:
        xai_overlay.save(grad_buffer, format="PNG")
        
    grad_buffer.seek(0)
        
    pdf_buffer = io.BytesIO()
    
    # Letter size is 612 x 792. Printable area: 540 x 720 (36pt/0.5" margins)
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        alignment=TA_LEFT
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=12,
        spaceAfter=6
    )
    
    normal_bold = ParagraphStyle(
        'NormalBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#1e293b')
    )
    
    normal_text = ParagraphStyle(
        'NormalText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#334155')
    )

    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7,
        leading=10,
        textColor=colors.HexColor('#94a3b8'),
        alignment=TA_CENTER
    )
    
    # --- HEADER BLOCK ---
    logo_path = os.path.join(os.path.dirname(__file__), "diu_logo.png")
    header_data = []
    if os.path.exists(logo_path):
        header_logo = Image(logo_path, width=130, height=35)
        header_logo.hAlign = 'RIGHT'
        header_data = [[
            Paragraph("TB-XAI CLINICAL DIAGNOSTIC REPORT<br/><font color='#64748b' size='8'>Hybrid Explainable Artificial Intelligence Framework</font>", title_style),
            header_logo
        ]]
        header_table = Table(header_data, colWidths=[380, 160])
    else:
        header_data = [[
            Paragraph("TB-XAI CLINICAL DIAGNOSTIC REPORT<br/><font color='#64748b' size='8'>Hybrid Explainable Artificial Intelligence Framework</font>", title_style)
        ]]
        header_table = Table(header_data, colWidths=[540])
        
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(header_table)
    
    # Decorative colored line
    divider = Table([[""]], colWidths=[540])
    divider.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), 2, colors.HexColor('#0ea5e9')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(divider)
    story.append(Spacer(1, 12))
    
    # --- PATIENT INFO TABLE ---
    story.append(Paragraph("Patient Information", section_heading))
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    patient_data = [
        [
            Paragraph("Patient Name:", normal_bold), Paragraph(str(patient_name) if patient_name else "N/A", normal_text),
            Paragraph("Patient ID / Reference:", normal_bold), Paragraph(str(patient_id) if patient_id else "N/A", normal_text)
        ],
        [
            Paragraph("Age / Gender:", normal_bold), Paragraph(f"{patient_age} / {patient_gender}", normal_text),
            Paragraph("Analysis Date:", normal_bold), Paragraph(date_str, normal_text)
        ]
    ]
    
    if symptoms and symptoms.strip():
        patient_data.append([
            Paragraph("Clinical History:", normal_bold), Paragraph(str(symptoms), normal_text),
            "", ""
        ])
        
    patient_table = Table(patient_data, colWidths=[90, 180, 130, 140])
    patient_table_styles = [
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
    ]
    if len(patient_data) > 2:
        patient_table_styles.append(('SPAN', (1, 2), (3, 2)))
        
    patient_table.setStyle(TableStyle(patient_table_styles))
    story.append(patient_table)
    story.append(Spacer(1, 12))
    
    # --- DIAGNOSTIC OUTCOME CARD ---
    story.append(Paragraph("Diagnostic Summary", section_heading))
    
    status_text = "Tuberculosis Signs DETECTED" if prediction else "NORMAL (No Signs of Tuberculosis)"
    confidence_text = f"Confidence Score: {confidence:.2f}%"
    bg_color = '#fee2e2' if prediction else '#dcfce7'
    border_color = '#ef4444' if prediction else '#22c55e'
    text_color = '#b91c1c' if prediction else '#15803d'
    
    result_style_bold = ParagraphStyle(
        'ResultBold',
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=colors.HexColor(text_color),
        alignment=TA_CENTER
    )
    result_style_normal = ParagraphStyle(
        'ResultNormal',
        fontName='Helvetica-Bold',
        fontSize=9.5,
        textColor=colors.HexColor('#1e293b'),
        alignment=TA_CENTER
    )
    
    result_data = [
        [Paragraph(status_text, result_style_bold)],
        [Paragraph(confidence_text, result_style_normal)]
    ]
    result_table = Table(result_data, colWidths=[540])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(bg_color)),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor(border_color)),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 12))
    
    # --- IMAGES SECTION (Side by side) ---
    story.append(Paragraph("Visual Interpretability (Explainable AI)", section_heading))
    
    img_w, img_h = 240, 240
    img_orig = Image(orig_buffer, width=img_w, height=img_h)
    img_grad = Image(grad_buffer, width=img_w, height=img_h)
    
    images_data = [[img_orig, img_grad]]
    images_labels = [[
        Paragraph("Original Chest X-Ray Scan", ParagraphStyle('Lbl1', fontName='Helvetica-Bold', fontSize=8, alignment=TA_CENTER, textColor=colors.HexColor('#475569'))),
        Paragraph(f"{xai_method_name} Interpretability Map", ParagraphStyle('Lbl2', fontName='Helvetica-Bold', fontSize=8, alignment=TA_CENTER, textColor=colors.HexColor('#475569')))
    ]]
    
    images_table = Table(images_data, colWidths=[260, 260])
    images_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOX', (0,0), (0,0), 0.5, colors.HexColor('#cbd5e1')),
        ('BOX', (1,0), (1,0), 0.5, colors.HexColor('#cbd5e1')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    
    labels_table = Table(images_labels, colWidths=[260, 260])
    labels_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    
    story.append(images_table)
    story.append(labels_table)
    story.append(Spacer(1, 10))
    
    # --- INSIGHTS / XAI INTERPRETATION ---
    story.append(Paragraph(f"{xai_method_name} AI Insights", section_heading))
    insights_clean = insights.replace("•", "").replace("<b>", "").replace("</b>", "").replace("<br>", "\n").replace("<br/>", "\n")
    insights_lines = [line.strip() for line in insights_clean.split("\n") if line.strip()]
    
    insights_bullet_style = ParagraphStyle(
        'InsightsBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#334155'),
        leftIndent=15,
        firstLineIndent=-10
    )
    
    for line in insights_lines:
        story.append(Paragraph(f"• {line}", insights_bullet_style))
        story.append(Spacer(1, 2.5))
        
    story.append(Spacer(1, 10))
    
    # --- SIGNATURE BLOCK & FOOTER ---
    sig_data = [
        [
            Paragraph("Attending Radiologist / Evaluator", normal_bold),
            Paragraph("Institutional Validation Details", normal_bold)
        ],
        [
            Paragraph("<br/><br/>________________________________________<br/>Signature & Seal", normal_text),
            Paragraph("<br/><br/>Date: ________________________<br/>Status: Verified / Unverified", normal_text)
        ]
    ]
    sig_table = Table(sig_data, colWidths=[270, 270])
    sig_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    
    footer_elements = []
    footer_elements.append(sig_table)
    footer_elements.append(Spacer(1, 12))
    footer_elements.append(Paragraph(
        "<b>Medical Disclaimer:</b> This report is generated by an explainable AI research prototype for educational evaluation. "
        "It does not constitute formal medical advice. All diagnostic findings and visual focus areas must be verified and "
        "validated by a certified medical practitioner prior to clinical decision-making.",
        disclaimer_style
    ))
    
    story.append(KeepTogether(footer_elements))
    
    # Build PDF doc
    doc.build(story)
    
    pdf_val = pdf_buffer.getvalue()
    pdf_buffer.close()
    return pdf_val
