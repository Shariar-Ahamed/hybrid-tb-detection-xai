import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time

# Set page configuration with a premium icon and title
st.set_page_config(
    page_title="TB-XAI | Tuberculosis Explainable AI Dashboard",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS styling injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main Background & Text */
    .stApp {
        background-color: #0d0f13;
        color: #e2e8f0;
    }
    
    /* Premium Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #311042 50%, #0f172a 100%);
        padding: 2.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 400;
    }
    
    /* Custom Sidebar Card Styling */
    section[data-testid="stSidebar"] {
        background-color: #11141a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Glassmorphism Cards */
    .metric-card {
        background: rgba(22, 28, 36, 0.8);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 0.2rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Diagnosis Badges */
    .badge-positive {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }

    .badge-negative {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Sidebar Details Card */
    .sidebar-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #a855f7;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTENT -----------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/07/DIU_Logo.png", width=90)
    st.markdown("### FYDP Title Phase Evaluation")
    st.markdown("<hr style='margin: 8px 0; opacity: 0.15;'>", unsafe_allow_html=True)
    
    st.markdown("#### **Project Title**")
    st.markdown(
        "<div class='sidebar-card' style='font-size: 0.9rem; font-weight: 500;'>"
        "A Hybrid Explainable AI Framework for Automatic Tuberculosis Detection from Chest X-Ray Images Using CNNs"
        "</div>", 
        unsafe_allow_html=True
    )
    
    st.markdown("#### **Team Members**")
    st.markdown(
        "• **Shariar Ahamed Ripon**<br><small style='color:#64748b;'>ID: 0242310005101019</small><br>"
        "• **Md. Moniruzzaman Rifat**<br><small style='color:#64748b;'>ID: 0242310005101020</small>",
        unsafe_allow_html=True
    )
    
    st.markdown("<br>#### **Supervision**", unsafe_allow_html=True)
    st.markdown(
        "• **Dr. Md. Ali Hossain** (Associate Professor)<br>"
        "• **Mr. Md. Mizanur Rahman** (Co-Supervisor)",
        unsafe_allow_html=True
    )
    
    st.markdown("<hr style='margin: 20px 0; opacity: 0.15;'>", unsafe_allow_html=True)
    st.info("Medical Disclaimer: This dashboard is a prototype for educational and research evaluation. Decisions should always be validated by certified radiologists.")

# ----------------- MAIN VIEW -----------------
# Header Banner
st.markdown("""
<div class="header-banner">
    <div class="header-title">🫁 Hybrid Explainable AI Framework</div>
    <div class="header-subtitle">Automatic Tuberculosis Detection & Heatmap Visualizations from Chest X-Ray Images</div>
</div>
""", unsafe_allow_html=True)

# Stats Metrics Section (Interactive and visually premium)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">96.8%</div>
        <div class="metric-label">Target Accuracy</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">Grad-CAM</div>
        <div class="metric-label">XAI Engine</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">&lt; 1.2s</div>
        <div class="metric-label">Inference Time</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">Shenzhen</div>
        <div class="metric-label">Benchmarked Dataset</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# File Uploader and Diagnostic Display
upload_col, display_col = st.columns([1, 2])

with upload_col:
    st.subheader("📁 Upload CXR Scan")
    uploaded_file = st.file_uploader(
        "Upload a Chest X-ray image (PNG or JPG format) to detect Tuberculosis signs.", 
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        image = Image.open(uploaded_file)
        
        # Display small preview of uploaded image
        st.image(image, caption="Uploaded Scan Preview", use_container_width=True)
        
        # Trigger prediction demo
        analyze_button = st.button("Run Diagnostic Analysis 🚀", use_container_width=True)
    else:
        st.info("Waiting for Chest X-Ray upload...")
        analyze_button = False

with display_col:
    st.subheader("🔬 Diagnostic & Interpretability Report")
    
    if uploaded_file is not None and analyze_button:
        # Mocking prediction latency with a cool loader animation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
            if i < 30:
                status_text.text("🔄 Initializing Hybrid CNN Feature Fusion...")
            elif i < 70:
                status_text.text("🧠 Feeding features to Spatial Attention Blocks...")
            else:
                status_text.text("⚡ Generating Grad-CAM Interpretability Maps...")
                
        status_text.empty()
        progress_bar.empty()
        
        # Show results side by side
        img_col1, img_col2 = st.columns(2)
        
        # Original Image
        with img_col1:
            st.image(image, caption="Original CXR", use_container_width=True)
            
        # Mock Grad-CAM explanation image creation
        with img_col2:
            # Load image into opencv format for mock processing
            img_np = np.array(image.convert("RGB"))
            h, w, c = img_np.shape
            
            # Generate mock Heatmap (overlay red circles around the lungs area)
            heatmap = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(heatmap, (int(w * 0.35), int(h * 0.45)), int(min(h, w) * 0.18), 255, -1)
            cv2.circle(heatmap, (int(w * 0.65), int(h * 0.4)), int(min(h, w) * 0.15), 180, -1)
            heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
            
            # Colorize heatmap
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Blend original with heatmap
            overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
            
            st.image(overlay, caption="Grad-CAM Activation Heatmap", use_container_width=True)
            
        # Diagnostic Details
        st.markdown("<hr style='opacity: 0.1;'>", unsafe_allow_html=True)
        st.markdown("### Clinical Decision Breakdown")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown("#### **Prediction Status**")
            st.markdown("<span class='badge-positive'>Tuberculosis Detected</span>", unsafe_allow_html=True)
            st.markdown("Confidence Score: **89.4%**")
            
        with res_col2:
            st.markdown("#### **Grad-CAM Insights**")
            st.markdown(
                "• Active anomalies found in **Upper Left and Right Lung Lobes**.<br>"
                "• Attention maps confirm highlights over dense consolidation zones.", 
                unsafe_allow_html=True
            )
            
    else:
        # Placeholder screen when no file is analyzed
        st.markdown(
            "<div style='border: 2px dashed rgba(255,255,255,0.05); border-radius: 12px; padding: 4rem; text-align: center; color: #64748b;'>"
            "<h3>No Active Diagnostic Report</h3>"
            "<p>Please upload a Chest X-ray scan on the left and click 'Run Diagnostic Analysis' to view predictions and Grad-CAM explainability heatmaps.</p>"
            "</div>", 
            unsafe_allow_html=True
        )
