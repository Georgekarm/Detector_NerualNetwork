import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import math
import os
import time
from malware_pipeline import build_model, load_checkpoint, get_transforms

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Gatekeeper AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CUSTOM CSS - HIGH-FIDELITY SOC THEME
# ==========================================
st.markdown("""
<style>
    /* IMPORTS */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@300;500;700&family=Roboto+Mono:wght@300;400;700&display=swap');

    /* GLOBAL RESET & THEME */
    .stApp {
        background-color: #050505;
        background-image: 
            linear-gradient(rgba(0, 20, 30, 0.9), rgba(0, 10, 15, 0.95)),
            url("https://www.transparenttextures.com/patterns/cubes.png"); /* Subtle texture */
        color: #e0f7fa;
        font-family: 'Rajdhani', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* HIDE STREAMLIT CHROME */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* CUSTOM SCROLLBAR */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0a0a0a; }
    ::-webkit-scrollbar-thumb { background: #00d4ff; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #00ff41; }

    /* UTILITIES & CONTAINERS */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* GLASSMORPHISM PANELS */
    .glass-panel {
        background: rgba(10, 15, 25, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 212, 255, 0.15);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }
    
    .glass-panel::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, #00d4ff, transparent);
        opacity: 0.5;
    }

    /* TYPOGRAPHY */
    .soc-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #ffffff, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
        margin-bottom: 0.5rem;
    }
    
    .soc-sub {
        color: #546e7a;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.9rem;
        letter-spacing: 1px;
    }

    /* FILE UPLOADER CUSTOMIZATION */
    div[data-testid="stFileUploader"] {
        width: 100%;
    }
    div[data-testid="stFileUploader"] section {
        background-color: rgba(0, 212, 255, 0.03);
        border: 1px dashed rgba(0, 212, 255, 0.4);
        border-radius: 8px;
        padding: 1rem;
    }
    div[data-testid="stFileUploader"] section:hover {
        background-color: rgba(0, 212, 255, 0.06);
        border-color: #00d4ff;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.1);
    }
    
    /* SCANNING ANIMATION - RADAR SWEEP */
    .scan-container {
        position: relative;
        width: 100%;
        height: 300px;
        overflow: hidden;
        border-radius: 8px;
        border: 1px solid #00d4ff;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #000;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.1);
    }
    
    .scan-img {
        max-width: 90%;
        max-height: 90%;
        z-index: 1;
        opacity: 0.7;
    }

    .scan-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(to bottom, 
            transparent 50%, 
            rgba(0, 212, 255, 0.2) 51%, 
            rgba(0, 212, 255, 0.6) 52%, 
            rgba(0, 212, 255, 0.2) 53%, 
            transparent 54%);
        background-size: 100% 200%;
        animation: scan 3s linear infinite;
        z-index: 2;
        pointer-events: none;
    }
    
    .grid-overlay {
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image: 
            linear-gradient(rgba(0, 212, 255, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 212, 255, 0.1) 1px, transparent 1px);
        background-size: 20px 20px;
        z-index: 0;
    }

    @keyframes scan {
        0% { background-position: 0% -100%; }
        100% { background-position: 0% 200%; }
    }

    /* RESULTS - STAGE 1 & 2 STYLING */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        font-family: 'Roboto Mono', monospace;
        font-weight: 700;
        border-radius: 4px;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-size: 0.9rem;
    }

    .badge-benign {
        border: 1px solid #00ff41;
        color: #00ff41;
        background: rgba(0, 255, 65, 0.1);
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.2);
    }

    .badge-malware {
        border: 1px solid #ff2b2b;
        color: #ff2b2b;
        background: rgba(255, 43, 43, 0.1);
        box-shadow: 0 0 10px rgba(255, 43, 43, 0.2);
        animation: glitch 2s infinite;
    }

    .badge-neutral {
        border: 1px solid #b0bec5;
        color: #b0bec5;
        background: rgba(176, 190, 197, 0.1);
    }
    
    @keyframes glitch {
        0% { transform: translate(0); }
        2% { transform: translate(-2px, 1px); }
        4% { transform: translate(2px, -1px); }
        6% { transform: translate(0); }
        100% { transform: translate(0); }
    }

    /* METRICS */
    .metric-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #e0f7fa;
    }
    .metric-label {
        font-family: 'Roboto Mono', monospace;
        color: #546e7a;
        font-size: 0.8rem;
        text-transform: uppercase;
    }
    
    /* PROGRESS BARS */
    .stProgress > div > div > div > div {
        background-color: #00d4ff;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CACHED MODEL LOADING
# ==========================================
@st.cache_resource
def load_engine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Class Definitions
    MALWARE_CLASSES = ['adware', 'backdoor', 'benign', 'downloader', 'spyware', 'trojan', 'virus', 'worm']
    
    # Load Stage 1
    s1 = build_model(num_classes=2)
    s1_path = os.path.join('checkpoints', 'Stage1_Gatekeeper_BEST.pth')
    if os.path.exists(s1_path):
        load_checkpoint(s1, s1_path)
    s1.eval()
    
    # Load Stage 2
    s2 = build_model(num_classes=len(MALWARE_CLASSES))
    s2_path = os.path.join('checkpoints', 'Stage2_Classifier_BEST.pth')
    if os.path.exists(s2_path):
        load_checkpoint(s2, s2_path)
    s2.eval()
    
    transform = get_transforms(augment=False)
    
    return s1, s2, MALWARE_CLASSES, transform, device

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def process_file_preview(file):
    """Generate image preview (real or binary viz)"""
    file.seek(0)
    try:
        if file.name.lower().endswith(('.exe', '.dll', '.bin', '.dat')):
            bytes_data = np.frombuffer(file.getvalue(), dtype=np.uint8)
            if len(bytes_data) == 0: return None, "Empty File"
            
            size = int(math.sqrt(len(bytes_data)))
            if size == 0: return None, "Too Small"
            
            image_matrix = bytes_data[:size*size].reshape((size, size))
            img = Image.fromarray(image_matrix).convert('RGB').resize((224, 224))
            return img, "Binary Visualization"
        else:
            img = Image.open(file).convert('RGB')
            return img, "Image Preview"
    except Exception:
        return None, "Error"

def run_analysis(file, s1, s2, classes, transform, device):
    """Core Inference Logic"""
    img, img_type = process_file_preview(file)
    if img is None: return {"error": "Invalid file content"}
    
    tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # STAGE 1
        s1_out = s1(tensor)
        s1_probs = torch.softmax(s1_out, dim=1)
        s1_pred = torch.argmax(s1_probs, dim=1).item() 
        s1_conf = s1_probs[0][s1_pred].item() * 100
        
        # SMART FILE-TYPE LOGIC
        if img_type == "Binary Visualization":
            # Rule 1: Executables/Binaries ALWAYS Pass Stage 1
            # (Override model if it thinks it's 'Natural')
            is_gatekeeper_pass = True
        else:
            # Rule 2: Images MUST pass Stage 1 Model Check
            # Label 1 = Digital/Malware (Pass), Label 0 = Natural (Reject)
            is_gatekeeper_pass = (s1_pred == 1)
        
        if not is_gatekeeper_pass:
            return {
                "stage": 1,
                "status": "REJECTED",
                "reason": "Natural Image Detected",
                "confidence": s1_conf
            }
            
        # STAGE 2
        s2_out = s2(tensor)
        s2_probs = torch.softmax(s2_out, dim=1)
        
        top3_prob, top3_idx = torch.topk(s2_probs[0], 3)
        
        preds = []
        for i in range(3):
            p_cls = classes[top3_idx[i].item()]
            p_conf = top3_prob[i].item() * 100
            preds.append((p_cls, p_conf))
            
        primary_class = preds[0][0]
        primary_conf = preds[0][1]
        is_benign = (primary_class == 'benign')
        
        return {
            "stage": 2,
            "status": "CLEAN" if is_benign else "THREAT",
            "class": primary_class,
            "confidence": primary_conf,
            "predictions": preds,
            "s1_conf": s1_conf
        }

# ==========================================
# MAIN APP LAYOUT
# ==========================================
def main():
    # HEADER
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div class="soc-header">GATEKEEPER <span style="color:#00d4ff;">AI</span></div>
            <div class="soc-sub">ADVANCED THREAT DETECTION & ANALYSIS TERMINAL</div>
        </div>
    """, unsafe_allow_html=True)

    # LOAD MODELS (Silent)
    try:
        s1_model, s2_model, malware_classes, transform, device = load_engine()
    except Exception as e:
        st.error(f"SYSTEM FAILURE: Model loading failed - {e}")
        st.stop()

    # MAIN UPLOAD AREA
    with st.container():
        uploaded_file = st.file_uploader("DROP PAYLOAD HERE", type=['exe', 'dll', 'bin', 'png', 'jpg'], label_visibility='collapsed')

    # AUTO-ANALYSIS LOGIC
    if uploaded_file is not None:
        
        # LAYOUT: 2 COLUMNS
        col_preview, col_results = st.columns([1, 1.6])
        
        # --- LEFT COLUMN: PREVIEW & SCANNING ---
        with col_preview:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown(f'<div style="color: #00d4ff; font-family: Roboto Mono; font-size: 0.8rem; margin-bottom: 10px;">ID: {uploaded_file.name[:15]}...</div>', unsafe_allow_html=True)
            
            # Generate Preview
            img_preview, img_type = process_file_preview(uploaded_file)
            
            if img_preview:
                # We save image to display it within HTML structure to apply custom classes easily
                # or we can use st.image, but custom HTML gives us the 'overlay' ability.
                # For simplicity in Streamlit, we'll use st.image but wrapped in a container for the scan effect.
                
                # However, to overlay the CSS animation perfectly, we can output the image as base64 or just use the structure below.
                # Since simple st.image doesn't allow easy overlay div injection, we use purely st.image for now 
                # OR we accept that the "Scanning" overlay sits ON TOP via CSS absolute positioning if we could target it.
                # A robust way is to use st.markdown with an img tag.
                import base64
                from io import BytesIO
                buffered = BytesIO()
                img_preview.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                st.markdown(f"""
                <div class="scan-container">
                    <div class="grid-overlay"></div>
                    <div class="scan-overlay"></div>
                    <img src="data:image/png;base64,{img_str}" class="scan-img">
                </div>
                <div style="text-align: center; margin-top: 10px; font-family: 'Roboto Mono'; color: #546e7a;">
                    TYPE: {img_type.upper()} | SIZE: {uploaded_file.size/1024:.1f} KB
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Preview Unavailable")
            
            st.markdown('</div>', unsafe_allow_html=True)

        # --- RIGHT COLUMN: RESULTS ---
        with col_results:
            st.markdown('<div class="glass-panel" style="min-height: 400px;">', unsafe_allow_html=True)
            
            # ANALYSIS PHASES
            status_container = st.empty()
            
            # Simulating "Scanning" steps for effect (very short delays)
            with status_container.container():
                st.markdown('<div style="color: #00d4ff; font-family: Roboto Mono;">> INITIALIZING HANDSHAKE...</div>', unsafe_allow_html=True)
                time.sleep(0.3)
                st.markdown('<div style="color: #00d4ff; font-family: Roboto Mono;">> EXTRACTING FEATURES...</div>', unsafe_allow_html=True)
                time.sleep(0.4)
                st.markdown('<div style="color: #00ff41; font-family: Roboto Mono;">> FEATURE TENSOR CONSTRUCTED.</div>', unsafe_allow_html=True)
            
            # RUN INFERENCE
            result = run_analysis(uploaded_file, s1_model, s2_model, malware_classes, transform, device)
            
            # Clear logs
            status_container.empty()
            
            if "error" in result:
                st.error(f"ANALYSIS FAILED: {result['error']}")
            
            elif result["stage"] == 1 and result["status"] == "REJECTED":
                # STAGE 1 FAIL
                st.markdown("""
                    <h3 style="color: #b0bec5;">SCAN RESULTS</h3>
                    <hr style="border-color: rgba(255,255,255,0.1);">
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem;">
                    <div class="status-badge badge-neutral">❌ REJECTED</div>
                    <div style="font-family: 'Roboto Mono'; color: #b0bec5;">GATEKEEPER PROTOCOL</div>
                </div>
                <div style="background: rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 8px;">
                    <div class="metric-label">REASON</div>
                    <div class="metric-value" style="color: #b0bec5; font-size: 1.5rem;">{result['reason']}</div>
                    <div style="margin-top: 1rem; color: #546e7a;">The file was identified as a natural image, not a binary data representation. Analysis halted.</div>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                # STAGE 2 RESULTS
                st.markdown("""
                    <h3 style="color: #00d4ff;">ANALYSIS REPORT</h3>
                    <hr style="border-color: rgba(0, 212, 255, 0.2);">
                """, unsafe_allow_html=True)
                
                # Primary Verdict
                status_class = "badge-benign" if result["status"] == "CLEAN" else "badge-malware"
                icon = "🛡️" if result["status"] == "CLEAN" else "☣️"
                color = "#00ff41" if result["status"] == "CLEAN" else "#ff2b2b"
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class="metric-label">THREAT LEVEL</div>
                    <div class="{status_class}" style="font-size: 1.2rem; margin-top: 0.5rem; margin-bottom: 1rem;">
                        {icon} {result['status']}
                    </div>
                    """, unsafe_allow_html=True)
                
                with c2:
                    st.markdown(f"""
                    <div class="metric-label">CLASSIFICATION</div>
                    <div class="metric-value" style="color: {color};">{result['class'].upper()}</div>
                    """, unsafe_allow_html=True)
                
                # Confidence Bar
                st.markdown(f"""
                <div style="margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span class="metric-label">CONFIDENCE SCORE</span>
                        <span style="color: {color}; font-weight: bold;">{result['confidence']:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(result['confidence'] / 100)
                
                # Top 3 Details
                with st.expander("DETAILED FORENSICS", expanded=True):
                    for cls, conf in result['predictions']:
                        bar_color = "#00ff41" if cls == "benign" else "#ff2b2b"
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; margin-top: 10px; font-family: 'Roboto Mono'; font-size: 0.9rem;">
                            <span style="color: {bar_color};">{cls.upper()}</span>
                            <span>{conf:.1f}%</span>
                        </div>
                        <div style="height: 4px; background: #1a1a1a; width: 100%; border-radius: 2px; margin-top: 4px;">
                            <div style="height: 100%; width: {conf}%; background: {bar_color}; border-radius: 2px;"></div>
                        </div>
                        """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
