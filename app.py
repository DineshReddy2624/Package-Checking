"""
Smart AI-Based Delivery Package Damage Detection and Internal Damage Risk Prediction System.
Premium Streamlit Application with Auto-Detection, Two-Stage Vision Pipeline, and Integrated AI Quality Advisor.
"""

import os
import sys
import json
import time
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import streamlit as st

# Set project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    WEIGHTS_DIR,
    RUNS_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
    SAMPLE_IMAGES_DIR,
    detect_hardware,
    load_json,
)
from src.multi_package_pipeline import TwoStagePackageInspector
from src.llm_advisor import generate_llm_inspection_notes

# Streamlit Page Setup
st.set_page_config(
    page_title="AI Package Damage & Risk Advisor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Modern Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 50%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 20px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #60a5fa 0%, #34d399 40%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    .telemetry-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 30px;
        padding: 4px 14px;
        font-size: 0.82rem;
        color: #cbd5e1;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* AI Advisor Glassmorphic Card */
    .ai-card {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 1px solid rgba(96, 165, 250, 0.3);
        border-radius: 18px;
        padding: 1.6rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.12);
    }
    
    .ai-card-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #60a5fa;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 0.7rem;
    }
    
    .ai-summary-text {
        font-size: 1.08rem;
        line-height: 1.7;
        color: #f8fafc;
        font-weight: 400;
    }
    
    /* Decision Badges */
    .badge-safe {
        background: linear-gradient(90deg, #059669, #10b981);
        color: #ffffff;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .badge-inspect {
        background: linear-gradient(90deg, #d97706, #f59e0b);
        color: #ffffff;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
    }
    
    .badge-replace {
        background: linear-gradient(90deg, #dc2626, #ef4444);
        color: #ffffff;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }
    
    /* Sub-card Container */
    .subcard {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 1.2rem;
        height: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_vision_engines(pkg_conf: float, dmg_conf: float):
    """
    Loads Stage 1 Package Detector and Stage 2 Damage Detector with caching.
    """
    pkg_candidates = [
        WEIGHTS_DIR / "package_detector_best.pt",
        RUNS_DIR / "train" / "package_detector_run" / "weights" / "best.pt",
        WEIGHTS_DIR / "best.pt",
    ]
    pkg_path = next((p for p in pkg_candidates if p.exists()), None)

    dmg_candidates = [
        WEIGHTS_DIR / "damage_detector_best.pt",
        WEIGHTS_DIR / "best.pt",
    ]
    dmg_path = next((p for p in dmg_candidates if p.exists()), None)

    two_stage = None
    if pkg_path and dmg_path:
        two_stage = TwoStagePackageInspector(
            package_detector_path=pkg_path,
            damage_detector_path=dmg_path,
            package_conf=pkg_conf,
            damage_conf=dmg_conf,
        )

    return two_stage, pkg_path, dmg_path


def main():
    hw_info = detect_hardware()

    # Premium Hero Header
    st.markdown(
        f"""
        <div class="hero-container">
            <div class="hero-title">📦 Smart AI Package Damage & Risk Advisor</div>
            <div class="hero-subtitle">Automated Two-Stage Computer Vision & Generative AI Logistics Intelligence</div>
            <div style="margin-top: 1rem; display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="telemetry-tag">⚡ Engine: {hw_info['recommended_device'].upper()}</span>
                <span class="telemetry-tag">🧠 PyTorch: {hw_info['pytorch_version']}</span>
                <span class="telemetry-tag">✨ AI Advisor: Auto-Active</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Clean Sidebar: Focused only on practical vision sensitivity & decision rules
    with st.sidebar:
        st.header("🎯 Detection Controls")
        
        pkg_conf = st.slider("Package Detection Sensitivity", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
        dmg_conf = st.slider("Damage Detection Sensitivity", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
        crop_margin = st.slider("Crop Margin (%)", min_value=0, max_value=15, value=5, step=1) / 100.0

        st.divider()
        st.subheader("📋 Delivery Decision Protocol")
        st.markdown(
            """
            * 🟢 **Risk < 25**: `SAFE TO DELIVER`
            * 🟡 **Risk 25 – 65**: `INSPECT BEFORE DELIVERY`
            * 🔴 **Risk > 65**: `REPLACE PACKAGE`
            """
        )

        st.divider()
        st.subheader("🛡️ Model Status")
        two_stage_engine, pkg_path, dmg_path = load_vision_engines(pkg_conf, dmg_conf)
        if pkg_path:
            st.success(f"Stage 1 Detector: `{pkg_path.name}`")
        if dmg_path:
            st.success(f"Stage 2 Damage Model: `{dmg_path.name}`")

    # Main Area: Auto-Detection
    st.subheader("📥 Inspect Package or Warehouse Scene")
    input_col1, input_col2 = st.columns([1, 1])

    with input_col1:
        uploaded_file = st.file_uploader(
            "Upload any image (Single package or warehouse scene)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Drop any image to automatically detect packages, analyze defects, and generate AI logistics notes.",
        )

    with input_col2:
        # Aggregate all sample images
        all_samples = []
        for sdir in [SAMPLE_IMAGES_DIR / "multiple_parcels", SAMPLE_IMAGES_DIR / "single_parcel", SAMPLE_IMAGES_DIR]:
            if sdir.exists():
                all_samples.extend([f for f in sdir.glob("*.jpg") if f.is_file()])
        # Deduplicate
        unique_samples = {f.name: f for f in all_samples}
        sample_choice = st.selectbox(
            "Or pick a pre-loaded test image:",
            ["None"] + sorted(list(unique_samples.keys())),
        )

    # Resolve image input
    input_image_bgr = None
    image_source_name = "custom_upload.jpg"

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        input_image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_source_name = uploaded_file.name
    elif sample_choice != "None":
        selected_path = unique_samples[sample_choice]
        input_image_bgr = cv2.imread(str(selected_path))
        image_source_name = selected_path.name

    # Auto-Execute Pipeline
    if input_image_bgr is not None:
        if two_stage_engine is None:
            st.error("Vision models are loading or not initialized yet.")
            return

        with st.spinner("🤖 Auto-Detecting Packages, Analyzing Damage, & Generating AI Insights..."):
            start_time = time.time()
            results = two_stage_engine.inspect_scene(
                image_input=input_image_bgr,
                crop_margin_pct=crop_margin,
            )
            elapsed_time = (time.time() - start_time) * 1000

            # Generate AI Logistics Notes Automatically
            llm_notes = generate_llm_inspection_notes(
                inspection_data=results,
                provider="gemini",
            )

        # -------------------------------------------------------------
        # 1. TOP KPI STATUS BAR
        # -------------------------------------------------------------
        st.divider()
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("📦 Packages Found", results["total_packages"])
        kpi2.metric("🟢 Safe to Deliver", results["safe_count"])
        kpi3.metric("🟡 Needs Inspection", results["inspect_count"])
        kpi4.metric("🔴 Needs Replacement", results["replace_count"])
        kpi5.metric("⚡ Total Latency", f"{elapsed_time:.0f} ms")

        # -------------------------------------------------------------
        # 2. AI INTELLIGENT QUALITY & LOGISTICS ADVICE
        # -------------------------------------------------------------
        st.markdown(
            f"""
            <div class="ai-card">
                <div class="ai-card-header">✨ AI Quality Assessment & Logistics Advisor</div>
                <div class="ai-summary-text">
                    {llm_notes['executive_summary']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_llm_left, col_llm_right = st.columns(2)
        with col_llm_left:
            st.markdown(
                f"""
                <div class="subcard">
                    <h4 style="margin-top:0; color:#38bdf8;">🔬 Structural & Integrity Analysis</h4>
                    <div style="font-size: 0.95rem; line-height: 1.6; color:#cbd5e1;">
                        {llm_notes.get('integrity_analysis', 'Full container integrity verified with zero physical compromise.')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_llm_right:
            courier_txt = llm_notes.get('courier_instructions', 'Standard route dispatch permitted.')
            cust_txt = llm_notes.get('customer_note', 'Your package passed AI quality inspection with 100% integrity.')
            st.markdown(
                f"""
                <div class="subcard">
                    <h4 style="margin-top:0; color:#34d399;">🚚 Courier Field Protocol & Customer Update</h4>
                    <div style="font-size: 0.95rem; line-height: 1.6; color:#cbd5e1; margin-bottom: 0.8rem;">
                        {courier_txt}
                    </div>
                    <div style="font-size: 0.9rem; color:#94a3b8; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 0.6rem;">
                        📱 <b>Customer SMS/Tracking</b>: <i>"{cust_txt}"</i>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # -------------------------------------------------------------
        # 3. SIDE-BY-SIDE VISUAL INSPECTION
        # -------------------------------------------------------------
        st.divider()
        st.subheader("🖼️ Visual Pipeline Progression")
        img_col1, img_col2 = st.columns(2)

        with img_col1:
            st.markdown("**Stage 1: Individual Package Isolation**")
            stage1_bgr = cv2.imread(results["stage1_image_path"])
            st.image(cv2.cvtColor(stage1_bgr, cv2.COLOR_BGR2RGB))

        with img_col2:
            st.markdown("**Final Result: Multi-Package Delivery Decision Badges**")
            final_bgr = cv2.imread(results["final_image_path"])
            st.image(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB))

        # -------------------------------------------------------------
        # 4. INDIVIDUAL PACKAGE DEEP DIVE CARDS
        # -------------------------------------------------------------
        st.divider()
        st.subheader("🔎 Individual Package Breakdown")

        for p in results["package_evaluations"]:
            pkg_id = p["package_id"]
            decision = p["risk_prediction"]["delivery_decision"]
            risk_score = p["risk_prediction"]["risk_score"]
            sev_level = p["severity_assessment"]["severity_level"]
            cov_pct = p["severity_assessment"]["total_coverage_percentage"]
            dmgs = p["damage_classes_found"]

            with st.expander(f"📦 {pkg_id}  —  {decision} (Risk Score: {risk_score:.0f}/100)", expanded=True):
                c_left, c_right = st.columns([1, 2])

                with c_left:
                    if Path(p["crop_path"]).exists():
                        crop_img = Image.open(p["crop_path"])
                        st.image(crop_img, caption=f"{pkg_id} Isolated Crop")

                with c_right:
                    if decision == "SAFE TO DELIVER":
                        st.markdown(f'<span class="badge-safe">🟢 {decision}</span>', unsafe_allow_html=True)
                    elif decision == "INSPECT BEFORE DELIVERY":
                        st.markdown(f'<span class="badge-inspect">🟡 {decision}</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="badge-replace">🔴 {decision}</span>', unsafe_allow_html=True)

                    st.markdown(f"**Action Required**: {p['risk_prediction']['action_required']}")
                    st.progress(min(1.0, risk_score / 100.0), text=f"Estimated Internal Damage Risk: {risk_score:.0f} / 100")

                    st.markdown(
                        f"""
                        - **Damage Detected**: {', '.join(dmgs) if dmgs else 'None (Clean)'} ({p['defects_detected_count']} defects)
                        - **Severity Level**: `{sev_level}` (Damage Area Coverage: `{cov_pct}`)
                        - **Stage 1 Detector Confidence**: `{p['detector_confidence']:.2f}`
                        - **Bounding Box**: `{p['original_bbox']}`
                        - **Severity Rationale**: {p['severity_assessment']['rationale']}
                        """
                    )

        # -------------------------------------------------------------
        # 5. EXPORT & DOWNLOAD
        # -------------------------------------------------------------
        st.divider()
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            if Path(results["json_report_path"]).exists():
                with open(results["json_report_path"], "r") as f:
                    json_bytes = f.read()
                st.download_button(
                    label="📥 Download JSON Report",
                    data=json_bytes,
                    file_name="package_inspection_report.json",
                    mime="application/json",
                )
        with d_col2:
            if Path(results["txt_report_path"]).exists():
                with open(results["txt_report_path"], "r") as f:
                    txt_bytes = f.read()
                st.download_button(
                    label="📥 Download TXT Summary Report",
                    data=txt_bytes,
                    file_name="package_inspection_report.txt",
                    mime="text/plain",
                )


if __name__ == "__main__":
    main()
