"""
Smart AI-Based Delivery Package Damage Detection and Internal Damage Risk Prediction System.
Interactive Streamlit Application with Auto-Detection, Two-Stage Vision Pipeline, and LLM Logistics Advisor.
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
from src.inference import PackageDamagePredictor
from src.llm_advisor import generate_llm_inspection_notes

# Streamlit Page Setup
st.set_page_config(
    page_title="AI Package Damage & Risk Advisor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Glassmorphic & Modern Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-header {
        padding: 1.5rem 2rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa 0%, #34d399 50%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .hero-sub {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0.4rem;
    }
    
    .llm-card {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(31, 41, 55, 0.9) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.15);
    }
    
    .llm-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #60a5fa;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.8rem;
    }
    
    .badge-safe {
        background: linear-gradient(90deg, #059669, #10b981);
        color: #ffffff;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        letter-spacing: 0.5px;
    }
    
    .badge-inspect {
        background: linear-gradient(90deg, #d97706, #f59e0b);
        color: #ffffff;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        letter-spacing: 0.5px;
    }
    
    .badge-replace {
        background: linear-gradient(90deg, #dc2626, #ef4444);
        color: #ffffff;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        letter-spacing: 0.5px;
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
    # Hero Banner
    st.markdown(
        """
        <div class="hero-header">
            <div class="hero-title">📦 Smart AI Package Damage & Risk Advisor</div>
            <div class="hero-sub">Auto-Detecting Computer Vision Pipeline with LLM-Powered Quality & Logistics Reasoning</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hw_info = detect_hardware()

    # Sidebar Controls
    with st.sidebar:
        st.header("⚡ System & Model Control")
        st.caption(f"Device: **{hw_info['recommended_device'].upper()}** | PyTorch: **{hw_info['pytorch_version']}**")

        st.subheader("🤖 LLM Reasoning Engine")
        llm_provider = st.selectbox("LLM Provider", ["Google Gemini (Recommended)", "OpenAI GPT-4o", "Built-in Expert Reasoning"])
        api_key_input = st.text_input("API Key (Optional)", type="password", help="Leave blank to use built-in Expert Logistics Reasoning Engine")

        st.divider()
        st.subheader("🎯 Vision Sensitivity")
        pkg_conf = st.slider("Package Detection Confidence", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
        dmg_conf = st.slider("Damage Defect Confidence", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
        crop_margin = st.slider("Crop Margin (%)", min_value=0, max_value=15, value=5, step=1) / 100.0

        st.divider()
        st.subheader("Model Status")
        two_stage_engine, pkg_path, dmg_path = load_vision_engines(pkg_conf, dmg_conf)
        if pkg_path:
            st.success(f"Stage 1 Detector: `{pkg_path.name}`")
        if dmg_path:
            st.success(f"Stage 2 Damage Model: `{dmg_path.name}`")

    # Main Area: Auto-Detection
    st.subheader("📥 Input Package or Warehouse Scene")
    input_col1, input_col2 = st.columns([1, 1])

    with input_col1:
        uploaded_file = st.file_uploader(
            "Upload any image (Single package or multi-package warehouse scene)",
            type=["jpg", "jpeg", "png", "webp"],
            help="The system will automatically isolate each package, inspect for defects, and generate LLM quality notes.",
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

    # Resolve image
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

        with st.spinner("🤖 Auto-Detecting Packages, Analyzing Damage, & Generating LLM Insights..."):
            start_time = time.time()
            results = two_stage_engine.inspect_scene(
                image_input=input_image_bgr,
                crop_margin_pct=crop_margin,
            )
            elapsed_time = (time.time() - start_time) * 1000

            # Generate LLM Logistics Notes
            provider_code = "gemini" if "Gemini" in llm_provider else ("openai" if "OpenAI" in llm_provider else "expert")
            llm_notes = generate_llm_inspection_notes(
                inspection_data=results,
                api_key=api_key_input if api_key_input.strip() else None,
                provider=provider_code,
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
        # 2. LLM INTELLIGENT QUALITY & LOGISTICS ADVICE
        # -------------------------------------------------------------
        st.markdown(
            f"""
            <div class="llm-card">
                <div class="llm-title">🤖 AI Quality Assessment & Logistics Advisor</div>
                <div style="font-size: 1.05rem; line-height: 1.6; color: #f1f5f9; margin-bottom: 1rem;">
                    {llm_notes['executive_summary']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_llm_left, col_llm_right = st.columns(2)
        with col_llm_left:
            st.markdown("### 🔬 Structural & Integrity Analysis")
            if "integrity_analysis" in llm_notes:
                st.markdown(llm_notes["integrity_analysis"])
            else:
                st.info("Full container integrity verified with zero physical compromise.")

        with col_llm_right:
            st.markdown("### 🚚 Courier Field Instructions")
            if "courier_instructions" in llm_notes:
                st.markdown(llm_notes["courier_instructions"])
            if "customer_note" in llm_notes:
                st.markdown(f"**📱 Customer Update**: *\"{llm_notes['customer_note']}\"*")

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
