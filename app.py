"""
Streamlit Web Application Interface.
Smart AI-Based Delivery Package Damage Detection and Internal Damage Risk Prediction System.
Two-Stage Computer Vision & Intelligent Risk Assessment Pipeline.
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
from src.crop_packages import crop_individual_packages

# Page configuration
st.set_page_config(
    page_title="AI Package Damage & Risk Prediction",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3b82f6, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.2rem;
    }
    
    .status-card {
        padding: 1.1rem;
        border-radius: 12px;
        background: #1e293b;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    
    .badge-safe {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .badge-inspect {
        background-color: #78350f;
        color: #fbbf24;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .badge-replace {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .crop-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_inspectors(pkg_conf: float, dmg_conf: float):
    """
    Caches model loading for high inference performance.
    """
    # Locate package detector weights
    pkg_candidates = [
        WEIGHTS_DIR / "package_detector_best.pt",
        RUNS_DIR / "train" / "package_detector_run" / "weights" / "best.pt",
        WEIGHTS_DIR / "best.pt",
    ]
    pkg_path = next((p for p in pkg_candidates if p.exists()), None)

    # Locate damage detector weights
    dmg_candidates = [
        WEIGHTS_DIR / "damage_detector_best.pt",
        WEIGHTS_DIR / "best.pt",
    ]
    dmg_path = next((p for p in dmg_candidates if p.exists()), None)

    two_stage = None
    single_stage = None

    if pkg_path and dmg_path:
        two_stage = TwoStagePackageInspector(
            package_detector_path=pkg_path,
            damage_detector_path=dmg_path,
            package_conf=pkg_conf,
            damage_conf=dmg_conf,
        )

    if dmg_path:
        single_stage = PackageDamagePredictor(
            model_path=dmg_path,
            conf_threshold=dmg_conf,
        )

    return two_stage, single_stage, pkg_path, dmg_path


def main():
    # Header
    st.markdown('<div class="main-title">📦 Smart AI Package Damage & Risk Prediction System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Two-Stage Computer Vision Pipeline: Stage 1 Individual Package Isolation & Stage 2 Deep Defect Analysis</div>',
        unsafe_allow_html=True,
    )

    # Hardware detection
    hw_info = detect_hardware()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Pipeline Configuration")
        st.caption(f"Hardware: **{hw_info['recommended_device'].upper()}** | PyTorch: **{hw_info['pytorch_version']}**")

        st.subheader("Confidence Thresholds")
        pkg_conf = st.slider("Stage 1 Package Detector Confidence", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
        dmg_conf = st.slider("Stage 2 Damage Detector Confidence", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
        crop_margin = st.slider("Package Crop Margin (%)", min_value=0, max_value=15, value=5, step=1) / 100.0

        st.divider()
        st.subheader("Model Status")
        two_stage_engine, single_stage_engine, pkg_path, dmg_path = load_inspectors(pkg_conf, dmg_conf)

        if pkg_path:
            st.success(f"Stage 1 Detector: `{pkg_path.name}`")
        else:
            st.warning("Stage 1 Detector: Training in progress...")

        if dmg_path:
            st.success(f"Stage 2 Damage Model: `{dmg_path.name}`")
        else:
            st.error("Stage 2 Damage Model: Not Found")

        st.divider()
        st.subheader("Delivery Decision Rules")
        st.markdown(
            """
            - 🟢 **Risk < 25**: `SAFE TO DELIVER`
            - 🟡 **Risk 25 – 65**: `INSPECT BEFORE DELIVERY`
            - 🔴 **Risk > 65**: `REPLACE PACKAGE`
            """
        )

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏢 Multi-Package Scene (Two-Stage)",
        "📦 Single Parcel Deep Inspection",
        "📁 Sample Gallery & Quick Tests",
        "📊 Model Analytics & Metrics",
    ])

    # =========================================================================
    # TAB 1: MULTI-PACKAGE SCENE INSPECTION
    # =========================================================================
    with tab1:
        st.subheader("Multi-Package Warehouse / Pallet Scene Inspection")
        st.caption("Stage 1 detects each package separately. Stage 2 executes defect detection on each crop.")

        col1, col2 = st.columns([1, 1])
        with col1:
            multi_upload = st.file_uploader(
                "Upload Scene Image (Warehouse / Pallet / Conveyor)",
                type=["jpg", "jpeg", "png", "webp"],
                key="multi_upload",
            )
        with col2:
            multi_sample_dir = SAMPLE_IMAGES_DIR / "multiple_parcels"
            multi_samples = list(multi_sample_dir.glob("*.jpg")) if multi_sample_dir.exists() else []
            sample_options = ["None"] + [s.name for s in multi_samples]
            selected_multi_sample = st.selectbox("Or choose a pre-loaded multi-package sample:", sample_options)

        target_multi_image = None
        if multi_upload:
            img_bytes = multi_upload.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            target_multi_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif selected_multi_sample != "None":
            target_multi_image = cv2.imread(str(multi_sample_dir / selected_multi_sample))

        if target_multi_image is not None:
            st.divider()
            run_btn = st.button("🚀 Run Two-Stage Multi-Package Inspection", type="primary", use_container_width=True)

            if run_btn:
                if two_stage_engine is None:
                    st.error("Two-Stage Pipeline models are not fully initialized yet. Please verify weights.")
                else:
                    with st.spinner("Executing Stage 1 Package Isolation + Stage 2 Defect Scoring..."):
                        start_t = time.time()
                        results = two_stage_engine.inspect_scene(
                            image_input=target_multi_image,
                            crop_margin_pct=crop_margin,
                        )
                        elapsed_ms = (time.time() - start_t) * 1000

                    st.success(f"Inspection Completed in {elapsed_ms:.1f} ms!")

                    # Top KPI Summary Cards
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    kpi1.metric("📦 Packages Isolated", results["total_packages"])
                    kpi2.metric("🟢 Safe to Deliver", results["safe_count"])
                    kpi3.metric("🟡 Inspect Before Delivery", results["inspect_count"])
                    kpi4.metric("🔴 Replace Package", results["replace_count"])

                    st.divider()

                    # Side-by-Side Visual Comparison
                    st.subheader("🖼️ Visual Pipeline Progression")
                    vis_col1, vis_col2 = st.columns(2)

                    with vis_col1:
                        st.markdown("**Stage 1: Individual Package Isolation**")
                        stage1_bgr = cv2.imread(results["stage1_image_path"])
                        st.image(cv2.cvtColor(stage1_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

                    with vis_col2:
                        st.markdown("**Final Result: Multi-Package Delivery Decision Badges**")
                        final_bgr = cv2.imread(results["final_image_path"])
                        st.image(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

                    st.divider()

                    # Individual Package Breakdown Gallery
                    st.subheader("🔎 Individual Package Deep Dive")
                    evals = results["package_evaluations"]

                    for p in evals:
                        pkg_id = p["package_id"]
                        decision = p["risk_prediction"]["delivery_decision"]
                        risk_score = p["risk_prediction"]["risk_score"]
                        sev_level = p["severity_assessment"]["severity_level"]
                        cov_pct = p["severity_assessment"]["total_coverage_percentage"]
                        dmgs = p["damage_classes_found"]

                        with st.expander(f"📦 {pkg_id}  —  {decision} (Risk: {risk_score:.0f}/100)", expanded=True):
                            c_left, c_right = st.columns([1, 2])

                            with c_left:
                                if Path(p["crop_path"]).exists():
                                    crop_img = Image.open(p["crop_path"])
                                    st.image(crop_img, caption=f"{pkg_id} Crop ({p['crop_filename']})", use_container_width=True)

                            with c_right:
                                # Status Badge
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

                    # Download Inspection Reports
                    st.divider()
                    st.subheader("📥 Export Inspection Reports")
                    d_col1, d_col2 = st.columns(2)
                    with d_col1:
                        if Path(results["json_report_path"]).exists():
                            with open(results["json_report_path"], "r") as f:
                                json_bytes = f.read()
                            st.download_button(
                                label="Download Structured JSON Report",
                                data=json_bytes,
                                file_name="multi_package_inspection.json",
                                mime="application/json",
                                use_container_width=True,
                            )
                    with d_col2:
                        if Path(results["txt_report_path"]).exists():
                            with open(results["txt_report_path"], "r") as f:
                                txt_bytes = f.read()
                            st.download_button(
                                label="Download Human-Readable TXT Report",
                                data=txt_bytes,
                                file_name="multi_package_inspection.txt",
                                mime="text/plain",
                                use_container_width=True,
                            )

    # =========================================================================
    # TAB 2: SINGLE PARCEL DEEP INSPECTION
    # =========================================================================
    with tab2:
        st.subheader("Single Parcel Direct Damage & Risk Inspection")
        st.caption("High-resolution damage defect localization, geometric area analysis, and internal risk calculation.")

        s_col1, s_col2 = st.columns([1, 1])
        with s_col1:
            single_upload = st.file_uploader(
                "Upload Single Package Image",
                type=["jpg", "jpeg", "png", "webp"],
                key="single_upload",
            )
        with s_col2:
            single_sample_dir = SAMPLE_IMAGES_DIR / "single_parcel"
            single_samples = list(single_sample_dir.glob("*.jpg")) if single_sample_dir.exists() else []
            single_sample_options = ["None"] + [s.name for s in single_samples]
            selected_single_sample = st.selectbox("Or choose a pre-loaded single parcel sample:", single_sample_options)

        target_single_image = None
        if single_upload:
            s_bytes = single_upload.read()
            nparr = np.frombuffer(s_bytes, np.uint8)
            target_single_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif selected_single_sample != "None":
            target_single_image = cv2.imread(str(single_sample_dir / selected_single_sample))

        if target_single_image is not None:
            st.divider()
            s_run_btn = st.button("🔍 Inspect Single Package", type="primary", use_container_width=True)

            if s_run_btn:
                if single_stage_engine is None:
                    st.error("Damage Detector weights not found.")
                else:
                    with st.spinner("Analyzing package damage features..."):
                        s_res = single_stage_engine.predict_image(
                            image_input=target_single_image,
                            save_annotated=True,
                            save_report=True,
                            package_id="PKG-SINGLE-TEST",
                        )

                    st.success("Single Parcel Inspection Complete!")

                    s_vis1, s_vis2 = st.columns(2)
                    with s_vis1:
                        st.markdown("**Original Package**")
                        st.image(cv2.cvtColor(target_single_image, cv2.COLOR_BGR2RGB), use_container_width=True)
                    with s_vis2:
                        st.markdown("**Annotated Damage Detection Overlay**")
                        pred_bgr = cv2.imread(s_res["annotated_image_path"])
                        st.image(cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

                    st.divider()
                    st.subheader("📋 Defect & Severity Telemetry")

                    s_kpi1, s_kpi2, s_kpi3 = st.columns(3)
                    decision = s_res["risk_info"]["delivery_decision"]
                    risk_pts = s_res["risk_info"]["risk_score"]
                    sev_level = s_res["severity_info"]["severity_level"]

                    s_kpi1.metric("Delivery Decision", decision)
                    s_kpi2.metric("Internal Risk Score", f"{risk_pts:.0f} / 100")
                    s_kpi3.metric("Severity Level", sev_level)

                    st.progress(min(1.0, risk_pts / 100.0), text=f"Estimated Internal Damage Risk: {risk_pts:.0f} / 100")
                    st.info(f"**Action Recommended**: {s_res['risk_info']['decision_action']}")

                    # Defect list
                    if s_res["defect_features"]:
                        st.markdown("#### Detected Damage Breakdown:")
                        for idx, df in enumerate(s_res["defect_features"]):
                            st.write(
                                f"- **Defect #{idx+1}**: `{df['damage_class']}` (Confidence: `{df['confidence']:.2f}`) | "
                                f"Area: `{df['area_percentage']}` | Location: `{df['spatial_placement']}`"
                            )
                    else:
                        st.success("✨ No physical damage defects detected. Package is in intact condition.")

    # =========================================================================
    # TAB 3: SAMPLE GALLERY & QUICK TESTS
    # =========================================================================
    with tab3:
        st.subheader("🖼️ Pre-loaded Test Sample Gallery")
        st.caption("Select any sample below to quickly preview test scenes.")

        st.markdown("### 🏢 Multiple Parcels Scenes")
        m_dir = SAMPLE_IMAGES_DIR / "multiple_parcels"
        if m_dir.exists():
            m_imgs = list(m_dir.glob("*.jpg"))[:6]
            if m_imgs:
                cols = st.columns(len(m_imgs))
                for i, p in enumerate(m_imgs):
                    with cols[i]:
                        st.image(Image.open(p), caption=p.name, use_container_width=True)

        st.divider()
        st.markdown("### 📦 Single Parcel Samples")
        s_dir = SAMPLE_IMAGES_DIR / "single_parcel"
        if s_dir.exists():
            s_imgs = list(s_dir.glob("*.jpg"))[:6]
            if s_imgs:
                cols = st.columns(len(s_imgs))
                for i, p in enumerate(s_imgs):
                    with cols[i]:
                        st.image(Image.open(p), caption=p.name, use_container_width=True)

    # =========================================================================
    # TAB 4: MODEL ANALYTICS & METRICS
    # =========================================================================
    with tab4:
        st.subheader("📊 Model Performance & Dataset Metrics")

        m_col1, m_col2 = st.columns(2)

        with m_col1:
            st.markdown("### Stage 1: Individual Package Detector")
            pkg_report_p = REPORTS_DIR / "package_detector_dataset_report.json"
            if pkg_report_p.exists():
                p_rep = load_json(pkg_report_p)
                st.json(p_rep)

        with m_col2:
            st.markdown("### Stage 2: Damage Detector Metrics")
            dmg_report_p = REPORTS_DIR / "model_metrics.json"
            if dmg_report_p.exists():
                d_rep = load_json(dmg_report_p)
                st.json(d_rep)

        st.divider()
        st.subheader("📈 Training & Validation Visualizations")
        vis_dir = OUTPUTS_DIR / "visualizations"
        if vis_dir.exists():
            v_imgs = list(vis_dir.glob("*.png")) + list(vis_dir.glob("*.jpg"))
            for v_img in v_imgs[:4]:
                st.image(Image.open(v_img), caption=v_img.name, use_container_width=True)


if __name__ == "__main__":
    main()
