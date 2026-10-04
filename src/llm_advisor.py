"""
LLM Logistics & Damage Assessment Advisor Module.
Generates comprehensive natural-language inspection insights, quality notes,
courier field instructions, and root-cause explanations using Generative AI (Gemini/OpenAI)
with a deterministic Expert Logistics Reasoning Engine fallback.
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger

logger = get_logger("LLMAdvisor")


def generate_llm_inspection_notes(
    inspection_data: Dict[str, Any],
    api_key: Optional[str] = None,
    provider: str = "gemini",
) -> Dict[str, str]:
    """
    Generates intelligent LLM quality notes and field handling advice.
    """
    total_packages = inspection_data.get("total_packages", 1)
    safe_count = inspection_data.get("safe_count", 0)
    inspect_count = inspection_data.get("inspect_count", 0)
    replace_count = inspection_data.get("replace_count", 0)
    package_evals = inspection_data.get("package_evaluations", [])

    # Check for live API Key (Gemini or OpenAI)
    gemini_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_key = api_key or os.getenv("OPENAI_API_KEY")

    prompt = build_inspection_prompt(inspection_data)

    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            if response and response.text:
                return parse_llm_response(response.text, inspection_data)
        except Exception as e:
            logger.warning(f"Gemini API generation encountered error: {e}. Using Expert Logistics Reasoning fallback.")

    if openai_key and provider == "openai":
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a senior logistics quality engineer and computer vision damage assessment expert."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            text = response.choices[0].message.content
            return parse_llm_response(text, inspection_data)
        except Exception as e:
            logger.warning(f"OpenAI API error: {e}. Using Expert Logistics Reasoning fallback.")

    # High-fidelity built-in Expert Logistics Reasoning Engine
    return generate_expert_rule_notes(inspection_data)


def build_inspection_prompt(data: Dict[str, Any]) -> str:
    """
    Builds a structured prompt for the LLM based strictly on observable external visual features.
    """
    evals = data.get("package_evaluations", [])
    summary = f"""You are a logistics quality engineer analyzing automated AI computer vision external damage inspection results.

IMPORTANT SCIENTIFIC CONSTRAINTS:
- The vision system only captures external camera views. Do NOT claim internal shock absorption, 100% internal protection, 100% integrity, or certified safe delivery.
- State clearly that the delivery decision is an AI-Based External Condition Assessment and heuristic model recommendation.
- When no damage is detected, use exact phrasing: "No externally visible damage was detected."

INSPECTION SUMMARY:
- Total Packages Detected: {data.get('total_packages', len(evals))}
- Recommended Safe to Deliver: {data.get('safe_count', 0)}
- Recommended Inspect Before Delivery: {data.get('inspect_count', 0)}
- Recommended Replace Package: {data.get('replace_count', 0)}

PER-PACKAGE TELEMETRY:
"""
    for p in evals:
        pkg_id = p.get("package_id", "Package")
        is_valid = p.get("is_valid_package", True)
        if not is_valid:
            area_pct = p.get("box_area_ratio_pct", "0.00%")
            summary += f"""
[{pkg_id}]
- Detection Status: INVALID PACKAGE DETECTION (Stage 1 Quality Gate Rejection)
- Bounding Box Area Ratio: {area_pct} (Exceeds maximum allowable package area threshold)
- Damage Analysis: Bypassed — individual package boundary could not be established
- Recommended Action: Review image or re-capture closer view of package
"""
        else:
            dmgs = p.get("damage_classes_found", [])
            sev = p.get("severity_assessment", {}).get("severity_level", "Unknown")
            cov = p.get("severity_assessment", {}).get("total_coverage_percentage", "0%")
            risk_val = p.get("risk_prediction", {}).get("risk_score")
            risk_str = f"{risk_val:.0f}/100" if isinstance(risk_val, (int, float)) else "N/A"
            dec = p.get("risk_prediction", {}).get("delivery_decision", "Unknown")
            summary += f"""
[{pkg_id}]
- Recommended Delivery Decision: {dec} (Heuristic Estimated Internal Damage Risk Score: {risk_str})
- External Severity: {sev} (External Damage Area Coverage: {cov})
- Detected Defect Types: {', '.join(dmgs) if dmgs else 'No externally visible damage detected'}
"""

    summary += """
Please generate a structured quality assessment report with these 4 distinct sections:
1. EXECUTIVE_STATUS: A clear overview of the external condition assessment and delivery recommendation.
2. INTEGRITY_ANALYSIS: Defensible analysis of observable external surface conditions and heuristic risk.
3. COURIER_INSTRUCTIONS: Practical handling steps based on external visual flags.
4. RECIPIENT_NOTIFICATION: Customer delivery update stating visual inspection was completed without making unsupported internal claims.
"""
    return summary


def parse_llm_response(text: str, fallback_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Parses sections from LLM response or falls back gracefully.
    """
    return {
        "executive_summary": text,
        "is_llm_generated": True,
    }


def generate_expert_rule_notes(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Deterministic Expert Logistics Reasoning Engine.
    Provides scientifically defensible logistics assessments based strictly on observable external visual features.
    """
    evals = data.get("package_evaluations", [])
    valid_evals = [p for p in evals if p.get("is_valid_package", True)]
    invalid_evals = [p for p in evals if not p.get("is_valid_package", True)]

    total_pkgs = len(evals)
    valid_pkgs_count = len(valid_evals)
    safe_pkgs = [p for p in valid_evals if p.get("delivery_decision") == "SAFE TO DELIVER"]
    inspect_pkgs = [p for p in valid_evals if p.get("delivery_decision") == "INSPECT BEFORE DELIVERY"]
    replace_pkgs = [p for p in valid_evals if p.get("delivery_decision") == "REPLACE PACKAGE"]

    # 1. Executive Summary
    if valid_pkgs_count == 0 and len(invalid_evals) > 0:
        exec_summary = (
            f"⚠️ **Package Detection Requires Review**: Stage 1 could not reliably isolate an individual package from the input image "
            f"({len(invalid_evals)} region(s) exceeded the maximum package area threshold). "
            f"Damage analysis and delivery decisions were not performed because individual package boundaries could not be established."
        )
    elif len(replace_pkgs) > 0:
        exec_summary = (
            f"⚠️ **Inspection Action Required**: The AI visual inspection identified {len(replace_pkgs)} package(s) exhibiting "
            f"significant external deformation or surface damage exceeding the heuristic risk threshold (Heuristic Risk Score > 65/100). "
            f"Dispatch quarantine is recommended for secondary evaluation or re-packaging."
        )
    elif len(inspect_pkgs) > 0:
        exec_summary = (
            f"🔍 **Inspection Recommended**: {len(inspect_pkgs)} package(s) exhibit observable surface defects or minor corner creasing "
            f"(Heuristic Risk Score: 25–65/100). Based on detected external visual features and business rules, manual physical check "
            f"by warehouse handlers is recommended before dispatch."
        )
    else:
        exec_summary = (
            f"✅ **AI-Based External Condition Assessment**: All {valid_pkgs_count} valid detected package(s) exhibited no externally visible damage "
            f"(Heuristic Estimated Internal Damage Risk Score < 25/100). Based on the detected external condition and the configured "
            f"heuristic risk threshold, the system recommends this shipment for delivery."
        )

    # 2. Structural & External Condition Analysis
    integrity_points = []
    for p in evals:
        pkg_id = p.get("package_id", "Package")
        is_valid = p.get("is_valid_package", True)

        if not is_valid:
            integrity_points.append(
                f"• **{pkg_id}**: ⚠️ Stage 1 Quality Gate Rejection — Package boundary could not be reliably established. Damage analysis was not performed."
            )
        else:
            dmgs = p.get("damage_classes_found", [])
            sev = p.get("severity_assessment", {}).get("severity_level", "No Damage")
            cov = p.get("severity_assessment", {}).get("total_coverage_percentage", "0%")
            risk_val = p.get("risk_prediction", {}).get("risk_score")
            risk_str = f"{risk_val:.0f}/100" if isinstance(risk_val, (int, float)) else "N/A"

            if not dmgs or sev == "No Damage":
                integrity_points.append(
                    f"• **{pkg_id}**: No externally visible damage was detected. Heuristic Estimated Internal Damage Risk Score: `{risk_str}`."
                )
            else:
                integrity_points.append(
                    f"• **{pkg_id}**: Observed external defect types: `[{', '.join(dmgs)}]` covering `{cov}` of detected package area. "
                    f"Assessed External Severity: `{sev}` (Heuristic Estimated Internal Damage Risk Score: `{risk_str}`)."
                )

    integrity_analysis = "\n".join(integrity_points)

    # 3. Courier & Hub Handling Instructions
    if valid_pkgs_count == 0 and len(invalid_evals) > 0:
        courier_instructions = (
            "1. **Quality Gate Review Required**: Image contains oversized / full-scene detection exceeding individual package boundary.\n"
            "2. Capture a closer image containing the individual package or perform manual check.\n"
            "3. Do not dispatch until package condition is visually confirmed."
        )
    elif len(replace_pkgs) > 0:
        courier_instructions = (
            "1. Hold packages flagged with `REPLACE PACKAGE` recommendation.\n"
            "2. Route to inspection / re-packaging station for verification.\n"
            "3. Update warehouse inventory status accordingly."
        )
    elif len(inspect_pkgs) > 0:
        courier_instructions = (
            "1. Perform standard manual external check on flagged items before loading.\n"
            "2. Maintain orientation markers during transit.\n"
            "3. If external surfaces remain secure, proceed with standard handling."
        )
    else:
        courier_instructions = (
            "1. **Recommended Delivery Decision**: Safe to deliver based on external visual inspection.\n"
            "2. Proceed with standard dispatch routing.\n"
            "3. Maintain standard handling procedures."
        )

    # 4. Customer Tracking Update
    if valid_pkgs_count == 0 and len(invalid_evals) > 0:
        customer_note = "Your package has arrived at our sorting hub and is currently undergoing facility processing."
    elif len(replace_pkgs) > 0:
        customer_note = "Your shipment completed automated visual inspection at our sorting facility. A routine packaging verification is underway prior to dispatch."
    elif len(inspect_pkgs) > 0:
        customer_note = "Your package has arrived at the facility and is undergoing final transit preparation for delivery."
    else:
        customer_note = "Your package has completed AI-based external visual inspection. No visible external damage was detected, and the system recommends it for delivery."

    return {
        "executive_summary": exec_summary,
        "integrity_analysis": integrity_analysis,
        "courier_instructions": courier_instructions,
        "customer_note": customer_note,
        "is_llm_generated": False,
    }
