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
    Builds a structured prompt for the LLM.
    """
    evals = data.get("package_evaluations", [])
    summary = f"""You are a senior logistics quality engineer analyzing automated AI computer vision damage inspection results.

INSPECTION SUMMARY:
- Total Packages Detected: {data.get('total_packages', len(evals))}
- Safe to Deliver: {data.get('safe_count', 0)}
- Inspect Before Delivery: {data.get('inspect_count', 0)}
- Replace Package: {data.get('replace_count', 0)}

PER-PACKAGE TELEMETRY:
"""
    for p in evals:
        pkg_id = p.get("package_id", "Package")
        dmgs = p.get("damage_classes_found", [])
        sev = p.get("severity_assessment", {}).get("severity_level", "Unknown")
        cov = p.get("severity_assessment", {}).get("total_coverage_percentage", "0%")
        risk = p.get("risk_prediction", {}).get("risk_score", 0)
        dec = p.get("risk_prediction", {}).get("delivery_decision", "Unknown")
        summary += f"""
[{pkg_id}]
- Decision: {dec} (Internal Damage Risk Score: {risk:.0f}/100)
- Severity: {sev} (Damage Area Coverage: {cov})
- Defect Types: {', '.join(dmgs) if dmgs else 'None (Intact)'}
"""

    summary += """
Please generate a structured quality assessment report with these 4 distinct sections:
1. EXECUTIVE_STATUS: A clear 2-3 sentence overview of why the shipment is in good condition or why specific items require attention.
2. INTEGRITY_ANALYSIS: Technical breakdown of cardboard structural strength, seal status, and risk to fragile internal contents.
3. COURIER_INSTRUCTIONS: Specific, actionable step-by-step guidance for the delivery driver or hub handling team.
4. RECIPIENT_NOTIFICATION: Professional, customer-friendly delivery tracking update message.
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
    Provides rich, professional logistics assessments when API keys are not supplied.
    """
    evals = data.get("package_evaluations", [])
    total_pkgs = len(evals)
    safe_pkgs = [p for p in evals if p.get("risk_prediction", {}).get("delivery_decision") == "SAFE TO DELIVER"]
    inspect_pkgs = [p for p in evals if p.get("risk_prediction", {}).get("delivery_decision") == "INSPECT BEFORE DELIVERY"]
    replace_pkgs = [p for p in evals if p.get("risk_prediction", {}).get("delivery_decision") == "REPLACE PACKAGE"]

    # 1. Executive Summary
    if len(replace_pkgs) > 0:
        exec_summary = (
            f"⚠️ **Critical Attention Required**: The inspection identified {len(replace_pkgs)} package(s) exhibiting "
            f"severe structural deformation or critical puncture defects exceeding safety thresholds (Risk Score > 65/100). "
            f"Immediate dispatch quarantine is enforced to prevent delivery of compromised merchandise."
        )
    elif len(inspect_pkgs) > 0:
        exec_summary = (
            f"🔍 **Conditional Clearance (Verification Recommended)**: {len(inspect_pkgs)} package(s) display moderate superficial wear, "
            f"minor corner compression, or tape irregularities (Risk Index: 25–65/100). The primary structural integrity remains largely viable, "
            f"but manual physical inspection by warehouse handlers is advised prior to final customer handover."
        )
    else:
        exec_summary = (
            f"✅ **Optimal Shipment Condition (Certified Safe to Deliver)**: All {total_pkgs} detected package(s) demonstrated "
            f"complete structural integrity with zero observable defect signatures (Risk Index < 25/100). The cardboard container geometry, "
            f"corner firmness, and tamper seals are in prime factory condition, ensuring 100% protection for internal contents."
        )

    # 2. Structural & Integrity Analysis
    integrity_points = []
    for p in evals:
        pkg_id = p.get("package_id", "Package")
        dmgs = p.get("damage_classes_found", [])
        sev = p.get("severity_assessment", {}).get("severity_level", "No Damage")
        cov = p.get("severity_assessment", {}).get("total_coverage_percentage", "0%")
        risk = p.get("risk_prediction", {}).get("risk_score", 0)

        if not dmgs or sev == "No Damage":
            integrity_points.append(
                f"• **{pkg_id}**: Outer corrugated walls show zero punctures, stress creasing, or moisture intrusion. Internal shock-absorption buffer is fully intact."
            )
        else:
            integrity_points.append(
                f"• **{pkg_id}**: Observed defect types: `[{', '.join(dmgs)}]` covering `{cov}` of package surface. Severity categorized as `{sev}` (Internal Risk: `{risk:.0f}/100`). Potential localized load vulnerability on affected edge/face."
            )

    integrity_analysis = "\n".join(integrity_points)

    # 3. Courier & Hub Handling Instructions
    if len(replace_pkgs) > 0:
        courier_instructions = (
            "1. **Do NOT load** marked `REPLACE` packages onto local delivery vans.\n"
            "2. Transfer flagged items to the **Claims & Return Merchandise Hub** for item re-boxing.\n"
            "3. Notify warehouse dispatch manager for automated merchant replacement fulfillment."
        )
    elif len(inspect_pkgs) > 0:
        courier_instructions = (
            "1. Perform a **10-second tactile check** on corners and tape seals before loading.\n"
            "2. Ensure fragile orientation arrows are maintained upright during transit.\n"
            "3. If container feels firm and intact, proceed with regular recipient delivery."
        )
    else:
        courier_instructions = (
            "1. **Direct Green-Light Dispatch**: Proceed with standard route loading.\n"
            "2. No special secondary verification needed; package meets prime delivery specifications.\n"
            "3. Maintain standard dry stacking procedures."
        )

    # 4. Customer Tracking Update
    if len(replace_pkgs) > 0:
        customer_note = "Your shipment was checked during hub sorting. To ensure pristine item quality, we are repackaging your order for safe transit."
    elif len(inspect_pkgs) > 0:
        customer_note = "Your package has arrived at the local facility in good condition and is undergoing final transit preparation for delivery."
    else:
        customer_note = "Your package passed AI quality inspection with 100% integrity and is on schedule for flawless delivery."

    return {
        "executive_summary": exec_summary,
        "integrity_analysis": integrity_analysis,
        "courier_instructions": courier_instructions,
        "customer_note": customer_note,
        "is_llm_generated": False,
    }
