"""
Internal Damage Risk Assessment & Delivery Decision Module.
Computes the Estimated Internal Damage Risk Score (0-100) and delivery recommendations.
"""

from typing import Dict, Any, List


def calculate_internal_damage_risk(
    severity_info: Dict[str, Any],
    defect_features: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calculates the Estimated Internal Damage Risk Score (0-100).
    
    IMPORTANT:
    This score is an engineered heuristic index based on external damage features.
    It is NOT a scientifically or statistically calibrated probability of internal failure.
    
    Formula Composition:
    1. Base Severity Contribution (0-50 pts):
       - No Damage: 0 pts
       - Minor: 15 pts
       - Moderate: 35 pts
       - Severe: 50 pts
    2. Area Coverage Contribution (0-25 pts):
       - (total_coverage_ratio / 0.20) * 25, clamped to 25 pts
    3. Defect Count & Density Contribution (0-15 pts):
       - min(15, num_defects * 5)
    4. Structural Class Multiplier / Location Vulnerability (0-10 pts):
       - Top/Corner impacts and severe damage classes add up to 10 pts.
    """
    severity_level = severity_info.get("severity_level", "No Damage")
    total_coverage = severity_info.get("total_coverage_ratio", 0.0)
    num_defects = len(defect_features)

    if severity_level == "No Damage" or num_defects == 0:
        risk_score = 0.0
        delivery_decision = "SAFE TO DELIVER"
        decision_action = "Package package appears structurally sound with no external defects. Safe for direct customer delivery."
    else:
        # 1. Base Severity
        base_map = {
            "Minor": 15.0,
            "Moderate": 35.0,
            "Severe": 50.0,
        }
        base_pts = base_map.get(severity_level, 10.0)

        # 2. Area Coverage (up to 25 pts)
        area_pts = min(25.0, (total_coverage / 0.20) * 25.0)

        # 3. Defect Count (up to 15 pts)
        count_pts = min(15.0, num_defects * 5.0)

        # 4. Critical location / type bonus (up to 10 pts)
        loc_type_pts = 0.0
        for f in defect_features:
            # Top impacts or corner areas often compromise internal content cushioning
            if f.get("location_vertical") == "Top":
                loc_type_pts += 3.0
            if any(k in f.get("damage_class", "").lower() for k in ["crush", "puncture", "hole", "break"]):
                loc_type_pts += 4.0
        loc_type_pts = min(10.0, loc_type_pts)

        # Total Raw Score
        raw_risk = base_pts + area_pts + count_pts + loc_type_pts
        risk_score = min(100.0, max(0.0, raw_risk))

        # Heuristic Business Delivery Decision
        if risk_score < 25.0:
            delivery_decision = "SAFE TO DELIVER"
            decision_action = "Minor cosmetic blemishes only. High probability internal contents remain intact. Proceed with delivery."
        elif risk_score <= 65.0:
            delivery_decision = "INSPECT BEFORE DELIVERY"
            decision_action = "Moderate package deformation or tearing. Flag for courier/warehouse visual check or recipient notice prior to handover."
        else:
            delivery_decision = "REPLACE PACKAGE"
            decision_action = "High risk of internal product compromise. Halt delivery, return to fulfillment center for replacement or internal verification."

    return {
        "risk_metric_name": "Estimated Internal Damage Risk Score",
        "risk_score": round(risk_score, 1),
        "score_range": "0 - 100",
        "delivery_decision": delivery_decision,
        "decision_action": decision_action,
        "decision_thresholds": {
            "safe_to_deliver": "Score < 25",
            "inspect_before_delivery": "Score 25 - 65",
            "replace_package": "Score > 65",
        },
        "score_breakdown": {
            "severity_level": severity_level,
            "total_coverage_ratio": total_coverage,
            "num_defects": num_defects,
        },
    }
