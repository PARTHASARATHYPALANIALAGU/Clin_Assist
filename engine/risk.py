"""
Risk categorization module for ClinAssist
Uses deterministic rule-based logic (NO LLM involvement)
"""
from typing import Dict, Any, List
import json


# Critical chief complaint keywords
CRITICAL_KEYWORDS = [
    "chest pain",
    "difficulty breathing",
    "shortness of breath",
    "loss of consciousness",
    "fainted",
    "seizure",
    "stroke",
    "slurred speech",
    "face drooping",
    "paralysis",
    "severe bleeding",
    "uncontrolled bleeding",
    "overdose",
    "allergic reaction",
    "throat swelling",
]


HIGH_PRIORITY_KEYWORDS = [
    "high fever",
    "vomiting blood",
    "black stool",
    "bloody stool",
    "severe abdominal pain",
    "confusion",
    "stiff neck",
]


def detect_urgent_keyword(text: str) -> Dict[str, str] | None:
    lowered = (text or "").lower()

    for keyword in CRITICAL_KEYWORDS:
        if keyword in lowered:
            return {"risk_level": "CRITICAL", "keyword": keyword}

    for keyword in HIGH_PRIORITY_KEYWORDS:
        if keyword in lowered:
            return {"risk_level": "HIGH", "keyword": keyword}

    return None


def build_urgent_assessment(detected: Dict[str, str], source_text: str) -> Dict[str, str]:
    """Build a deterministic urgent risk assessment from detected keyword."""
    risk_level = detected["risk_level"]
    keyword = detected["keyword"]
    complaint = (source_text or "").strip()

    if risk_level == "CRITICAL":
        return {
            "risk_level": "CRITICAL",
            "reason": f"Urgent symptom detected: '{keyword}' in complaint '{complaint}'.",
            "recommended_action": "Seek emergency care now and contact a doctor immediately."
        }

    return {
        "risk_level": "HIGH",
        "reason": f"High-priority symptom detected: '{keyword}' in complaint '{complaint}'.",
        "recommended_action": "Please see a doctor urgently today for evaluation and treatment."
    }


def categorize_risk(symptom_record: Dict[str, Any]) -> Dict[str, str]:
    """
    Categorize risk level using deterministic rules
    
    Rules (in priority order):
    CRITICAL: severity >= 9 OR chief_complaint contains critical keywords
    HIGH: severity >= 7 OR (progression == "worsening" AND duration <= 2 days)
    MODERATE: severity >= 4 OR associated_symptoms count >= 3
    LOW: everything else
    
    Args:
        symptom_record: Dictionary containing all symptom attributes
    
    Returns:
        Dictionary with risk_level, reason, and recommended_action
    """
    severity = symptom_record.get("severity")
    chief_complaint = (symptom_record.get("chief_complaint") or "").lower()
    progression = symptom_record.get("progression")
    duration = symptom_record.get("duration") or ""
    associated_symptoms = symptom_record.get("associated_symptoms") or []
    
    # Parse associated symptoms if stored as JSON string
    if isinstance(associated_symptoms, str):
        try:
            associated_symptoms = json.loads(associated_symptoms)
        except:
            associated_symptoms = []
    
    # CRITICAL: Check severity >= 9
    if severity is not None and severity >= 9:
        return {
            "risk_level": "CRITICAL",
            "reason": f"Severity level {severity}/10 indicates critical condition",
            "recommended_action": "Seek emergency care immediately. Call 911 or go to the nearest emergency department."
        }
    
    # CRITICAL: Check for critical keywords in chief complaint
    for keyword in CRITICAL_KEYWORDS:
        if keyword in chief_complaint:
            return {
                "risk_level": "CRITICAL",
                "reason": f"Chief complaint '{chief_complaint}' contains critical symptom: '{keyword}'",
                "recommended_action": "Seek emergency care now. Contact a doctor immediately for urgent evaluation and treatment."
            }

    # HIGH: Check for high-priority keywords in chief complaint
    for keyword in HIGH_PRIORITY_KEYWORDS:
        if keyword in chief_complaint:
            return {
                "risk_level": "HIGH",
                "reason": f"Chief complaint '{chief_complaint}' contains high-priority symptom: '{keyword}'",
                "recommended_action": "See a doctor urgently today for evaluation and treatment."
            }
    
    # HIGH: Check severity >= 7
    if severity is not None and severity >= 7:
        return {
            "risk_level": "HIGH",
            "reason": f"Severity level {severity}/10 requires urgent attention",
            "recommended_action": "Schedule urgent appointment within 24 hours or visit urgent care."
        }
    
    # HIGH: Check for worsening symptoms with short duration
    if progression == "worsening" and duration:
        duration_lower = duration.lower()
        # Check if duration mentions hours, 1 day, or 2 days
        if ("hour" in duration_lower or 
            "1 day" in duration_lower or 
            "2 day" in duration_lower or
            "today" in duration_lower or
            "yesterday" in duration_lower):
            return {
                "risk_level": "HIGH",
                "reason": f"Symptoms worsening over short duration ({duration}) requires prompt evaluation",
                "recommended_action": "Schedule urgent appointment within 24 hours or visit urgent care."
            }
    
    # MODERATE: Check severity >= 4
    if severity is not None and severity >= 4:
        return {
            "risk_level": "MODERATE",
            "reason": f"Severity level {severity}/10 with moderate symptoms",
            "recommended_action": "See physician within 2-3 days for evaluation."
        }
    
    # MODERATE: Check for multiple associated symptoms
    if len(associated_symptoms) >= 3:
        return {
            "risk_level": "MODERATE",
            "reason": f"Multiple associated symptoms ({len(associated_symptoms)}) present",
            "recommended_action": "See physician within 2-3 days for evaluation."
        }
    
    # LOW: Default case
    return {
        "risk_level": "LOW",
        "reason": "Symptoms appear manageable with low severity indicators",
        "recommended_action": "Monitor symptoms. Schedule routine appointment if symptoms persist or worsen."
    }
