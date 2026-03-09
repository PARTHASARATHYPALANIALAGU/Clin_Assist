"""
Intake state machine for ClinAssist
Manages conversation flow: GREETING → COLLECTING → CLARIFYING → SUMMARIZING → COMPLETE
"""
from typing import Dict, Any, Optional, List
import re
from config import STATES, FIELD_PRIORITY
from .memory import SessionMemory
from database import save_turn, update_session_state, get_session_state, update_symptom_record
from . import llm
from . import risk


def process_interaction(session_id: str, user_input: str) -> Dict[str, Any]:
    """
    Process user interaction through the intake state machine
    
    Args:
        session_id: Current session ID
        user_input: User's text input
    
    Returns:
        Dictionary with response_text, state, is_complete, and optional risk_assessment
    """
    # Load session memory
    memory = SessionMemory(session_id)
    current_state = get_session_state(session_id)
    
    # Save user input
    if user_input and user_input.strip():
        save_turn(session_id, "user", user_input)
    
    response_text = ""
    risk_assessment = None
    wellness_tip = None
    trigger_consult_transition = False
    show_report_popup = False

    def _extract_severity_value(text: str) -> Optional[int]:
        if not text:
            return None

        lowered = text.lower()
        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }

        range_match = re.search(r"\b(10|[1-9])\s*(?:to|-|–)\s*(10|[1-9])\b", lowered)
        if range_match:
            left = int(range_match.group(1))
            right = int(range_match.group(2))
            return max(1, min(10, round((left + right) / 2)))

        word_range_match = re.search(
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:to|-|–)\s*"
            r"(one|two|three|four|five|six|seven|eight|nine|ten)\b",
            lowered,
        )
        if word_range_match:
            left = word_to_num[word_range_match.group(1)]
            right = word_to_num[word_range_match.group(2)]
            return max(1, min(10, round((left + right) / 2)))

        single_num = re.search(r"\b(10|[1-9])\b", lowered)
        if single_num:
            return int(single_num.group(1))

        for word, value in word_to_num.items():
            if re.search(rf"\b{word}\b", lowered):
                return value

        return None

    def _extract_associated_symptoms(text: str) -> Optional[List[str]]:
        if not text:
            return None

        lowered = text.lower()

        negative_markers = [
            "no other symptoms",
            "no additional symptoms",
            "no more symptoms",
            "none",
            "nothing else",
            "just fever",
            "only fever",
        ]
        if any(marker in lowered for marker in negative_markers):
            return ["none reported"]

        symptom_aliases = [
            ("chills", ["chill", "chills"]),
            ("body aches", ["body ache", "body aches", "aches", "achy"]),
            ("headache", ["headache", "head pain"]),
            ("fatigue", ["fatigue", "tired", "weak"]),
            ("sweating", ["sweating", "sweats"]),
            ("cough", ["cough"]),
            ("sore throat", ["sore throat", "throat pain"]),
            ("runny nose", ["runny nose", "congestion"]),
            ("nausea", ["nausea", "nauseous"]),
            ("vomiting", ["vomit", "vomiting"]),
            ("diarrhea", ["diarrhea", "loose stools"]),
            ("dizziness", ["dizzy", "dizziness"]),
            ("shortness of breath", ["shortness of breath", "breathless"]),
        ]

        found: List[str] = []
        for canonical, aliases in symptom_aliases:
            if any(alias in lowered for alias in aliases):
                found.append(canonical)

        if not found:
            return None

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for symptom in found:
            if symptom not in seen:
                deduped.append(symptom)
                seen.add(symptom)

        return deduped

    def _finalize_intake(symptom_data: Dict[str, Any], partial: bool = False):
        nonlocal risk_assessment, response_text, wellness_tip

        update_session_state(session_id, STATES["SUMMARIZING"])

        # Risk assessment with safety
        try:
            risk_result = risk.categorize_risk(symptom_data)
        except Exception as e:
            risk_result = {"risk_level": "LOW", "reason": "System error during assessment", "recommended_action": "Please see a doctor."}
        risk_assessment = risk_result

        # Save risk assessment
        update_symptom_record(session_id, {
            "risk_level": risk_result["risk_level"],
            "risk_reason": risk_result["reason"],
            "recommended_action": risk_result["recommended_action"]
        })

        # Summary with safety
        try:
            summary = llm.generate_summary(symptom_data, session_id)
        except Exception:
            summary = "Summary generation failed."
        update_symptom_record(session_id, {"summary": summary})

        # Wellness tip with safety
        try:
            if risk_result['risk_level'] in ["LOW", "MODERATE"]:
                wellness_tip = llm.generate_health_advice(symptom_data, session_id)
            else:
                wellness_tip = f"Please prioritize seeking medical evaluation as advised. {risk_result['recommended_action']}"
        except Exception:
            wellness_tip = "Stay well and consult a professional if concerns persist."

        # Move to offering consult instead of complete immediately
        response_text = "I've completed your symptom check based on the details provided. Do you have any questions or doubts about your overall health right now?"
        update_session_state(session_id, STATES["OFFERING_CONSULT"])
    
    # State machine logic
    if current_state == STATES["GREETING"]:
        # Welcome and ask for chief complaint
        response_text = (
            "Hi, I'm ClinAssist. I'll ask a few simple questions about how you feel. "
            "What problem are you having today?"
        )
        update_session_state(session_id, STATES["COLLECTING"])
    
    elif current_state == STATES["COLLECTING"] or current_state == STATES["CLARIFYING"]:
        user_lower = (user_input or "").lower()
        user_cleaned = user_lower.strip().strip('.,!?')
        
        # Check for early exit / abort commands
        exit_keywords = ["end", "stop", "cancel", "that's enough", "finish", "quit", "end conversation", "end our conversation", "end now"]
        if any(user_cleaned == kw or user_cleaned.startswith(kw + " ") or user_cleaned.endswith(" " + kw) for kw in exit_keywords):
            symptom_data = memory.get_symptom_data()
            _finalize_intake(symptom_data, partial=True)
            # Flag that we've already set the response and are skipping the rest
            pass
        
        elif not response_text:
            # Extract symptoms from user input
            extracted = llm.extract_symptoms(
                memory.conversation_history,
                user_input,
                session_id
            )

            # Update memory with extracted data
            if extracted:
                is_sufficient = extracted.pop("is_sufficient", True)
                if is_sufficient:
                    memory.update_fields(extracted)
                else:
                    # If LLM says it's insufficient (typos/nonsense), we mark it as failed turn
                    # This will trigger the failed_field re-prompt below
                    pass

            # Deterministic fallback extraction for short/natural replies that LLM may miss
            fallback_updates: Dict[str, Any] = {}
            currently_missing = memory.get_missing_fields()

            if "severity" in currently_missing:
                severity_value = _extract_severity_value(user_input or "")
                if severity_value is not None:
                    fallback_updates["severity"] = severity_value

            if "associated_symptoms" in currently_missing:
                associated = _extract_associated_symptoms(user_input or "")
                if associated is not None:
                    fallback_updates["associated_symptoms"] = associated

            if fallback_updates:
                memory.update_fields(fallback_updates)
                
            # Deterministic fallback for "No" or "None" to prevent repeated questions
            negatives = ["no", "nope", "nah", "none", "nothing", "no more", "don't have any", "that's it"]
            is_pure_negative = any(word == user_lower.strip().strip('.,!?') or user_lower.strip().startswith(word + " ") for word in negatives) and len(user_lower.split()) <= 4
            
            if is_pure_negative and memory.asked_fields:
                last_asked = memory.asked_fields[-1]
                if last_asked in memory.get_missing_fields():
                    if last_asked == "associated_symptoms":
                        memory.update_fields({"associated_symptoms": ["none reported"]})
                    else:
                        memory.update_fields({last_asked: "None reported"})

            from config import EMERGENCY_CONTEXTS, EMERGENCY_FIELDS
            context_name = llm._detect_symptom_context(memory.conversation_history)
            is_emergency = context_name in EMERGENCY_CONTEXTS

            if is_emergency:
                missing_emergency_fields = [f for f in EMERGENCY_FIELDS if f in memory.get_missing_fields()]
                
                if not missing_emergency_fields:
                    # We have gathered the 3 emergency context fields. Now evaluate TRUE risk.
                    symptom_data = memory.get_symptom_data()
                    temp_risk = risk.categorize_risk(symptom_data)
                    
                    if temp_risk['risk_level'] in ["CRITICAL", "HIGH"]:
                        # It is truly severe. Finalize partially.
                        missing_fields = [] # Force completion
                        _finalize_intake(symptom_data, partial=True)
                    else:
                        # It's an emergency keyword context but severity/duration is mild.
                        # Drop the emergency flag and fall back to asking the remaining normal fields.
                        is_emergency = False
                        missing_fields = memory.get_missing_fields()
                else:
                    # Still gathering emergency fields
                    missing_fields = missing_emergency_fields
                    
            if not is_emergency:
                missing_fields = memory.get_missing_fields()

            # Check if intake is complete for the current context
            if not missing_fields and not response_text:
                symptom_data = memory.get_symptom_data()
                _finalize_intake(symptom_data, partial=(is_emergency or not memory.is_intake_complete()))
            
            else:
                # --- SMART VALIDATION ---
                # Determine if we just asked a specific field and if it failed to extract
                failed_field = None
                last_input_insufficient = False
                
                if extracted and not extracted.get("is_sufficient", True):
                    last_input_insufficient = True

                if memory.asked_fields and current_state == STATES["CLARIFYING"]:
                    last_asked = memory.asked_fields[-1]
                    if last_asked in missing_fields:
                        # User was asked but the field is still missing after extraction
                        failed_field = last_asked

                if missing_fields and not response_text:
                    # Ask clarification questions for the remaining fields
                    # CRITICAL: If chief_complaint is missing, STAY on it.
                    if "chief_complaint" in missing_fields:
                        # Only mark chief_complaint as asked if it wasn't already.
                        # But we'll keep asking it until we get it.
                        if "chief_complaint" not in memory.asked_fields:
                            memory.mark_field_asked("chief_complaint")
                        
                        response_text = llm.generate_clarification_question(
                            ["chief_complaint"],
                            memory.conversation_history,
                            session_id,
                            failed_field=("chief_complaint" if failed_field == "chief_complaint" else None),
                            last_user_input=(user_input if failed_field == "chief_complaint" else None),
                            was_insufficient=last_input_insufficient
                        )
                        update_session_state(session_id, STATES["CLARIFYING"])
                    
                    else:
                        # Chief complaint is present, look for other UNASKED fields
                        unasked_missing = [f for f in missing_fields if f not in memory.asked_fields]

                        if unasked_missing:
                            # Pick ONLY the highest priority unasked field to mark as asked
                            # Sort by priority and pick the first
                            next_field = sorted(unasked_missing, key=lambda x: FIELD_PRIORITY.get(x, 99))[0]
                            memory.mark_field_asked(next_field)
                            
                            # Generate clarification question for only the next targeted field
                            response_text = llm.generate_clarification_question(
                                [next_field],
                                memory.conversation_history,
                                session_id,
                                failed_field=failed_field,
                                last_user_input=user_input,
                                was_insufficient=last_input_insufficient
                            )
                            update_session_state(session_id, STATES["CLARIFYING"])
                        else:
                            # All missing fields were asked before, but still not filled.
                            # We might be stuck in a loop where the user isn't giving clear data.
                            # Break out of the loop and finalize with what we have.
                            symptom_data = memory.get_symptom_data()
                            _finalize_intake(symptom_data, partial=True)
    
    elif current_state == STATES["OFFERING_CONSULT"]:
        user_lower = (user_input or "").lower().strip().strip('.,!?')
        # Check if the user is declining questions
        negatives = ["no", "nope", "nah", "none", "nothing", "that's it", "thats it", "all good", "i'm good", "im good", "show the report"]
        
        # If user exactly says they have no doubts, we move to COMPLETE
        if any(word == user_lower or user_lower.startswith(word + " ") or user_lower.endswith(" " + word) for word in negatives) and len(user_lower.split()) < 6:
            update_session_state(session_id, STATES["COMPLETE"])
            response_text = llm.generate_warm_signoff(session_id)
            show_report_popup = True
        else:
            # User likely has a question or said "yes"
            trigger_consult_transition = True
            update_session_state(session_id, STATES["COMPLETE"])
            response_text = "Switching you to the Health Consult tab..."

    elif current_state == STATES["COMPLETE"]:
        # Session already complete
        user_lower = (user_input or "").lower().strip().strip('.,!?')
        affirmatives = ["yes", "yep", "y", "ok", "okay", "sure", "show", "report", "please"]
        if any(word in user_lower for word in affirmatives):
            symptom_data = memory.get_symptom_data()
            summary = symptom_data.get("summary", "Summary not available.")
            response_text = f"Here is your summary report:\n\n{summary}\n\nYou can view the full report using the 'Clinical Report' button."
        else:
            response_text = "Your symptom check is complete. You can view your report or switch to the Health Consult tab."
    
    else:
        # Unknown state, reset to greeting
        response_text = "Sorry, something went wrong. Let's start again. What problem are you having today?"
        update_session_state(session_id, STATES["COLLECTING"])
    
    # Save assistant response
    if response_text:
        save_turn(session_id, "assistant", response_text)
    
    # Get updated state
    final_state = get_session_state(session_id)
    is_complete = (final_state == STATES["COMPLETE"])

    # If in COMPLETE but we don't have risk/wellness in memory (subsequent turn), load them
    if (is_complete) and (risk_assessment is None or wellness_tip is None):
        try:
            symptom_data = memory.get_symptom_data()
            risk_assessment = risk.categorize_risk(symptom_data)
            # For wellness tips, we can re-generate if LOW/MODERATE
            if risk_assessment['risk_level'] in ["LOW", "MODERATE"]:
                wellness_tip = llm.generate_health_advice(symptom_data, session_id)
            else:
                wellness_tip = f"Please prioritize seeking medical evaluation as advised. {risk_assessment['recommended_action']}"
        except Exception as e:
            # Fallback for completion phase load errors
            if risk_assessment is None:
                risk_assessment = {"risk_level": "LOW", "reason": "Data load error", "recommended_action": ""}
            if wellness_tip is None:
                wellness_tip = "Please consult your records or start a new session."
    
    return {
        "response_text": response_text,
        "state": final_state,
        "is_complete": is_complete,
        "risk_assessment": risk_assessment,
        "wellness_tip": wellness_tip,
        "trigger_consult_transition": trigger_consult_transition,
        "show_report_popup": show_report_popup
    }
