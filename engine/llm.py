
import time
import json
import requests
from requests.adapters import HTTPAdapter
from typing import List, Dict, Any, Optional
from config import (
    NEXUS_BASE_URL, 
    NEXUS_API_KEY, 
    NEXUS_CHAT_COMPLETIONS_PATH,
    LLM_MODEL, 
    LLM_TEMPERATURE, 
    LLM_MAX_RETRIES,
    LLM_TIMEOUT_SECONDS,
    LLM_SAFETY_INSTRUCTIONS,
    FIELD_PRIORITY,
    CONTEXT_FIELD_KEYWORDS,
    GENERIC_FIELD_KEYWORDS
)
from database import log_latency


HTTP_SESSION = requests.Session()
HTTP_ADAPTER = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=0)
HTTP_SESSION.mount("http://", HTTP_ADAPTER)
HTTP_SESSION.mount("https://", HTTP_ADAPTER)




SYMPTOM_CONTEXT_KEYWORDS = {
    "injury": ["cut", "wound", "laceration", "scratch", "bleeding", "burn", "sprain", "twisted", "bruise", "injury", "fall"],
    "respiratory": ["cough", "breathing", "shortness of breath", "wheeze", "chest tightness", "sore throat", "congestion", "phlegm", "runny nose"],
    "gastro": ["stomach", "abdominal", "nausea", "vomit", "vomiting", "diarrhea", "constipation", "bloating", "stool", "indigestion", "heartburn"],
    "neuro": ["headache", "migraine", "dizzy", "dizziness", "vertigo", "numb", "tingling", "weakness", "vision", "slurred", "faint", "confusion"],
    "skin": ["rash", "itch", "itching", "hives", "redness", "swelling", "blister", "skin", "pimple", "lesion"],
    "fever": ["fever", "chills", "body ache", "fatigue", "tired", "temperature", "sweats"],
    "cardiac": ["chest pain", "palpitations", "heart racing", "heart", "pressure in chest"],
    "musculoskeletal": ["joint pain", "back pain", "neck pain", "muscle pain", "stiffness", "cramp", "shoulder pain", "knee pain"],
    "urinary": ["urine", "urinary", "burning pee", "pain while peeing", "frequent urination", "blood in urine", "flank pain"],
    "eye": ["eye pain", "red eye", "blurred vision", "double vision", "eye discharge", "light sensitivity"],
    "ear_nose_throat": ["ear pain", "ear discharge", "hearing", "sinus", "nasal", "throat pain", "hoarseness"],
    "dental": ["tooth", "toothache", "gum", "jaw pain", "mouth ulcer", "oral swelling"],
    "mental_health": ["anxiety", "panic", "depressed", "stress", "insomnia", "sleep", "low mood"],
    "reproductive": ["period pain", "menstrual", "vaginal", "pelvic pain", "pregnant", "pregnancy", "testicular", "discharge"],
}


def _detect_symptom_context(conversation_history: List[Dict[str, str]]) -> str:
    """Detect likely symptom context from conversation text."""
    if not conversation_history:
        return "general"

    combined_text = " ".join(
        (turn.get("content") or "")
        for turn in conversation_history
        if isinstance(turn, dict)
    ).lower()

    best_context = "general"
    best_score = 0

    for context_name, keywords in SYMPTOM_CONTEXT_KEYWORDS.items():
        score = sum(1 for word in keywords if word in combined_text)
        if score > best_score:
            best_score = score
            best_context = context_name

    return best_context




from groq import Groq
from config import GROQ_API_KEY

_groq_client = None

def get_groq_client():
    global _groq_client
    if _groq_client is None:
        if GROQ_API_KEY:
            _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def call_nexus_llm(messages: List[Dict[str, str]], session_id: str, operation_type: str = "llm") -> Optional[str]:
    """
    Call Groq API (kept function name call_nexus_llm to avoid rewriting caller logic)
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        session_id: Session ID for latency logging
        operation_type: Type of operation for logging
    
    Returns:
        Response text from LLM or None on error
    """
    start_time = time.time()
    
    try:
        client = get_groq_client()
        if not client:
            print("LLM configuration missing: set GROQ_API_KEY in .env.")
            latency_ms = (time.time() - start_time) * 1000
            log_latency(session_id, f"{operation_type}_error", latency_ms)
            return None
            
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=LLM_MODEL,
            temperature=float(LLM_TEMPERATURE),
            timeout=float(LLM_TIMEOUT_SECONDS)
        )
        
        latency_ms = (time.time() - start_time) * 1000
        log_latency(session_id, operation_type, latency_ms)
        
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        print(f"LLM Error (Groq): {e}")
        latency_ms = (time.time() - start_time) * 1000
        log_latency(session_id, f"{operation_type}_error", latency_ms)
        return None


def extract_symptoms(conversation_history: List[Dict[str, str]], user_input: str, session_id: str) -> Dict[str, Any]:
    """
    Extract structured symptom attributes from conversation
    
    Args:
        conversation_history: Previous conversation turns
        user_input: Latest user input
        session_id: Session ID for latency logging
    
    Returns:
        Dictionary with extracted symptom fields
    """
    system_prompt = f"""You are a medical intake assistant. Extract ONLY structured symptom information.

{LLM_SAFETY_INSTRUCTIONS}

The user speaks in everyday language. Understand natural, simple speech and map it to the required fields.
Guidelines:
- **IGNORE NONSENSE**: If the user provides an answer that is logically impossible, medically irrelevant, or nonsense (e.g., "pain comes from the sky", "I am a robot", "it started 100 years ago"), do NOT extract it into any field. Leave the field as null.
- **NO EQUIPMENT**: Do not extract measurements that clearly require medical equipment (e.g., specific blood pressure numbers) unless the user specifically mentions an at-home device.
- **NATURAL MAPPING**:
  - "it started this morning" -> duration
  - "it's really bad, like 8" -> severity = 8
  - "it's getting worse" -> progression = "worsening"
  - "it came out of nowhere" -> onset_type = "sudden"

From the conversation, extract up to these 9 attributes (only extract what is relevant and logical; leave irrelevant or nonsensical fields as null):
1. chief_complaint (main health concern, string)
2. duration (how long symptoms present, string like "3 days" or "2 weeks")
3. severity (pain/discomfort level 1-10, integer)
4. progression (one of: "improving", "worsening", "stable")
5. associated_symptoms (list of additional symptoms, array of strings)
6. affected_body_part (body location, string)
7. onset_type (one of: "sudden", "gradual")
8. aggravating_alleviating_factors (what makes symptoms better or worse, string)
9. relevant_medical_history (past medical conditions, current medications, or previous instances, string)

CRITICAL: Add a 10th field `is_sufficient` (boolean). Set it to false if the user's latest input is nonsense, too vague, contains garbled text, or contains typos that make it medically weird (e.g., "paying" instead of "pain"). If the user's answer is logically disconnected from the question, set `is_sufficient` to false. If `is_sufficient` is false, set the relevant extracted field to null.

Return ONLY valid JSON with these 10 fields.
Do NOT include markdown code fences.
Do NOT add explanations.
"""
    
    # Build message history
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_input})
    
    no_symptom_phrases = [
        "no other symptoms",
        "no more symptoms",
        "no additional symptoms",
        "no symptoms",
        "nothing else",
        "that's it",
        "none",
        "no",
        "no i",
        "i don't have any",
        "dont have any",
    ]

    def _last_assistant_asked_associated_symptoms(history: List[Dict[str, str]]) -> bool:
        for turn in reversed(history):
            if isinstance(turn, dict) and (turn.get("role") == "assistant"):
                content = (turn.get("content") or "").lower()
                hints = [
                    "other symptoms",
                    "associated symptoms",
                    "any other symptoms",
                    "also have",
                ]
                return any(hint in content for hint in hints)
        return False

    def _is_negative_response(text: str) -> bool:
        cleaned = (text or "").strip().lower()
        if not cleaned:
            return False
        negatives = [
            "no",
            "none",
            "nope",
            "nah",
            "nothing",
            "none at all",
            "i don't have any",
            "i dont have any",
            "no i don't have",
            "no i dont have",
        ]
        return any(phrase in cleaned for phrase in negatives)

    # Try extraction with retries
    for attempt in range(LLM_MAX_RETRIES + 1):
        response_text = call_nexus_llm(messages, session_id, "extract_symptoms")
        
        if not response_text:
            continue
        
        # Strip markdown code fences if present
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        try:
            extracted = json.loads(cleaned)

            # Handle simple negative response for associated symptoms
            user_lower = (user_input or "").lower()
            if (
                any(phrase in user_lower for phrase in no_symptom_phrases)
                and (
                    extracted.get("associated_symptoms") is None
                    or extracted.get("associated_symptoms") == []
                )
            ):
                extracted["associated_symptoms"] = ["none reported"]

            # If user gave a short negative reply right after associated symptoms question,
            # explicitly mark associated_symptoms as completed to avoid repeated asking.
            if (
                _last_assistant_asked_associated_symptoms(conversation_history)
                and _is_negative_response(user_input)
                and (
                    extracted.get("associated_symptoms") is None
                    or extracted.get("associated_symptoms") == []
                )
            ):
                extracted["associated_symptoms"] = ["none reported"]

            return extracted
        except json.JSONDecodeError as e:
            print(f"JSON parse error (attempt {attempt + 1}): {e}")
            if attempt < LLM_MAX_RETRIES:
                continue
    
    # Return empty dict on all failures
    return {}


def generate_clarification_question(missing_fields: List[str], conversation_history: List[Dict[str, str]], session_id: str, failed_field: Optional[str] = None, last_user_input: Optional[str] = None, was_insufficient: bool = False) -> str:
    """
    Generate a dynamic clarification question for missing fields based on user context.
    
    Args:
        missing_fields: List of fields still needed
        conversation_history: Previous conversation turns
        session_id: Session ID for latency logging
        failed_field: If a specific field was just asked but not successfully extracted
        last_user_input: The user's previous (insufficient) answer
        was_insufficient: Whether the previous answer was considered nonsense or medically unclear
    
    Returns:
        Clarification question string
    """
    if not missing_fields:
        return "Thank you for providing all the information."
    
    # Identify the conversation context to grab the right keyword hints
    context_name = _detect_symptom_context(conversation_history)
    
    # Collect keyword hints for the LLM
    hints = []
    for field in missing_fields:
        field_hints = CONTEXT_FIELD_KEYWORDS.get(context_name, {}).get(field)
        if not field_hints:
            field_hints = GENERIC_FIELD_KEYWORDS.get(field, [field.replace("_", " ")])
        hints.append(f"- {field}: {', '.join(field_hints)}")
    
    hints_text = "\n".join(hints)
    
    recent_history = conversation_history[-8:] if conversation_history else []

    system_prompt = f"""You create a natural follow-up question (or questions) for symptom intake.

{LLM_SAFETY_INSTRUCTIONS}

We need to gather the following missing information:
{hints_text}

Validation Context:
{f"FAILED FIELD: {failed_field}" if failed_field else ""}
{f"USER'S PREVIOUS ANSWER: '{last_user_input}'" if last_user_input else ""}

Rules:
- EMPATHY AND WARMTH: Adopt a highly empathetic, conversational, and caring tone. Do not sound like a robotic medical form. Use phrases like "I'm so sorry you're dealing with that", "That sounds uncomfortable", or "To help me understand better...".
- CONVERSATIONAL FLOW: If the user is just saying "hi", "hello", or asking social questions (e.g., "how are you?", "what is your name?"), respond naturally and friendly first, then gently ask how you can help with their health.
- **HISTORY & RECORDS**: If the user asks to "analyze my history", "check my records", or "you have my data", explain politely that you are currently conducting a **fresh symptom intake session** to capture their current state. Gently steer them back to answering the specific missing questions.
- **NONSENSE PROTECTION**: If `was_insufficient` is true, or the user's previous response was nonsense or medically impossible (like "pain comes from sky"), you MUST start your response with a polite apology, acknowledging you didn't quite catch that. For example: "I'm sorry, I didn't quite catch that...", "I'm a bit confused by that last part...", or "I missed that, could you tell me again?". Then re-ask the question simply.
- Ask a natural, conversational question to gather these missing details.
- MEDICAL JUDGMENT: Only ask for fields that are medically relevant to the user's specific problem. 
- ACUTE VS CHRONIC: 
  * If the symptom is an ACUTE event (e.g., bleeding, cut, injury, sudden fall, "just happened"), avoid generic phrases like "how long have you been experiencing these symptoms". 
  * Instead, use natural event-based phrasing like "When did this happen?", "Is it still bleeding?", or "How long ago was the injury?".
- ADAPT QUESTION STYLE TO THE SEVERITY/CONTEXT:
  * For emergency/critical issues (e.g., cardiac, neural, left chest pain, important organs): ask direct, critical questions focusing on the most important details instantly, but remain calm and reassuring.
  * For mild issues (e.g., headache, mild fever, a little scratch, minor cuts): ask more conversational and contextual questions.
- AVOID asking for measurements that require medical equipment (e.g., blood pressure, heart rate monitor, pulse oximetry, thermometer). Ask about symptoms the patient can feel or observe instead.
- **SMART RE-PROMPTING**: If `FAILED FIELD` is provided, it means the user's last answer ('{last_user_input}') was not enough to fill that field. 
  * Re-prompt the user with context. Say something like "I'm sorry, I didn't quite get [field] from that..." or "You mentioned [context], but could you clarify [field] differently?".
  * Be helpful and guide them on what kind of answer we need (e.g., if severity failed, ask for a number 1-10).
- Do NOT ask about anything else not listed above.
- Do NOT diagnose, prescribe, or give treatment advice.
- Keep it concise but friendly.

Return ONLY the question text.
"""

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(recent_history)

    response_text = call_nexus_llm(messages, session_id, "clarification_dynamic")
    if response_text and response_text.strip():
        return response_text.strip()

    # Improved fallback if LLM fails
    if missing_fields:
        field_to_ask = missing_fields[0].replace("_", " ")
        return f"To help me understand, could you tell me more about your {field_to_ask}?"
    
    return "Could you provide a few more details about your symptoms?"


def generate_summary(symptom_record: Dict[str, Any], session_id: str) -> str:
    """
    Generate a clinical-style summary from structured symptom data
    
    Args:
        symptom_record: Complete symptom record dictionary
        session_id: Session ID for latency logging
    
    Returns:
        Clinical summary string (3-5 sentences)
    """
    system_prompt = f"""You are a senior medical documentation specialist.

{LLM_SAFETY_INSTRUCTIONS}

Create a COMPREHENSIVE and PROFESSIONAL clinical handoff note from the provided symptom data.

Required Sections:
1. **Clinical Narrative (HPI)**: A detailed chronological account of the patient's symptoms, including onset, character, location, radiation, and any patterns.
2. **Symptom Profile**: Clear breakdown of severity (1-10), duration, and progression.
3. **Clinical Context**: Include aggravating/alleviating factors and relevant medical history.
4. **Lifestyle & Wellness Insight**: Briefly mention how these symptoms might relate to the patient's reported activity or wellness context if applicable.

Guidelines:
- Use formal medical terminology (e.g., 'acute', 'intermittent', 'localized').
- Keep the tone objective, clinical, and precise.
- Do NOT diagnose and do NOT give treatment advice.
- DO NOT INCLUDE a field in the summary if the value is missing or not reported.
- Ensure the output is highly professional and ready for a clinician's review.

Return ONLY the summary text, no formatting symbols like `###` or explanations.
"""
    
    # Format symptom record as readable text, omitting missing values
    associated = symptom_record.get("associated_symptoms") or []
    if isinstance(associated, str):
        try:
            associated = json.loads(associated)
        except:
            associated = []
    
    symptom_text_lines = []
    if symptom_record.get('chief_complaint'):
        symptom_text_lines.append(f"Chief Complaint: {symptom_record['chief_complaint']}")
    if symptom_record.get('duration'):
        symptom_text_lines.append(f"Duration: {symptom_record['duration']}")
    if symptom_record.get('severity') is not None:
        symptom_text_lines.append(f"Severity: {symptom_record['severity']}/10")
    if symptom_record.get('progression'):
        symptom_text_lines.append(f"Progression: {symptom_record['progression']}")
    if symptom_record.get('affected_body_part'):
        symptom_text_lines.append(f"Affected Body Part: {symptom_record['affected_body_part']}")
    if symptom_record.get('onset_type'):
        symptom_text_lines.append(f"Onset Type: {symptom_record['onset_type']}")
    if symptom_record.get('aggravating_alleviating_factors'):
        symptom_text_lines.append(f"Aggravating/Alleviating Factors: {symptom_record['aggravating_alleviating_factors']}")
    if symptom_record.get('relevant_medical_history'):
        symptom_text_lines.append(f"Medical History: {symptom_record['relevant_medical_history']}")
    if associated and associated != ["none reported"]:
        symptom_text_lines.append(f"Associated Symptoms: {', '.join(associated)}")
    
    symptom_text = "\n".join(symptom_text_lines)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Create the doctor handoff note from this data:\n{symptom_text}"}
    ]
    
    response_text = call_nexus_llm(messages, session_id, "summary")

    if response_text and response_text.strip():
        return response_text.strip()

    # Deterministic fallback so clinicians always get a usable summary
    fallback_lines = []
    if symptom_record.get('chief_complaint'):
        fallback_lines.append(f"Chief Complaint: {symptom_record['chief_complaint']}")
    if symptom_record.get('duration'):
        fallback_lines.append(f"Duration: {symptom_record['duration']}")
    if symptom_record.get('severity') is not None:
        fallback_lines.append(f"Severity (1-10): {symptom_record['severity']}")
    if symptom_record.get('progression'):
        fallback_lines.append(f"Progression: {symptom_record['progression']}")
    if symptom_record.get('affected_body_part'):
        fallback_lines.append(f"Affected Body Part: {symptom_record['affected_body_part']}")
    if symptom_record.get('onset_type'):
        fallback_lines.append(f"Onset Type: {symptom_record['onset_type']}")
    if symptom_record.get('aggravating_alleviating_factors'):
        fallback_lines.append(f"Aggravating/Alleviating Factors: {symptom_record['aggravating_alleviating_factors']}")
    if symptom_record.get('relevant_medical_history'):
        fallback_lines.append(f"Medical History: {symptom_record['relevant_medical_history']}")
    if associated and associated != ["none reported"]:
        fallback_lines.append(f"Associated Symptoms: {', '.join(associated)}")

    fallback_lines.append("Clinical Note: Symptom details collected from patient interview.")

    return "\n".join(fallback_lines)


def generate_health_advice(symptom_record: Dict[str, Any], session_id: str) -> str:
    """
    Generate general health advice for mild/low risk symptoms (e.g., drink hot water for mild fever)
    
    Args:
        symptom_record: Complete symptom record dictionary
        session_id: Session ID for latency logging
    
    Returns:
        A short string with safe, general health advice.
    """
    system_prompt = f"""You are a caring and deeply empathetic health assistant.
    
{LLM_SAFETY_INSTRUCTIONS}

The user is experiencing a mild health concern. Your goal is to provide them with comfort and safe, gentle wellness advice to help them feel supported.

Guidelines:
- Use a very warm, supportive, and caring tone.
- Keep it to 1-2 short, friendly sentences.
- Do NOT diagnose.
- STRICT PROHIBITION: Do NOT mention any medication names, brand names, or chemical names.
- ONLY suggest safe, natural comfort measures like rest, hydration, or warm compresses.
- Frame it as gentle care support rather than a medical prescription.

Return ONLY the advice text.
"""
    
    symptom_text = f"Chief Complaint: {symptom_record.get('chief_complaint', 'mild issue')}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Please provide safe wellness tips for this symptom:\n{symptom_text}"}
    ]
    
    response_text = call_nexus_llm(messages, session_id, "health_advice")

    if response_text and response_text.strip():
        return response_text.strip()
    
    # Generic fallback
    return "Make sure to get plenty of rest and stay hydrated."

def generate_warm_signoff(session_id: str) -> str:
    """Generate a warm, caring closing message when the user finishes the intake."""
    system_prompt = f"""You are a caring and empathetic medical assistant.
    
{LLM_SAFETY_INSTRUCTIONS}

The user has just completed their symptom intake and confirmed they have no further doubts or questions. 
Write a very brief (1-2 sentences), warm, and caring sign-off message thanking them for providing the information and wishing them well. Let them know their report is ready. 

Return ONLY the message text.
"""
    messages = [{"role": "system", "content": system_prompt}]
    
    response_text = call_nexus_llm(messages, session_id, "warm_signoff")
    if response_text and response_text.strip():
        return response_text.strip()
        
    return "Thank you for sharing that with me. Please take care of yourself. I've prepared your clinical report."
def respond_to_consult(user_question: str, session_id: str, symptom_record: Optional[Dict[str, Any]] = None, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Answer user's general health, care, and fitness questions safely and professionally.
    Works independently or with previous session context.
    """
    history = conversation_history or []
    
    context_str = ""
    if symptom_record:
        context_str = f"\nPrevious Session Context (For Reference Only):\nThe user recently completed a symptom check with these details:\n{json.dumps(symptom_record, indent=2)}\nUse this context if relevant, but prioritize answering their current question."

    system_prompt = f"""You are a supportive, versatile, and professional Health Consult Assistant. You provide caring medical information with empathy and kindness.

{LLM_SAFETY_INSTRUCTIONS}

Your role is to provide educational medical information based on reported symptoms while making the user feel heard and cared for. You must not provide definitive diagnoses or treatment instructions.

Capabilities:
- General Health Info: Explain symptoms, body functions, and common mechanisms.
- Care & Comfort: Share general, caring self-care guidance (e.g., rest, hydration, ice/heat) without prescribing medication.
- Fitness & Wellness: Provide general fitness, nutrition, and wellness education.
- Symptom Education: Describe possible causes in non-definitive, supportive language.
- Clinical Clarification: Explain common medical terms in plain, easy-to-understand language.

Guidelines:
- TONE: Be empathetic, supportive, and professional. Use caring language (e.g., "I'm sorry you're feeling this way," "I hope this information helps you feel a bit more at ease").
- STRICT SCOPE: You are exclusively a health, wellness, and care assistant for both humans and animals (including pets, birds, etc.). You MUST NOT answer questions about politics, current events, general history, programming, or any topic not directly related to human or animal health.
- OUT-OF-SCOPE FALLBACK: If a user asks an out-of-scope question, politely decline and steer them back to health topics in a kind way.
- Use simple, clear medical language.
- Keep answers concise and easy to read.
- Prefer short bullets over long paragraphs.
- STRICT PROHIBITION: Never mention any medication names, brands, or chemical drugs.
- If symptoms suggest a condition, describe it as a possibility (e.g., "This can sometimes be associated with...").
- Never present a condition as confirmed.
- If red-flag symptoms are present (e.g., chest pain, stroke signs, severe breathing issues), immediately and firmly but kindly advise emergency care.
{context_str}
- Output plain text only. Do not use Markdown formatting (no **bold**, no headings, no code blocks).

Answer the user directly with care and professionalism.
"""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add relevant history for context (last few turns)
    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    
    # Add current question
    messages.append({"role": "user", "content": user_question})
    
    response = call_nexus_llm(messages, session_id, "consult")
    if not response:
        return "I'm sorry, I couldn't process your question at the moment. Please consult a doctor for specific medical advice."

    cleaned_response = response.replace("**", "").strip()
    return cleaned_response
