"""
Configuration module for ClinAssist
Loads environment variables and defines system constants
"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv()

# Nexus API Configuration
NEXUS_API_KEY = os.getenv("NEXUS_API_KEY") or os.getenv("API_KEY")
NEXUS_BASE_URL = (os.getenv("NEXUS_BASE_URL") or os.getenv("BASE_URL", "")).rstrip("/")

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Endpoint paths (override in .env if Nexus uses non-standard routes)
NEXUS_CHAT_COMPLETIONS_PATH = os.getenv("NEXUS_CHAT_COMPLETIONS_PATH", "/v1/chat/completions")
NEXUS_TTS_PATH = os.getenv("NEXUS_TTS_PATH", "/v1/audio/speech")
NEXUS_STT_PATH = os.getenv("NEXUS_STT_PATH", "/v1/audio/transcriptions")


def _normalize_path(path: str) -> str:
    return path if path.startswith("/") else f"/{path}"


NEXUS_CHAT_COMPLETIONS_PATH = _normalize_path(NEXUS_CHAT_COMPLETIONS_PATH)
NEXUS_TTS_PATH = _normalize_path(NEXUS_TTS_PATH)
NEXUS_STT_PATH = _normalize_path(NEXUS_STT_PATH)

if not NEXUS_API_KEY or not NEXUS_BASE_URL:
    print("Warning: STT/LLM/TTS credentials are missing. Set NEXUS_API_KEY + NEXUS_BASE_URL (or API_KEY + BASE_URL) in .env.")

# API Headers for Nexus
NEXUS_HEADERS = {
    "Authorization": f"Bearer {NEXUS_API_KEY}",
    "Content-Type": "application/json"
}

# Model Configuration
STT_MODEL = os.getenv("STT_MODEL", "whisper-1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
TTS_MODEL = "piper"
# Models now stored in assets/
TTS_VOICE_PATH = BASE_DIR / "assets" / (os.getenv("TTS_VOICE", "en_US-lessac-medium") + ".onnx")
TTS_VOICE_CONFIG_PATH = BASE_DIR / "assets" / (os.getenv("TTS_VOICE", "en_US-lessac-medium") + ".onnx.json")
TTS_VOICE = os.getenv("TTS_VOICE", "en_US-lessac-medium")

# STT request behavior
# auto: try multipart then json fallback
# multipart: only multipart (fastest when gateway is OpenAI-compatible)
# json: only json/base64 (fastest when gateway expects JSON)
STT_REQUEST_MODE = os.getenv("STT_REQUEST_MODE", "auto").strip().lower()
if STT_REQUEST_MODE not in {"auto", "multipart", "json"}:
    STT_REQUEST_MODE = "auto"

STT_TIMEOUT_SECONDS = int(os.getenv("STT_TIMEOUT_SECONDS", "30"))

# LLM and TTS timeout tuning
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "25"))
TTS_TIMEOUT_SECONDS = int(os.getenv("TTS_TIMEOUT_SECONDS", "20"))

# Latency optimization: for text endpoint, skip TTS by default (does not affect extraction/risk accuracy)
ENABLE_TTS_FOR_TEXT = os.getenv("ENABLE_TTS_FOR_TEXT", "0").strip().lower() in {"1", "true", "yes", "on"}

# LLM Parameters
LLM_TEMPERATURE = 0.2
LLM_MAX_RETRIES = 2

# Required Symptom Fields (9 attributes)
REQUIRED_FIELDS = [
    "chief_complaint",
    "duration",
    "severity",
    "progression",
    "associated_symptoms",
    "affected_body_part",
    "onset_type",
    "aggravating_alleviating_factors",
    "relevant_medical_history"
]

# Emergency Configuration
EMERGENCY_CONTEXTS = ["cardiac", "neuro", "fever", "injury"]
EMERGENCY_FIELDS = ["onset_type", "severity", "duration"]

# Keywords for LLM Prompt Injection per Context
CONTEXT_FIELD_KEYWORDS = {
    "injury": {
        "duration": ["when did this happen", "how long has it been bleeding", "time of injury"],
        "severity": ["pain scale", "1 to 10", "how bad"],
        "progression": ["getting worse", "bleeding more", "swelling more"],
        "associated_symptoms": ["numbness", "tingling", "cannot move it", "other issues"],
        "affected_body_part": ["exact location", "which part"],
        "onset_type": ["happened suddenly", "accident"],
    },
    "respiratory": {
        "duration": ["when did the breathing issue start", "how many days"],
        "severity": ["breathing difficulty scale", "1 to 10"],
        "progression": ["breathing getting harder", "worse over time"],
        "associated_symptoms": ["coughing", "fever", "chest tightness", "wheezing"],
        "affected_body_part": ["throat", "chest", "lungs"],
        "onset_type": ["started suddenly", "came on gradually"],
    },
    "gastro": {
        "duration": ["when did the stomach pain start"],
        "severity": ["pain scale", "1-10"],
        "progression": ["pain getting sharper", "more frequent vomiting"],
        "associated_symptoms": ["nausea", "diarrhea", "fever", "blood in stool"],
        "affected_body_part": ["upper stomach", "lower belly", "left or right side"],
        "onset_type": ["after eating", "sudden pain", "gradual ache"],
    },
    "neuro": {
        "duration": ["when did the headache or numbness start"],
        "severity": ["intensity", "1-10"],
        "progression": ["spreading", "getting worse", "more frequent"],
        "associated_symptoms": ["vision changes", "dizziness", "confusion", "balance loss"],
        "affected_body_part": ["head", "face", "limbs", "one side of body"],
        "onset_type": ["thunderclap", "sudden", "gradual"],
    },
    "skin": {
        "duration": ["when did the rash appear"],
        "severity": ["itchiness or pain level", "1-10", "how irritated"],
        "progression": ["spreading", "getting redder", "changing color"],
        "associated_symptoms": ["fever", "swelling", "blisters", "oozing"],
        "affected_body_part": ["where on the body", "legs, arms, back"],
        "onset_type": ["sudden breakout", "slowly formed"],
    },
    "fever": {
        "duration": ["how long have you had the fever"],
        "severity": ["how high is the temperature", "how bad do you feel", "1-10"],
        "progression": ["fever spiking", "going up and down"],
        "associated_symptoms": ["chills", "sweats", "body aches", "cough"],
        "affected_body_part": ["full body", "headache"],
        "onset_type": ["sudden spike", "gradual climb"],
    },
    "cardiac": {
        "duration": ["when did the chest pain start"],
        "severity": ["chest pain scale", "1-10", "crushing pressure"],
        "progression": ["pain increasing", "spreading to arm or jaw"],
        "associated_symptoms": ["sweating", "shortness of breath", "nausea", "dizzy"],
        "affected_body_part": ["center of chest", "left arm", "jaw", "back"],
        "onset_type": ["sudden onset", "woke up with it"],
    },
    "musculoskeletal": {
        "duration": ["when did the joint or muscle pain start"],
        "severity": ["pain scale", "1-10"],
        "progression": ["getting stiffer", "hurts more to move"],
        "associated_symptoms": ["swelling", "locking", "clicking", "muscle spasms"],
        "affected_body_part": ["knee", "back", "shoulder", "exact joint"],
        "onset_type": ["after a specific movement", "sudden twinge", "gradual wear"],
    },
    "urinary": {
        "duration": ["when did the urination issue start"],
        "severity": ["pain level while urinating", "1-10"],
        "progression": ["going more often", "getting more painful"],
        "associated_symptoms": ["fever", "blood in urine", "lower back pain", "foul smell"],
        "affected_body_part": ["lower belly", "flank", "groin"],
        "onset_type": ["sudden urge", "gradual worsening"],
    },
    "eye": {
        "duration": ["when did the eye issue start"],
        "severity": ["pain or irritation out of 10"],
        "progression": ["vision getting blurrier", "redness spreading"],
        "associated_symptoms": ["discharge", "tearing", "light sensitivity"],
        "affected_body_part": ["left eye", "right eye", "both eyes", "eyelid"],
        "onset_type": ["woke up like this", "sudden trauma"],
    },
    "ear_nose_throat": {
        "duration": ["when did the ear pain or sore throat start"],
        "severity": ["pain out of 10", "how hard to swallow"],
        "progression": ["getting more clogged", "hearing decreasing"],
        "associated_symptoms": ["fever", "runny nose", "ear discharge", "ringing"],
        "affected_body_part": ["left or right ear", "sinuses", "throat"],
        "onset_type": ["sudden pop", "gradual congestion"],
    },
    "dental": {
        "duration": ["when did the toothache start"],
        "severity": ["pain out of 10"],
        "progression": ["swelling increasing", "kept up at night"],
        "associated_symptoms": ["fever", "bad taste", "bleeding gums", "jaw stiffness"],
        "affected_body_part": ["top teeth", "bottom teeth", "left", "right"],
        "onset_type": ["sudden sharp pain", "gradual ache while chewing"],
    },
    "mental_health": {
        "duration": ["how long have you felt this way"],
        "severity": ["intensity of feelings", "1-10", "how overwhelming"],
        "progression": ["getting harder to cope", "constant vs coming and going"],
        "associated_symptoms": ["trouble sleeping", "appetite changes", "panic attacks"],
        "affected_body_part": ["heart racing", "stomach knots"],
        "onset_type": ["after an event", "gradually over weeks"],
    },
    "reproductive": {
        "duration": ["when did the issue start"],
        "severity": ["pain level", "1-10"],
        "progression": ["cramping getting worse", "bleeding increasing"],
        "associated_symptoms": ["fever", "nausea", "dizziness", "unusual discharge"],
        "affected_body_part": ["pelvis", "lower abdomen"],
        "onset_type": ["sudden cramps", "gradual onset"],
    }
}

# Generic Keywords fallback for fields not defined in specific context
GENERIC_FIELD_KEYWORDS = {
    "duration": ["how long", "when it started"],
    "severity": ["how bad", "scale 1 to 10"],
    "progression": ["getting better or worse", "changing"],
    "associated_symptoms": ["any other symptoms", "anything else wrong"],
    "affected_body_part": ["where on your body", "location"],
    "onset_type": ["sudden", "gradual"],
    "aggravating_alleviating_factors": ["what makes it worse", "what makes it better", "does anything help"],
    "relevant_medical_history": ["any past conditions", "current medications", "medical history"]
}

# Field Priority for Clarification Questions
FIELD_PRIORITY = {
    "chief_complaint": 1,
    "severity": 2,
    "duration": 3,
    "progression": 4,
    "onset_type": 5,
    "affected_body_part": 6,
    "associated_symptoms": 7,
    "aggravating_alleviating_factors": 8,
    "relevant_medical_history": 9
}

# Valid Enum Values
VALID_PROGRESSIONS = ["improving", "worsening", "stable"]
VALID_ONSET_TYPES = ["sudden", "gradual"]

# Database Configuration
BASE_DIR = Path(__file__).parent
DATABASE_PATH = BASE_DIR / "clinassist.db"

# Audio Output & Cache Configuration
AUDIO_OUTPUT_DIR = BASE_DIR / "audio_output"
AUDIO_CACHE_DIR = BASE_DIR / "audio_cache"
AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)
AUDIO_CACHE_DIR.mkdir(exist_ok=True)

# Voice Persona Mapping
VOICES = {
    "default": "en_US-lessac-medium",      # Warm, professional
    "emergency": "en_US-kristin-medium",   # Alternative for high risk (if available) - falling back to lessac if kristin not found
    "pediatric": "en_US-lessac-medium"     # Placeholder for now
}

# Evaluation Configuration
EVALUATION_SAMPLES_DIR = BASE_DIR / "evaluation_samples"
EVALUATION_SAMPLES_DIR.mkdir(exist_ok=True)

# System States
STATES = {
    "GREETING": "greeting",
    "COLLECTING": "collecting",
    "CLARIFYING": "clarifying",
    "SUMMARIZING": "summarizing",
    "OFFERING_CONSULT": "offering_consult",
    "COMPLETE": "complete",
    "QA": "qa"
}

# Safety Disclaimers
SAFETY_DISCLAIMER = (
    "ClinAssist is a structured pre-screening support tool and does not provide "
    "medical diagnosis or treatment advice. This system does not replace a physician."
)

LLM_SAFETY_INSTRUCTIONS = """
CRITICAL SAFETY RULES:
- Do NOT diagnose conditions.
- Do NOT prescribe treatment.
- Do NOT provide medical advice.
- STRICT PROHIBITION: Do NOT mention any specific medication names, brand names, or chemical names (e.g., Tylenol, Ibuprofen, Acetaminophen, Naproxen, etc.).
- ONLY provide general, safe wellness advice (hydration, rest, etc.).
- Only extract structured symptom attributes.
- Only summarize provided information.
- This system does not replace a physician.
"""
