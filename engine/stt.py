"""
Speech-to-Text module for ClinAssist
Uses whisper-1 model via Nexus API for audio transcription
"""
import time
import requests
import base64
import io
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional
from config import (
    NEXUS_BASE_URL,
    NEXUS_API_KEY,
    NEXUS_STT_PATH,
    STT_MODEL,
    STT_REQUEST_MODE,
    STT_TIMEOUT_SECONDS,
)
from database import log_latency




HTTP_SESSION = requests.Session()
HTTP_ADAPTER = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=Retry(total=2, backoff_factor=1))
HTTP_SESSION.mount("http://", HTTP_ADAPTER)
HTTP_SESSION.mount("https://", HTTP_ADAPTER)


def transcribe_audio(audio_bytes: bytes, session_id: str) -> str:
    """
    Transcribe audio to text using whisper-1 via Nexus API
    
    Args:
        audio_bytes: Raw audio bytes (WAV format, 16kHz, 16-bit PCM mono)
        session_id: Session ID for latency logging
    
    Returns:
        Transcribed text string (empty string on error)
    """
    start_time = time.time()
    url = f"{NEXUS_BASE_URL}{NEXUS_STT_PATH}"
    headers = {"Authorization": f"Bearer {NEXUS_API_KEY}"}
    
    try:
        # Check if we should use multipart or JSON (base64)
        if STT_REQUEST_MODE in ["multipart", "auto"]:
            files = {
                "file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")
            }
            data = {"model": STT_MODEL}
            
            response = HTTP_SESSION.post(
                url, 
                headers=headers, 
                files=files, 
                data=data,
                timeout=STT_TIMEOUT_SECONDS
            )
        else:
            # JSON/Base64 mode
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            payload = {
                "model": STT_MODEL,
                "file": audio_b64
            }
            response = HTTP_SESSION.post(
                url,
                headers={**headers, "Content-Type": "application/json"},
                json=payload,
                timeout=STT_TIMEOUT_SECONDS
            )

        if response.status_code != 200:
            print(f"STT API Error ({STT_REQUEST_MODE}): {response.status_code} - {response.text}")
            
            # If auto and multipart failed, try a quick JSON fallback
            if STT_REQUEST_MODE == "auto" and response.status_code >= 400:
                print("Attempting JSON fallback for STT...")
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                response = HTTP_SESSION.post(
                    url,
                    headers={**headers, "Content-Type": "application/json"},
                    json={"model": STT_MODEL, "file": audio_b64},
                    timeout=STT_TIMEOUT_SECONDS
                )
        
        response.raise_for_status()
        result = response.json()
        transcript = result.get("text", "").strip()
        
        latency_ms = (time.time() - start_time) * 1000
        log_latency(session_id, "stt", latency_ms)
        
        return transcript

    except Exception as e:
        print(f"STT Error (Nexus): {e}")
        latency_ms = (time.time() - start_time) * 1000
        log_latency(session_id, "stt_error", latency_ms)
        return ""


def validate_audio_format(audio_bytes: bytes) -> bool:
    """
    Validate that audio is in WAV format
    Simple check for WAV header (RIFF)
    
    Args:
        audio_bytes: Raw audio bytes
    
    Returns:
        True if valid WAV format, False otherwise
    """
    if len(audio_bytes) < 44:
        return False
    
    # Check for RIFF header
    if audio_bytes[0:4] != b'RIFF':
        return False
    
    # Check for WAVE format
    if audio_bytes[8:12] != b'WAVE':
        return False
    
    return True
