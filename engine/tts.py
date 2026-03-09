"""
Text-to-Speech module for ClinAssist
Uses Piper TTS for native, low-latency, and offline speech synthesis
"""
import hashlib
import os
import time
import wave
import io
from datetime import datetime
from typing import Optional

from config import TTS_MODEL, TTS_VOICE, AUDIO_OUTPUT_DIR, AUDIO_CACHE_DIR, VOICES
from database import log_latency, get_cached_tts, save_tts_to_cache

try:
    from piper.voice import PiperVoice
except ImportError:
    PiperVoice = None

# Cache for loaded models
_voice_models = {}

def get_piper_voice(voice_alias: str = "default"):
    """Lazy load specific Piper TTS model by alias"""
    global _voice_models
    
    # Resolve alias to actual voice name from config
    voice_name = VOICES.get(voice_alias, VOICES["default"])
    
    if voice_name not in _voice_models and PiperVoice is not None:
        # Resolve the assets path correctly regardless of where the script is run
        from config import BASE_DIR
        model_path = str(BASE_DIR / "assets" / f"{voice_name}.onnx")
        
        if os.path.exists(model_path):
            print(f"Loading Piper TTS model '{voice_name}' from {model_path}...")
            _voice_models[voice_name] = PiperVoice.load(model_path)
            print(f"Model '{voice_name}' loaded successfully.")
        else:
            print(f"Piper TTS Model '{voice_name}' not found at {model_path}.")
            # Fallback to default if it's already loaded
            default_voice = VOICES["default"]
            if default_voice in _voice_models:
                return _voice_models[default_voice]
            
            # If default not loaded, try to load it
            if voice_name != default_voice:
                return get_piper_voice("default")
            
            print(f"CRITICAL: Default voice {default_voice} also missing.")
            
    return _voice_models.get(voice_name)


def generate_speech(text: str, session_id: str, length_scale: float = 1.0, voice_alias: str = "default") -> Optional[bytes]:
    """
    Generate speech audio from text using Piper TTS locally with caching
    
    Args:
        text: Text to convert to speech
        session_id: Session ID for latency logging
        length_scale: Speed of speech (smaller is faster, default 1.0)
        voice_alias: Alias of the voice persona to use (default, emergency, pediatric)
    """
    start_time = time.time()
    
    try:
        if not text or not text.strip():
            return None

        # 1. Cache Lookup
        settings = {"length_scale": length_scale, "voice": voice_alias}
        cache_key_str = f"{text}|{voice_alias}|{length_scale}"
        text_hash = hashlib.md5(cache_key_str.encode('utf-8')).hexdigest()
        
        cached_path = get_cached_tts(text_hash)
        if cached_path and os.path.exists(cached_path):
            with open(cached_path, "rb") as f:
                audio_bytes = f.read()
            latency_ms = (time.time() - start_time) * 1000
            log_latency(session_id, "tts_cache_hit", latency_ms)
            return audio_bytes

        # 2. Generation (Cache Miss)
        voice = get_piper_voice(voice_alias)
        if not voice:
            print("Piper TTS not initialized or model missing.")
            return None
            
        # Use BytesIO for in-memory synthesis to avoid disk latency for raw result
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wav_file:
            voice.synthesize(text, wav_file, length_scale=length_scale)
            
        audio_bytes = audio_buffer.getvalue()
        
        # Save to cache in background (or immediately, but after we have the bytes)
        cache_path = AUDIO_CACHE_DIR / f"{text_hash}.wav"
        with open(cache_path, "wb") as f:
            f.write(audio_bytes)
            
        # Save to database
        save_tts_to_cache(text_hash, text, voice_alias, settings, str(cache_path))
        
        latency_ms = (time.time() - start_time) * 1000
        log_latency(session_id, "tts", latency_ms)
        
        return audio_bytes
    
    except Exception as e:
        print(f"TTS Error (Piper): {e}")
        latency_ms = (time.time() - start_time) * 1000
        log_latency(session_id, "tts_error", latency_ms)
        return None

