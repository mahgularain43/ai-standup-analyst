import os
from typing import Optional

def _lazy_engine():
    try:
        import pyttsx3
        eng = pyttsx3.init()
        eng.setProperty("rate", 175)   # speaking speed
        eng.setProperty("volume", 1.0)
        return eng
    except Exception:
        return None

_ENGINE = None

def speak_to_file(text: str, out_path: str, voice_name: Optional[str] = None) -> Optional[str]:
    """
    Synthesize text to a WAV/MP3 file. Returns path or None on failure.
    Falls back to writing a .txt if TTS engine is unavailable.
    """
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = _lazy_engine()
    try:
        if _ENGINE is None:
            with open(out_path + ".txt", "w", encoding="utf-8") as f:
                f.write(text)
            return out_path + ".txt"

        if voice_name:
            for v in _ENGINE.getProperty("voices"):
                if voice_name.lower() in (v.name or "").lower():
                    _ENGINE.setProperty("voice", v.id)
                    break

        _ENGINE.save_to_file(text, out_path)
        _ENGINE.runAndWait()
        return out_path
    except Exception:
        return None

def add_laugh_track(audio_path: str, out_path: str, intensity: float = 0.5) -> Optional[str]:
    """
    Overlay a simple laugh track if pydub is available.
    If unavailable, returns None to keep the app stable.
    """
    try:
        from pydub import AudioSegment
        base = AudioSegment.from_file(audio_path)
        # For a real laugh, add an actual file here and overlay:
        # laugh = AudioSegment.from_file("assets/laugh.mp3") - int((1-intensity)*10)
        # mixed = base.overlay(laugh)
        # Minimal “ambience” approach (safe no-extra-file fallback):
        mixed = base - int(10 + (1 - intensity) * 10)
        mixed = mixed.overlay(base)
        fmt = os.path.splitext(out_path)[1].lstrip(".") or "wav"
        mixed.export(out_path, format=fmt)
        return out_path
    except Exception:
        return None
