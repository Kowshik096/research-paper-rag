import threading

from google import genai

from src.config import get_settings

_gemini_client = None
_gemini_client_lock = threading.Lock()


def get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        with _gemini_client_lock:
            if _gemini_client is None:
                _gemini_client = genai.Client(api_key=get_settings().gemini_api_key)
    return _gemini_client
