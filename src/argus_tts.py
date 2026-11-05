"""Text-to-speech for ARGUS.

Two engines:

    piper  (default) runs on this PC, no internet needed. Voices are .onnx files
           from https://huggingface.co/rhasspy/piper-voices, downloaded once into
           models/piper/ on first use (or placed there by hand).
    edge   Microsoft Edge's online read-aloud voices through edge-tts. Needs
           internet and waits for the whole clip, so it is slower.

Pick one with ARGUS_TTS=piper|edge. If Piper is not installed or has no voice
for a language, ARGUS falls back to edge-tts for that reply and logs why.

synthesize() returns (audio_bytes, mime_type); the server plays the bytes on the
PC or sends them to the phone.
"""

import asyncio
import io
import os
import threading
import urllib.request
import wave

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_ROOT = os.path.abspath(os.getenv("ARGUS_HOME", REPO_ROOT))
ENGINE = os.getenv("ARGUS_TTS", "piper").lower()
PIPER_DIR = os.getenv("ARGUS_PIPER_DIR", os.path.join(DATA_ROOT, "models", "piper"))
VOICE_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# language -> (piper voice name, edge-tts voice)
VOICES = {
    "en": (os.getenv("ARGUS_PIPER_VOICE_EN", "en_US-ryan-medium"), "en-US-GuyNeural"),
    "de": (os.getenv("ARGUS_PIPER_VOICE_DE", "de_DE-thorsten-medium"), "de-DE-ConradNeural"),
    # Piper has no Bangla voice yet, so Bangla captions use edge-tts when online.
    "bn": (os.getenv("ARGUS_PIPER_VOICE_BN", ""), "bn-BD-PradeepNeural"),
}

_piper_voices = {}
_piper_lock = threading.Lock()
_piper_missing_reported = set()


def _voice_url(name):
    # en_US-ryan-medium -> en/en_US/ryan/medium/en_US-ryan-medium.onnx
    locale, speaker, quality = name.split("-", 2)
    return f"{VOICE_BASE_URL}/{locale.split('_')[0]}/{locale}/{speaker}/{quality}/{name}.onnx"


def _load_piper(lang):
    name = VOICES.get(lang, VOICES["en"])[0]
    if not name:
        return None
    with _piper_lock:
        if name in _piper_voices:
            return _piper_voices[name]
        from piper import PiperVoice  # pip install piper-tts
        path = os.path.join(PIPER_DIR, name + ".onnx")
        if not os.path.isfile(path) or not os.path.isfile(path + ".json"):
            os.makedirs(PIPER_DIR, exist_ok=True)
            url = _voice_url(name)
            print(f"[TTS] Downloading Piper voice {name} (one time)...")
            urllib.request.urlretrieve(url + ".json", path + ".json")
            urllib.request.urlretrieve(url, path)
        voice = PiperVoice.load(path)
        _piper_voices[name] = voice
        print(f"[TTS] Piper voice ready: {name}")
        return voice


def _rate_to_length_scale(rate):
    # edge-tts style "+12%" -> Piper length_scale (smaller is faster)
    try:
        pct = float(rate.strip().rstrip("%"))
    except (AttributeError, ValueError):
        return 1.0
    return max(0.5, min(2.0, 1.0 / (1.0 + pct / 100.0)))


def _piper_wav(voice, text, rate):
    buf = io.BytesIO()
    length_scale = _rate_to_length_scale(rate)
    with wave.open(buf, "wb") as wav:
        if hasattr(voice, "synthesize_wav"):  # piper-tts 1.3+
            try:
                from piper import SynthesisConfig
                voice.synthesize_wav(text, wav, syn_config=SynthesisConfig(length_scale=length_scale))
            except ImportError:
                voice.synthesize_wav(text, wav)
        else:  # piper-tts 1.2
            voice.synthesize(text, wav, length_scale=length_scale)
    return buf.getvalue()


def _edge_mp3(text, lang, rate):
    import edge_tts

    async def run():
        communicate = edge_tts.Communicate(text, VOICES.get(lang, VOICES["en"])[1], rate=rate)
        out = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                out.extend(chunk["data"])
        return bytes(out)

    return asyncio.run(run())


def synthesize(text, lang="en", rate="+8%"):
    """Return (audio_bytes, mime_type) for text, or raise if no engine could speak it."""
    if ENGINE == "piper":
        try:
            voice = _load_piper(lang)
            if voice is not None:
                return _piper_wav(voice, text, rate), "audio/wav"
        except Exception as exc:
            key = (lang, type(exc).__name__)
            if key not in _piper_missing_reported:
                _piper_missing_reported.add(key)
                print(f"[TTS] Piper unavailable for '{lang}' ({type(exc).__name__}: {exc}); using edge-tts.")
    return _edge_mp3(text, lang, rate), "audio/mpeg"


def warm_up(lang="en"):
    """Load the voice at startup so the first reply isn't slowed by loading it."""
    if ENGINE == "piper":
        try:
            _load_piper(lang)
        except Exception as exc:
            print(f"[TTS] Piper voice not loaded ({type(exc).__name__}: {exc}). Replies will use edge-tts.")
