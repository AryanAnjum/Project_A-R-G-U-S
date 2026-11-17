"""Text-to-speech for ARGUS.

Engines:

    piper  (default) runs on this PC, no internet needed. Voices are .onnx files
           from https://huggingface.co/rhasspy/piper-voices, downloaded once into
           models/piper/ on first use (or placed there by hand).
    edge   Microsoft Edge's online read-aloud voices through edge-tts. Needs
           internet and waits for the whole clip, so it is slower.
    mms    Meta's MMS voice (facebook/mms-tts-ben), used for Bangla without
           internet. Runs on this PC through transformers, downloaded once.

Pick one with ARGUS_TTS=piper|edge. If Piper is not installed or has no voice
for a language, ARGUS falls back to edge-tts for that reply and logs why.

Bangla: Piper has no Bangla voice, so ARGUS_TTS_BN decides how Bangla is spoken:
    auto  (default) the online Edge voice, and the offline MMS voice when there
          is no internet,
    edge  the online voice only,
    mms   the offline voice only (nothing leaves this PC).

synthesize() returns (audio_bytes, mime_type); the server plays the bytes on the
PC or sends them to the phone.
"""

import asyncio
import io
import os
import threading
import time
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
    # Piper has no Bangla voice yet: Bangla uses edge-tts when online and MMS when offline.
    "bn": (os.getenv("ARGUS_PIPER_VOICE_BN", ""), os.getenv("ARGUS_EDGE_VOICE_BN", "bn-BD-PradeepNeural")),
}
BN_ENGINE = os.getenv("ARGUS_TTS_BN", "auto").lower()   # auto | edge | mms
MMS_MODEL_BN = os.getenv("ARGUS_MMS_VOICE_BN", "facebook/mms-tts-ben")
MMS_DEVICE = os.getenv("ARGUS_MMS_DEVICE", "cpu").lower()  # the voice is small; the CPU keeps VRAM free

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


_mms = {}
_mms_lock = threading.Lock()
_edge_down_until = 0.0


_MMS_FILES = dict(required=["config.json", "vocab.json"], optional=["tokenizer_config.json", "special_tokens_map.json"],
                  weights=["model.safetensors", "pytorch_model.bin"])


def _mms_source():
    """Folder (or cached model id) of the offline Bangla voice. ARGUS downloads it into models/ itself,
    so it does not depend on the Hugging Face cache or on that library's offline mode."""
    if os.path.isdir(MMS_MODEL_BN):
        return MMS_MODEL_BN, {}
    from argus_language import fetch_model
    try:
        return fetch_model(MMS_MODEL_BN, **_MMS_FILES), {}
    except Exception as exc:
        print(f"[TTS] Could not download the offline Bangla voice ({type(exc).__name__}: {exc}).")
        return MMS_MODEL_BN, {"local_files_only": True}


def prepare_bangla():
    """Called when Bengali is chosen: fetch the offline Bangla voice while there is internet, so it is
    there when the internet is not. With ARGUS_TTS_BN=mms it is also loaded right away."""
    if VOICES["bn"][0] or BN_ENGINE == "edge":
        return
    try:
        if BN_ENGINE == "mms":
            _load_mms()
        elif not os.path.isdir(MMS_MODEL_BN):
            from argus_language import fetch_model
            fetch_model(MMS_MODEL_BN, **_MMS_FILES)
    except Exception as exc:
        print(f"[TTS] Offline Bangla voice not ready ({type(exc).__name__}: {exc}).")


def _load_mms():
    with _mms_lock:
        if "model" not in _mms:
            import torch
            from transformers import AutoTokenizer, VitsModel
            print(f"[TTS] Loading the offline Bangla voice {MMS_MODEL_BN} (one-time download, about 150 MB)...")
            source, extra = _mms_source()
            tok = AutoTokenizer.from_pretrained(source, **extra)
            if getattr(tok, "is_uroman", False):
                raise RuntimeError(f"{MMS_MODEL_BN} needs romanised text, which ARGUS does not produce")
            device = MMS_DEVICE if MMS_DEVICE != "cuda" or torch.cuda.is_available() else "cpu"
            model = VitsModel.from_pretrained(source, **extra).to(device).eval()
            _mms.update(model=model, tok=tok, device=device, torch=torch)
            print(f"[TTS] Offline Bangla voice ready ({device}).")
        return _mms


def _mms_wav(text, rate):
    """Bangla speech from MMS (VITS). It cannot read digits, so numbers are spelled out first."""
    import numpy as np
    from argus_language import spell_numbers_bn

    m = _load_mms()
    torch, model, tok = m["torch"], m["model"], m["tok"]
    inputs = tok(spell_numbers_bn(text), return_tensors="pt").to(m["device"])
    if inputs["input_ids"].shape[1] < 2:
        raise ValueError("no Bangla text to speak")
    with _mms_lock, torch.inference_mode():
        model.speaking_rate = 1.0 / _rate_to_length_scale(rate)
        wave_f = model(**inputs).waveform[0].float().cpu().numpy()
    peak = float(np.abs(wave_f).max()) or 1.0
    pcm = (wave_f / max(1.0, peak) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(model.config.sampling_rate))
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()


def _bangla(text, rate):
    """Bangla: the online Edge voice first (it sounds better), the offline MMS voice when there is no
    internet. After a failed online try, the offline voice is used directly for the next two minutes."""
    global _edge_down_until
    if BN_ENGINE != "mms" and time.time() >= _edge_down_until:
        try:
            audio = _edge_mp3(text, "bn", rate)
            if audio:
                return audio, "audio/mpeg"
            raise RuntimeError("the online voice returned no audio")
        except Exception as exc:
            if BN_ENGINE == "edge":
                raise
            _edge_down_until = time.time() + 120
            print(f"[TTS] Online Bangla voice unavailable ({type(exc).__name__}: {exc}); "
                  "using the offline voice for now.")
    return _mms_wav(text, rate), "audio/wav"


def synthesize(text, lang="en", rate="+8%"):
    """Return (audio_bytes, mime_type) for text, or raise if no engine could speak it."""
    if lang == "bn" and not VOICES["bn"][0]:
        return _bangla(text, rate)
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
    if lang == "bn" and not VOICES["bn"][0]:
        if BN_ENGINE == "mms":
            try:
                _mms_wav("প্রস্তুত।", "+0%")
            except Exception as exc:
                print(f"[TTS] Offline Bangla voice not loaded ({type(exc).__name__}: {exc}).")
        return
    if ENGINE == "piper":
        try:
            voice = _load_piper(lang)
            if voice is not None:
                _piper_wav(voice, "Ready.", "+0%")  # first synthesis also initialises ONNX Runtime
        except Exception as exc:
            print(f"[TTS] Piper voice not loaded ({type(exc).__name__}: {exc}). Replies will use edge-tts.")
