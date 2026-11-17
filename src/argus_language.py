"""Reply language for ARGUS: English or Bengali (Bangla).

The vision model, the walking guidance and the voice commands all work in English.
When a device picks Bengali, every sentence ARGUS is about to say is turned into
Bangla here, just before it is spoken:

    1. fixed sentences (mode names, guidance instructions, error messages) come from
       a hand-written table, so they are instant and always the same,
    2. obstacle warnings are built from a template and a small word list,
    3. everything else (the model's own answers) goes through a translation model,
       NLLB-200 by default, which runs on this PC and is downloaded once.

Bangla questions typed into the app are translated to English the same way, so the
model can answer them.

The translation model is downloaded by ARGUS itself into models/<model name>/ (about
2.5 GB, once, with a progress figure, and it resumes after an interruption). It does
not go through the Hugging Face cache, so it also works in the installed desktop app,
which runs the Hugging Face libraries in offline mode. A PC without internet can get
the folder copied from another PC.

Settings (environment variables):
    ARGUS_LANGUAGE            en | bn    language a device starts with (default en)
    ARGUS_TRANSLATE_MODEL     Hugging Face id (default facebook/nllb-200-distilled-600M),
                              or the path of a folder that already holds the model
    ARGUS_TRANSLATE_DEVICE    auto | cuda | cpu   (auto = GPU when there is one)
    ARGUS_TRANSLATE_BEAMS     beam size, 1 is fastest (default 2)
"""

import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict

LANGUAGES = {"en": "English", "bn": "Bengali"}
NATIVE_NAMES = {"en": "English", "bn": "বাংলা"}
DEFAULT_LANGUAGE = os.getenv("ARGUS_LANGUAGE", "en").lower()
if DEFAULT_LANGUAGE in ("bangla", "bengali"):
    DEFAULT_LANGUAGE = "bn"
if DEFAULT_LANGUAGE not in LANGUAGES:
    DEFAULT_LANGUAGE = "en"

MODEL_NAME = os.getenv("ARGUS_TRANSLATE_MODEL", "facebook/nllb-200-distilled-600M")
DEVICE = os.getenv("ARGUS_TRANSLATE_DEVICE", "auto").lower()
BEAMS = max(1, int(os.getenv("ARGUS_TRANSLATE_BEAMS", "2")))
NLLB_CODES = {"en": "eng_Latn", "bn": "ben_Beng", "de": "deu_Latn"}
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_ROOT = os.path.abspath(os.getenv("ARGUS_HOME", REPO_ROOT))
MODELS_DIR = os.getenv("ARGUS_MODELS_DIR", os.path.join(DATA_ROOT, "models"))
HF_URL = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
# Files of an NLLB model on Hugging Face. The weights are whichever of the two exists.
NLLB_FILES = dict(required=["config.json", "tokenizer.json", "tokenizer_config.json"],
                  optional=["generation_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"],
                  weights=["model.safetensors", "pytorch_model.bin"])

BENGALI_CHARS = re.compile("[\u0980-\u09FF]")  # the Bengali Unicode block
SWITCHED = {"bn": "ঠিক আছে, এখন থেকে আমি বাংলায় কথা বলব।", "en": "Okay, I will speak English now."}


def normalize_language(value):
    value = str(value or "").strip().lower()
    value = {"bangla": "bn", "bengali": "bn", "english": "en"}.get(value, value)
    return value if value in LANGUAGES else None


def has_bengali(text):
    """True when the text is written (mostly) in Bangla script."""
    letters = [c for c in text or "" if c.isalpha()]
    if not letters:
        return False
    return sum(1 for c in letters if BENGALI_CHARS.match(c)) >= 0.4 * len(letters)


# ------------------------------------------------------------------
# "Speak Bengali" / "speak English" (spoken or typed)
# ------------------------------------------------------------------
_BN_WORD = r"bangla|bengali|bangali|bangala|bengoli|bengal|bangladeshi"
_LANG = rf"(?P<lang>{_BN_WORD}|english)"
_WAKE = r"(?:(?:hey|ok|okay|hi)\s+)?(?:argus\s+)?(?:please\s+)?(?:can you\s+|could you\s+)?"
_TAIL = r"(?:\s+(?:language|now|please|from now on|to me))*"
LANGUAGE_COMMAND = re.compile(
    rf"^{_WAKE}(?:"
    rf"(?:speak|talk|reply|answer|respond|say it|tell me)(?:\s+to me)?\s+(?:in\s+)?{_LANG}{_TAIL}"
    rf"|(?:switch|change|set|turn)\s+(?:the\s+|your\s+)?(?:language|voice|speech)\s+(?:to|into)\s+{_LANG.replace('lang', 'l2')}{_TAIL}"
    rf"|(?:switch|change|go)\s+(?:back\s+)?(?:to|into)\s+{_LANG.replace('lang', 'l3')}(?:\s+(?:language|voice|speech))?{_TAIL}"
    rf"|{_LANG.replace('lang', 'l4')}\s+(?:language|voice|speech){_TAIL}"
    rf")$")


def parse_language_command(text):
    """'speak Bengali', 'switch language to English', 'Bangla language' -> "bn" / "en", else None.
    `text` is lower-case with punctuation removed (the server's normalize_command_text)."""
    m = LANGUAGE_COMMAND.match(" ".join((text or "").lower().split()))
    if not m:
        return None
    word = next(v for v in m.groupdict().values() if v)
    return "en" if word == "english" else "bn"


# ------------------------------------------------------------------
# Fixed sentences
# ------------------------------------------------------------------
PHRASES_BN = {
    # modes
    "Screen mode.": "স্ক্রিন মোড।",
    "Live mode.": "লাইভ মোড।",
    "Assistive mode.": "সহায়ক মোড।",
    "Sign mode.": "সাইন মোড।",
    "Sign mode, Bangla.": "সাইন মোড, বাংলা।",
    "Sign mode, German.": "সাইন মোড, জার্মান।",
    # general
    "Ready.": "প্রস্তুত।",
    "I haven't said anything yet.": "আমি এখনো কিছু বলিনি।",
    "Sorry, something went wrong while answering.": "দুঃখিত, উত্তর দিতে গিয়ে একটি সমস্যা হয়েছে।",
    "Sorry, I didn't get a clear answer.": "দুঃখিত, আমি পরিষ্কার উত্তর পাইনি।",
    "Please ask me again.": "অনুগ্রহ করে আবার জিজ্ঞাসা করুন।",
    "Okay, I will speak English now.": "ঠিক আছে, এখন থেকে আমি ইংরেজিতে কথা বলব।",
    # sign language
    "I'm not sure.": "আমি নিশ্চিত নই।",
    "Please sign again.": "অনুগ্রহ করে আবার সাইন করুন।",
    # cameras
    "I can't see your phone's screen.": "আমি আপনার ফোনের স্ক্রিন দেখতে পাচ্ছি না।",
    "Tap Share screen in the ARGUS app, then ask again.": "আর্গাস অ্যাপে শেয়ার স্ক্রিন চাপুন, তারপর আবার জিজ্ঞাসা করুন।",
    "I can't see the screen right now.": "আমি এখন স্ক্রিন দেখতে পাচ্ছি না।",
    "I can't see the head-mount camera.": "আমি হেড-মাউন্ট ক্যামেরা দেখতে পাচ্ছি না।",
    "Check that it is plugged into the PC.": "এটি কম্পিউটারে লাগানো আছে কি না দেখুন।",
    "I can't see the phone's camera.": "আমি ফোনের ক্যামেরা দেখতে পাচ্ছি না।",
    "Open the ARGUS app on the phone, or pick another camera.": "ফোনে আর্গাস অ্যাপ খুলুন, অথবা অন্য ক্যামেরা বেছে নিন।",
    "I can't see the camera right now.": "আমি এখন ক্যামেরা দেখতে পাচ্ছি না।",
    "Turn the camera on and ask again.": "ক্যামেরা চালু করে আবার জিজ্ঞাসা করুন।",
    # walking guidance (the commands stay English words, written in Bangla letters so the voice can say them)
    "Guiding you.": "আমি আপনাকে পথ দেখাচ্ছি।",
    "Hold the camera at chest height, pointing ahead.": "ক্যামেরা বুকের উচ্চতায় সামনের দিকে ধরে রাখুন।",
    "Say stop guiding when you arrive.": "পৌঁছে গেলে বলুন, স্টপ গাইডিং।",
    "I can't see the camera, so I stopped guiding.": "আমি ক্যামেরা দেখতে পাচ্ছি না, তাই পথ দেখানো বন্ধ করেছি।",
    "Turn the camera on and say guide me.": "ক্যামেরা চালু করে বলুন, গাইড মি।",
    "Okay, I stopped guiding.": "ঠিক আছে, পথ দেখানো বন্ধ করেছি।",
    "Careful!": "সাবধান!",
    "Stand still until it passes.": "এটি চলে না যাওয়া পর্যন্ত স্থির হয়ে দাঁড়িয়ে থাকুন।",
    "Move a little to your left to go around it.": "এটি এড়াতে একটু বাম দিকে সরে যান।",
    "Move a little to your right to go around it.": "এটি এড়াতে একটু ডান দিকে সরে যান।",
    "Slow down and stand still, I will check again.": "ধীরে চলুন এবং স্থির হয়ে দাঁড়ান, আমি আবার দেখছি।",
    "The road is on your left, keep to the right.": "রাস্তা আপনার বাম দিকে, ডান দিক ঘেঁষে চলুন।",
    "The road is on your right, keep to the left.": "রাস্তা আপনার ডান দিকে, বাম দিক ঘেঁষে চলুন।",
    "The way ahead looks clear, keep walking.": "সামনের পথ পরিষ্কার, হাঁটতে থাকুন।",
    "Keep walking slowly.": "ধীরে ধীরে হাঁটতে থাকুন।",
}
_PHRASE_KEYS = {" ".join(k.lower().split()): v for k, v in PHRASES_BN.items()}

# Obstacles the walking guidance warns about (the same words as HAZARD_WORDS in allnew.py).
HAZARDS_BN = {
    "car": "গাড়ি", "bus": "বাস", "truck": "ট্রাক", "lorry": "লরি", "van": "ভ্যান", "rickshaw": "রিকশা",
    "cng": "সিএনজি", "auto rickshaw": "অটোরিকশা", "motorbike": "মোটরসাইকেল", "motorcycle": "মোটরসাইকেল",
    "bike": "বাইক", "bicycle": "সাইকেল", "scooter": "স্কুটার", "vehicle": "গাড়ি", "jeep": "জিপ",
    "dog": "কুকুর", "cat": "বিড়াল", "cow": "গরু", "goat": "ছাগল", "animal": "প্রাণী", "horse": "ঘোড়া",
    "table": "টেবিল", "chair": "চেয়ার", "bench": "বেঞ্চ", "stool": "টুল", "shelf": "তাক", "pole": "খুঁটি",
    "post": "খুঁটি", "pillar": "পিলার", "column": "স্তম্ভ", "tree": "গাছ", "tree trunk": "গাছের গুঁড়ি",
    "wall": "দেয়াল", "stairs": "সিঁড়ি", "staircase": "সিঁড়ি", "step": "সিঁড়ির ধাপ", "kerb": "ফুটপাতের কিনারা",
    "curb": "ফুটপাতের কিনারা", "hole": "গর্ত", "drain": "ড্রেন", "pothole": "গর্ত", "door": "দরজা",
    "barrier": "ব্যারিকেড", "fence": "বেড়া", "bollard": "খুঁটি", "box": "বাক্স", "bin": "ডাস্টবিন",
    "dustbin": "ডাস্টবিন", "railing": "রেলিং", "construction": "নির্মাণকাজ", "ladder": "মই", "bag": "ব্যাগ",
}
_PLURALS = {"buses": "bus", "lorries": "lorry", "benches": "bench", "shelves": "shelf", "boxes": "box"}
_THERE_IS = re.compile(r"^there is (?:(a|an) )?(.+?) ahead of you[.!]?$", re.IGNORECASE)
_COMING = re.compile(r"^(?:(a|an) )?(.+?) is coming toward you[.!]?$", re.IGNORECASE)


def _hazard_bn(article, thing):
    """'a', 'car' -> 'একটি গাড়ি';  None, 'cars' -> 'গাড়ি'. None when the word isn't in the list."""
    word = thing.lower().strip()
    for candidate in (word, _PLURALS.get(word), word[:-1] if word.endswith("s") else None):
        if candidate in HAZARDS_BN:
            bn = HAZARDS_BN[candidate]
            return f"একটি {bn}" if article else bn
    return None


def _fixed_bn(sentence):
    """Bangla for a sentence that needs no translation model, or None."""
    key = " ".join(sentence.lower().split())
    if key in _PHRASE_KEYS:
        return _PHRASE_KEYS[key]
    m = _THERE_IS.match(sentence.strip())
    if m:
        thing = _hazard_bn(m.group(1), m.group(2))
        if thing:
            return f"আপনার সামনে {thing} আছে।"
    m = _COMING.match(sentence.strip())
    if m:
        thing = _hazard_bn(m.group(1), m.group(2))
        if thing:
            return f"{thing} আপনার দিকে আসছে।"
    return None


_SENTENCE_SPLIT = re.compile("(?<=[.!?\u0964])" + r"""["')\]]*\s+""")  # \u0964 is the Bangla full stop


def split_sentences(text):
    return [s.strip() for s in _SENTENCE_SPLIT.split(text or "") if s.strip()]


# ------------------------------------------------------------------
# Downloading a model as a plain folder
# ------------------------------------------------------------------
class DownloadError(RuntimeError):
    pass


def model_folder(repo):
    return os.path.join(MODELS_DIR, repo.rstrip("/").split("/")[-1])


def _download(url, path, on_progress=None):
    """Download one file to `path`. An interrupted download continues from its .part file."""
    if os.path.isfile(path):
        return
    part = path + ".part"
    have = os.path.getsize(part) if os.path.isfile(part) else 0
    request = urllib.request.Request(url, headers={"User-Agent": "ARGUS"})
    if have:
        request.add_header("Range", f"bytes={have}-")
    try:
        response = urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and have:  # the server refuses to continue this .part: start the file again
            os.remove(part)
            return _download(url, path, on_progress)
        raise
    with response:
        if have and response.status != 206:
            have = 0  # the server sent the whole file again
        total = have + int(response.headers.get("Content-Length") or 0)
        with open(part, "ab" if have else "wb") as f:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                have += len(chunk)
                if on_progress:
                    on_progress(have, total)
    if total and have < total:
        raise DownloadError(f"the download of {os.path.basename(path)} stopped at {have} of {total} bytes")
    os.replace(part, path)


def fetch_model(repo, required, weights, optional=(), on_progress=None, attempts=4):
    """Make sure models/<name>/ holds the Hugging Face model `repo` and return that folder.

    `weights` lists the possible weight files, best first; the first one the repository has is taken.
    `on_progress(done_bytes, total_bytes)` follows the weight file, which is nearly all of the download."""
    folder = model_folder(repo)
    marker = os.path.join(folder, ".complete")
    if os.path.isfile(marker):
        return folder
    os.makedirs(folder, exist_ok=True)
    base = f"{HF_URL}/{repo}/resolve/main/"
    for name in required:
        _download(base + name, os.path.join(folder, name))
    for name in optional:
        try:
            _download(base + name, os.path.join(folder, name))
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
    for name in weights:
        last = None
        for attempt in range(attempts):
            try:
                _download(base + name, os.path.join(folder, name), on_progress)
                last = None
                break
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code == 404:
                    break  # this repository has the other weight format
            except (OSError, DownloadError) as exc:  # connection dropped: wait, then continue the .part
                last = exc
                print(f"[DOWNLOAD] {name}: {type(exc).__name__}: {exc}; trying again...")
                time.sleep(3 * (attempt + 1))
        if last is None:
            with open(marker, "w", encoding="utf-8") as f:
                f.write(repo)
            return folder
        if not (isinstance(last, urllib.error.HTTPError) and last.code == 404):
            raise last
    raise DownloadError(f"{repo} has none of the weight files {', '.join(weights)}")


# ------------------------------------------------------------------
# Translation model
# ------------------------------------------------------------------
class Translator:
    """NLLB-200 on this PC. Loads in the background the first time Bengali is chosen."""

    def __init__(self):
        self.state = "off"      # off | loading | ready | error
        self.error = ""
        self.device = ""
        self.progress = None    # 0-100 while the model is being downloaded
        self.on_change = None   # called with status() whenever the state changes
        self._fn = None         # (text, src, tgt) -> text
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._cache = OrderedDict()
        self._failed_at = 0.0

    def status(self):
        return {"state": self.state, "error": self.error, "model": MODEL_NAME, "device": self.device,
                "progress": self.progress}

    @property
    def ready(self):
        return self.state == "ready"

    def _set(self, state, error=""):
        self.state, self.error = state, error
        if self.on_change:
            try:
                self.on_change(self.status())
            except Exception:
                pass

    def ensure_loading(self, retry=False):
        """Start loading if it hasn't been (returns at once). After a failure it tries again when asked
        to (`retry`) or a minute later, so fixing the internet connection is enough."""
        with self._load_lock:
            if self.state in ("loading", "ready"):
                return
            if self.state == "error" and not retry and time.time() - self._failed_at < 60:
                return
            self.state, self.error = "loading", ""
        if self.on_change:
            self.on_change(self.status())
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        start = time.time()
        try:
            print(f"[LANGUAGE] Loading the translation model {MODEL_NAME} "
                  "(the first time downloads about 2.5 GB)...")
            self.progress = None
            self._fn = self._build()
            self.progress = None
            sample = self._fn("Good morning, how are you?", "en", "bn")
            if not BENGALI_CHARS.search(sample or ""):
                raise RuntimeError(f"the model's test translation was not Bangla: {sample!r}")
            print(f"[LANGUAGE] Translation model ready on {self.device} in {time.time() - start:.1f} s.")
            self._set("ready")
        except Exception as exc:
            self._fn = None
            self.progress = None
            self._failed_at = time.time()
            print(f"[LANGUAGE] Translation model not available ({type(exc).__name__}: {exc}). "
                  "Replies stay in English until it loads.")
            self._set("error", f"{type(exc).__name__}: {exc}")

    def _on_download(self, done, total):
        pct = int(done * 100 / total) if total else 0
        if pct != self.progress:
            self.progress = pct
            if pct % 10 == 0:
                print(f"[LANGUAGE] Downloading the translation model: {pct}% "
                      f"({done >> 20} of {total >> 20} MB)")
            if self.on_change:
                try:
                    self.on_change(self.status())
                except Exception:
                    pass

    def _build(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = DEVICE if DEVICE in ("cuda", "cpu") else ("cuda" if torch.cuda.is_available() else "cpu")
        source, extra, download_error = MODEL_NAME, {}, None
        if not os.path.isdir(MODEL_NAME):
            try:
                source = fetch_model(MODEL_NAME, on_progress=self._on_download, **NLLB_FILES)
            except Exception as exc:
                # No download: the model may still be in the Hugging Face cache from an earlier run.
                download_error, extra = exc, {"local_files_only": True}
                print(f"[LANGUAGE] Download failed ({type(exc).__name__}: {exc}).")
        try:
            tok = AutoTokenizer.from_pretrained(source, **extra)
            model = AutoModelForSeq2SeqLM.from_pretrained(source, **extra)
        except Exception:
            if download_error is None:
                raise
            raise RuntimeError(
                f"This PC could not download the Bengali translation model from {HF_URL} "
                f"({type(download_error).__name__}: {download_error}). Check the PC's internet connection and "
                f"choose Bengali again; the download continues where it stopped. Or copy the folder "
                f"{model_folder(MODEL_NAME)} from a PC that has it.") from download_error
        model.eval()

        def make(model):
            def run(text, src, tgt):
                tok.src_lang = NLLB_CODES[src]
                enc = tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
                with torch.inference_mode():
                    out = model.generate(
                        **enc, forced_bos_token_id=tok.convert_tokens_to_ids(NLLB_CODES[tgt]),
                        num_beams=BEAMS, do_sample=False,
                        max_new_tokens=min(256, int(enc["input_ids"].shape[1] * 2.5) + 16))
                return tok.batch_decode(out, skip_special_tokens=True)[0].strip()
            return run

        if device == "cuda":
            # Half precision needs about 1.3 GB of VRAM. If its output is not Bangla, use full precision.
            try:
                run = make(model.half().to(device))
                if BENGALI_CHARS.search(run("Good morning, how are you?", "en", "bn")):
                    self.device = "cuda (half precision)"
                    return run
                print("[LANGUAGE] Half precision gave a bad translation; using full precision.")
            except Exception as exc:
                print(f"[LANGUAGE] Half precision failed ({type(exc).__name__}: {exc}); using full precision.")
            model = model.float()
        self.device = device
        return make(model.to(device))

    def translate(self, text, src, tgt):
        """Translate one sentence or short text. Raises RuntimeError when the model isn't loaded."""
        text = (text or "").strip()
        if not text or src == tgt:
            return text
        key = (src, tgt, text)
        with self._run_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            if self._fn is None:
                raise RuntimeError("translation model not loaded")
            out = self._fn(text, src, tgt)
            self._cache[key] = out
            while len(self._cache) > 500:
                self._cache.popitem(last=False)
            return out


translator = Translator()


def to_bangla(text, source="en"):
    """English (or German) text -> Bangla, sentence by sentence. Returns None when a sentence needs the
    translation model and it isn't ready yet; the caller then speaks the original text."""
    if not (text or "").strip():
        return text
    if has_bengali(text):
        return text
    out = []
    for sentence in split_sentences(text):
        fixed = _fixed_bn(sentence) if source == "en" else None
        if fixed is None:
            if not translator.ready:
                return None
            try:
                fixed = translator.translate(sentence, source, "bn")
            except Exception as exc:
                print(f"[LANGUAGE] Translation failed ({type(exc).__name__}: {exc}): {sentence!r}")
                return None
        if fixed:
            out.append(fixed)
    return " ".join(out)


def to_english(text):
    """A Bangla question -> English for the model. Returns None when the model isn't ready."""
    if not translator.ready:
        return None
    try:
        return translator.translate(" ".join(text.split()), "bn", "en")
    except Exception as exc:
        print(f"[LANGUAGE] Translation failed ({type(exc).__name__}: {exc}): {text!r}")
        return None


# ------------------------------------------------------------------
# Numbers in Bangla words (for the offline Bangla voice, which cannot read digits)
# ------------------------------------------------------------------
_BN_DIGITS = "০১২৩৪৫৬৭৮৯"
_DIGIT_WORDS = ["শূন্য", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়"]
_BELOW_100 = (
    "শূন্য এক দুই তিন চার পাঁচ ছয় সাত আট নয় দশ "
    "এগারো বারো তেরো চৌদ্দ পনেরো ষোলো সতেরো আঠারো উনিশ বিশ "
    "একুশ বাইশ তেইশ চব্বিশ পঁচিশ ছাব্বিশ সাতাশ আটাশ ঊনত্রিশ ত্রিশ "
    "একত্রিশ বত্রিশ তেত্রিশ চৌত্রিশ পঁয়ত্রিশ ছত্রিশ সাঁইত্রিশ আটত্রিশ ঊনচল্লিশ চল্লিশ "
    "একচল্লিশ বিয়াল্লিশ তেতাল্লিশ চুয়াল্লিশ পঁয়তাল্লিশ ছেচল্লিশ সাতচল্লিশ আটচল্লিশ ঊনপঞ্চাশ পঞ্চাশ "
    "একান্ন বায়ান্ন তিপ্পান্ন চুয়ান্ন পঞ্চান্ন ছাপ্পান্ন সাতান্ন আটান্ন ঊনষাট ষাট "
    "একষট্টি বাষট্টি তেষট্টি চৌষট্টি পঁয়ষট্টি ছেষট্টি সাতষট্টি আটষট্টি ঊনসত্তর সত্তর "
    "একাত্তর বাহাত্তর তিয়াত্তর চুয়াত্তর পঁচাত্তর ছিয়াত্তর সাতাত্তর আটাত্তর ঊনআশি আশি "
    "একাশি বিরাশি তিরাশি চুরাশি পঁচাশি ছিয়াশি সাতাশি আটাশি ঊননব্বই নব্বই "
    "একানব্বই বিরানব্বই তিরানব্বই চুরানব্বই পঁচানব্বই ছিয়ানব্বই সাতানব্বই আটানব্বই নিরানব্বই").split()
assert len(_BELOW_100) == 100


def number_words_bn(n):
    """0..999,999,999 in Bangla words (crore / lakh / hazar / sho)."""
    if n < 100:
        return _BELOW_100[n]
    parts = []
    for size, name in ((10_000_000, "কোটি"), (100_000, "লক্ষ"), (1000, "হাজার")):
        if n >= size:
            parts.append(f"{_BELOW_100[n // size]} {name}")
            n %= size
    if n >= 100:
        parts.append(f"{_DIGIT_WORDS[n // 100]}শ")
        n %= 100
    if n:
        parts.append(_BELOW_100[n])
    return " ".join(parts)


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def spell_numbers_bn(text):
    """Replace digits (Latin or Bangla) with Bangla number words: '৫০০ টাকা' -> 'পাঁচশ টাকা'."""
    text = (text or "").translate({ord(b): str(i) for i, b in enumerate(_BN_DIGITS)})

    def spell(m):
        raw = m.group(0).rstrip(",")
        tail = m.group(0)[len(raw):]
        whole, _, frac = raw.replace(",", "").partition(".")
        if (len(whole) > 1 and whole.startswith("0")) or len(whole) > 9:
            words = " ".join(_DIGIT_WORDS[int(d)] for d in whole)  # phone numbers, codes
        else:
            words = number_words_bn(int(whole))
        if frac:
            words += " দশমিক " + " ".join(_DIGIT_WORDS[int(d)] for d in frac)
        return f" {words} {tail}"

    text = _NUMBER.sub(spell, text).replace("%", " শতাংশ ")
    return " ".join(text.split())
