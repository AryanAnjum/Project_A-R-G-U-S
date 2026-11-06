"""Collect every model A.R.G.U.S. needs into one data folder, so the installer works offline.

build_installer.ps1 runs this with the bundled Python. It can also be run by hand:

    python installer/collect_models.py --data installer/build/ARGUS/data

Layout written under --data (the same layout the app reads through ARGUS_HOME):

    argus_phased_adapters/part_5_adapter/   your Qwen2-VL LoRA fine-tune (copied)
    runs/<name>/best/                       sign language checkpoints (copied)
    models/piper/*.onnx                     offline voices (downloaded)
    models/holistic_landmarker.task         MediaPipe holistic model (downloaded)
    models/bsl1k.pth.tar                    I3D video features for sign language (copied or downloaded)
    hf/<model>/                             Qwen2-VL-2B, Whisper small.en, mBART-50 tokenizer (plain folders)
    .argus_data                             marker the desktop app looks for
"""

import argparse
import json
import os
import shutil
import sys
import urllib.request

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

PIPER_VOICES = ["en_US-ryan-medium", "de_DE-thorsten-medium"]
PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
HOLISTIC_URL = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
                "holistic_landmarker/float16/latest/holistic_landmarker.task")
BSL1K_URL = "https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/pretrained_models/bsl1k.pth.tar"

# Hugging Face repos saved as plain folders (no cache symlinks, so the installer never doubles them).
# The desktop app points the matching setting at each folder. None = whole repo.
HF_REPOS = [
    ("ARGUS_BASE_MODEL", "Qwen/Qwen2-VL-2B-Instruct", None),
    ("ARGUS_WHISPER_MODEL", "Systran/faster-whisper-small.en", None),
    # Sign language decoder only needs the mBART-50 tokenizer and config, not its weights.
    ("ARGUS_SLT_MBART", "facebook/mbart-large-50", ["*.json", "*.model", "*.txt"]),
]

# Sign language checkpoints copied when present (same order as src/argus_slt.py CHECKPOINTS).
SLT_CHECKPOINTS = ["ablation_pose_only", "ablation_small_decoder_s1",
                   "bansign_signer_pose_s0", "bansign_signer_s0", "bansign_sentence_pose_s0", "bansign_sentence_s0"]


def step(msg):
    print(f"[BUILD] {msg}", flush=True)


def copy_dir(src, dst, required=True):
    if not os.path.isdir(src):
        if required:
            sys.exit(f"[BUILD] Missing folder: {src}")
        step(f"Skipping {src} (not found)")
        return False
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    step(f"Copied {src} -> {dst}")
    return True


def download(url, path):
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    step(f"Downloading {url}")
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)


def piper_url(name):
    locale, speaker, quality = name.split("-", 2)
    return f"{PIPER_BASE}/{locale.split('_')[0]}/{locale}/{speaker}/{quality}/{name}.onnx"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="output data folder")
    ap.add_argument("--adapter", default=os.path.join(REPO_ROOT, "argus_phased_adapters", "part_5_adapter"))
    ap.add_argument("--runs", default=os.path.join(REPO_ROOT, "runs"), help="folder with sign language runs")
    ap.add_argument("--i3d", default=os.path.join(REPO_ROOT, "models", "bsl1k.pth.tar"))
    args = ap.parse_args()

    data = os.path.abspath(args.data)
    os.makedirs(data, exist_ok=True)

    # 1. The fine-tune: must be a PEFT adapter folder with its config.
    if not os.path.isfile(os.path.join(args.adapter, "adapter_config.json")):
        sys.exit(f"[BUILD] {args.adapter} has no adapter_config.json. Pass --adapter <your part_5_adapter folder>.")
    copy_dir(args.adapter, os.path.join(data, "argus_phased_adapters", "part_5_adapter"))

    # 2. Sign language checkpoints (only the best/ folder is needed at runtime).
    have_slt = False
    for name in SLT_CHECKPOINTS:
        src = os.path.join(args.runs, name, "best")
        if os.path.isfile(os.path.join(src, "model.pt")):
            have_slt |= copy_dir(src, os.path.join(data, "runs", name, "best"))
        else:
            step(f"No sign language checkpoint at {src} (that language stays off)")
    if not have_slt:
        step("WARNING: no sign language checkpoint found, Sign mode will be disabled in the installed app.")

    # 3. I3D features model for sign language.
    i3d_dst = os.path.join(data, "models", "bsl1k.pth.tar")
    if os.path.isfile(args.i3d):
        os.makedirs(os.path.dirname(i3d_dst), exist_ok=True)
        shutil.copy2(args.i3d, i3d_dst)
        step(f"Copied {args.i3d}")
    else:
        download(BSL1K_URL, i3d_dst)

    # 4. Voices and MediaPipe.
    for name in PIPER_VOICES:
        base = os.path.join(data, "models", "piper", name + ".onnx")
        download(piper_url(name) + ".json", base + ".json")
        download(piper_url(name), base)
    download(HOLISTIC_URL, os.path.join(data, "models", "holistic_landmarker.task"))

    # 5. Hugging Face models as plain folders inside the data folder.
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import snapshot_download
    settings = {}
    for env, repo, patterns in HF_REPOS:
        folder = os.path.join("hf", repo.split("/")[-1])
        step(f"Fetching {repo} from Hugging Face")
        snapshot_download(repo, allow_patterns=patterns, local_dir=os.path.join(data, folder))
        shutil.rmtree(os.path.join(data, folder, ".cache"), ignore_errors=True)
        settings[env] = folder

    os.makedirs(os.path.join(data, "logs"), exist_ok=True)
    with open(os.path.join(data, ".argus_data"), "w", encoding="utf-8") as f:
        # Settings are paths relative to the data folder; argus_desktop.py turns them into env vars.
        json.dump({"settings": settings, "piper": PIPER_VOICES}, f, indent=2)

    total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(data) for f in fs)
    step(f"Models collected in {data} ({total / 1024 ** 3:.1f} GB)")


if __name__ == "__main__":
    main()
