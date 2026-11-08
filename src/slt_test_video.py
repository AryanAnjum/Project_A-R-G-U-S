"""Run sign language videos through the exact live ARGUS pipeline and print the captions.

This answers "does live mode give the same result as the offline test?". Each video goes
through the same SignSession code the server uses (MediaPipe Holistic in VIDEO mode, hand-based
start, resampling to 25 fps, the live checkpoint), fed at the frame rate the webcam uses.

    python src/slt_test_video.py path\\to\\video.mp4 --lang bn
    python src/slt_test_video.py E:\\datasets\\Ban-Sign-Sent-9K-V1\\Sign_Videos --lang bn --limit 20 \\
        --refs runs\\bansign_signer_pose_s0\\predictions_test.tsv

Options:
    --lang de|bn     sign language (default bn)
    --fps 13         frame rate to feed, like the dashboard camera (0 = every frame of the video)
    --refs file.tsv  a predictions_*.tsv from training: its videos are tested and the reference
                     sentence is printed next to the live caption
    --limit N        test at most N videos

The confidence column is what ARGUS_SLT_MIN_CONF compares against. Run a few videos the model
knows (test split) and a few it doesn't (greetings), and set ARGUS_SLT_MIN_CONF between the two.
"""

import argparse
import csv
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import argus_slt  # noqa: E402

VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def find_videos(paths, refs, limit):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                files += [os.path.join(root, n) for n in sorted(names) if n.lower().endswith(VIDEO_EXT)]
        else:
            files.append(p)
    if refs:
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in refs]
    return files[:limit] if limit else files


def load_refs(path):
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return {row["id"]: row["reference"] for row in csv.DictReader(f, delimiter="\t")}


def run_video(engine, path, lang, feed_fps):
    events = []
    session = argus_slt.SignSession(engine, events.append, lang)
    session.close()  # stop its worker thread: frames are fed here, one by one, in order
    session.thread.join(timeout=1.0)
    cap = cv2.VideoCapture(path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = video_fps / feed_fps if feed_fps and video_fps > feed_fps else 1.0
    i, next_keep, t0 = 0, 0.0, 1000.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= next_keep:
            next_keep += step
            session._on_frame(frame, t0 + i / video_fps)
        i += 1
    cap.release()
    if session.signing and session.kp:
        session._finish()  # like pressing "Translate now" at the end of the video
    captions = [e for e in events if e.get("event") == "slt_caption"]
    errors = [e["text"] for e in events if e.get("event") == "slt_error"]
    return captions, errors, i / video_fps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", help="video files or folders")
    ap.add_argument("--lang", default="bn", choices=sorted(argus_slt.LANGUAGES))
    ap.add_argument("--fps", type=float, default=13.0)
    ap.add_argument("--refs", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    lang = argus_slt.LANGUAGES[args.lang]
    if not lang.ready:
        sys.exit(f"No {lang.label} checkpoint found ({lang.checkpoint}).")
    refs = load_refs(args.refs)
    files = find_videos(args.videos, refs, args.limit)
    if not files:
        sys.exit("No videos found." + (" (none of them are listed in --refs)" if refs else ""))
    print(f"{lang.label} model: {lang.checkpoint}")
    print(f"Feeding {args.fps:g} fps; 'not sure' below {argus_slt.MIN_CONFIDENCE:g}% (ARGUS_SLT_MIN_CONF)\n")

    engine = argus_slt.SLTEngine(args.device)
    exact, confs_ok, confs_bad = 0, [], []
    for n, path in enumerate(files, 1):
        start = time.time()
        captions, errors, seconds = run_video(engine, path, args.lang, args.fps)
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"[{n}/{len(files)}] {name} ({seconds:.1f} s video, {time.time() - start:.1f} s to run)")
        for e in errors:
            print(f"    error: {e}")
        if not captions:
            print("    no caption (no hands found, or the signing was shorter than ARGUS_SLT_MIN_SECONDS)")
        for c in captions:
            flag = "  NOT SURE" if c.get("unsure") else ""
            print(f"    live:      {c['guess']}   ({c['confidence']:.0f}% sure{flag}, {c['translate_ms']} ms)")
        ref = refs.get(name)
        if ref is not None:
            print(f"    reference: {ref}")
            if captions:
                hit = captions[-1]["guess"].strip() == ref.strip()
                exact += hit
                (confs_ok if hit else confs_bad).append(captions[-1]["confidence"])
                print(f"    {'MATCH' if hit else 'different'}")
    if refs:
        print(f"\nExact matches: {exact}/{len(files)}")
        if confs_ok:
            print(f"Confidence when right: median {sorted(confs_ok)[len(confs_ok) // 2]:.0f}%")
        if confs_bad:
            print(f"Confidence when wrong: median {sorted(confs_bad)[len(confs_bad) // 2]:.0f}%")


if __name__ == "__main__":
    main()
