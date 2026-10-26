"""Extract MediaPipe Holistic keypoints from sign language videos.

Each input video (or folder of frame images, as in PHOENIX-2014T) becomes one
float16 .npy file of shape (T, 107, 3) under --out, mirroring the input's
relative path. Files that already exist are skipped, so the run can be resumed.

    python -m sign_language.extract_keypoints --src D:/BornilDB/videos --out D:/BornilDB/keypoints --workers 8
    python -m sign_language.extract_keypoints --src D:/PHOENIX-2014-T/features/fullFrame-210x260px --out D:/phoenix_kp --frame-dirs
"""

import argparse
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

from sign_language.keypoints import NUM_POINTS, PARTS, from_holistic

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXT = {".png", ".jpg", ".jpeg"}

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
             "holistic_landmarker/float16/latest/holistic_landmarker.task")

_model_path = None


def ensure_model(path):
    """Download MediaPipe's holistic landmarker model once. The phone app uses the same file."""
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        print(f"[EXTRACT] downloading {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def _init_worker(model_path):
    global _model_path
    _model_path = model_path


def _new_landmarker():
    # VIDEO mode tracks landmarks across frames; timestamps must increase, so
    # each sample gets a fresh landmarker.
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision
    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_model_path),
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def _frames_from_video(path, target_fps):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = fps / target_fps if target_fps and fps > target_fps else 1.0
    next_keep, i = 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= next_keep:
            next_keep += step
            yield frame
        i += 1
    cap.release()


def _frames_from_dir(path):
    for name in sorted(os.listdir(path)):
        if os.path.splitext(name)[1].lower() in IMAGE_EXT:
            frame = cv2.imread(os.path.join(path, name))
            if frame is not None:
                yield frame


def _process(job):
    src, dst, is_dir, target_fps = job
    if os.path.exists(dst):
        return dst, "skip", None
    frames = _frames_from_dir(src) if is_dir else _frames_from_video(src, target_fps)
    import mediapipe as mp
    seq = []
    with _new_landmarker() as landmarker:
        for i, frame in enumerate(frames):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = landmarker.detect_for_video(image, int(i * 1000 / 25))
            seq.append(from_holistic(results))
    if not seq:
        return dst, "empty", None
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp.npy"
    arr = np.stack(seq)
    np.save(tmp, arr.astype(np.float16))
    os.replace(tmp, dst)
    seen = np.abs(arr).sum(-1) > 0
    hands = seen[:, PARTS["lhand"]].any(-1) | seen[:, PARTS["rhand"]].any(-1)
    return dst, "ok", (len(arr), int(hands.sum()))


def find_jobs(src_root, out_root, frame_dirs, target_fps):
    jobs = []
    for root, dirs, files in os.walk(src_root):
        if frame_dirs:
            if any(os.path.splitext(f)[1].lower() in IMAGE_EXT for f in files):
                rel = os.path.relpath(root, src_root)
                jobs.append((root, os.path.join(out_root, rel + ".npy"), True, target_fps))
            continue
        for f in files:
            if os.path.splitext(f)[1].lower() in VIDEO_EXT:
                rel = os.path.splitext(os.path.relpath(os.path.join(root, f), src_root))[0]
                jobs.append((os.path.join(root, f), os.path.join(out_root, rel + ".npy"), False, target_fps))
    return sorted(jobs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder of videos, or of frame folders with --frame-dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame-dirs", action="store_true", help="each leaf folder of images is one sample")
    ap.add_argument("--fps", type=float, default=25.0, help="subsample videos to at most this frame rate (0 = keep all)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--model", default=os.path.join("models", "holistic_landmarker.task"),
                    help="MediaPipe holistic landmarker .task file; downloaded if missing")
    args = ap.parse_args()

    model_path = os.path.abspath(ensure_model(args.model))
    jobs = find_jobs(args.src, args.out, args.frame_dirs, args.fps)
    print(f"[EXTRACT] {len(jobs)} samples, {args.workers} workers, {NUM_POINTS} points per frame")
    done, frames, hand_frames = 0, 0, 0
    with Pool(args.workers, initializer=_init_worker, initargs=(model_path,)) as pool:
        for dst, status, stats in pool.imap_unordered(_process, jobs, chunksize=4):
            done += 1
            if stats:
                frames += stats[0]
                hand_frames += stats[1]
            if status == "empty":
                print(f"[EXTRACT] no frames read: {dst}", file=sys.stderr)
            if done % 100 == 0 or done == len(jobs):
                print(f"[EXTRACT] {done}/{len(jobs)} | at least one hand found in "
                      f"{100 * hand_frames / max(1, frames):.1f}% of new frames")


if __name__ == "__main__":
    main()
