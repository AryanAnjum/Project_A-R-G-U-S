"""Extract I3D video features for RGB + pose training.

SignFormer-GCN feeds I3D features of 16-frame clips into its encoder. This does
the same with the public BSL-1K I3D checkpoint (trained on British Sign
Language, no PHOENIX glosses), so the comparison stays gloss-free:

    https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/pretrained_models/bsl1k.pth.tar

Each sample becomes a float16 .npy of shape (clips, 1024) under --out, one clip
every --stride frames, mirroring the input's relative path like
extract_keypoints.py. Existing files are skipped, so the run can be resumed.

    python -m sign_language.extract_i3d --src D:/PHOENIX-2014-T/features/fullFrame-210x260px \
        --out D:/phoenix_i3d --frame-dirs --checkpoint models/bsl1k.pth.tar
"""

import argparse
import os
import sys
import threading
import queue

import cv2
import numpy as np
import torch

from sign_language.extract_keypoints import IMAGE_EXT, find_jobs
from sign_language.i3d import InceptionI3d

RESIZE, CROP = 256, 224


def load_i3d(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    num_classes = state["logits.conv3d.weight"].shape[0]
    model = InceptionI3d(num_classes=num_classes, num_in_frames=16, include_embds=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def read_frames(src, is_dir):
    """All frames as uint8 (T, 224, 224, 3) RGB, resized to 256x256 and centre-cropped as in BSL-1K."""
    if is_dir:
        names = sorted(n for n in os.listdir(src) if os.path.splitext(n)[1].lower() in IMAGE_EXT)
        frames = (cv2.imread(os.path.join(src, n)) for n in names)
    else:
        cap = cv2.VideoCapture(src)
        frames = iter(lambda: cap.read()[1], None)
    out, o = [], (RESIZE - CROP) // 2
    for f in frames:
        if f is None:
            continue
        f = cv2.resize(f, (RESIZE, RESIZE), interpolation=cv2.INTER_LINEAR)
        out.append(cv2.cvtColor(f[o:o + CROP, o:o + CROP], cv2.COLOR_BGR2RGB))
    return np.stack(out) if out else None


def clip_starts(n, clip, stride):
    if n <= clip:
        return [0]
    starts = list(range(0, n - clip + 1, stride))
    if starts[-1] != n - clip:
        starts.append(n - clip)
    return starts


@torch.no_grad()
def features(model, frames, device, clip=16, stride=4, batch=32):
    video = torch.from_numpy(frames).to(device)  # (T, H, W, 3) uint8
    if len(video) < clip:  # repeat the last frame so short samples still fill one clip
        video = torch.cat([video, video[-1:].expand(clip - len(video), -1, -1, -1)])
    video = video.permute(3, 0, 1, 2).float().div_(255).sub_(0.5)  # (3, T, H, W), BSL-1K normalisation
    starts = clip_starts(video.shape[1], clip, stride)
    out = []
    for i in range(0, len(starts), batch):
        x = torch.stack([video[:, s:s + clip] for s in starts[i:i + batch]])
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            emb = model(x)["embds"]
        out.append(emb.flatten(1).float().cpu())
    return torch.cat(out).numpy().astype(np.float16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder of videos, or of frame folders with --frame-dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame-dirs", action="store_true", help="each leaf folder of images is one sample")
    ap.add_argument("--checkpoint", default=os.path.join("models", "bsl1k.pth.tar"))
    ap.add_argument("--stride", type=int, default=4, help="frames between clip starts; 4 matches the pose encoder")
    ap.add_argument("--batch", type=int, default=32, help="clips per GPU batch")
    ap.add_argument("--readers", type=int, default=4, help="threads decoding frames")
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"[I3D] checkpoint not found: {args.checkpoint}\n"
                 "Download bsl1k.pth.tar from the link at the top of this file.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_i3d(args.checkpoint, device)
    jobs = [j for j in find_jobs(args.src, args.out, args.frame_dirs, 0) if not os.path.exists(j[1])]
    print(f"[I3D] {len(jobs)} samples to do on {device}")

    # decode frames on CPU threads while the GPU runs I3D
    todo, ready = queue.Queue(), queue.Queue(maxsize=8)
    for job in jobs:
        todo.put(job)

    def reader():
        while True:
            try:
                src, dst, is_dir, _ = todo.get_nowait()
            except queue.Empty:
                ready.put(None)
                return
            ready.put((dst, read_frames(src, is_dir)))

    for _ in range(args.readers):
        threading.Thread(target=reader, daemon=True).start()

    done, finished = 0, 0
    while finished < args.readers:
        item = ready.get()
        if item is None:
            finished += 1
            continue
        dst, frames = item
        done += 1
        if frames is None:
            print(f"[I3D] no frames read: {dst}", file=sys.stderr)
            continue
        feats = features(model, frames, device, stride=args.stride, batch=args.batch)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".tmp.npy"
        np.save(tmp, feats)
        os.replace(tmp, dst)
        if done % 100 == 0 or done == len(jobs):
            print(f"[I3D] {done}/{len(jobs)}")


if __name__ == "__main__":
    main()
