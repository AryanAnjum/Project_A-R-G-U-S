"""Keypoint dataset, augmentation and batching."""

import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from sign_language.keypoints import PARTS, normalize, to_features
from sign_language.manifests import read_manifest


def augment(seq, rng):
    """Random affine, speed change and hand dropout on a normalised (T, V, 3) sequence."""
    present = np.abs(seq).sum(-1) > 0
    angle = math.radians(rng.uniform(-15, 15))
    sx, sy = rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15)
    shear = rng.uniform(-0.15, 0.15)
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    mat = rot @ np.array([[sx, shear], [0, sy]])
    seq = seq.copy()
    seq[..., :2] = seq[..., :2] @ mat.T.astype(np.float32)
    seq[..., :2] += rng.uniform(-0.1, 0.1, size=2).astype(np.float32)

    # speed change by resampling the time axis
    t = len(seq)
    new_t = max(4, int(round(t * rng.uniform(0.8, 1.2))))
    idx = np.clip(np.round(np.linspace(0, t - 1, new_t)).astype(int), 0, t - 1)
    seq, present = seq[idx], present[idx]

    # occasionally hide one hand for a span, as happens when it leaves the frame
    if rng.random() < 0.2:
        part = PARTS["lhand"] if rng.random() < 0.5 else PARTS["rhand"]
        span = rng.integers(1, max(2, len(seq) // 3))
        start = rng.integers(0, max(1, len(seq) - span))
        present[start:start + span, part] = False

    seq[~present] = 0
    return seq


class KeypointDataset(Dataset):
    def __init__(self, manifest, kp_root, split, max_frames=400, train=False, seed=0):
        self.rows = [r for r in read_manifest(manifest) if r["split"] == split]
        self.kp_root = kp_root
        self.max_frames = max_frames
        self.train = train
        self.rng = np.random.default_rng(seed)
        missing = [r for r in self.rows if not os.path.exists(os.path.join(kp_root, r["keypoints"]))]
        if missing:
            print(f"[DATA] {split}: skipping {len(missing)} rows with no keypoint file "
                  f"(first: {missing[0]['keypoints']})")
            self.rows = [r for r in self.rows if r not in missing]
        print(f"[DATA] {split}: {len(self.rows)} samples")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        seq = np.load(os.path.join(self.kp_root, row["keypoints"])).astype(np.float32)
        seq = normalize(seq)
        if self.train:
            seq = augment(seq, self.rng)
        if len(seq) > self.max_frames:
            idx = np.round(np.linspace(0, len(seq) - 1, self.max_frames)).astype(int)
            seq = seq[idx]
        return {"id": row["id"], "features": torch.from_numpy(to_features(seq)), "text": row["text"]}


def collate(batch):
    lengths = torch.tensor([b["features"].shape[0] for b in batch])
    t_max = int(lengths.max())
    v, c = batch[0]["features"].shape[1:]
    feats = torch.zeros(len(batch), t_max, v, c)
    for i, b in enumerate(batch):
        feats[i, :lengths[i]] = b["features"]
    return {"ids": [b["id"] for b in batch], "features": feats, "lengths": lengths,
            "texts": [b["text"] for b in batch]}


def worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % (2 ** 32))
