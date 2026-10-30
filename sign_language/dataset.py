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


def resample(feats, length):
    """Linearly resample (N, D) features along time to `length` steps."""
    if len(feats) == length:
        return feats
    pos = np.linspace(0, len(feats) - 1, length)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(feats) - 1)
    w = (pos - lo)[:, None].astype(np.float32)
    return feats[lo] * (1 - w) + feats[hi] * w


class KeypointDataset(Dataset):
    """Keypoint sequences, plus I3D clip features when rgb_root is given."""

    def __init__(self, manifest, kp_root, split, max_frames=400, train=False, seed=0, rgb_root=None,
                 time_mask=0.0):
        self.rows = [r for r in read_manifest(manifest) if r["split"] == split]
        self.kp_root = kp_root
        self.rgb_root = rgb_root
        self.time_mask = time_mask
        self.max_frames = max_frames
        self.train = train
        self.rng = np.random.default_rng(seed)
        missing = [r for r in self.rows if not os.path.exists(os.path.join(kp_root, r["keypoints"]))
                   or rgb_root and not os.path.exists(os.path.join(rgb_root, r["keypoints"]))]
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
        item = {"id": row["id"], "features": torch.from_numpy(to_features(seq)), "text": row["text"]}
        if self.rgb_root:
            from sign_language.model import PoseEncoder
            # one I3D clip per 4 frames, stretched to the pose encoder's output length
            rgb = np.load(os.path.join(self.rgb_root, row["keypoints"])).astype(np.float32)
            n = int(PoseEncoder.out_lengths(torch.tensor(len(seq))))
            item["rgb"] = torch.from_numpy(resample(rgb, n))
        if self.train and self.time_mask > 0:
            self._mask_time(item)
        return item

    def _mask_time(self, item):
        """Blank two random spans, together about time_mask of the sample, in every input stream.

        The model then has to fill gaps from context instead of memorising whole
        training sequences; dev loss rose after about 30 epochs without it.
        """
        t = item["features"].shape[0]
        for _ in range(2):
            span = int(self.rng.integers(0, max(1, int(t * self.time_mask / 2)) + 1))
            if span == 0:
                continue
            start = int(self.rng.integers(0, t - span + 1))
            item["features"][start:start + span] = 0
            if "rgb" in item:
                n = item["rgb"].shape[0]
                item["rgb"][start * n // t:-(-(start + span) * n // t)] = 0


def collate(batch):
    lengths = torch.tensor([b["features"].shape[0] for b in batch])
    t_max = int(lengths.max())
    v, c = batch[0]["features"].shape[1:]
    feats = torch.zeros(len(batch), t_max, v, c)
    for i, b in enumerate(batch):
        feats[i, :lengths[i]] = b["features"]
    out = {"ids": [b["id"] for b in batch], "features": feats, "lengths": lengths,
           "texts": [b["text"] for b in batch]}
    if "rgb" in batch[0]:
        n = max(b["rgb"].shape[0] for b in batch)
        rgb = torch.zeros(len(batch), n, batch[0]["rgb"].shape[1])
        for i, b in enumerate(batch):
            rgb[i, :len(b["rgb"])] = b["rgb"]
        out["rgb"] = rgb
    return out


def worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % (2 ** 32))
