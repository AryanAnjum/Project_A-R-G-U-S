"""Keypoint layout shared by offline extraction, training and the live ARGUS mode.

Every sequence is stored as a float16 array of shape (T, V, 3) holding MediaPipe
(x, y, z) coordinates, with V = 107 points in this order:

    body  13  upper-body pose landmarks
    lhand 21  left hand
    rhand 21  right hand
    face  52  lips, eyebrows, nose tip and chin from the face mesh

A part that MediaPipe did not detect in a frame is stored as zeros.
SignFormer-GCN used only the 33 pose landmarks; the hands and mouth are the
main addition here.
"""

import numpy as np

# MediaPipe pose indices: nose, eyes, ears, shoulders, elbows, wrists, hips.
POSE_IDX = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]

# Face mesh indices. Outer and inner lips are listed lower arc (left to right)
# then upper arc (left to right), so each ring can be closed in the graph.
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
              185, 40, 39, 37, 0, 267, 269, 270, 409]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
              191, 80, 81, 82, 13, 312, 311, 310, 415]
BROW_LEFT = [70, 63, 105, 66, 107]
BROW_RIGHT = [336, 296, 334, 293, 300]
FACE_IDX = LIPS_OUTER + LIPS_INNER + BROW_LEFT + BROW_RIGHT + [1, 152]

N_BODY, N_HAND, N_FACE = len(POSE_IDX), 21, len(FACE_IDX)
PARTS = {
    "body": slice(0, N_BODY),
    "lhand": slice(N_BODY, N_BODY + N_HAND),
    "rhand": slice(N_BODY + N_HAND, N_BODY + 2 * N_HAND),
    "face": slice(N_BODY + 2 * N_HAND, N_BODY + 2 * N_HAND + N_FACE),
}
NUM_POINTS = N_BODY + 2 * N_HAND + N_FACE

# Local indices inside each part, used to build the skeleton graphs.
BODY_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), (5, 7), (7, 9),
              (6, 8), (8, 10), (5, 11), (6, 12), (11, 12)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9),
              (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
              (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def _lip_ring(offset):
    # lower arc 0..10 left to right, upper arc 11..19 left to right
    edges = [(offset + i, offset + i + 1) for i in range(10)]
    edges += [(offset + 10, offset + 19), (offset + 11, offset)]
    edges += [(offset + i, offset + i + 1) for i in range(11, 19)]
    return edges


FACE_EDGES = (_lip_ring(0) + _lip_ring(20)
              + [(40 + i, 41 + i) for i in range(4)]
              + [(45 + i, 46 + i) for i in range(4)]
              + [(50, 15), (50, 44), (50, 45), (51, 5)])
PART_EDGES = {"body": BODY_EDGES, "lhand": HAND_EDGES, "rhand": HAND_EDGES, "face": FACE_EDGES}

# Root joint of each part, used for the part-relative channels.
PART_ROOT = {"body": 0, "lhand": 0, "rhand": 0, "face": 50}
# Shoulders inside the body part, used to normalise position and scale.
L_SHOULDER, R_SHOULDER = 5, 6


def _landmarks(lms, idx=None, n=None):
    # Accepts MediaPipe Tasks results (plain lists) and legacy solution results (.landmark).
    pts = getattr(lms, "landmark", lms)
    if not pts:
        return np.zeros((n if idx is None else len(idx), 3), np.float32)
    if idx is None:
        return np.array([[p.x, p.y, p.z] for p in pts], np.float32)
    return np.array([[pts[i].x, pts[i].y, pts[i].z] for i in idx], np.float32)


def from_holistic(results):
    """Convert one MediaPipe Holistic result (Tasks or legacy API) into a (V, 3) array."""
    return np.concatenate([
        _landmarks(results.pose_landmarks, POSE_IDX),
        _landmarks(results.left_hand_landmarks, n=N_HAND),
        _landmarks(results.right_hand_landmarks, n=N_HAND),
        _landmarks(results.face_landmarks, FACE_IDX),
    ])


def from_lists(pose, left_hand, right_hand, face):
    """Build a (V, 3) frame from plain landmark lists, as sent by the phone.

    pose has 33 points, hands 21 each, face 468 or 478; any may be empty.
    """
    def pick(points, idx, n):
        if not points:
            return np.zeros((n, 3), np.float32)
        arr = np.asarray(points, np.float32).reshape(-1, 3)
        return arr if idx is None else arr[idx]
    return np.concatenate([
        pick(pose, POSE_IDX, N_BODY),
        pick(left_hand, None, N_HAND),
        pick(right_hand, None, N_HAND),
        pick(face, FACE_IDX, N_FACE),
    ])


def normalize(seq):
    """Centre on the shoulders and scale by shoulder width, per sequence.

    seq: (T, V, 3). Missing points (all zero) stay zero.
    """
    seq = seq.astype(np.float32)
    present = np.abs(seq).sum(-1) > 0
    body = seq[:, PARTS["body"]]
    ok = present[:, L_SHOULDER] & present[:, R_SHOULDER]
    if ok.any():
        ls, rs = body[ok, L_SHOULDER], body[ok, R_SHOULDER]
        center = ((ls + rs) / 2).mean(0)
        scale = np.linalg.norm((ls - rs)[:, :2], axis=-1).mean()
    else:
        pts = seq[present]
        center = pts.mean(0) if len(pts) else np.zeros(3, np.float32)
        scale = pts[:, :2].std() * 4 if len(pts) else 1.0
    scale = max(float(scale), 1e-3)
    out = (seq - center) / scale
    out[~present] = 0
    return out


def to_features(seq):
    """(T, V, 3) normalised keypoints -> (T, V, 9) model input.

    Channels: position, frame-to-frame motion, position relative to the part's
    root joint (wrist for hands), which carries handshape independent of where
    the hand is.
    """
    present = (np.abs(seq).sum(-1, keepdims=True) > 0).astype(np.float32)
    motion = np.zeros_like(seq)
    motion[1:] = (seq[1:] - seq[:-1]) * present[1:] * present[:-1]
    local = np.zeros_like(seq)
    for name, sl in PARTS.items():
        part = seq[:, sl]
        root = part[:, PART_ROOT[name]:PART_ROOT[name] + 1]
        local[:, sl] = (part - root) * present[:, sl]
    return np.concatenate([seq, motion, local], -1)
