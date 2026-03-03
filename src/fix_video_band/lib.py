from __future__ import annotations

import re
import subprocess  # noqa: S404
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Band:
    x0: int
    x1: int


@dataclass(frozen=True)
class Shot:
    t0: float
    t1: float


def run_ffmpeg_scene_detect(input_path: str, thresh: float) -> list[float]:
    """
    Returns sorted unique cut timestamps (seconds) from ffmpeg's scene detection.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        input_path,
        "-vf",
        f"select='gt(scene,{thresh})',showinfo",
        "-f",
        "null",
        "-",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)  # noqa: S603
    stderr = p.stderr or ""

    times: list[float] = []
    for line in stderr.splitlines():
        m = re.search(r"pts_time:([0-9]*\.?[0-9]+)", line)
        if m:
            times.append(float(m.group(1)))

    return sorted(set(times))


def build_shots(cut_times: list[float], duration: float, min_shot_len: float) -> list[Shot]:
    """
    Builds shot ranges [t0,t1) from cut timestamps, merging cuts that produce tiny shots.
    """
    cut_times = [t for t in cut_times if 0.0 < t < duration]
    cut_times.sort()

    boundaries = [0.0, *cut_times, duration]

    merged: list[float] = [boundaries[0]]
    for b in boundaries[1:]:
        if b - merged[-1] < min_shot_len:
            continue
        merged.append(b)

    if merged[-1] != duration:
        merged.append(duration)

    shots: list[Shot] = []
    for i in range(len(merged) - 1):
        t0, t1 = merged[i], merged[i + 1]
        if t1 - t0 >= min_shot_len:
            shots.append(Shot(t0=t0, t1=t1))

    return shots


def rolling_median_1d(arr: np.ndarray, k: int) -> np.ndarray:
    assert k % 2 == 1  # noqa: S101
    assert k >= 3  # noqa: PLR2004, S101
    pad = k // 2
    a = np.pad(arr, (pad, pad), mode="reflect")
    windows = np.stack([a[i : i + arr.shape[0]] for i in range(k)], axis=0)
    return np.median(windows, axis=0)  # type:ignore[no-any-return]


def find_band_candidates(  # noqa: PLR0914
    col_means: np.ndarray,
    win: int,
    z_thresh: float,
    band_width: int,
    top_k: int = 8,
) -> list[tuple[Band, float]]:
    """
    Return up to top_k non-overlapping band candidates sorted by score descending.

    Uses per-channel Z-scores combined by channel-wise max, then a matched-filter
    (sliding mean of exactly band_width columns).  Greedy non-maximum suppression
    picks the best non-overlapping peaks above threshold.
    """
    if col_means.ndim == 1:
        col_means = col_means[:, None]

    win = max(3, win | 1)
    W, C = col_means.shape  # noqa: N806

    z_per_channel = np.zeros((W, C), dtype=np.float32)
    for c in range(C):
        ch = col_means[:, c]
        baseline = rolling_median_1d(ch, win)
        dev = np.abs(ch - baseline)
        med = float(np.median(dev))
        mad = float(np.median(np.abs(dev - med))) + 1e-6
        z_per_channel[:, c] = dev / (1.4826 * mad)

    combined = z_per_channel.max(axis=1)
    kernel = np.ones(band_width, dtype=np.float32) / band_width
    windowed = np.convolve(combined, kernel, mode="valid")

    med_w = float(np.median(windowed))
    mad_w = float(np.median(np.abs(windowed - med_w))) + 1e-6
    thresh = med_w + z_thresh * (1.4826 * mad_w)

    candidates: list[tuple[Band, float]] = []
    scores = windowed.astype(np.float64)
    for _ in range(top_k):
        best = int(np.argmax(scores))
        if scores[best] <= thresh:
            break
        candidates.append((Band(best, best + band_width - 1), float(scores[best])))
        # Suppress all windows that overlap with this one
        lo = max(0, best - band_width + 1)
        hi = min(len(scores), best + band_width)
        scores[lo:hi] = -np.inf

    return candidates


def detect_band(
    col_means: np.ndarray,
    win: int,
    z_thresh: float,
    band_width: int,
) -> Band | None:
    """Return the single best band candidate from a column-mean profile, or None."""
    candidates = find_band_candidates(col_means, win, z_thresh, band_width, top_k=1)
    return candidates[0][0] if candidates else None


def detect_band_frame(
    frame_bgr: np.ndarray,
    win: int,
    z_thresh: float,
    band_width: int,
) -> Band | None:
    """Detect the defect band in a single BGR frame."""
    col_means = frame_bgr.astype(np.float32).mean(axis=0)  # (W, 3)
    return detect_band(col_means, win, z_thresh, band_width)


def repair_band_linear(frame_bgr: np.ndarray, band: Band, pad: int) -> np.ndarray:
    """
    Replace pixels in [x0..x1] by row-wise linear interpolation between endpoints outside the band.
    pad controls how far outside the band we sample (helps when near-band pixels are also contaminated).
    """
    out = frame_bgr.copy()
    _, w, _ = out.shape
    x0, x1 = band.x0, band.x1

    left = max(0, x0 - pad)
    right = min(w - 1, x1 + pad)

    xl = max(0, left - 1)
    xr = min(w - 1, right + 1)

    if xl >= x0 or xr <= x1:
        return out

    width = x1 - x0 + 1
    t = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]  # (1, width, 1)

    left_px = out[:, xl : xl + 1, :].astype(np.float32)   # (H,1,3)
    right_px = out[:, xr : xr + 1, :].astype(np.float32)  # (H,1,3)

    fill = (1.0 - t) * left_px + t * right_px
    out[:, x0 : x1 + 1, :] = np.clip(fill, 0, 255).astype(np.uint8)
    return out


def repair_band_inpaint(
    frame_bgr: np.ndarray,
    band: Band,
    radius: int = 3,
) -> np.ndarray:
    """
    Repair the defect band using OpenCV's Telea (Fast Marching Method) inpainting.

    Telea propagates local gradient information inward from the unmasked boundary
    pixels.  This makes it correct for any scene content — edges crossing the band,
    smooth gradients, fine texture — without any assumption about the defect's nature.
    For a narrow 4-pixel mask, the boundary pixels immediately adjacent to the band
    dominate, so a radius of 3 is sufficient.
    """
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    mask[:, band.x0 : band.x1 + 1] = 255
    return cv2.inpaint(frame_bgr, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)


def get_duration_seconds(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    if fps > 0 and frames > 0:
        return float(frames / fps)
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
    msec = cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0.0)
    return float(msec / 1000.0)
