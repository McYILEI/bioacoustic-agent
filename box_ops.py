"""Box-level operations: IoU, validation, merge/NMS, split, refine, quality scoring,
DSP candidate proposal, pixel conversion."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from config import MIN_BAT_CALL_BANDWIDTH_HZ, MIN_BAT_CALL_DURATION_MS
from schemas import (
    BBox,
    BoxQualityScore,
    BoxValidationReport,
    CandidateBox,
    SpectrogramWindowResult,
)


# ==========================================
# IoU
# ==========================================

def calculate_iou(a: BBox, b: BBox) -> float:
    x_left = max(a.t_min, b.t_min)
    x_right = min(a.t_max, b.t_max)
    y_bottom = max(a.f_min, b.f_min)
    y_top = min(a.f_max, b.f_max)
    if x_right <= x_left or y_top <= y_bottom:
        return 0.0
    inter = (x_right - x_left) * (y_top - y_bottom)
    area_a = (a.t_max - a.t_min) * (a.f_max - a.f_min)
    area_b = (b.t_max - b.t_min) * (b.f_max - b.f_min)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _intersection_area(a: BBox, b: BBox) -> float:
    x_left = max(a.t_min, b.t_min)
    x_right = min(a.t_max, b.t_max)
    y_bottom = max(a.f_min, b.f_min)
    y_top = min(a.f_max, b.f_max)
    if x_right <= x_left or y_top <= y_bottom:
        return 0.0
    return (x_right - x_left) * (y_top - y_bottom)


# ==========================================
# Validation
# ==========================================

def validate_boxes(
    boxes: List[BBox],
    duration_seconds: float,
    fmax_hz: float,
    min_duration_ms: float = MIN_BAT_CALL_DURATION_MS,
    min_bandwidth_hz: float = MIN_BAT_CALL_BANDWIDTH_HZ,
) -> BoxValidationReport:
    """Check every box for legal coordinates, minimum duration/bandwidth, and audio bounds.

    Boxes that pass are clamped to [0, duration] x [0, fmax_hz] before returning.
    Boxes that fail are reported in rejected_boxes with a reason string.
    """
    valid: List[BBox] = []
    rejected: List[dict] = []
    warnings: List[str] = []

    for box in boxes:
        reason: Optional[str] = None
        if box.t_min < 0:
            reason = f"t_min={box.t_min:.4f}s is negative"
        elif box.t_max > duration_seconds + 1e-3:
            reason = f"t_max={box.t_max:.4f}s exceeds audio duration {duration_seconds:.4f}s"
        elif box.f_max > fmax_hz + 1.0:
            reason = f"f_max={box.f_max:.0f}Hz exceeds spectrogram fmax {fmax_hz:.0f}Hz"
        elif (box.t_max - box.t_min) * 1000 < min_duration_ms:
            reason = (
                f"Duration {(box.t_max - box.t_min) * 1000:.1f}ms "
                f"< minimum {min_duration_ms}ms"
            )
        elif (box.f_max - box.f_min) < min_bandwidth_hz:
            reason = (
                f"Bandwidth {box.f_max - box.f_min:.0f}Hz "
                f"< minimum {min_bandwidth_hz:.0f}Hz"
            )

        if reason:
            rejected.append({"box": box.model_dump(), "reason": reason})
        else:
            valid.append(
                box.model_copy(update={
                    "t_min": max(0.0, box.t_min),
                    "t_max": min(duration_seconds, box.t_max),
                    "f_min": max(0.0, box.f_min),
                    "f_max": min(fmax_hz, box.f_max),
                })
            )

    if rejected:
        warnings.append(f"{len(rejected)} box(es) rejected — check coordinates or thresholds")
    if not valid and boxes:
        warnings.append("All boxes were rejected; annotation output will be empty")

    return BoxValidationReport(valid_boxes=valid, rejected_boxes=rejected, warnings=warnings)


# ==========================================
# Merge / NMS
# ==========================================

def merge_overlapping_boxes(
    boxes: List[BBox],
    iou_threshold: float = 0.3,
    containment_threshold: float = 0.8,
) -> List[BBox]:
    """NMS-style merge: high-confidence boxes suppress overlapping/contained ones.

    Boxes with IoU >= iou_threshold are merged into their union bounding box.
    Boxes where a smaller box is >= containment_threshold contained inside a larger
    one are also absorbed.
    """
    if not boxes:
        return boxes

    # Sort descending by confidence (None treated as 0.5)
    boxes = sorted(boxes, key=lambda b: (b.confidence or 0.5), reverse=True)
    suppressed: set[int] = set()
    keep: List[BBox] = []

    for i, anchor in enumerate(boxes):
        if i in suppressed:
            continue
        group = [anchor]
        for j in range(i + 1, len(boxes)):
            if j in suppressed:
                continue
            candidate = boxes[j]
            if calculate_iou(anchor, candidate) >= iou_threshold:
                group.append(candidate)
                suppressed.add(j)
            else:
                # Containment check: is candidate mostly inside anchor?
                inter = _intersection_area(anchor, candidate)
                area_c = (candidate.t_max - candidate.t_min) * (candidate.f_max - candidate.f_min)
                if area_c > 0 and inter / area_c >= containment_threshold:
                    suppressed.add(j)

        keep.append(_union_box(group))

    return keep


def _union_box(boxes: List[BBox]) -> BBox:
    if len(boxes) == 1:
        return boxes[0]
    conf = max((b.confidence for b in boxes if b.confidence is not None), default=None)
    return BBox(
        t_min=min(b.t_min for b in boxes),
        t_max=max(b.t_max for b in boxes),
        f_min=min(b.f_min for b in boxes),
        f_max=max(b.f_max for b in boxes),
        label=boxes[0].label,
        confidence=conf,
        source="merged",
    )


# ==========================================
# Refinement
# ==========================================

def refine_box(
    box: BBox,
    spectrogram: SpectrogramWindowResult,
    energy_threshold_db: float = -30.0,
    padding_seconds: float = 0.001,
    padding_hz: float = 500.0,
) -> BBox:
    """Tighten a box to the actual energy boundaries in the spectrogram.

    Finds the outermost STFT frames/bins above energy_threshold_db within the box
    region and returns a tighter copy with small padding preserved.
    Returns the original box unchanged if refinement cannot be applied.
    """
    freqs = spectrogram.freqs
    times = spectrogram.times
    db = spectrogram.db

    t_mask = (times >= box.t_min) & (times <= box.t_max)
    f_mask = (freqs >= box.f_min) & (freqs <= box.f_max)

    if not t_mask.any() or not f_mask.any():
        return box

    region = db[f_mask, :][:, t_mask]
    above = region > energy_threshold_db

    if not above.any():
        return box

    f_indices = np.where(f_mask)[0]
    t_indices = np.where(t_mask)[0]

    active_rows = np.where(above.any(axis=1))[0]
    active_cols = np.where(above.any(axis=0))[0]

    new_f_min = float(freqs[f_indices[active_rows[0]]]) - padding_hz
    new_f_max = float(freqs[f_indices[active_rows[-1]]]) + padding_hz
    new_t_min = float(times[t_indices[active_cols[0]]]) - padding_seconds
    new_t_max = float(times[t_indices[active_cols[-1]]]) + padding_seconds

    # Clamp to spectrogram and audio bounds
    new_t_min = max(new_t_min, 0.0)
    new_t_max = min(new_t_max, spectrogram.duration_seconds)
    new_f_min = max(new_f_min, spectrogram.freq_min_hz)
    new_f_max = min(new_f_max, spectrogram.freq_max_hz)

    if new_t_max <= new_t_min or new_f_max <= new_f_min:
        return box

    return box.model_copy(update={
        "t_min": new_t_min,
        "t_max": new_t_max,
        "f_min": new_f_min,
        "f_max": new_f_max,
        "source": "refined",
    })


# ==========================================
# Split
# ==========================================

def split_merged_calls(
    box: BBox,
    spectrogram: SpectrogramWindowResult,
    min_gap_db: float = 10.0,
    min_split_duration_s: float = 0.003,
) -> List[BBox]:
    """Attempt to split a box containing multiple merged calls by finding energy gaps.

    Detects temporal valleys below (peak - min_gap_db) within the box region.
    Returns the original single-element list if no valid split is found.
    """
    freqs = spectrogram.freqs
    times = spectrogram.times
    db = spectrogram.db

    t_mask = (times >= box.t_min) & (times <= box.t_max)
    f_mask = (freqs >= box.f_min) & (freqs <= box.f_max)

    if not t_mask.any() or not f_mask.any():
        return [box]

    region = db[f_mask, :][:, t_mask]
    t_indices = np.where(t_mask)[0]
    f_indices = np.where(f_mask)[0]

    frame_max = np.max(region, axis=0)
    peak_e = float(np.max(frame_max))
    valley_thresh = peak_e - min_gap_db
    is_gap = frame_max < valley_thresh

    # Find contiguous non-gap segments
    segments: List[Tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    for i, gap in enumerate(is_gap):
        if not gap and not in_seg:
            seg_start = i
            in_seg = True
        elif gap and in_seg:
            segments.append((seg_start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((seg_start, len(is_gap) - 1))

    if len(segments) <= 1:
        return [box]

    result: List[BBox] = []
    for s_start, s_end in segments:
        t_start = float(times[t_indices[s_start]])
        t_end = float(times[t_indices[s_end]])
        if (t_end - t_start) < min_split_duration_s:
            continue

        seg_region = region[:, s_start : s_end + 1]
        seg_peak = float(np.max(frame_max[s_start : s_end + 1]))
        active_rows = np.where(np.max(seg_region, axis=1) > seg_peak - min_gap_db)[0]
        if not active_rows.size:
            continue

        f_min = float(freqs[f_indices[active_rows[0]]])
        f_max = float(freqs[f_indices[active_rows[-1]]])
        if f_max <= f_min:
            continue

        result.append(BBox(
            t_min=t_start, t_max=t_end,
            f_min=f_min, f_max=f_max,
            label=box.label, confidence=box.confidence, source="split",
        ))

    return result if result else [box]


# ==========================================
# Quality scoring
# ==========================================

def score_box_quality(
    boxes: List[BBox],
    spectrogram: SpectrogramWindowResult,
    noise_floor_db: float = -60.0,
) -> List[BoxQualityScore]:
    """Score each box on tightness, SNR, duration, bandwidth, and isolation.

    Scores are in [0, 1]. recommendation is 'keep' (>=0.6), 'review' (>=0.35), 'reject' (<0.35).
    """
    scores: List[BoxQualityScore] = []

    for idx, box in enumerate(boxes):
        dur = box.t_max - box.t_min
        bw = box.f_max - box.f_min

        # Duration score: optimum 5–100ms for bat calls
        if dur < 0.005:
            duration_score = dur / 0.005
        elif dur > 0.1:
            duration_score = max(0.0, 1.0 - (dur - 0.1) / 0.1)
        else:
            duration_score = 1.0

        # Bandwidth score: optimum 1–80kHz
        if bw < 1000:
            bandwidth_score = bw / 1000.0
        elif bw > 80000:
            bandwidth_score = max(0.0, 1.0 - (bw - 80000) / 80000)
        else:
            bandwidth_score = 1.0

        tightness_score = 0.5
        snr_score = 0.5

        t_mask = (spectrogram.times >= box.t_min) & (spectrogram.times <= box.t_max)
        f_mask = (spectrogram.freqs >= box.f_min) & (spectrogram.freqs <= box.f_max)

        if t_mask.any() and f_mask.any():
            region = spectrogram.db[f_mask, :][:, t_mask]
            box_mean_db = float(np.mean(region))
            outside_mean_db = (
                float(np.mean(spectrogram.db[~f_mask, :]))
                if (~f_mask).any()
                else noise_floor_db
            )
            snr = box_mean_db - outside_mean_db
            snr_score = float(np.clip(snr / 20.0, 0.0, 1.0))

            # Tightness: fraction of box cells above noise + 10dB
            active_thresh = noise_floor_db + 10.0
            tightness_score = float(np.mean(region > active_thresh))

        # Isolation: fraction of time frames in the box without energy outside the freq range
        isolation_score = 1.0  # simplified; full version needs adjacent-band check

        overall = float(np.mean([
            tightness_score, isolation_score, snr_score, duration_score, bandwidth_score,
        ]))
        recommendation = "keep" if overall >= 0.6 else ("review" if overall >= 0.35 else "reject")

        scores.append(BoxQualityScore(
            box_index=idx,
            tightness_score=round(tightness_score, 4),
            isolation_score=round(isolation_score, 4),
            snr_score=round(snr_score, 4),
            duration_score=round(duration_score, 4),
            bandwidth_score=round(bandwidth_score, 4),
            overall_score=round(overall, 4),
            recommendation=recommendation,
        ))

    return scores


# ==========================================
# DSP candidate proposal
# ==========================================

def propose_candidate_boxes(
    spectrogram: SpectrogramWindowResult,
    energy_percentile: float = 85.0,
    min_duration_s: float = 0.002,
    min_bandwidth_hz: float = 500.0,
    max_boxes: int = 50,
) -> List[CandidateBox]:
    """Energy-threshold + time-projection candidate detection.

    Finds contiguous active time regions, then extracts the frequency extent
    of each region. Returns raw candidates for the agent to review and refine.
    """
    db = spectrogram.db
    freqs = spectrogram.freqs
    times = spectrogram.times

    threshold = float(np.percentile(db, energy_percentile))
    binary = db > threshold

    # Find contiguous active time slices
    col_active = binary.any(axis=0)
    regions: List[Tuple[int, int]] = []
    in_reg = False
    reg_start = 0
    for i, active in enumerate(col_active):
        if active and not in_reg:
            reg_start = i
            in_reg = True
        elif not active and in_reg:
            regions.append((reg_start, i - 1))
            in_reg = False
    if in_reg:
        regions.append((reg_start, len(col_active) - 1))

    candidates: List[CandidateBox] = []
    for t_start_idx, t_end_idx in regions:
        t_start = float(times[t_start_idx])
        t_end = float(times[t_end_idx])
        if (t_end - t_start) < min_duration_s:
            continue

        col_slice = binary[:, t_start_idx : t_end_idx + 1]
        row_active = col_slice.any(axis=1)
        f_rows = np.where(row_active)[0]
        if not f_rows.size:
            continue

        f_min = float(freqs[f_rows[0]])
        f_max = float(freqs[f_rows[-1]])
        if (f_max - f_min) < min_bandwidth_hz:
            continue

        region_db = db[row_active, :][:, t_start_idx : t_end_idx + 1]
        score = float(np.clip((float(np.mean(region_db)) - threshold) / 20.0, 0.0, 1.0))

        candidates.append(CandidateBox(
            t_min=t_start, t_max=t_end, f_min=f_min, f_max=f_max, score=score,
        ))

        if len(candidates) >= max_boxes:
            break

    return candidates


# ==========================================
# Pixel-to-unit conversion
# ==========================================

def convert_pixels_to_units(
    x_pixel: int,
    y_pixel: int,
    spectrogram: SpectrogramWindowResult,
) -> dict:
    """Convert pixel coordinates (image space) to absolute seconds and Hz.

    Args:
        x_pixel: Horizontal pixel (0 = left = time_min).
        y_pixel: Vertical pixel (0 = top = freq_max, because images are top-down).
        spectrogram: The SpectrogramWindowResult the image came from.
    """
    time_s = spectrogram.time_min_seconds + x_pixel * spectrogram.seconds_per_pixel
    # y=0 is image top = freq_max (imshow origin=lower flips the data but not the pixel coords)
    freq_hz = spectrogram.freq_max_hz - y_pixel * spectrogram.hz_per_pixel

    return {
        "time_seconds": float(np.clip(time_s, spectrogram.time_min_seconds, spectrogram.time_max_seconds)),
        "freq_hz": float(np.clip(freq_hz, spectrogram.freq_min_hz, spectrogram.freq_max_hz)),
    }


# ==========================================
# Call type classification (heuristic)
# ==========================================

def classify_call_type(
    box: BBox,
    spectrogram: SpectrogramWindowResult,
) -> dict:
    """Heuristic FM/CF/QCF classification based on spectral shape.

    FM: high rate of frequency change relative to bandwidth.
    CF: most energy in a narrow frequency band over most of the duration.
    QCF: hybrid — relatively flat call with FM tails.

    Returns a dict with 'call_type' and 'confidence'.
    """
    t_mask = (spectrogram.times >= box.t_min) & (spectrogram.times <= box.t_max)
    f_mask = (spectrogram.freqs >= box.f_min) & (spectrogram.freqs <= box.f_max)

    if not t_mask.any() or not f_mask.any():
        return {"call_type": "unknown", "confidence": 0.0}

    region = spectrogram.db[f_mask, :][:, t_mask]
    freqs_in = spectrogram.freqs[f_mask]

    # Dominant frequency per time frame
    dominant_row = np.argmax(region, axis=0)
    dominant_freq = freqs_in[dominant_row]

    if len(dominant_freq) < 2:
        return {"call_type": "unknown", "confidence": 0.0}

    freq_range = float(dominant_freq.max() - dominant_freq.min())
    total_bw = box.f_max - box.f_min
    freq_sweep_ratio = freq_range / total_bw if total_bw > 0 else 0.0

    # CF: dominant frequency barely moves
    # FM: dominant frequency sweeps strongly
    # QCF: mid-range sweep ratio
    if freq_sweep_ratio > 0.6:
        call_type = "FM"
        conf = min(1.0, freq_sweep_ratio)
    elif freq_sweep_ratio < 0.2:
        call_type = "CF"
        conf = min(1.0, 1.0 - freq_sweep_ratio * 3)
    else:
        call_type = "QCF"
        conf = 0.6

    return {"call_type": call_type, "confidence": round(conf, 3), "freq_sweep_ratio": round(freq_sweep_ratio, 3)}
