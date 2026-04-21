"""Audio pre-analysis: health inspection, noise floor, active region detection."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from audio_io import read_audio
from config import (
    ACTIVE_REGION_ENERGY_PERCENTILE,
    CLIPPING_THRESHOLD,
    KNOWN_BAT_SAMPLE_RATES,
    MIN_AUDIO_DURATION_SECONDS,
    NOISE_FLOOR_PERCENTILE,
    SILENCE_THRESHOLD_RMS,
)
from dsp import compute_spectrogram
from schemas import ActiveRegion, AudioHealthReport, SpectrogramConfig


def inspect_audio(audio_path: str | Path) -> AudioHealthReport:
    """Run a health check on an audio file before any spectrogram work.

    Checks duration, sample rate, amplitude, clipping, silence, and recommends
    a window strategy. Should be called before starting the annotation pipeline.
    """
    y, sr, channels = read_audio(audio_path)
    duration = len(y) / sr
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y ** 2)))
    clipping_ratio = float(np.mean(np.abs(y) >= CLIPPING_THRESHOLD))
    silence_ratio = float(np.mean(np.abs(y) < SILENCE_THRESHOLD_RMS))

    warnings: List[str] = []
    if duration < MIN_AUDIO_DURATION_SECONDS:
        warnings.append(f"Audio too short: {duration:.4f}s (min {MIN_AUDIO_DURATION_SECONDS}s)")
    if rms < SILENCE_THRESHOLD_RMS:
        warnings.append(f"Near-silent: RMS={rms:.6f} below threshold {SILENCE_THRESHOLD_RMS}")
    if clipping_ratio > 0.001:
        warnings.append(f"Clipping: {clipping_ratio * 100:.2f}% of samples at saturation")
    if sr not in KNOWN_BAT_SAMPLE_RATES:
        warnings.append(f"Unusual sample rate {sr}Hz — expected one of {sorted(KNOWN_BAT_SAMPLE_RATES)}")
    if channels > 1:
        warnings.append(f"Multi-channel audio ({channels}ch) downmixed to mono")

    # Suggest fmax at Nyquist
    recommended_fmax = float(sr) / 2.0

    # Suggest window strategy based on duration
    if duration < 0.1:
        strategy = "single_overview_only"
    elif duration < 1.0:
        strategy = "overview_plus_fine_scan"
    else:
        strategy = "overview_plus_sliding_scan"

    return AudioHealthReport(
        duration_seconds=duration,
        sample_rate=sr,
        channels=channels,
        num_samples=len(y),
        peak_amplitude=peak,
        rms_amplitude=rms,
        is_clipping=clipping_ratio > 0.001,
        is_near_silent=rms < SILENCE_THRESHOLD_RMS,
        is_too_short=duration < MIN_AUDIO_DURATION_SECONDS,
        has_abnormal_sample_rate=sr not in KNOWN_BAT_SAMPLE_RATES,
        clipping_ratio=clipping_ratio,
        silence_ratio=silence_ratio,
        recommended_fmax_hz=recommended_fmax,
        warnings=warnings,
        suggested_window_strategy=strategy,
    )


def estimate_noise_floor(
    y: np.ndarray,
    sr: int,
    config: SpectrogramConfig,
) -> dict:
    """Estimate background noise level from the spectrogram.

    Returns a dict with noise_floor_db, dynamic_range_db, and a list of
    persistently noisy frequency bands (potential false-positive sources).
    """
    freqs, times, db = compute_spectrogram(y, sr, config)

    noise_floor_db = float(np.percentile(db, NOISE_FLOOR_PERCENTILE))
    mean_db = float(np.mean(db))
    peak_db = float(np.max(db))
    dynamic_range_db = peak_db - noise_floor_db

    # Frequencies where the time-averaged energy is persistently above noise + 6dB
    freq_mean_db = np.mean(db, axis=1)
    noisy_mask = freq_mean_db > (noise_floor_db + 6.0)
    noisy_freqs = freqs[noisy_mask].tolist()[:20]  # cap for readability

    return {
        "noise_floor_db": round(noise_floor_db, 2),
        "mean_db": round(mean_db, 2),
        "peak_db": round(peak_db, 2),
        "dynamic_range_db": round(dynamic_range_db, 2),
        "noisy_frequency_bands_hz": noisy_freqs,
        "recommended_detection_threshold_db": round(noise_floor_db + 10.0, 2),
    }


def detect_active_regions(
    y: np.ndarray,
    sr: int,
    config: SpectrogramConfig,
    min_duration_s: float = 0.002,
    merge_gap_s: float = 0.010,
) -> List[ActiveRegion]:
    """Detect time regions with significant acoustic activity.

    Strategy: per-frame max energy in the target frequency band, adaptive threshold,
    contiguous region extraction, then gap-based merging.

    Returns a list of ActiveRegion sorted by start time.
    """
    freqs, times, db = compute_spectrogram(y, sr, config)

    # Focus on bat-call frequencies (>15kHz if available, else full band)
    bat_mask = freqs >= max(config.fmin_hz, 15000.0)
    if not bat_mask.any():
        bat_mask = np.ones(len(freqs), dtype=bool)
    db_bat = db[bat_mask, :]

    frame_energy = np.max(db_bat, axis=0)
    threshold = float(np.percentile(frame_energy, ACTIVE_REGION_ENERGY_PERCENTILE)) + 3.0
    active_mask = frame_energy > threshold

    bat_freqs = freqs[bat_mask]

    regions: List[ActiveRegion] = []
    in_region = False
    seg_start = 0

    for i, is_active in enumerate(active_mask):
        if is_active and not in_region:
            seg_start = i
            in_region = True
        elif not is_active and in_region:
            _maybe_add_region(
                regions, times, frame_energy, db_bat, bat_freqs,
                seg_start, i - 1, threshold, min_duration_s,
            )
            in_region = False

    if in_region and len(times) > 0:
        _maybe_add_region(
            regions, times, frame_energy, db_bat, bat_freqs,
            seg_start, len(active_mask) - 1, threshold, min_duration_s,
        )

    return _merge_close_regions(sorted(regions, key=lambda r: r.start), merge_gap_s)


def _maybe_add_region(
    regions: List[ActiveRegion],
    times: np.ndarray,
    frame_energy: np.ndarray,
    db_bat: np.ndarray,
    bat_freqs: np.ndarray,
    seg_start: int,
    seg_end: int,
    threshold: float,
    min_duration_s: float,
) -> None:
    t_start = float(times[seg_start])
    t_end = float(times[seg_end])
    if (t_end - t_start) < min_duration_s:
        return

    seg_slice = db_bat[:, seg_start : seg_end + 1]
    peak_freq_row = int(np.argmax(np.max(seg_slice, axis=1)))
    peak_freq_hz = float(bat_freqs[peak_freq_row]) if bat_freqs.size > 0 else None

    max_energy = float(np.max(frame_energy[seg_start : seg_end + 1]))
    score = float(np.clip((max_energy - threshold) / 20.0, 0.0, 1.0))

    regions.append(ActiveRegion(start=t_start, end=t_end, score=score, peak_freq_hz=peak_freq_hz))


def _merge_close_regions(regions: List[ActiveRegion], gap_s: float) -> List[ActiveRegion]:
    if not regions:
        return regions
    merged = [regions[0]]
    for r in regions[1:]:
        last = merged[-1]
        if r.start - last.end <= gap_s:
            merged[-1] = ActiveRegion(
                start=last.start,
                end=max(last.end, r.end),
                score=max(last.score, r.score),
                peak_freq_hz=last.peak_freq_hz or r.peak_freq_hz,
            )
        else:
            merged.append(r)
    return merged
