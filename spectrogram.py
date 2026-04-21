"""Spectrogram image generation and bounding-box overlay rendering."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audio_io import read_audio
from dsp import compute_spectrogram
from schemas import BBox, SpectrogramConfig, SpectrogramWindowResult


def generate_spectrogram_window(
    audio_path: str | Path,
    output_image_path: str | Path,
    config: SpectrogramConfig,
    window_start: float = 0.0,
    window_end: Optional[float] = None,
    title: str = "Bat Audio Spectrogram",
) -> SpectrogramWindowResult:
    """Generate a spectrogram PNG for a time window and return metadata + pixel mapping.

    The returned SpectrogramWindowResult carries the raw db array and pixel-to-unit
    mapping so downstream tools can operate on the same data without re-reading the file.
    """
    y, sr, _ = read_audio(audio_path)
    duration_seconds = len(y) / sr

    if window_start < 0:
        raise ValueError("window_start must be >= 0")
    if window_end is None:
        window_end = duration_seconds
    if window_end <= window_start:
        raise ValueError("window_end must be > window_start")
    if window_start >= duration_seconds:
        raise ValueError("window_start is beyond audio duration")

    clipped_end = min(window_end, duration_seconds)
    start_sample = int(round(window_start * sr))
    end_sample = int(round(clipped_end * sr))
    window_y = y[start_sample:end_sample]
    if len(window_y) == 0:
        raise ValueError("Selected audio window is empty after sampling.")

    freqs, local_times, db = compute_spectrogram(window_y, sr, config)
    absolute_times = local_times + window_start

    output_image_path = Path(output_image_path)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=config.figsize)
    extent = [window_start, clipped_end, float(freqs.min()), float(freqs.max())]
    im = ax.imshow(
        db,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=config.cmap,
        interpolation="nearest",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Amplitude (dB)")
    plt.tight_layout()
    fig.savefig(output_image_path, dpi=config.dpi)
    plt.close(fig)

    # Pixel-to-unit mapping (approximate; ignores matplotlib margins)
    w_px = int(config.figsize[0] * config.dpi)
    h_px = int(config.figsize[1] * config.dpi)
    time_span = clipped_end - window_start
    freq_span = float(freqs.max()) - float(freqs.min())

    return SpectrogramWindowResult(
        image_path=output_image_path,
        audio_path=Path(audio_path),
        duration_seconds=duration_seconds,
        window_start_seconds=window_start,
        window_end_seconds=clipped_end,
        window_duration_seconds=clipped_end - window_start,
        time_min_seconds=window_start,
        time_max_seconds=clipped_end,
        freq_min_hz=float(freqs.min()),
        freq_max_hz=float(freqs.max()),
        sample_rate=sr,
        times=absolute_times,
        freqs=freqs,
        db=db,
        config=config,
        seconds_per_pixel=time_span / w_px if w_px > 0 else 0.0,
        hz_per_pixel=freq_span / h_px if h_px > 0 else 0.0,
        image_width_px=w_px,
        image_height_px=h_px,
    )


def draw_bboxes_on_spectrogram(
    spectrogram: SpectrogramWindowResult,
    boxes: Iterable[BBox],
    output_image_path: str | Path,
) -> Path:
    """Render annotation overlay: high-quality boxes on lime, low-quality on orange."""
    output_image_path = Path(output_image_path)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=spectrogram.config.figsize)
    extent = [
        spectrogram.time_min_seconds,
        spectrogram.time_max_seconds,
        spectrogram.freq_min_hz,
        spectrogram.freq_max_hz,
    ]
    ax.imshow(
        spectrogram.db,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=spectrogram.config.cmap,
        interpolation="nearest",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Predicted Bounding Boxes")

    for box in boxes:
        quality = box.quality_score if box.quality_score is not None else 1.0
        color = "lime" if quality >= 0.6 else ("orange" if quality >= 0.35 else "red")
        rect = plt.Rectangle(
            (box.t_min, box.f_min),
            box.t_max - box.t_min,
            box.f_max - box.f_min,
            fill=False,
            linewidth=2,
            edgecolor=color,
        )
        ax.add_patch(rect)
        label_text = box.label
        if box.confidence is not None:
            label_text += f" {box.confidence:.2f}"
        ax.text(
            box.t_min, box.f_max, label_text,
            fontsize=7, va="bottom", color="white",
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )

    plt.tight_layout()
    fig.savefig(output_image_path, dpi=spectrogram.config.dpi)
    plt.close(fig)
    return output_image_path
