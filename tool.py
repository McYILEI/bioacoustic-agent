from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field, field_validator, model_validator
from scipy.signal import stft


# ==========================================
# 1. Structured schemas
# ==========================================
class BBox(BaseModel):
    """Bounding box on a spectrogram using absolute seconds and Hz."""

    t_min: float = Field(..., description="Start time in seconds")
    t_max: float = Field(..., description="End time in seconds")
    f_min: float = Field(..., description="Minimum frequency in Hz")
    f_max: float = Field(..., description="Maximum frequency in Hz")
    label: str = Field(default="bat_call", description="Class label")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("label")
    @classmethod
    def strip_label(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("label cannot be empty")
        return value

    @model_validator(mode="after")
    def validate_box(self) -> "BBox":
        if self.t_max <= self.t_min:
            raise ValueError("t_max must be greater than t_min")
        if self.f_max <= self.f_min:
            raise ValueError("f_max must be greater than f_min")
        if self.t_min < 0:
            raise ValueError("t_min cannot be negative")
        if self.f_min < 0:
            raise ValueError("f_min cannot be negative")
        return self


class AnnotationOutput(BaseModel):
    """Final structured output from the model."""

    boxes: List[BBox] = Field(default_factory=list)
    reasoning_summary: str = Field(
        default="",
        description="Very short summary of what was detected and how certain the model is.",
    )


class EvaluationResult(BaseModel):
    precision: float
    recall: float
    f1_score: float
    mean_iou: float
    tp: int
    fp: int
    fn: int
    iou_threshold: float
    matched_pairs: list[dict]


class PipelineResult(BaseModel):
    audio_path: str
    spectrogram_image_path: str
    overlay_image_path: Optional[str] = None
    prediction_json_path: Optional[str] = None
    boxes: List[BBox]
    metrics: Optional[EvaluationResult] = None
    duration_seconds: float
    sample_rate: int


@dataclass(slots=True)
class SpectrogramConfig:
    n_fft: int = 1024
    hop_length: int = 256
    window: str = "hann"
    fmin_hz: float = 0.0
    fmax_hz: Optional[float] = None
    dpi: int = 220
    figsize: tuple[float, float] = (12.0, 4.0)
    cmap: str = "viridis"


@dataclass(slots=True)
class SpectrogramWindowResult:
    image_path: Path
    audio_path: Path
    duration_seconds: float
    window_start_seconds: float
    window_end_seconds: float
    window_duration_seconds: float
    time_min_seconds: float
    time_max_seconds: float
    freq_min_hz: float
    freq_max_hz: float
    sample_rate: int
    times: np.ndarray
    freqs: np.ndarray
    db: np.ndarray
    config: SpectrogramConfig


# ==========================================
# 2. Audio + spectrogram tools
# ==========================================
def read_audio(audio_path: str | Path) -> tuple[np.ndarray, int]:
    """Read audio with the original sample rate and downmix stereo to mono."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    y, sr = sf.read(audio_path)
    if y.ndim == 2:
        y = y.mean(axis=1)

    y = y.astype(np.float32)
    return y, int(sr)


def amplitude_to_db(magnitude: np.ndarray, ref: float | None = None, amin: float = 1e-10) -> np.ndarray:
    """Convert amplitude spectrogram to dB scale."""
    magnitude = np.maximum(np.asarray(magnitude, dtype=np.float32), amin)
    ref_value = float(np.max(magnitude) if ref is None else max(ref, amin))
    return 20.0 * np.log10(magnitude / ref_value)


def compute_spectrogram(
    y: np.ndarray,
    sr: int,
    config: SpectrogramConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a consistent STFT-based spectrogram using SciPy."""
    if len(y) == 0:
        raise ValueError("Audio segment is empty after windowing.")

    noverlap = config.n_fft - config.hop_length
    if noverlap < 0:
        raise ValueError("hop_length must be <= n_fft")

    freqs, times, zxx = stft(
        y,
        fs=sr,
        window=config.window,
        nperseg=config.n_fft,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(zxx)
    db = amplitude_to_db(magnitude)

    if config.fmax_hz is not None:
        mask = (freqs >= config.fmin_hz) & (freqs <= config.fmax_hz)
        freqs = freqs[mask]
        db = db[mask, :]
    elif config.fmin_hz > 0:
        mask = freqs >= config.fmin_hz
        freqs = freqs[mask]
        db = db[mask, :]

    if freqs.size == 0:
        raise ValueError("No frequencies remain after applying fmin/fmax filters.")

    return freqs, times, db


def generate_spectrogram_window(
    audio_path: str | Path,
    output_image_path: str | Path,
    config: SpectrogramConfig,
    window_start: float = 0.0,
    window_end: float | None = None,
) -> SpectrogramWindowResult:
    """Create a spectrogram image for a selected time window and return metadata."""
    y, sr = read_audio(audio_path)
    duration_seconds = len(y) / sr

    if window_start < 0:
        raise ValueError("window_start must be >= 0")
    if window_end is None:
        window_end = duration_seconds
    if window_end <= window_start:
        raise ValueError("window_end must be greater than window_start")
    if window_start >= duration_seconds:
        raise ValueError("window_start is outside the audio duration")

    clipped_end = min(window_end, duration_seconds)

    start_sample = int(round(window_start * sr))
    end_sample = int(round(clipped_end * sr))
    window_y = y[start_sample:end_sample]
    if len(window_y) == 0:
        raise ValueError("Selected audio window is empty.")

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
    ax.set_title("Bat Audio Spectrogram")
    fig.colorbar(im, ax=ax, label="Amplitude (dB)")
    plt.tight_layout()
    fig.savefig(output_image_path, dpi=config.dpi)
    plt.close(fig)

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
    )


def draw_bboxes_on_spectrogram(
    spectrogram: SpectrogramWindowResult,
    boxes: Iterable[BBox],
    output_image_path: str | Path,
) -> Path:
    """Create an overlay image for qualitative review."""
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
        rect = plt.Rectangle(
            (box.t_min, box.f_min),
            box.t_max - box.t_min,
            box.f_max - box.f_min,
            fill=False,
            linewidth=2,
        )
        ax.add_patch(rect)
        text = box.label if box.confidence is None else f"{box.label} ({box.confidence:.2f})"
        ax.text(box.t_min, box.f_max, text, fontsize=8, va="bottom")

    plt.tight_layout()
    fig.savefig(output_image_path, dpi=spectrogram.config.dpi)
    plt.close(fig)
    return output_image_path


# ==========================================
# 3. Persistence helpers
# ==========================================
def save_annotation_json(output: AnnotationOutput, save_path: str | Path) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(output.model_dump_json(indent=2), encoding="utf-8")
    return save_path


def load_annotation_json(json_path: str | Path) -> AnnotationOutput:
    json_path = Path(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return AnnotationOutput.model_validate(data)


# ==========================================
# 4. IoU + evaluation tools
# ==========================================
def calculate_iou(box1: BBox, box2: BBox) -> float:
    """Intersection over Union on the time-frequency plane."""
    x_left = max(box1.t_min, box2.t_min)
    x_right = min(box1.t_max, box2.t_max)
    y_bottom = max(box1.f_min, box2.f_min)
    y_top = min(box1.f_max, box2.f_max)

    if x_right <= x_left or y_top <= y_bottom:
        return 0.0

    intersection_area = (x_right - x_left) * (y_top - y_bottom)
    box1_area = (box1.t_max - box1.t_min) * (box1.f_max - box1.f_min)
    box2_area = (box2.t_max - box2.t_min) * (box2.f_max - box2.f_min)
    union_area = box1_area + box2_area - intersection_area
    if union_area <= 0:
        return 0.0
    return float(intersection_area / union_area)


def evaluate_predictions(
    predictions: List[BBox],
    ground_truths: List[BBox],
    iou_threshold: float = 0.5,
) -> EvaluationResult:
    """Greedy one-to-one matching by highest IoU under label consistency."""
    candidate_matches: list[tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(predictions):
        for gt_idx, gt in enumerate(ground_truths):
            if pred.label != gt.label:
                continue
            iou = calculate_iou(pred, gt)
            if iou >= iou_threshold:
                candidate_matches.append((iou, pred_idx, gt_idx))

    candidate_matches.sort(key=lambda x: x[0], reverse=True)

    matched_preds: set[int] = set()
    matched_gts: set[int] = set()
    matched_pairs: list[dict] = []

    for iou, pred_idx, gt_idx in candidate_matches:
        if pred_idx in matched_preds or gt_idx in matched_gts:
            continue
        matched_preds.add(pred_idx)
        matched_gts.add(gt_idx)
        matched_pairs.append(
            {
                "pred_index": pred_idx,
                "gt_index": gt_idx,
                "iou": round(iou, 6),
                "label": predictions[pred_idx].label,
            }
        )

    tp = len(matched_pairs)
    fp = len(predictions) - tp
    fn = len(ground_truths) - tp
    precision = tp / len(predictions) if predictions else 0.0
    recall = tp / len(ground_truths) if ground_truths else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    mean_iou = sum(item["iou"] for item in matched_pairs) / tp if tp else 0.0

    return EvaluationResult(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        mean_iou=mean_iou,
        tp=tp,
        fp=fp,
        fn=fn,
        iou_threshold=iou_threshold,
        matched_pairs=matched_pairs,
    )