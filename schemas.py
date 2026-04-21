from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


# ---- Core annotation types ----

class BBox(BaseModel):
    t_min: float = Field(..., description="Start time in seconds")
    t_max: float = Field(..., description="End time in seconds")
    f_min: float = Field(..., description="Minimum frequency in Hz")
    f_max: float = Field(..., description="Maximum frequency in Hz")
    label: str = Field(default="bat_call")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    detection_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    localization_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    source: str = Field(default="agent", description="'agent' | 'dsp_candidate' | 'refined' | 'merged' | 'split'")

    @field_validator("label")
    @classmethod
    def strip_label(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("label cannot be empty")
        return v

    @model_validator(mode="after")
    def check_coords(self) -> "BBox":
        if self.t_max <= self.t_min:
            raise ValueError("t_max must be > t_min")
        if self.f_max <= self.f_min:
            raise ValueError("f_max must be > f_min")
        if self.t_min < 0:
            raise ValueError("t_min cannot be negative")
        if self.f_min < 0:
            raise ValueError("f_min cannot be negative")
        return self

    @property
    def duration(self) -> float:
        return self.t_max - self.t_min

    @property
    def bandwidth(self) -> float:
        return self.f_max - self.f_min


class AnnotationOutput(BaseModel):
    boxes: List[BBox] = Field(default_factory=list)
    reasoning_summary: str = Field(default="")


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
    audio_health: Optional[dict] = None
    active_regions: Optional[list] = None
    box_quality_scores: Optional[list] = None


# ---- Audio analysis results ----

class AudioHealthReport(BaseModel):
    duration_seconds: float
    sample_rate: int
    channels: int
    num_samples: int
    peak_amplitude: float
    rms_amplitude: float
    is_clipping: bool
    is_near_silent: bool
    is_too_short: bool
    has_abnormal_sample_rate: bool
    clipping_ratio: float
    silence_ratio: float
    recommended_fmax_hz: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)
    suggested_window_strategy: str = "standard"


class ActiveRegion(BaseModel):
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    score: float = Field(..., ge=0.0, le=1.0)
    peak_freq_hz: Optional[float] = None


class WindowPlan(BaseModel):
    start: float
    end: float
    purpose: str = Field(..., description="'overview' | 'scan' | 'focus'")
    priority: int = Field(default=0, description="Higher = more important to view first")


class CandidateBox(BaseModel):
    t_min: float
    t_max: float
    f_min: float
    f_max: float
    score: float = Field(..., ge=0.0, le=1.0)
    source: str = "dsp"


class BoxValidationReport(BaseModel):
    valid_boxes: List[BBox]
    rejected_boxes: List[dict]
    warnings: List[str]


class BoxQualityScore(BaseModel):
    box_index: int
    tightness_score: float
    isolation_score: float
    snr_score: float
    duration_score: float
    bandwidth_score: float
    overall_score: float
    recommendation: str  # 'keep' | 'review' | 'reject'


# ---- Spectrogram types ----

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
    seconds_per_pixel: float = 0.0
    hz_per_pixel: float = 0.0
    image_width_px: int = 0
    image_height_px: int = 0
