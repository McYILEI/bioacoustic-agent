from __future__ import annotations

import os
from pathlib import Path
from typing import List

from schemas import SpectrogramConfig

# ---- Model / service ----
MODEL_NAME: str = os.getenv("MODEL_NAME", "qwen3-vl:latest")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "outputs"))
DEFAULT_IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.5"))

# ---- Spectrogram ----
SPECTROGRAM_CONFIG = SpectrogramConfig(
    n_fft=int(os.getenv("N_FFT", "1024")),
    hop_length=int(os.getenv("HOP_LENGTH", "256")),
    fmin_hz=float(os.getenv("FMIN_HZ", "0")),
    fmax_hz=float(os.getenv("FMAX_HZ", "120000")),
    dpi=int(os.getenv("DPI", "220")),
    figsize=(12.0, 4.0),
    cmap=os.getenv("CMAP", "viridis"),
)

# ---- Audio health thresholds ----
MIN_AUDIO_DURATION_SECONDS: float = float(os.getenv("MIN_AUDIO_DURATION", "0.05"))
SILENCE_THRESHOLD_RMS: float = float(os.getenv("SILENCE_THRESHOLD_RMS", "0.001"))
CLIPPING_THRESHOLD: float = float(os.getenv("CLIPPING_THRESHOLD", "0.999"))

# ---- Box constraints ----
MIN_BAT_CALL_DURATION_MS: float = float(os.getenv("MIN_BAT_CALL_DURATION_MS", "2.0"))
MIN_BAT_CALL_BANDWIDTH_HZ: float = float(os.getenv("MIN_BAT_CALL_BANDWIDTH_HZ", "1000.0"))

# ---- DSP detection thresholds ----
ACTIVE_REGION_ENERGY_PERCENTILE: float = float(os.getenv("ACTIVE_REGION_ENERGY_PERCENTILE", "70.0"))
CANDIDATE_BOX_ENERGY_PERCENTILE: float = float(os.getenv("CANDIDATE_BOX_ENERGY_PERCENTILE", "85.0"))
NOISE_FLOOR_PERCENTILE: float = float(os.getenv("NOISE_FLOOR_PERCENTILE", "20.0"))

# Known reasonable bat recording sample rates
KNOWN_BAT_SAMPLE_RATES: frozenset[int] = frozenset(
    {44100, 48000, 96000, 192000, 250000, 384000, 500000}
)


def validate_runtime_config(config: SpectrogramConfig, output_dir: Path) -> List[str]:
    """Validate runtime configuration. Returns warnings list. Raises ValueError on fatal errors."""
    warnings: List[str] = []

    if config.fmax_hz is not None and config.fmin_hz >= config.fmax_hz:
        raise ValueError(
            f"fmin_hz ({config.fmin_hz}) must be < fmax_hz ({config.fmax_hz})"
        )
    if config.hop_length > config.n_fft:
        raise ValueError(
            f"hop_length ({config.hop_length}) must be <= n_fft ({config.n_fft})"
        )
    if not (0.0 < DEFAULT_IOU_THRESHOLD < 1.0):
        raise ValueError(
            f"iou_threshold must be in (0, 1), got {DEFAULT_IOU_THRESHOLD}"
        )
    if config.dpi < 72:
        warnings.append(f"DPI={config.dpi} is very low; spectrogram images may be blurry for the VLM")
    if config.fmax_hz is not None and config.fmax_hz > 500000:
        warnings.append(f"fmax_hz={config.fmax_hz} exceeds typical ultrasonic range")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".write_probe"
        probe.touch()
        probe.unlink()
    except OSError as exc:
        raise OSError(f"Output directory {output_dir} is not writable: {exc}") from exc

    return warnings
