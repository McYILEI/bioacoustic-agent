"""Audio file reading — isolated from analysis and DSP logic."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf


def read_audio(audio_path: str | Path) -> Tuple[np.ndarray, int, int]:
    """Read an audio file and downmix to mono float32.

    Returns:
        y: Mono signal, shape (N,), dtype float32
        sr: Sample rate in Hz
        channels: Original channel count before downmix
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    y, sr = sf.read(audio_path)
    channels = 1 if y.ndim == 1 else y.shape[1]
    if y.ndim == 2:
        y = y.mean(axis=1)

    return y.astype(np.float32), int(sr), channels
