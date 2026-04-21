"""Pure signal-processing layer — no file I/O, no plotting."""
from __future__ import annotations

import numpy as np
from scipy.signal import stft

from schemas import SpectrogramConfig


def amplitude_to_db(magnitude: np.ndarray, amin: float = 1e-10) -> np.ndarray:
    magnitude = np.maximum(np.asarray(magnitude, dtype=np.float32), amin)
    ref_value = float(np.max(magnitude))
    return 20.0 * np.log10(magnitude / max(ref_value, amin))


def compute_spectrogram(
    y: np.ndarray,
    sr: int,
    config: SpectrogramConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute STFT-based spectrogram and apply frequency band filter.

    Returns:
        freqs: Frequency array (Hz), shape (F,)
        times: Time array (s), shape (T,)
        db: dB-scaled magnitude, shape (F, T)
    """
    if len(y) == 0:
        raise ValueError("Audio segment is empty.")
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
    db = amplitude_to_db(np.abs(zxx))

    fmin = config.fmin_hz
    fmax = config.fmax_hz
    if fmax is not None:
        mask = (freqs >= fmin) & (freqs <= fmax)
    else:
        mask = freqs >= fmin

    freqs = freqs[mask]
    db = db[mask, :]

    if freqs.size == 0:
        raise ValueError(
            f"No frequencies remain after fmin={fmin}Hz / fmax={fmax}Hz filter. "
            f"Check config vs audio sample rate."
        )

    return freqs, times, db
