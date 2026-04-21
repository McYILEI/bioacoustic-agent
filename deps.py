"""Agent runtime dependencies — passed to every tool via RunContext.deps."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import DEFAULT_IOU_THRESHOLD
from schemas import AudioHealthReport, SpectrogramConfig, SpectrogramWindowResult


@dataclass(slots=True)
class RunDeps:
    # Required — must be provided at pipeline construction time
    audio_path: Path
    output_dir: Path
    spectrogram_config: SpectrogramConfig
    audio_health: AudioHealthReport

    # Optional
    ground_truth_json_path: Optional[Path] = None
    iou_threshold: float = DEFAULT_IOU_THRESHOLD

    # Mutable state written by tools during the agent run.
    # Keys are user-supplied tags (e.g. "overview", "focus_0", "scan_500ms").
    spectrogram_cache: dict[str, SpectrogramWindowResult] = field(default_factory=dict)
