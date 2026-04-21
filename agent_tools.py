"""Agent tool registrations.

Call register_tools(agent) once during build_agent() to attach all tools.
Tools are grouped by pipeline stage:
  Stage A — Perception  : overview, zoom spectrogram generation
  Stage B — DSP assist  : energy measurement
  Stage C — Refinement  : refine, split, merge
  Stage D — Validation  : validate, score quality, assign confidence, classify
  Stage E — Coordinate  : pixel-to-unit conversion
  Stage F — I/O         : save predictions, load ground truth, evaluate
"""
from __future__ import annotations

from pydantic_ai import Agent, BinaryContent, RunContext

from box_ops import (
    classify_call_type,
    convert_pixels_to_units,
    merge_overlapping_boxes,
    refine_box,
    score_box_quality,
    split_merged_calls,
    validate_boxes,
)
from deps import RunDeps
from evaluation import evaluate_predictions
from persistence import load_annotation_json, save_annotation_json
from schemas import (
    AnnotationOutput,
    BBox,
    SpectrogramWindowResult,
)
from spectrogram import generate_spectrogram_window


def register_tools(agent: Agent[RunDeps, AnnotationOutput]) -> None:

    # ==========================================
    # Stage A — Perception
    # ==========================================

    @agent.tool
    def tool_generate_overview_spectrogram(
        ctx: RunContext[RunDeps],
    ) -> tuple[str, BinaryContent]:
        """Generate a full-audio overview spectrogram.

        ALWAYS call this first. The result is cached under key 'overview' for
        use by all subsequent tools that accept spectrogram_key.
        """
        deps = ctx.deps
        out = deps.output_dir / f"{deps.audio_path.stem}_overview.png"
        result = generate_spectrogram_window(
            audio_path=deps.audio_path,
            output_image_path=out,
            config=deps.spectrogram_config,
            window_start=0.0,
            window_end=None,
            title=f"Overview — {deps.audio_path.name}",
        )
        deps.spectrogram_cache["overview"] = result
        return (_window_summary(result, "overview"), BinaryContent(data=out.read_bytes(), media_type="image/png"))

    @agent.tool
    def tool_generate_zoom_spectrogram(
        ctx: RunContext[RunDeps],
        window_start: float,
        window_end: float,
        tag: str = "zoom",
    ) -> tuple[str, BinaryContent]:
        """Generate a zoomed spectrogram for a specific time window.

        Use after the overview to inspect active regions or individual calls.
        The result is cached under the supplied tag for refinement/scoring tools.

        Args:
            window_start: Start time in seconds (absolute, not window-relative).
            window_end: End time in seconds.
            tag: Short label, e.g. 'focus_0', 'scan_500ms'. Must be unique per window.
        """
        deps = ctx.deps
        out = deps.output_dir / f"{deps.audio_path.stem}_{tag}_{int(window_start * 1000)}ms.png"
        result = generate_spectrogram_window(
            audio_path=deps.audio_path,
            output_image_path=out,
            config=deps.spectrogram_config,
            window_start=window_start,
            window_end=window_end,
            title=f"{tag}  [{window_start:.3f}s – {window_end:.3f}s]",
        )
        cache_key = f"{tag}_{int(window_start * 1000)}"
        deps.spectrogram_cache[cache_key] = result
        return (_window_summary(result, cache_key), BinaryContent(data=out.read_bytes(), media_type="image/png"))

    # ==========================================
    # Stage B — DSP assist (energy measurement)
    # ==========================================

    @agent.tool
    def tool_measure_energy(
        ctx: RunContext[RunDeps],
        t_min: float,
        t_max: float,
        f_min: float,
        f_max: float,
        spectrogram_key: str = "overview",
    ) -> dict:
        """Measure energy statistics in a region you have visually identified as a potential call.

        Use this to confirm that a region spotted in the spectrogram has real signal
        above the noise floor before committing to annotating it.

        Args:
            t_min, t_max: Time bounds in seconds (absolute).
            f_min, f_max: Frequency bounds in Hz.
            spectrogram_key: Which cached spectrogram to measure.
        Returns:
            mean_db, peak_db, noise_floor_db, snr_db, active_fraction, has_signal
        """
        spec = _get_spec(ctx, spectrogram_key)
        if isinstance(spec, dict):
            return spec

        t_mask = (spec.times >= t_min) & (spec.times <= t_max)
        f_mask = (spec.freqs >= f_min) & (spec.freqs <= f_max)

        if not t_mask.any() or not f_mask.any():
            return {"error": "No spectrogram data in the specified region — check coordinate bounds."}

        import numpy as np  # noqa: PLC0415

        region = spec.db[f_mask, :][:, t_mask]
        outside = spec.db[~f_mask, :] if (~f_mask).any() else None

        mean_db = float(np.mean(region))
        peak_db = float(np.max(region))
        noise_floor_db = (
            float(np.percentile(outside, 20)) if outside is not None and outside.size > 0
            else float(np.percentile(spec.db, 10))
        )
        snr_db = mean_db - noise_floor_db
        active_fraction = float(np.mean(region > noise_floor_db + 6.0))

        return {
            "mean_db": round(mean_db, 2),
            "peak_db": round(peak_db, 2),
            "noise_floor_db": round(noise_floor_db, 2),
            "snr_db": round(snr_db, 2),
            "active_fraction": round(active_fraction, 3),
            "has_signal": snr_db > 6.0 and active_fraction > 0.1,
        }

    # ==========================================
    # Stage C — Refinement
    # ==========================================

    @agent.tool
    def tool_refine_box(
        ctx: RunContext[RunDeps],
        t_min: float,
        t_max: float,
        f_min: float,
        f_max: float,
        label: str = "bat_call",
        spectrogram_key: str = "overview",
    ) -> dict:
        """Tighten a bounding box to the actual energy boundaries in the spectrogram.

        Use this for every box before finalising — especially important for 'very tight' boxes
        required for training a good detector.

        Args:
            t_min, t_max: Absolute time bounds in seconds.
            f_min, f_max: Frequency bounds in Hz.
            label: Box label.
            spectrogram_key: Which cached spectrogram to refine against.
        Returns:
            Refined box dict, or error if the spectrogram key is not found.
        """
        spec = _get_spec(ctx, spectrogram_key)
        if isinstance(spec, dict):
            return spec
        box = BBox(t_min=t_min, t_max=t_max, f_min=f_min, f_max=f_max, label=label)
        return refine_box(box, spec).model_dump()

    @agent.tool
    def tool_split_merged_calls(
        ctx: RunContext[RunDeps],
        t_min: float,
        t_max: float,
        f_min: float,
        f_max: float,
        label: str = "bat_call",
        spectrogram_key: str = "overview",
    ) -> list[dict]:
        """Attempt to split a box that visually appears to contain multiple merged calls.

        Uses energy gap detection to find temporal valleys within the box.
        Returns the original single box if no valid split is found.

        Args:
            t_min, t_max, f_min, f_max: Box coordinates.
            label: Box label.
            spectrogram_key: Which cached spectrogram to analyse.
        """
        spec = _get_spec(ctx, spectrogram_key)
        if isinstance(spec, dict):
            return [spec]
        box = BBox(t_min=t_min, t_max=t_max, f_min=f_min, f_max=f_max, label=label)
        return [b.model_dump() for b in split_merged_calls(box, spec)]

    @agent.tool
    def tool_merge_overlapping_boxes(
        ctx: RunContext[RunDeps],
        boxes: list[dict],
        iou_threshold: float = 0.3,
    ) -> list[dict]:
        """Merge overlapping or contained boxes using IoU-based NMS.

        Call this after collecting all boxes from multiple windows to remove duplicates
        before validation.

        Args:
            boxes: List of box dicts with t_min, t_max, f_min, f_max, label, confidence.
            iou_threshold: Boxes with IoU >= this are merged into their union (default 0.3).
        """
        parsed = [BBox(**b) for b in boxes]
        return [b.model_dump() for b in merge_overlapping_boxes(parsed, iou_threshold=iou_threshold)]

    # ==========================================
    # Stage E — Validation & scoring
    # ==========================================

    @agent.tool
    def tool_validate_boxes(
        ctx: RunContext[RunDeps],
        boxes: list[dict],
    ) -> dict:
        """Validate boxes for legal coordinates, minimum duration/bandwidth, and audio bounds.

        Call this before finalising output. Boxes that fail are reported with reasons.

        Returns:
            {valid_boxes: [...], rejected_boxes: [{box, reason}, ...], warnings: [...]}
        """
        deps = ctx.deps
        health = deps.audio_health
        fmax = deps.spectrogram_config.fmax_hz or health.recommended_fmax_hz or 120000.0
        parsed = [BBox(**b) for b in boxes]
        report = validate_boxes(parsed, duration_seconds=health.duration_seconds, fmax_hz=fmax)
        return report.model_dump()

    @agent.tool
    def tool_score_box_quality(
        ctx: RunContext[RunDeps],
        boxes: list[dict],
        spectrogram_key: str = "overview",
    ) -> list[dict]:
        """Score each box for tightness, SNR, duration, bandwidth, and isolation.

        Returns a quality score per box with a 'keep' / 'review' / 'reject' recommendation.
        Use this to filter pseudo-labels for detector training.

        Args:
            boxes: List of box dicts.
            spectrogram_key: Which cached spectrogram to measure quality against.
        """
        spec = _get_spec(ctx, spectrogram_key)
        if isinstance(spec, dict):
            return [spec]
        noise_db = -60.0  # fallback; ideally pass in from noise floor analysis
        parsed = [BBox(**b) for b in boxes]
        return [s.model_dump() for s in score_box_quality(parsed, spec, noise_floor_db=noise_db)]

    @agent.tool
    def tool_assign_confidence(
        ctx: RunContext[RunDeps],
        boxes: list[dict],
        spectrogram_key: str = "overview",
    ) -> list[dict]:
        """Populate confidence, detection_confidence, and localization_confidence on each box.

        Confidence is derived from the quality score (SNR, tightness). Call this as the
        last step before returning final AnnotationOutput.

        Args:
            boxes: List of box dicts.
            spectrogram_key: Which cached spectrogram to compute quality from.
        Returns:
            Boxes with confidence fields populated.
        """
        spec = _get_spec(ctx, spectrogram_key)
        if isinstance(spec, dict):
            return [spec]
        parsed = [BBox(**b) for b in boxes]
        quality = score_box_quality(parsed, spec)
        result = []
        for box, qs in zip(parsed, quality):
            result.append(box.model_copy(update={
                "confidence": round(qs.overall_score, 3),
                "detection_confidence": round(qs.snr_score, 3),
                "localization_confidence": round(qs.tightness_score, 3),
                "quality_score": round(qs.overall_score, 3),
            }).model_dump())
        return result

    @agent.tool
    def tool_classify_call_type(
        ctx: RunContext[RunDeps],
        t_min: float,
        t_max: float,
        f_min: float,
        f_max: float,
        spectrogram_key: str = "overview",
    ) -> dict:
        """Classify a call as FM, CF, or QCF using spectral shape heuristics.

        Args:
            t_min, t_max, f_min, f_max: Box coordinates.
            spectrogram_key: Which cached spectrogram to analyse.
        Returns:
            {call_type: 'FM'|'CF'|'QCF'|'unknown', confidence: float, freq_sweep_ratio: float}
        """
        spec = _get_spec(ctx, spectrogram_key)
        if isinstance(spec, dict):
            return spec
        box = BBox(t_min=t_min, t_max=t_max, f_min=f_min, f_max=f_max)
        return classify_call_type(box, spec)

    # ==========================================
    # Stage F — Coordinate helpers
    # ==========================================

    @agent.tool
    def tool_convert_pixels_to_units(
        ctx: RunContext[RunDeps],
        x_pixel: int,
        y_pixel: int,
        spectrogram_key: str = "overview",
    ) -> dict:
        """Convert pixel coordinates on a spectrogram image to absolute seconds and Hz.

        Use this if the model naturally identifies regions by pixel position rather than
        physical coordinates. More robust than asking the model to estimate units directly.

        Args:
            x_pixel: Horizontal pixel (0 = left = time_min).
            y_pixel: Vertical pixel (0 = top = freq_max).
            spectrogram_key: Which cached spectrogram image to reference.
        Returns:
            {time_seconds: float, freq_hz: float}
        """
        spec = _get_spec(ctx, spectrogram_key)
        if isinstance(spec, dict):
            return spec
        return convert_pixels_to_units(x_pixel, y_pixel, spec)

    # ==========================================
    # Stage G — I/O
    # ==========================================

    @agent.tool
    def tool_save_predictions(
        ctx: RunContext[RunDeps],
        predictions: AnnotationOutput,
        filename: str = "agent_saved_predictions.json",
    ) -> str:
        """Save predicted annotations to JSON inside the output directory.

        The pipeline also saves automatically at the end; this tool is useful for
        checkpointing intermediate results during a long agent run.
        """
        save_path = ctx.deps.output_dir / filename
        save_annotation_json(predictions, save_path)
        return str(save_path)

    @agent.tool
    def tool_load_ground_truth(ctx: RunContext[RunDeps]) -> AnnotationOutput:
        """Load the ground-truth annotation JSON for the current audio, if provided."""
        deps = ctx.deps
        if deps.ground_truth_json_path is None:
            raise ValueError("No ground-truth JSON was provided for this run.")
        return load_annotation_json(deps.ground_truth_json_path)

    @agent.tool
    def tool_evaluate_predictions(
        ctx: RunContext[RunDeps],
        predictions: AnnotationOutput,
    ) -> dict:
        """Evaluate predicted boxes against ground-truth using IoU matching.

        Requires a ground-truth JSON to have been provided at pipeline start.
        Returns precision, recall, F1, and per-match details.
        """
        deps = ctx.deps
        if deps.ground_truth_json_path is None:
            raise ValueError("No ground-truth JSON was provided for this run.")
        gt = load_annotation_json(deps.ground_truth_json_path)
        return evaluate_predictions(
            predictions=predictions.boxes,
            ground_truths=gt.boxes,
            iou_threshold=deps.iou_threshold,
        ).model_dump()


# ==========================================
# Private helpers
# ==========================================

def _get_spec(ctx: RunContext[RunDeps], key: str) -> SpectrogramWindowResult | dict:
    """Retrieve a cached spectrogram; return an error dict if not found."""
    spec = ctx.deps.spectrogram_cache.get(key)
    if spec is None:
        available = list(ctx.deps.spectrogram_cache.keys())
        return {
            "error": (
                f"No cached spectrogram with key '{key}'. "
                f"Available keys: {available}. "
                "Generate the spectrogram first with tool_generate_overview_spectrogram "
                "or tool_generate_zoom_spectrogram."
            )
        }
    return spec


def _window_summary(result: SpectrogramWindowResult, key: str) -> str:
    return (
        f"spectrogram_key={key}\n"
        f"image_path={result.image_path}\n"
        f"time_range_seconds=[{result.time_min_seconds:.6f}, {result.time_max_seconds:.6f}]\n"
        f"frequency_range_hz=[{result.freq_min_hz:.1f}, {result.freq_max_hz:.1f}]\n"
        f"duration_seconds={result.duration_seconds:.6f}\n"
        f"window_duration_seconds={result.window_duration_seconds:.6f}\n"
        f"sample_rate={result.sample_rate}\n"
        f"image_size_px={result.image_width_px}x{result.image_height_px}\n"
        f"seconds_per_pixel={result.seconds_per_pixel:.9f}\n"
        f"hz_per_pixel={result.hz_per_pixel:.4f}"
    )
