"""Entry point: pipeline orchestration, FastAPI app, CLI.

Execution stages in annotate_audio():
  Stage 0 — Config validation
  Stage 1 — Audio pre-analysis   (inspect, noise floor, active regions)
  Stage 2 — Deps assembly
  Stage 3 — Agent annotation     (multi-scale spectrogram + VLM reasoning)
  Stage 4 — Post-processing      (validate, merge, safety-net cleanup)
  Stage 5 — Quality scoring
  Stage 6 — Persist + visualise
  Stage 7 — Evaluation           (if ground-truth provided)
"""
from __future__ import annotations

import argparse
import asyncio
import tempfile
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from agent_tools import register_tools
from audio_analysis import estimate_noise_floor, inspect_audio
from audio_io import read_audio
from box_ops import merge_overlapping_boxes, score_box_quality, validate_boxes
from config import (
    DEFAULT_IOU_THRESHOLD,
    MODEL_NAME,
    OLLAMA_BASE_URL,
    OUTPUT_DIR,
    SPECTROGRAM_CONFIG,
    validate_runtime_config,
)
from deps import RunDeps
from evaluation import evaluate_predictions
from persistence import load_annotation_json, save_annotation_json
from schemas import AnnotationOutput, PipelineResult
from spectrogram import draw_bboxes_on_spectrogram, generate_spectrogram_window

# ==========================================
# Agent instructions
# ==========================================
_INSTRUCTIONS = """\
You are a bioacoustic annotation agent specialised in bat echolocation call detection.
Your primary capability is visual perception. You look at spectrograms and identify
bat calls based on their appearance: FM sweeps, CF tones, narrow-band pulses, etc.

DSP tools are assistants — they help you quantify and refine what YOU have already
identified visually. They do not decide where calls are. YOU do.

TOOLS AVAILABLE:
  tool_generate_overview_spectrogram  — see the full audio at once
  tool_generate_zoom_spectrogram      — zoom into any time window for a closer look
  tool_measure_energy                 — quantify signal strength in a region you spotted
  tool_refine_box                     — tighten YOUR proposed box to exact energy boundaries
  tool_split_merged_calls             — split a box that contains multiple distinct calls
  tool_merge_overlapping_boxes        — deduplicate boxes before finalising
  tool_validate_boxes                 — check coordinate legality
  tool_assign_confidence              — score final boxes
  tool_classify_call_type             — determine FM / CF / QCF call type
  tool_convert_pixels_to_units        — convert image pixels to seconds / Hz

WORKFLOW (adapt as your visual judgment requires):

1. Generate the overview spectrogram. Observe carefully:
   - Where do you see energy blobs or sweeps? What time and frequency ranges?
   - Are signals dense or sparse? Clean background or noisy?

2. Based on what you see, decide your next steps freely:
   - Zoom in if you need a closer look at a region.
   - Use tool_measure_energy to confirm a visually-spotted region has real signal.
   - Zoom more than once if needed — dense or complex regions deserve tight windows.

3. For each region you believe contains a bat call, propose a box from your visual
   estimate (time and frequency extent), then call tool_refine_box to tighten it.

4. Review each refined box:
   - Spans multiple distinct calls? → tool_split_merged_calls
   - Still too loose? → zoom in further and refine again

5. Deduplicate with tool_merge_overlapping_boxes, validate with tool_validate_boxes,
   then score with tool_assign_confidence.

6. Return AnnotationOutput with final boxes and a brief reasoning_summary describing
   what you observed and how you decided where the calls are.

BOX RULES:
- All coordinates are ABSOLUTE: seconds (time), Hz (frequency). Never window-relative pixels.
- Boxes must be TIGHT — closely wrapping visible call energy, not loose guesses.
- Minimum duration: 2 ms. Minimum bandwidth: 1 kHz.
- Do not annotate noise, broadband transients, or ambiguous blobs.
- Default label: 'bat_call'. Use 'noise', 'insect', or 'artifact' only if clearly justified.
- If you see no bat calls, return an empty boxes list.
"""


# ==========================================
# Agent construction (module-level singleton)
# ==========================================

def build_agent() -> Agent[RunDeps, AnnotationOutput]:
    model = OpenAIChatModel(
        MODEL_NAME,
        provider=OllamaProvider(base_url=OLLAMA_BASE_URL),
    )
    agent: Agent[RunDeps, AnnotationOutput] = Agent(
        model=model,
        deps_type=RunDeps,
        output_type=AnnotationOutput,
        instructions=_INSTRUCTIONS,
    )
    register_tools(agent)
    return agent


ANNOTATION_AGENT = build_agent()
app = FastAPI(title="Bat Echolocation Annotation Agent", version="3.0.0")


# ==========================================
# Core pipeline
# ==========================================

async def annotate_audio(
    audio_path: str | Path,
    ground_truth_json_path: str | Path | None = None,
    output_dir: str | Path = OUTPUT_DIR,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> PipelineResult:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_dir = Path(output_dir)

    # --- Stage 0: Config + output-dir validation ---
    config_warnings = validate_runtime_config(SPECTROGRAM_CONFIG, output_dir)
    for w in config_warnings:
        print(f"[CONFIG WARNING] {w}")

    gt_path = Path(ground_truth_json_path) if ground_truth_json_path else None
    if gt_path is not None and not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth JSON not found: {gt_path}")

    # --- Stage 1: Audio pre-analysis ---
    health = inspect_audio(audio_path)
    for w in health.warnings:
        print(f"[AUDIO WARNING] {w}")

    if health.is_too_short:
        raise ValueError(
            f"Audio too short ({health.duration_seconds:.4f}s) to annotate."
        )

    y, sr, _ = read_audio(audio_path)
    noise_info = estimate_noise_floor(y, sr, SPECTROGRAM_CONFIG)

    print(
        f"[PRE-ANALYSIS] duration={health.duration_seconds:.3f}s | "
        f"sr={sr}Hz | "
        f"noise_floor={noise_info['noise_floor_db']:.1f}dB | "
        f"strategy={health.suggested_window_strategy}"
    )
    if health.is_near_silent:
        print("[AUDIO WARNING] Near-silent audio — no calls expected; agent will likely return empty boxes.")

    # --- Stage 2: Assemble deps ---
    deps = RunDeps(
        audio_path=audio_path,
        output_dir=output_dir,
        spectrogram_config=SPECTROGRAM_CONFIG,
        audio_health=health,
        ground_truth_json_path=gt_path,
        iou_threshold=iou_threshold,
    )

    prompt = (
        f"Annotate bat echolocation calls in: {audio_path.name}\n"
        f"duration={health.duration_seconds:.3f}s  "
        f"sample_rate={sr}Hz  "
        f"noise_floor={noise_info['noise_floor_db']:.1f}dB\n"
        "Start with the overview spectrogram and use your visual judgment to find bat calls."
    )

    # --- Stage 3: Agent annotation ---
    result = await ANNOTATION_AGENT.run(prompt, deps=deps)
    annotation_output: AnnotationOutput = result.output

    # --- Stage 4: Post-processing safety net ---
    fmax = SPECTROGRAM_CONFIG.fmax_hz or health.recommended_fmax_hz or 120000.0
    validation_report = validate_boxes(
        annotation_output.boxes,
        duration_seconds=health.duration_seconds,
        fmax_hz=fmax,
    )
    for w in validation_report.warnings:
        print(f"[POST-PROCESS] {w}")

    clean_boxes = merge_overlapping_boxes(validation_report.valid_boxes, iou_threshold=0.3)
    final_output = AnnotationOutput(
        boxes=clean_boxes,
        reasoning_summary=annotation_output.reasoning_summary,
    )

    # --- Stage 5: Quality scoring ---
    overview_spec = deps.spectrogram_cache.get("overview")
    quality_scores = None
    if overview_spec is not None and clean_boxes:
        quality_scores = score_box_quality(
            clean_boxes, overview_spec,
            noise_floor_db=noise_info["noise_floor_db"],
        )

    # --- Stage 6: Persist + visualise ---
    prediction_json_path = output_dir / f"{audio_path.stem}_predictions.json"
    save_annotation_json(final_output, prediction_json_path)

    # Generate overview if the agent didn't (e.g. it crashed mid-run)
    if overview_spec is None:
        overview_spec = generate_spectrogram_window(
            audio_path=audio_path,
            output_image_path=output_dir / f"{audio_path.stem}_overview.png",
            config=SPECTROGRAM_CONFIG,
        )

    overlay_path = output_dir / f"{audio_path.stem}_overlay.png"
    draw_bboxes_on_spectrogram(overview_spec, final_output.boxes, overlay_path)

    # --- Stage 7: Evaluation (optional) ---
    metrics = None
    if gt_path is not None:
        gt = load_annotation_json(gt_path)
        metrics = evaluate_predictions(final_output.boxes, gt.boxes, iou_threshold)

    return PipelineResult(
        audio_path=str(audio_path.resolve()),
        spectrogram_image_path=str(overview_spec.image_path.resolve()),
        overlay_image_path=str(overlay_path.resolve()),
        prediction_json_path=str(prediction_json_path.resolve()),
        boxes=final_output.boxes,
        metrics=metrics,
        duration_seconds=health.duration_seconds,
        sample_rate=sr,
        audio_health=health.model_dump(),
        active_regions=None,
        box_quality_scores=[s.model_dump() for s in quality_scores] if quality_scores else None,
    )


# ==========================================
# FastAPI endpoints
# ==========================================

@app.get("/health")
async def health_endpoint() -> dict:
    return {"status": "ok", "model_name": MODEL_NAME, "ollama_base_url": OLLAMA_BASE_URL}


@app.post("/annotate-by-path")
async def annotate_by_path(
    audio_path: str = Form(...),
    ground_truth_json_path: Optional[str] = Form(default=None),
    iou_threshold: float = Form(default=DEFAULT_IOU_THRESHOLD),
):
    try:
        result = await annotate_audio(audio_path, ground_truth_json_path, iou_threshold=iou_threshold)
        return JSONResponse(content=result.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/annotate-upload")
async def annotate_upload(
    audio_file: UploadFile = File(...),
    ground_truth_file: Optional[UploadFile] = File(default=None),
    iou_threshold: float = Form(default=DEFAULT_IOU_THRESHOLD),
):
    suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        audio_path = tmp_dir / f"input{suffix}"
        audio_path.write_bytes(await audio_file.read())

        gt_path = None
        if ground_truth_file is not None:
            gt_path = tmp_dir / (ground_truth_file.filename or "gt.json")
            gt_path.write_bytes(await ground_truth_file.read())

        try:
            result = await annotate_audio(audio_path, gt_path, iou_threshold=iou_threshold)
            return JSONResponse(content=result.model_dump())
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc


# ==========================================
# CLI
# ==========================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bat echolocation annotation agent")
    p.add_argument("--audio-path", type=str, help="Path to input audio file")
    p.add_argument("--ground-truth-json", type=str, default=None)
    p.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    p.add_argument("--serve", action="store_true", help="Start FastAPI server")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


async def _run_cli(audio_path: str, gt_json: str | None, iou_threshold: float) -> None:
    result = await annotate_audio(audio_path, gt_json, iou_threshold=iou_threshold)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    args = _parse_args()
    if args.serve:
        uvicorn.run("main:app", host=args.host, port=args.port, reload=False)
    else:
        if not args.audio_path:
            raise SystemExit("--audio-path is required unless --serve is used")
        asyncio.run(_run_cli(args.audio_path, args.ground_truth_json, args.iou_threshold))
