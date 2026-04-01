from __future__ import annotations

import argparse
import asyncio
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic_ai import Agent, BinaryContent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from tool import (
    AnnotationOutput,
    EvaluationResult,
    PipelineResult,
    SpectrogramConfig,
    draw_bboxes_on_spectrogram,
    evaluate_predictions,
    generate_spectrogram_window,
    load_annotation_json,
    save_annotation_json,
)

# ==========================================
# 1. Runtime configuration
# ==========================================
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-vl:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
DEFAULT_IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.5"))

SPECTROGRAM_CONFIG = SpectrogramConfig(
    n_fft=int(os.getenv("N_FFT", "1024")),
    hop_length=int(os.getenv("HOP_LENGTH", "256")),
    fmin_hz=float(os.getenv("FMIN_HZ", "0")),
    fmax_hz=float(os.getenv("FMAX_HZ", "120000")),
    dpi=int(os.getenv("DPI", "220")),
    figsize=(12.0, 4.0),
    cmap=os.getenv("CMAP", "viridis"),
)


# ==========================================
# 2. Agent deps
# ==========================================
@dataclass(slots=True)
class RunDeps:
    audio_path: Path
    output_dir: Path
    spectrogram_config: SpectrogramConfig
    ground_truth_json_path: Path | None = None
    iou_threshold: float = DEFAULT_IOU_THRESHOLD


# ==========================================
# 3. Agent construction
# ==========================================
def build_agent() -> Agent[RunDeps, AnnotationOutput]:
    model = OpenAIChatModel(
        MODEL_NAME,
        provider=OllamaProvider(base_url=OLLAMA_BASE_URL),
    )

    instructions = (
        "You are a bioacoustic annotation agent specialized in bat echolocation. "
        "Do not answer from memory. Use the available tools to inspect the audio via spectrograms before returning final boxes. "
        "Recommended workflow: first generate one overview spectrogram covering the full file, inspect it, and if needed generate one or more narrower windows for a closer view. "
        "Bounding boxes must use absolute coordinates in seconds and Hz, not window-relative pixel coordinates. "
        "Only annotate visible bat calls inside the plotted region. Keep boxes tight. "
        "If there are no clear calls, return an empty boxes list. "
        "Use label 'bat_call' unless another label is justified by explicit context. "
        "You may save or evaluate predictions with tools if useful, but your final answer must be valid structured output only."
    )

    agent: Agent[RunDeps, AnnotationOutput] = Agent(
        model=model,
        deps_type=RunDeps,
        output_type=AnnotationOutput,
        instructions=instructions,
    )

    @agent.tool
    def tool_generate_spectrogram_window(
        ctx: RunContext[RunDeps],
        window_start: float = 0.0,
        window_end: float | None = None,
        tag: str = "view",
    ) -> tuple[str, BinaryContent]:
        """Generate a spectrogram image for an audio time window and return metadata plus the image.

        Args:
            window_start: Absolute start time in seconds.
            window_end: Absolute end time in seconds. Use null to indicate the end of the file.
            tag: Short suffix used in the generated filename.
        """
        deps = ctx.deps
        output_path = deps.output_dir / f"{deps.audio_path.stem}_{tag}_{int(window_start * 1000)}ms.png"

        result = generate_spectrogram_window(
            audio_path=deps.audio_path,
            output_image_path=output_path,
            config=deps.spectrogram_config,
            window_start=window_start,
            window_end=window_end,
        )

        summary = (
            f"spectrogram_path={result.image_path}\n"
            f"time_range_seconds=[{result.time_min_seconds:.6f}, {result.time_max_seconds:.6f}]\n"
            f"frequency_range_hz=[{result.freq_min_hz:.2f}, {result.freq_max_hz:.2f}]\n"
            f"duration_seconds={result.duration_seconds:.6f}\n"
            f"sample_rate={result.sample_rate}\n"
            f"window_duration_seconds={result.window_duration_seconds:.6f}"
        )
        image_part = BinaryContent(data=result.image_path.read_bytes(), media_type="image/png")
        return (summary, image_part)

    @agent.tool
    def tool_save_predictions(
        ctx: RunContext[RunDeps],
        predictions: AnnotationOutput,
        filename: str = "agent_saved_predictions.json",
    ) -> str:
        """Save predicted annotations to JSON inside the output directory.

        Args:
            predictions: Structured predictions to save.
            filename: Output JSON filename relative to the current output directory.
        """
        save_path = ctx.deps.output_dir / filename
        save_annotation_json(predictions, save_path)
        return str(save_path)

    @agent.tool
    def tool_load_ground_truth(ctx: RunContext[RunDeps]) -> AnnotationOutput:
        """Load the ground-truth annotation JSON for the current audio if one was provided."""
        deps = ctx.deps
        if deps.ground_truth_json_path is None:
            raise ValueError("No ground-truth JSON was provided for this run.")
        return load_annotation_json(deps.ground_truth_json_path)

    @agent.tool
    def tool_evaluate_predictions(
        ctx: RunContext[RunDeps],
        predictions: AnnotationOutput,
    ) -> EvaluationResult:
        """Evaluate predicted boxes against the provided ground-truth boxes using IoU matching."""
        deps = ctx.deps
        if deps.ground_truth_json_path is None:
            raise ValueError("Cannot evaluate predictions because no ground-truth JSON was provided.")
        ground_truth = load_annotation_json(deps.ground_truth_json_path)
        return evaluate_predictions(
            predictions=predictions.boxes,
            ground_truths=ground_truth.boxes,
            iou_threshold=deps.iou_threshold,
        )

    return agent


ANNOTATION_AGENT = build_agent()
app = FastAPI(title="AI Annotator Agent", version="2.0.0")


# ==========================================
# 4. Core pipeline
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
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_path = Path(ground_truth_json_path) if ground_truth_json_path is not None else None
    if gt_path is not None and not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth JSON not found: {gt_path}")

    deps = RunDeps(
        audio_path=audio_path,
        output_dir=output_dir,
        spectrogram_config=SPECTROGRAM_CONFIG,
        ground_truth_json_path=gt_path,
        iou_threshold=iou_threshold,
    )

    prompt = (
        f"Annotate bat calls in the audio file at path {audio_path.name}. "
        "Use the tools to inspect the audio with spectrograms before you produce the final structured output. "
        "Start with an overview of the full file. If the overview is too dense or unclear, request one or more zoomed-in windows. "
        "Return absolute time coordinates in seconds and frequency coordinates in Hz."
    )

    result = await ANNOTATION_AGENT.run(prompt, deps=deps)
    annotation_output = result.output

    # 这里仍然在 Python 层做一次固定保存，保证 API/CLI 一定有落盘结果
    prediction_json_path = output_dir / f"{audio_path.stem}_predictions.json"
    save_annotation_json(annotation_output, prediction_json_path)

    # 生成一张全局 overview 用于可视化 overlay
    overview = generate_spectrogram_window(
        audio_path=audio_path,
        output_image_path=output_dir / f"{audio_path.stem}_overview.png",
        config=SPECTROGRAM_CONFIG,
        window_start=0.0,
        window_end=None,
    )

    overlay_image_path = output_dir / f"{audio_path.stem}_overlay.png"
    draw_bboxes_on_spectrogram(overview, annotation_output.boxes, overlay_image_path)

    metrics = None
    if gt_path is not None:
        gt = load_annotation_json(gt_path)
        metrics = evaluate_predictions(
            predictions=annotation_output.boxes,
            ground_truths=gt.boxes,
            iou_threshold=iou_threshold,
        )

    return PipelineResult(
        audio_path=str(audio_path.resolve()),
        spectrogram_image_path=str(overview.image_path.resolve()),
        overlay_image_path=str(overlay_image_path.resolve()),
        prediction_json_path=str(prediction_json_path.resolve()),
        boxes=annotation_output.boxes,
        metrics=metrics,
        duration_seconds=overview.duration_seconds,
        sample_rate=overview.sample_rate,
    )


# ==========================================
# 5. FastAPI endpoints
# ==========================================
@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "ollama_base_url": OLLAMA_BASE_URL,
    }


@app.post("/annotate-by-path")
async def annotate_by_path(
    audio_path: str = Form(...),
    ground_truth_json_path: Optional[str] = Form(default=None),
    iou_threshold: float = Form(default=DEFAULT_IOU_THRESHOLD),
):
    try:
        result = await annotate_audio(
            audio_path=audio_path,
            ground_truth_json_path=ground_truth_json_path,
            iou_threshold=iou_threshold,
        )
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
    ground_truth_file: UploadFile | None = File(default=None),
    iou_threshold: float = Form(default=DEFAULT_IOU_THRESHOLD),
):
    suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        audio_path = tmp_dir / f"input{suffix}"
        audio_path.write_bytes(await audio_file.read())

        gt_path = None
        if ground_truth_file is not None:
            gt_path = tmp_dir / (ground_truth_file.filename or "ground_truth.json")
            gt_path.write_bytes(await ground_truth_file.read())

        try:
            result = await annotate_audio(
                audio_path=audio_path,
                ground_truth_json_path=gt_path,
                iou_threshold=iou_threshold,
            )
            return JSONResponse(content=result.model_dump())
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc


# ==========================================
# 6. CLI entry point
# ==========================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local AI annotator agent for bat spectrograms")
    parser.add_argument("--audio-path", type=str, help="Path to the input audio file")
    parser.add_argument("--ground-truth-json", type=str, default=None, help="Optional ground-truth JSON file")
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--serve", action="store_true", help="Start FastAPI service instead of running one file")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


async def run_cli(audio_path: str, ground_truth_json: str | None, iou_threshold: float) -> None:
    result = await annotate_audio(
        audio_path=audio_path,
        ground_truth_json_path=ground_truth_json,
        iou_threshold=iou_threshold,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    args = parse_args()

    if args.serve:
        uvicorn.run("main:app", host=args.host, port=args.port, reload=False)
    else:
        if not args.audio_path:
            raise SystemExit("--audio-path is required unless --serve is used")
        asyncio.run(run_cli(args.audio_path, args.ground_truth_json, args.iou_threshold))