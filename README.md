# Bioacoustic Annotation Agent

An AI-powered pipeline for automated detection and annotation of bat echolocation calls in audio recordings. A vision-language model (VLM) running locally via Ollama visually inspects spectrograms and outputs time-frequency bounding boxes for each detected call.

---

## How It Works

The agent follows a visual-first workflow:

1. Generate an overview spectrogram of the full audio file.
2. Zoom into regions of interest for a closer look.
3. Propose bounding boxes from visual estimates, then tighten them with DSP tools.
4. Deduplicate, validate, score, and persist the final annotations.

The model reasons like a human annotator — it decides *where* calls are visually, and uses signal-processing tools only to refine and confirm those decisions.

---

## Pipeline Stages

| Stage | Description |
|-------|-------------|
| 0 | Config validation |
| 1 | Audio pre-analysis (duration, noise floor, silence detection) |
| 2 | Runtime dependency assembly |
| 3 | Agent annotation — multi-scale spectrogram + VLM reasoning |
| 4 | Post-processing — validate, merge overlapping boxes, clean up |
| 5 | Quality scoring per box |
| 6 | Persist JSON + render annotated overlay image |
| 7 | Optional evaluation against ground-truth labels |

---

## Project Structure

```
├── main.py            # Entry point: pipeline orchestration, FastAPI app, CLI
├── agent_tools.py     # Tool registrations for the PydanticAI agent
├── audio_analysis.py  # Audio health inspection and noise floor estimation
├── audio_io.py        # Audio file reading (mono float32)
├── box_ops.py         # Bounding box operations: refine, split, merge, score, classify
├── config.py          # Runtime configuration and environment variable overrides
├── deps.py            # Agent runtime dependencies (RunDeps, spectrogram cache)
├── dsp.py             # DSP utilities: STFT, energy measurement, active region detection
├── evaluation.py      # Greedy IoU matching, precision / recall / F1 metrics
├── persistence.py     # Load / save annotation JSON
├── schemas.py         # Pydantic models: BBox, AnnotationOutput, PipelineResult, etc.
└── spectrogram.py     # Spectrogram generation and bounding box overlay rendering
```

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/) running locally with a vision-capable model (default: `qwen3-vl:latest`)
- System dependencies for audio processing: `libsndfile`

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Key packages: `fastapi`, `uvicorn`, `pydantic-ai`, `soundfile`, `numpy`, `matplotlib`, `librosa`.

---

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `qwen3-vl:latest` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama API endpoint |
| `OUTPUT_DIR` | `outputs` | Directory for results |
| `IOU_THRESHOLD` | `0.5` | IoU threshold for evaluation and merging |
| `N_FFT` | `1024` | STFT window size |
| `HOP_LENGTH` | `256` | STFT hop length |
| `FMIN_HZ` | `0` | Spectrogram minimum frequency |
| `FMAX_HZ` | `120000` | Spectrogram maximum frequency |

---

## Usage

### CLI

```bash
python main.py --audio-path recording.wav
```

With ground-truth evaluation:

```bash
python main.py --audio-path recording.wav --ground-truth-json labels.json --iou-threshold 0.5
```

### HTTP Server

```bash
python main.py --serve --host 0.0.0.0 --port 8000
```

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service status and model info |
| `POST` | `/annotate-by-path` | Annotate audio by file path (form field: `audio_path`) |
| `POST` | `/annotate-upload` | Annotate uploaded audio file |

Example request:

```bash
curl -X POST http://localhost:8000/annotate-by-path \
  -F "audio_path=/data/recording.wav"
```

---

## Output

Each run produces:

- `<stem>_predictions.json` — detected bounding boxes with confidence scores
- `<stem>_overview.png` — spectrogram of the full recording
- `<stem>_overlay.png` — spectrogram with annotated boxes drawn

### Bounding Box Format

```json
{
  "t_min": 0.312,
  "t_max": 0.327,
  "f_min": 42000,
  "f_max": 85000,
  "label": "bat_call",
  "confidence": 0.91,
  "source": "refined"
}
```

Coordinates are absolute: seconds (time) and Hz (frequency).

---

## Evaluation

If a ground-truth JSON file is provided, the pipeline computes:

- **Precision** — fraction of predicted boxes that match a ground-truth box
- **Recall** — fraction of ground-truth boxes that were detected
- **F1 score** — harmonic mean of precision and recall

Matching uses greedy one-to-one IoU assignment with a configurable threshold (default 0.5). Only boxes with the same label are considered for matching.

---

## Annotation Rules

The agent follows strict constraints to minimise false positives:

- All coordinates are **absolute** (seconds / Hz), never relative to a window.
- Boxes must be **tight** — closely wrapping visible call energy.
- Minimum duration: **2 ms**. Minimum bandwidth: **1 kHz**.
- Noise, broadband transients, and ambiguous blobs are not annotated.
- Default label: `bat_call`. Labels `noise`, `insect`, `artifact` require clear justification.
