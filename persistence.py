"""Annotation JSON persistence — save and load AnnotationOutput."""
from __future__ import annotations

import json
from pathlib import Path

from schemas import AnnotationOutput


def save_annotation_json(output: AnnotationOutput, save_path: str | Path) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(output.model_dump_json(indent=2), encoding="utf-8")
    return save_path


def load_annotation_json(json_path: str | Path) -> AnnotationOutput:
    json_path = Path(json_path)
    return AnnotationOutput.model_validate(
        json.loads(json_path.read_text(encoding="utf-8"))
    )
