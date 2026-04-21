"""Prediction evaluation: greedy IoU matching and F1/precision/recall metrics."""
from __future__ import annotations

from typing import List

from box_ops import calculate_iou
from schemas import BBox, EvaluationResult


def evaluate_predictions(
    predictions: List[BBox],
    ground_truths: List[BBox],
    iou_threshold: float = 0.5,
) -> EvaluationResult:
    """Greedy one-to-one matching by descending IoU, restricted to matching labels.

    A predicted box is a TP if it matches a GT box of the same label with IoU >= threshold.
    Each GT and prediction can be matched at most once.
    """
    candidates: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(predictions):
        for gi, gt in enumerate(ground_truths):
            if pred.label != gt.label:
                continue
            iou = calculate_iou(pred, gt)
            if iou >= iou_threshold:
                candidates.append((iou, pi, gi))

    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_preds: set[int] = set()
    matched_gts: set[int] = set()
    matched_pairs: list[dict] = []

    for iou, pi, gi in candidates:
        if pi in matched_preds or gi in matched_gts:
            continue
        matched_preds.add(pi)
        matched_gts.add(gi)
        matched_pairs.append({
            "pred_index": pi,
            "gt_index": gi,
            "iou": round(iou, 6),
            "label": predictions[pi].label,
        })

    tp = len(matched_pairs)
    fp = len(predictions) - tp
    fn = len(ground_truths) - tp
    precision = tp / len(predictions) if predictions else 0.0
    recall = tp / len(ground_truths) if ground_truths else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    mean_iou = sum(p["iou"] for p in matched_pairs) / tp if tp else 0.0

    return EvaluationResult(
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1_score=round(f1, 6),
        mean_iou=round(mean_iou, 6),
        tp=tp, fp=fp, fn=fn,
        iou_threshold=iou_threshold,
        matched_pairs=matched_pairs,
    )
