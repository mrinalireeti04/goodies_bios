"""Evaluation metrics for the Task Intelligence module.

Provides temporal Intersection-over-Union (tIoU) and action recognition
metrics for comparing predicted timelines against manually labelled ground
truth.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from forge_trace.logging_config import get_logger
from forge_trace.schemas import TaskRecord

logger = get_logger("evaluation")


# ---------------------------------------------------------------------------
# Temporal IoU
# ---------------------------------------------------------------------------

def temporal_iou(
    pred_start: float, pred_end: float,
    gt_start: float, gt_end: float,
) -> float:
    """Compute temporal Intersection-over-Union for two time intervals.

    Returns a value in [0.0, 1.0].
    """
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0.0, inter_end - inter_start)

    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    if union <= 0:
        return 0.0
    return intersection / union


# ---------------------------------------------------------------------------
# Ground truth schema
# ---------------------------------------------------------------------------

def _normalise_gt(
    ground_truth: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalise ground-truth entries to a consistent schema.

    Each entry must have at least: ``verb``, ``start_s``, ``end_s``.
    Optional: ``object``, ``hand``.
    """
    out: List[Dict[str, Any]] = []
    for idx, gt in enumerate(ground_truth):
        try:
            out.append({
                "verb": str(gt["verb"]).strip().lower(),
                "object": str(gt.get("object", "")).strip().lower() or None,
                "hand": str(gt.get("hand", "")).strip().lower() or None,
                "start_s": float(gt["start_s"]),
                "end_s": float(gt["end_s"]),
            })
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Skipping ground-truth entry %d: %s", idx, exc)
    return out


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_predictions_to_ground_truth(
    predictions: List[TaskRecord],
    ground_truth: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Match predicted tasks to ground-truth actions using tIoU + verb match.

    Returns a dict with per-match details plus aggregate metrics.
    """
    gt = _normalise_gt(ground_truth)
    matches: List[Dict[str, Any]] = []
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    # Greedy matching: for each prediction, find the best gt match
    for pi, pred in enumerate(predictions):
        best_iou = 0.0
        best_gi = -1
        for gi, g in enumerate(gt):
            if gi in matched_gt:
                continue
            if pred.verb.lower() != g["verb"]:
                continue
            iou = temporal_iou(
                pred.timeline.start_s, pred.timeline.end_s,
                g["start_s"], g["end_s"],
            )
            if iou > best_iou:
                best_iou = iou
                best_gi = gi

        if best_gi >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_gi)
            matched_pred.add(pi)
            matches.append({
                "prediction": pred.task_id,
                "ground_truth_index": best_gi,
                "verb": pred.verb,
                "iou": round(best_iou, 4),
                "pred_start": pred.timeline.start_s,
                "pred_end": pred.timeline.end_s,
                "gt_start": gt[best_gi]["start_s"],
                "gt_end": gt[best_gi]["end_s"],
            })

    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(gt) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    mean_iou = (
        sum(m["iou"] for m in matches) / len(matches)
        if matches
        else 0.0
    )

    return {
        "iou_threshold": iou_threshold,
        "total_predictions": len(predictions),
        "total_ground_truth": len(gt),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
        "matches": matches,
    }


# ---------------------------------------------------------------------------
# Multi-threshold evaluation
# ---------------------------------------------------------------------------

def evaluate_at_multiple_thresholds(
    predictions: List[TaskRecord],
    ground_truth: List[Dict[str, Any]],
    thresholds: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate at several tIoU thresholds.

    Default thresholds: [0.3, 0.5, 0.7].
    """
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7]

    results = []
    for thr in thresholds:
        result = match_predictions_to_ground_truth(
            predictions, ground_truth, iou_threshold=thr,
        )
        results.append(result)
        logger.info(
            "tIoU@%.1f — P=%.3f  R=%.3f  F1=%.3f  mIoU=%.3f",
            thr, result["precision"], result["recall"],
            result["f1_score"], result["mean_iou"],
        )
    return results


def evaluate_timeline(
    predictions: List[TaskRecord],
    ground_truth: List[Dict[str, Any]],
    thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Full evaluation report for a predicted timeline.

    Returns a structured dict suitable for JSON serialisation.
    """
    per_threshold = evaluate_at_multiple_thresholds(
        predictions, ground_truth, thresholds,
    )
    return {
        "evaluation": per_threshold,
        "summary": {
            "best_f1": max(r["f1_score"] for r in per_threshold),
            "best_f1_threshold": per_threshold[
                max(range(len(per_threshold)), key=lambda i: per_threshold[i]["f1_score"])
            ]["iou_threshold"],
        },
    }
