"""Tests for the evaluation module (tIoU metrics)."""

import pytest

from forge_trace.evaluation import (
    evaluate_timeline,
    match_predictions_to_ground_truth,
    temporal_iou,
)
from forge_trace.schemas import TaskRecord, TimelineInfo


class TestTemporalIoU:
    def test_perfect_overlap(self):
        assert temporal_iou(2.0, 5.0, 2.0, 5.0) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert temporal_iou(0.0, 3.0, 5.0, 8.0) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # [2, 6] vs [4, 8] → intersection=2, union=6
        iou = temporal_iou(2.0, 6.0, 4.0, 8.0)
        assert iou == pytest.approx(2.0 / 6.0, abs=0.01)

    def test_contained(self):
        # [2, 8] vs [3, 7] → intersection=4, union=6
        iou = temporal_iou(2.0, 8.0, 3.0, 7.0)
        assert iou == pytest.approx(4.0 / 6.0, abs=0.01)

    def test_zero_length_intervals(self):
        assert temporal_iou(5.0, 5.0, 5.0, 5.0) == pytest.approx(0.0)


class TestMatchPredictions:
    def test_perfect_matches(self, sample_task_records, sample_ground_truth):
        result = match_predictions_to_ground_truth(
            sample_task_records, sample_ground_truth, iou_threshold=0.5,
        )
        assert result["true_positives"] == 4
        assert result["false_positives"] == 0
        assert result["false_negatives"] == 0
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["f1_score"] == pytest.approx(1.0)

    def test_no_matches_high_threshold(self, sample_task_records, sample_ground_truth):
        # With an impossibly high threshold, nothing matches
        result = match_predictions_to_ground_truth(
            sample_task_records, sample_ground_truth, iou_threshold=0.999,
        )
        assert result["true_positives"] == 0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_extra_predictions(self, sample_ground_truth):
        """Extra predictions count as false positives."""
        predictions = [
            TaskRecord(
                task_id=f"T{i:03d}", task_name=f"A{i}", verb=gt["verb"],
                timeline=TimelineInfo(
                    start_s=gt["start_s"], end_s=gt["end_s"],
                    duration_s=round(gt["end_s"] - gt["start_s"], 4),
                ),
                confidence=0.8,
            )
            for i, gt in enumerate(sample_ground_truth, 1)
        ]
        # Add an extra prediction with no matching GT
        predictions.append(
            TaskRecord(
                task_id="T005", task_name="Extra", verb="weld",
                timeline=TimelineInfo(start_s=20.0, end_s=25.0, duration_s=5.0),
                confidence=0.7,
            )
        )
        result = match_predictions_to_ground_truth(
            predictions, sample_ground_truth, iou_threshold=0.3,
        )
        assert result["false_positives"] == 1
        assert result["true_positives"] == 4

    def test_missing_predictions(self, sample_ground_truth):
        """Missing predictions count as false negatives."""
        predictions = [
            TaskRecord(
                task_id="T001", task_name="A1", verb=sample_ground_truth[0]["verb"],
                timeline=TimelineInfo(
                    start_s=sample_ground_truth[0]["start_s"],
                    end_s=sample_ground_truth[0]["end_s"],
                    duration_s=round(
                        sample_ground_truth[0]["end_s"] - sample_ground_truth[0]["start_s"], 4
                    ),
                ),
                confidence=0.8,
            )
        ]
        result = match_predictions_to_ground_truth(
            predictions, sample_ground_truth, iou_threshold=0.3,
        )
        assert result["true_positives"] == 1
        assert result["false_negatives"] == 3

    def test_verb_mismatch_no_match(self, sample_ground_truth):
        """Predictions with wrong verbs should not match."""
        predictions = [
            TaskRecord(
                task_id="T001", task_name="Wrong verb", verb="dance",
                timeline=TimelineInfo(
                    start_s=sample_ground_truth[0]["start_s"],
                    end_s=sample_ground_truth[0]["end_s"],
                    duration_s=round(
                        sample_ground_truth[0]["end_s"] - sample_ground_truth[0]["start_s"], 4
                    ),
                ),
                confidence=0.8,
            )
        ]
        result = match_predictions_to_ground_truth(
            predictions, sample_ground_truth, iou_threshold=0.3,
        )
        assert result["true_positives"] == 0

    def test_empty_predictions(self, sample_ground_truth):
        result = match_predictions_to_ground_truth(
            [], sample_ground_truth, iou_threshold=0.5,
        )
        assert result["true_positives"] == 0
        assert result["false_negatives"] == 4
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_empty_ground_truth(self, sample_task_records):
        result = match_predictions_to_ground_truth(
            sample_task_records, [], iou_threshold=0.5,
        )
        assert result["true_positives"] == 0
        assert result["false_positives"] == 4
        assert result["recall"] == 0.0


class TestEvaluateTimeline:
    def test_multi_threshold_evaluation(self, sample_task_records, sample_ground_truth):
        report = evaluate_timeline(
            sample_task_records, sample_ground_truth,
            thresholds=[0.3, 0.5, 0.7],
        )
        assert "evaluation" in report
        assert "summary" in report
        assert len(report["evaluation"]) == 3
        assert report["summary"]["best_f1"] >= 0.0
