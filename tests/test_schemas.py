"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from forge_trace.schemas import (
    ActionPrediction,
    RefinedAction,
    TaskRecord,
    TaskTimeline,
    TimelineInfo,
    VideoMetadata,
    VideoSegment,
    ProcessingMetadata,
)


class TestVideoSegment:
    def test_valid_segment(self):
        seg = VideoSegment(index=0, start_s=0.0, end_s=10.0, duration_s=10.0)
        assert seg.start_s == 0.0
        assert seg.end_s == 10.0

    def test_invalid_ordering(self):
        with pytest.raises(ValidationError, match="end_s must be greater"):
            VideoSegment(index=0, start_s=10.0, end_s=5.0, duration_s=5.0)

    def test_zero_duration(self):
        with pytest.raises(ValidationError):
            VideoSegment(index=0, start_s=5.0, end_s=5.0, duration_s=0.0)


class TestActionPrediction:
    def test_valid_prediction(self):
        ap = ActionPrediction(
            verb="pick up", start_s=1.0, end_s=3.0,
            object="bolt", hand="right", confidence=0.9,
        )
        assert ap.duration_s == 2.0
        assert ap.hand == "right"

    def test_hand_normalisation(self):
        ap = ActionPrediction(verb="pick up", start_s=0, end_s=1, hand="LEFT")
        assert ap.hand == "left"

    def test_invalid_hand_becomes_none(self):
        ap = ActionPrediction(verb="pick up", start_s=0, end_s=1, hand="maybe")
        assert ap.hand is None

    def test_end_before_start_rejected(self):
        with pytest.raises(ValidationError, match="end_s.*must be .*greater than start_s"):
            ActionPrediction(verb="pick up", start_s=5.0, end_s=3.0)

    def test_confidence_clamped(self):
        with pytest.raises(ValidationError):
            ActionPrediction(verb="x", start_s=0, end_s=1, confidence=1.5)

    def test_empty_verb_rejected(self):
        with pytest.raises(ValidationError):
            ActionPrediction(verb="", start_s=0, end_s=1)


class TestTimelineInfo:
    def test_valid(self):
        ti = TimelineInfo(start_s=1.0, end_s=3.0, duration_s=2.0)
        assert ti.duration_s == 2.0

    def test_duration_mismatch(self):
        with pytest.raises(ValidationError, match="duration_s.*!="):
            TimelineInfo(start_s=1.0, end_s=3.0, duration_s=5.0)

    def test_end_before_start(self):
        with pytest.raises(ValidationError):
            TimelineInfo(start_s=5.0, end_s=2.0, duration_s=3.0)


class TestTaskTimeline:
    def test_count_validation(self, sample_task_records):
        tl = TaskTimeline(
            clip_id="test",
            video_path="test.mp4",
            video_duration=20.0,
            total_tasks=4,
            unique_task_types=4,
            unique_task_list=["Pick up bolt", "Position bolt", "Fasten bolt", "Inspect assembly"],
            tasks=sample_task_records,
        )
        assert tl.total_tasks == 4

    def test_mismatched_total_raises(self, sample_task_records):
        with pytest.raises(ValidationError, match="total_tasks"):
            TaskTimeline(
                clip_id="test",
                video_path="test.mp4",
                video_duration=20.0,
                total_tasks=99,
                unique_task_types=4,
                tasks=sample_task_records,
            )

    def test_empty_timeline(self):
        tl = TaskTimeline(
            clip_id="empty",
            video_path="x.mp4",
            video_duration=10.0,
            total_tasks=0,
            unique_task_types=0,
            tasks=[],
        )
        assert tl.total_tasks == 0


class TestModelConfidenceAndToolSchemas:
    def test_action_prediction_syncs_confidence(self):
        ap1 = ActionPrediction(verb="fasten", start_s=0, end_s=1, model_confidence_score=0.82)
        assert ap1.confidence == 0.82
        assert ap1.model_confidence_score == 0.82

        ap2 = ActionPrediction(verb="fasten", start_s=0, end_s=1, confidence=0.75)
        assert ap2.model_confidence_score == 0.75
        assert ap2.confidence == 0.75

    def test_refined_action_syncs_confidence(self):
        ra = RefinedAction(verb="fasten", start_s=0, end_s=1, duration_s=1.0, model_confidence_score=0.91)
        assert ra.confidence == 0.91
        assert ra.model_confidence_score == 0.91

    def test_task_record_ps2_tool_fields(self):
        tr = TaskRecord(
            task_id="T001",
            task_name="Cut wood piece",
            verb="cut",
            object="wood piece",
            hand="right",
            timeline=TimelineInfo(start_s=0.0, end_s=2.0, duration_s=2.0),
            model_confidence_score=0.85,
            tool="circular saw",
            tool_hand="right",
            tool_match_confidence=0.92,
            tool_association_status="associated",
            tool_evidence="Tool hold associated with high temporal overlap",
        )
        assert tr.tool == "circular saw"
        assert tr.tool_hand == "right"
        assert tr.tool_association_status == "associated"
        assert tr.tool_match_confidence == 0.92
        assert tr.confidence == 0.85
        assert tr.model_confidence_score == 0.85

    def test_processing_metadata_refinement_status_sync(self):
        pm_applied = ProcessingMetadata(timestamp_refinement_status="applied")
        assert pm_applied.timestamp_refinement_applied is True

        pm_skipped = ProcessingMetadata(timestamp_refinement_status="skipped")
        assert pm_skipped.timestamp_refinement_applied is False

        pm_failed = ProcessingMetadata(timestamp_refinement_status="failed")
        assert pm_failed.timestamp_refinement_applied is False

        pm_partial = ProcessingMetadata(timestamp_refinement_status="partial")
        assert pm_partial.timestamp_refinement_applied is True
