"""Tests for input/output validators."""

import pytest

from forge_trace.config import ForgeTraceConfig
from forge_trace.exceptions import (
    TimelineValidationError,
    VideoValidationError,
)
from forge_trace.schemas import (
    ProcessingMetadata,
    TaskRecord,
    TaskTimeline,
    TimelineInfo,
)
from forge_trace.validators import (
    validate_enrichment_events,
    validate_task_record,
    validate_task_timeline,
    validate_video_duration,
    validate_video_path,
)


class TestValidateVideoPath:
    def test_valid_video(self, synthetic_video, config):
        path = validate_video_path(synthetic_video, config)
        assert path.exists()

    def test_missing_file(self, config):
        with pytest.raises(VideoValidationError, match="does not exist"):
            validate_video_path("/no/such/file.mp4", config)

    def test_empty_video(self, empty_video, config):
        with pytest.raises(VideoValidationError, match="empty"):
            validate_video_path(empty_video, config)

    def test_unsupported_extension(self, non_video_file, config):
        with pytest.raises(VideoValidationError, match="Unsupported"):
            validate_video_path(non_video_file, config)


class TestValidateVideoDuration:
    def test_valid_duration(self, config):
        validate_video_duration(60.0, config)

    def test_zero_duration(self, config):
        with pytest.raises(VideoValidationError, match="positive"):
            validate_video_duration(0.0, config)

    def test_negative_duration(self, config):
        with pytest.raises(VideoValidationError, match="positive"):
            validate_video_duration(-5.0, config)

    def test_exceeds_maximum(self, config):
        config.max_video_duration_s = 100.0
        with pytest.raises(VideoValidationError, match="exceeds maximum"):
            validate_video_duration(200.0, config)


class TestValidateEnrichmentEvents:
    def test_none_returns_empty(self):
        assert validate_enrichment_events(None, "test", 100.0) == []

    def test_valid_events_pass(self):
        events = [
            {"timestamp": 5.0, "label": "bolt"},
            {"start_s": 10.0, "label": "wrench"},
        ]
        result = validate_enrichment_events(events, "test", 100.0)
        assert len(result) == 2

    def test_missing_temporal_field_dropped(self):
        events = [{"label": "bolt"}]  # no timestamp
        result = validate_enrichment_events(events, "test", 100.0)
        assert len(result) == 0

    def test_non_dict_dropped(self):
        events = ["bad", 42, {"timestamp": 1.0}]
        result = validate_enrichment_events(events, "test", 100.0)
        assert len(result) == 1

    def test_timestamps_clamped(self):
        events = [{"timestamp": 200.0, "label": "x"}]
        result = validate_enrichment_events(events, "test", 100.0)
        assert result[0]["timestamp"] == 100.0


class TestValidateTaskRecord:
    def test_valid_record(self):
        tr = TaskRecord(
            task_id="T001", task_name="Pick up bolt", verb="pick up",
            object="bolt", hand="right",
            timeline=TimelineInfo(start_s=2.0, end_s=4.0, duration_s=2.0),
            confidence=0.85, description="test",
        )
        warnings = validate_task_record(tr, video_duration=20.0)
        assert warnings == []

    def test_start_after_duration(self):
        tr = TaskRecord(
            task_id="T001", task_name="X", verb="x",
            timeline=TimelineInfo(start_s=25.0, end_s=27.0, duration_s=2.0),
            confidence=0.5,
        )
        warnings = validate_task_record(tr, video_duration=20.0)
        assert any("start_s" in w for w in warnings)


class TestValidateTaskTimeline:
    def test_valid_timeline(self, sample_task_records):
        tl = TaskTimeline(
            clip_id="test",
            video_path="test.mp4",
            video_duration=20.0,
            total_tasks=4,
            unique_task_types=4,
            unique_task_list=["Pick up bolt", "Position bolt", "Fasten bolt", "Inspect assembly"],
            tasks=sample_task_records,
        )
        result = validate_task_timeline(tl)
        assert result.total_tasks == 4

    def test_mismatched_total_raises(self, sample_task_records):
        with pytest.raises(Exception):
            # TaskTimeline's own model_validator will catch this
            TaskTimeline(
                clip_id="test",
                video_path="test.mp4",
                video_duration=20.0,
                total_tasks=10,
                unique_task_types=4,
                tasks=sample_task_records,
            )
