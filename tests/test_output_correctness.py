"""Tests for FORGE TRACE output correctness and review requirements.

Covers:
- Missing timestamps handling.
- Zero-duration task rejection.
- Timestamps exceeding video duration.
- Timestamp clamping without degenerate zero-duration collapse.
- Preservation of repeated actions vs long continuous actions.
- Elimination of duplicate words in task names.
- Exclusion of duplicate confidence fields in JSON.
- Boundary refinement failure handling.
- Final JSON timeline validation.
"""

import json
import pytest
from pydantic import ValidationError

from forge_trace.action_normalizer import (
    merge_cross_segment_actions,
    normalize_actions,
    normalize_and_deduplicate,
)
from forge_trace.action_recognizer import parse_action_list
from forge_trace.config import ForgeTraceConfig
from forge_trace.exceptions import TimelineValidationError
from forge_trace.pipeline import _clamp_to_duration
from forge_trace.schemas import (
    ActionPrediction,
    RefinedAction,
    TaskRecord,
    TaskTimeline,
    TimelineInfo,
    VideoSegment,
)
from forge_trace.task_builder import _build_task_name, assign_task_ids
from forge_trace.temporal_refiner import refine_boundaries_with_frames
from forge_trace.validators import validate_task_record, validate_task_timeline
from unittest.mock import MagicMock


def _make_action(verb="fasten", start=5.0, end=10.0, obj="wood piece", conf=0.85, tool=None, tool_hint=None):
    return RefinedAction(
        verb=verb,
        object=obj,
        hand="right",
        start_s=start,
        end_s=end,
        duration_s=round(end - start, 4),
        model_confidence_score=conf,
        confidence=conf,
        description=f"{verb} {obj}",
        evidence="Gemini video analysis",
        tool=tool,
        tool_hint=tool_hint,
    )


class TestMissingTimestamps:
    def test_missing_start_or_end_dropped(self):
        raw = [
            {"verb": "fasten", "object": "bolt", "start_s": None, "end_s": 5.0},
            {"verb": "cut", "object": "wood piece", "start_s": 2.0},  # missing end_s
            {"verb": "align", "object": "panel"},  # missing both
            {"verb": "mark", "object": "wood piece", "start_s": 10.0, "end_s": 15.0},
        ]
        parsed = parse_action_list(raw)
        assert len(parsed) == 1
        assert parsed[0].verb == "mark"
        assert parsed[0].start_s == 10.0
        assert parsed[0].end_s == 15.0


class TestZeroDurationRejection:
    def test_schema_rejects_zero_duration(self):
        with pytest.raises(ValidationError):
            TimelineInfo(start_s=180.0, end_s=180.0, duration_s=0.0)

        with pytest.raises(ValidationError):
            ActionPrediction(verb="mark", start_s=180.0, end_s=180.0)

        with pytest.raises(ValidationError):
            RefinedAction(verb="mark", start_s=180.0, end_s=180.0, duration_s=0.0)

    def test_parser_rejects_zero_duration(self):
        raw = [
            {"verb": "mark", "start_s": 180.0, "end_s": 180.0},
            {"verb": "cut", "start_s": 5.0, "end_s": 5.0},
        ]
        parsed = parse_action_list(raw)
        assert len(parsed) == 0

    def test_normalizer_filters_zero_duration(self):
        cfg = ForgeTraceConfig(gemini_api_key="mock")
        actions = [
            _make_action("align", 10.0, 15.0),
        ]
        # Even if an invalid action were manually injected, normalizer drops it
        actions_with_degenerate = list(actions)
        # Directly bypass constructor using model_construct to test normalizer defense
        degen = RefinedAction.model_construct(
            verb="cut", object="wood", hand="both", start_s=180.0, end_s=180.0, duration_s=0.0,
            model_confidence_score=0.9, confidence=0.9, description="", evidence="",
            start_uncertainty_s=0.0, end_uncertainty_s=0.0, is_refined=False, tool=None, tool_hint=None,
        )
        actions_with_degenerate.append(degen)

        cleaned = normalize_and_deduplicate(actions_with_degenerate, cfg)
        assert len(cleaned) == 1
        assert cleaned[0].verb == "align"


class TestTimestampsExceedingDuration:
    def test_validators_reject_timestamps_exceeding_duration(self):
        tr = TaskRecord(
            task_id="T012",
            task_name="Mark wood piece",
            verb="mark",
            object="wood piece",
            timeline=TimelineInfo(start_s=180.0, end_s=185.0, duration_s=5.0),
        )
        with pytest.raises(TimelineValidationError, match="start_s.*>= video duration"):
            validate_task_record(tr, video_duration=180.0)

    def test_validators_reject_end_exceeding_duration(self):
        tr = TaskRecord(
            task_id="T012",
            task_name="Mark wood piece",
            verb="mark",
            object="wood piece",
            timeline=TimelineInfo(start_s=175.0, end_s=185.0, duration_s=10.0),
        )
        with pytest.raises(TimelineValidationError, match="end_s.*exceeds video duration"):
            validate_task_record(tr, video_duration=180.0)


class TestTimestampClamping:
    def test_clamping_drops_out_of_bounds_instead_of_zero_duration(self):
        actions = [
            _make_action("align", 170.0, 178.0),
            _make_action("mark", 180.0, 185.0),  # starts at boundary
            _make_action("cut", 185.0, 190.0),   # starts past boundary
        ]
        clamped = _clamp_to_duration(actions, duration_s=180.0)
        assert len(clamped) == 1
        assert clamped[0].verb == "align"
        assert clamped[0].start_s == 170.0
        assert clamped[0].end_s == 178.0
        assert clamped[0].duration_s == 8.0


class TestRepeatedActionsVsContinuous:
    def test_repeated_fastening_actions_not_merged_in_single_request_mode(self):
        cfg = ForgeTraceConfig(
            gemini_api_key="mock",
            single_request_mode=True,
            segment_overlap_s=15.0,
        )
        # Distinct fastening cycles separated by 2 seconds
        actions = [
            _make_action("fasten", 51.0, 62.0),
            _make_action("fasten", 64.0, 75.0),
            _make_action("fasten", 78.0, 103.0),
        ]
        cleaned = normalize_and_deduplicate(actions, cfg)
        assert len(cleaned) == 3
        assert [a.start_s for a in cleaned] == [51.0, 64.0, 78.0]
        assert [a.end_s for a in cleaned] == [62.0, 75.0, 103.0]


class TestTaskNamingAndVerbNormalization:
    def test_no_duplicate_words_in_task_name(self):
        a1 = _make_action(verb="align wood piece", obj="wood piece")
        a2 = _make_action(verb="cut wood piece", obj="wood piece")
        a3 = _make_action(verb="mark wood piece", obj="wood piece")
        a4 = _make_action(verb="fasten with nail gun", obj="wood piece")

        assert _build_task_name(a1) == "Align wood piece"
        assert _build_task_name(a2) == "Cut wood piece"
        assert _build_task_name(a3) == "Mark wood piece"
        assert _build_task_name(a4) == "Fasten wood piece with nail gun"

    def test_assign_task_ids_cleans_verb_field(self):
        actions = [
            _make_action(verb="align wood piece", obj="wood piece"),
            _make_action(verb="cut wood piece", obj="wood piece"),
            _make_action(verb="mark wood piece", obj="wood piece"),
            _make_action(verb="fasten with nail gun", obj="wood piece"),
        ]
        records = assign_task_ids(actions)
        assert records[0].verb == "align"
        assert records[0].task_name == "Align wood piece"

        assert records[1].verb == "cut"
        assert records[1].task_name == "Cut wood piece"

        assert records[2].verb == "mark"
        assert records[2].task_name == "Mark wood piece"

        assert records[3].verb == "fasten"
        assert records[3].task_name == "Fasten wood piece with nail gun"


class TestConfidenceFieldExclusion:
    def test_duplicate_confidence_field_excluded_from_json(self):
        tr = TaskRecord(
            task_id="T001",
            task_name="Align wood piece",
            verb="align",
            object="wood piece",
            timeline=TimelineInfo(start_s=0.0, end_s=5.0, duration_s=5.0),
            model_confidence_score=0.88,
        )
        dumped = tr.model_dump()
        assert "model_confidence_score" in dumped
        assert dumped["model_confidence_score"] == 0.88
        assert "confidence" not in dumped

        json_str = tr.model_dump_json()
        data = json.loads(json_str)
        assert "model_confidence_score" in data
        assert "confidence" not in data


class TestRefinementFailureSafety:
    def test_refinement_failure_retains_coarse_timestamps(self):
        actions = [_make_action("fasten", start=5.0, end=10.0)]
        cfg = ForgeTraceConfig(gemini_api_key="mock", enable_frame_refinement=True)
        client = MagicMock()

        # Mock frame refinement to return degenerate/inverted boundary (e.g. start > end)
        from unittest.mock import patch
        with patch("forge_trace.temporal_refiner.refine_boundary_with_frames") as mock_refine:
            mock_refine.side_effect = lambda client, path, act, b_type, cfg: (
                10.0 if b_type == "start" else 5.0
            )
            updated, status = refine_boundaries_with_frames(client, "dummy.mp4", actions, cfg)

            # Succeeded should be 0, status failed
            assert status == "failed"
            # Original coarse timestamps preserved
            assert updated[0].start_s == 5.0
            assert updated[0].end_s == 10.0
            assert updated[0].duration_s == 5.0


class TestFinalJsonValidation:
    def test_final_timeline_passes_validation(self):
        records = [
            TaskRecord(
                task_id="T001",
                task_name="Align wood piece",
                verb="align",
                object="wood piece",
                hand="both",
                timeline=TimelineInfo(start_s=0.0, end_s=5.0, duration_s=5.0),
                model_confidence_score=0.85,
            ),
            TaskRecord(
                task_id="T002",
                task_name="Fasten wood piece with nail gun",
                verb="fasten",
                object="wood piece",
                hand="right",
                timeline=TimelineInfo(start_s=7.0, end_s=15.0, duration_s=8.0),
                model_confidence_score=0.92,
            ),
        ]
        timeline = TaskTimeline(
            clip_id="test_clip",
            video_path="sample.mp4",
            video_duration=180.0,
            total_tasks=2,
            unique_task_types=2,
            unique_task_list=["Align wood piece", "Fasten wood piece with nail gun"],
            tasks=records,
        )
        validated = validate_task_timeline(timeline)
        assert validated.total_tasks == 2

        # Verify JSON output structure
        json_output = json.loads(validated.model_dump_json())
        for task in json_output["tasks"]:
            assert "confidence" not in task
            assert "model_confidence_score" in task
            assert task["timeline"]["duration_s"] > 0
            assert task["timeline"]["end_s"] <= 180.0
