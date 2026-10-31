"""Tests for the action recogniser (parsing, prompt building)."""

import pytest

from forge_trace.action_recognizer import (
    build_coarse_prompt,
    parse_action_list,
)
from forge_trace.exceptions import GeminiResponseParseError
from forge_trace.schemas import VideoSegment


class TestBuildCoarsePrompt:
    def test_no_segment(self):
        prompt = build_coarse_prompt()
        assert "action verb" in prompt.lower() or "verb" in prompt.lower()
        assert "JSON" in prompt

    def test_with_segment(self):
        seg = VideoSegment(index=1, start_s=60.0, end_s=120.0, duration_s=60.0)
        prompt = build_coarse_prompt(seg)
        assert "60.0" in prompt
        assert "120.0" in prompt


class TestParseActionList:
    def test_valid_array(self):
        raw = [
            {
                "verb": "pick up",
                "object": "bolt",
                "hand": "right",
                "start_s": 2.0,
                "end_s": 4.0,
                "confidence": 0.9,
                "description": "Picks up a bolt",
            },
            {
                "verb": "place",
                "object": "bolt",
                "start_s": 5.0,
                "end_s": 7.0,
                "confidence": 0.8,
                "description": "Places the bolt",
            },
        ]
        actions = parse_action_list(raw)
        assert len(actions) == 2
        assert actions[0].verb == "pick up"
        assert actions[1].verb == "place"

    def test_wrapped_in_dict(self):
        raw = {
            "actions": [
                {"verb": "inspect", "start_s": 0, "end_s": 2, "confidence": 0.7},
            ]
        }
        actions = parse_action_list(raw)
        assert len(actions) == 1

    def test_with_segment_offset(self):
        seg = VideoSegment(index=1, start_s=60.0, end_s=120.0, duration_s=60.0)
        raw = [
            {"verb": "pick up", "start_s": 5.0, "end_s": 8.0, "confidence": 0.8},
        ]
        actions = parse_action_list(raw, segment=seg)
        # Timestamps should be offset by 60s
        assert actions[0].start_s == 65.0
        assert actions[0].end_s == 68.0

    def test_clamped_to_segment_end(self):
        seg = VideoSegment(index=0, start_s=0.0, end_s=60.0, duration_s=60.0)
        raw = [
            {"verb": "fasten", "start_s": 55.0, "end_s": 75.0, "confidence": 0.8},
        ]
        actions = parse_action_list(raw, segment=seg)
        assert actions[0].end_s == 60.0

    def test_invalid_entries_skipped(self):
        raw = [
            {"verb": "pick up", "start_s": 0, "end_s": 2, "confidence": 0.9},
            "not a dict",
            {"verb": "", "start_s": 0, "end_s": 1},  # empty verb
            {"verb": "place", "start_s": "bad", "end_s": 1},  # bad type
        ]
        actions = parse_action_list(raw)
        assert len(actions) == 1

    def test_empty_list(self):
        actions = parse_action_list([])
        assert actions == []

    def test_unexpected_type_raises(self):
        with pytest.raises(GeminiResponseParseError):
            parse_action_list("not json")

    def test_dict_without_known_key_raises(self):
        with pytest.raises(GeminiResponseParseError):
            parse_action_list({"foo": "bar"})

    def test_null_object_and_hand(self):
        raw = [
            {"verb": "move", "start_s": 1.0, "end_s": 3.0,
             "object": None, "hand": None, "confidence": 0.6},
        ]
        actions = parse_action_list(raw)
        assert actions[0].object is None
        assert actions[0].hand is None

    def test_unknown_strings_cleaned(self):
        raw = [
            {"verb": "move", "start_s": 1.0, "end_s": 3.0,
             "object": "unknown", "hand": "N/A", "confidence": 0.6},
        ]
        actions = parse_action_list(raw)
        assert actions[0].object is None
        assert actions[0].hand is None

    def test_confidence_clamped(self):
        raw = [
            {"verb": "push", "start_s": 0, "end_s": 1, "confidence": 5.0},
        ]
        actions = parse_action_list(raw)
        assert actions[0].confidence == 1.0
