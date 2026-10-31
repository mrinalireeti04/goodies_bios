"""Tests for action normaliser: normalisation, deduplication, cross-segment merge."""

import pytest

from forge_trace.action_normalizer import (
    filter_low_confidence,
    merge_cross_segment_actions,
    merge_duplicate_actions,
    normalize_actions,
    normalize_and_deduplicate,
)
from forge_trace.config import ForgeTraceConfig
from forge_trace.schemas import RefinedAction


def _make_action(verb, start, end, obj=None, confidence=0.8):
    return RefinedAction(
        verb=verb, object=obj, start_s=start, end_s=end,
        duration_s=round(end - start, 4), confidence=confidence,
        description=f"{verb} action",
    )


class TestNormalizeActions:
    def test_verb_aliases(self):
        actions = [
            _make_action("picks up", 0, 1),
            _make_action("grabs", 2, 3),
            _make_action("tightens", 4, 5),
            _make_action("inspects", 6, 7),
        ]
        result = normalize_actions(actions)
        assert result[0].verb == "pick up"
        assert result[1].verb == "pick up"
        assert result[2].verb == "fasten"
        assert result[3].verb == "inspect"

    def test_unknown_verb_unchanged(self):
        actions = [_make_action("calibrate", 0, 1)]
        result = normalize_actions(actions)
        assert result[0].verb == "calibrate"


class TestMergeDuplicates:
    def test_exact_duplicates_merged(self, config):
        actions = [
            _make_action("pick up", 2.0, 4.0, confidence=0.85),
            _make_action("pick up", 2.0, 4.0, confidence=0.80),
        ]
        result = merge_duplicate_actions(actions, config)
        assert len(result) == 1
        # Confidence is averaged
        assert result[0].confidence == pytest.approx(0.825, abs=0.01)

    def test_different_verbs_not_merged(self, config):
        actions = [
            _make_action("pick up", 2.0, 4.0),
            _make_action("place", 2.0, 4.0),
        ]
        result = merge_duplicate_actions(actions, config)
        assert len(result) == 2

    def test_non_overlapping_same_verb_kept(self, config):
        # Same verb, but well separated in time — different instances
        actions = [
            _make_action("pick up", 2.0, 4.0),
            _make_action("pick up", 20.0, 22.0),
        ]
        result = merge_duplicate_actions(actions, config)
        assert len(result) == 2

    def test_high_iou_merged(self, config):
        config.duplicate_iou_threshold = 0.5
        actions = [
            _make_action("pick up", 2.0, 5.0),
            _make_action("pick up", 2.5, 5.5),
        ]
        result = merge_duplicate_actions(actions, config)
        assert len(result) == 1
        # Merged span should cover both
        assert result[0].start_s == 2.0
        assert result[0].end_s == 5.5

    def test_low_iou_not_merged(self, config):
        config.duplicate_iou_threshold = 0.9
        actions = [
            _make_action("pick up", 2.0, 5.0),
            _make_action("pick up", 4.0, 7.0),
        ]
        result = merge_duplicate_actions(actions, config)
        assert len(result) == 2


class TestMergeCrossSegment:
    def test_adjacent_same_verb_merged(self):
        actions = [
            _make_action("fasten", 50.0, 60.0),
            _make_action("fasten", 55.0, 65.0),  # overlaps in 10s window
        ]
        result = merge_cross_segment_actions(actions, overlap_s=10.0)
        assert len(result) == 1
        assert result[0].start_s == 50.0
        assert result[0].end_s == 65.0

    def test_different_verbs_not_merged(self):
        actions = [
            _make_action("fasten", 50.0, 60.0),
            _make_action("inspect", 58.0, 65.0),
        ]
        result = merge_cross_segment_actions(actions, overlap_s=10.0)
        assert len(result) == 2

    def test_no_overlap_no_merge(self):
        actions = [
            _make_action("pick up", 10.0, 15.0),
            _make_action("pick up", 30.0, 35.0),
        ]
        result = merge_cross_segment_actions(actions, overlap_s=10.0)
        assert len(result) == 2


class TestFilterLowConfidence:
    def test_filters_below_threshold(self, config):
        config.confidence_threshold = 0.5
        actions = [
            _make_action("pick up", 0, 1, confidence=0.9),
            _make_action("place", 2, 3, confidence=0.3),
            _make_action("inspect", 4, 5, confidence=0.5),
        ]
        result = filter_low_confidence(actions, config)
        assert len(result) == 2
        assert all(a.confidence >= 0.5 for a in result)


class TestNormalizeAndDeduplicate:
    def test_full_pipeline(self, config):
        config.duplicate_iou_threshold = 0.5
        config.confidence_threshold = 0.2
        actions = [
            _make_action("picks up", 2.0, 4.0, confidence=0.9),
            _make_action("grab", 2.0, 4.0, confidence=0.85),  # dup after normalise
            _make_action("places", 5.0, 7.0, confidence=0.8),
            _make_action("inspect", 10.0, 11.0, confidence=0.1),  # low conf
        ]
        result = normalize_and_deduplicate(actions, config)
        # "picks up" and "grab" should merge to one "pick up"
        # "places" becomes "place"
        # "inspect" at 0.1 should be filtered
        assert len(result) == 2
        assert result[0].verb == "pick up"
        assert result[1].verb == "place"

    def test_repeated_actions_preserved(self, config):
        """Same verb but at different times must remain separate."""
        actions = [
            _make_action("pick up", 2.0, 4.0),
            _make_action("pick up", 10.0, 12.0),
            _make_action("pick up", 20.0, 22.0),
        ]
        result = normalize_and_deduplicate(actions, config)
        assert len(result) == 3

    def test_chronological_ordering(self, config):
        actions = [
            _make_action("inspect", 10.0, 12.0),
            _make_action("pick up", 2.0, 4.0),
            _make_action("place", 6.0, 8.0),
        ]
        result = normalize_and_deduplicate(actions, config)
        starts = [a.start_s for a in result]
        assert starts == sorted(starts)
