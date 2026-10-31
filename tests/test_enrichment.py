"""Tests for the enrichment module."""

import pytest

from forge_trace.enrichment import enrich_actions
from forge_trace.schemas import RefinedAction


def _action(verb, start, end, obj=None, hand=None):
    return RefinedAction(
        verb=verb, object=obj, hand=hand,
        start_s=start, end_s=end, duration_s=round(end - start, 4),
        confidence=0.8, description=f"{verb} action",
    )


class TestEnrichWithObjects:
    def test_fills_missing_object(self):
        actions = [_action("pick up", 2.0, 4.0)]  # no object
        objects = [{"label": "bolt", "timestamp": 3.0, "confidence": 0.9}]
        result = enrich_actions(actions, 20.0, object_events=objects)
        assert result[0].object == "bolt"
        assert "YOLO" in result[0].evidence

    def test_does_not_override_existing_object(self):
        actions = [_action("pick up", 2.0, 4.0, obj="nut")]
        objects = [{"label": "bolt", "timestamp": 3.0, "confidence": 0.9}]
        result = enrich_actions(actions, 20.0, object_events=objects)
        assert result[0].object == "nut"

    def test_out_of_range_detection_ignored(self):
        actions = [_action("pick up", 2.0, 4.0)]
        objects = [{"label": "bolt", "timestamp": 10.0}]
        result = enrich_actions(actions, 20.0, object_events=objects)
        assert result[0].object is None


class TestEnrichWithHands:
    def test_fills_missing_hand(self):
        actions = [_action("pick up", 2.0, 4.0)]  # no hand
        hands = [
            {"hand": "right", "timestamp": 2.5},
            {"hand": "right", "timestamp": 3.0},
            {"hand": "right", "timestamp": 3.5},
        ]
        result = enrich_actions(actions, 20.0, hand_tracks=hands)
        assert result[0].hand == "right"
        assert "MediaPipe" in result[0].evidence

    def test_both_hands(self):
        actions = [_action("fasten", 5.0, 10.0)]
        hands = [
            {"hand": "right", "timestamp": 6.0},
            {"hand": "left", "timestamp": 7.0},
            {"hand": "right", "timestamp": 8.0},
            {"hand": "left", "timestamp": 9.0},
        ]
        result = enrich_actions(actions, 20.0, hand_tracks=hands)
        assert result[0].hand == "both"

    def test_does_not_override_existing_hand(self):
        actions = [_action("pick up", 2.0, 4.0, hand="left")]
        hands = [{"hand": "right", "timestamp": 3.0}]
        result = enrich_actions(actions, 20.0, hand_tracks=hands)
        assert result[0].hand == "left"


class TestEnrichWithInteractions:
    def test_confidence_boosted(self):
        actions = [_action("pick up", 2.0, 4.0, obj="bolt")]
        interactions = [
            {"object": "bolt", "hand": "right", "timestamp": 3.0, "type": "grasp"},
        ]
        result = enrich_actions(actions, 20.0, interaction_events=interactions)
        assert result[0].confidence > 0.8

    def test_fills_missing_from_interaction(self):
        actions = [_action("pick up", 2.0, 4.0)]  # no obj, no hand
        interactions = [
            {"object": "screw", "hand": "left", "timestamp": 3.0},
        ]
        result = enrich_actions(
            actions, 20.0, interaction_events=interactions,
        )
        assert result[0].object == "screw"
        assert result[0].hand == "left"


class TestNoEnrichment:
    def test_unchanged_without_sources(self, sample_refined_actions):
        result = enrich_actions(sample_refined_actions, 20.0)
        assert len(result) == len(sample_refined_actions)
        for orig, enriched in zip(sample_refined_actions, result):
            assert orig.verb == enriched.verb
            assert orig.start_s == enriched.start_s


class TestCombinedEnrichment:
    def test_all_sources(self):
        actions = [_action("pick up", 2.0, 4.0)]
        objects = [{"label": "bolt", "timestamp": 3.0, "confidence": 0.9}]
        hands = [{"hand": "right", "timestamp": 3.0}]
        interactions = [{"object": "bolt", "hand": "right", "timestamp": 3.0}]
        result = enrich_actions(
            actions, 20.0,
            object_events=objects,
            hand_tracks=hands,
            interaction_events=interactions,
        )
        assert result[0].object == "bolt"
        assert result[0].hand == "right"
        assert result[0].confidence > 0.8
