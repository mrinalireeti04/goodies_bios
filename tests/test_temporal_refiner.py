"""Tests for temporal refinement and boundary tightening."""

from unittest.mock import MagicMock, patch
import pytest

from forge_trace.config import ForgeTraceConfig
from forge_trace.schemas import ActionPrediction, RefinedAction
from forge_trace.temporal_refiner import (
    _coarse_to_refined,
    refine_actions_pass2,
    refine_boundaries_with_frames,
)


def _make_refined_action(verb="fasten", start=5.0, end=10.0, conf=0.85):
    return RefinedAction(
        verb=verb,
        object="panel",
        hand="right",
        start_s=start,
        end_s=end,
        duration_s=round(end - start, 4),
        model_confidence_score=conf,
        confidence=conf,
        description=f"{verb} panel",
        evidence="Gemini video analysis",
    )


class TestRefineBoundariesWithFrames:
    def test_refinement_skipped_when_disabled(self):
        cfg = ForgeTraceConfig(
            gemini_api_key="mock",
            enable_frame_refinement=False,
        )
        actions = [_make_refined_action()]
        client = MagicMock()

        updated, status = refine_boundaries_with_frames(
            client, "dummy_path.mp4", actions, cfg
        )
        assert status == "skipped"
        assert updated == actions
        client.analyze_frames.assert_not_called()

    @patch("forge_trace.temporal_refiner.refine_boundary_with_frames")
    def test_refinement_actually_called(self, mock_refine_bound):
        # Configure boundary returns
        mock_refine_bound.side_effect = lambda client, path, act, b_type, cfg: (
            4.8 if b_type == "start" else 10.2
        )

        cfg = ForgeTraceConfig(
            gemini_api_key="mock",
            enable_frame_refinement=True,
            max_frame_refinement_actions=5,
        )
        actions = [_make_refined_action(start=5.0, end=10.0)]
        client = MagicMock()

        updated, status = refine_boundaries_with_frames(
            client, "dummy_path.mp4", actions, cfg
        )

        assert mock_refine_bound.call_count == 2
        assert status == "applied"
        assert updated[0].start_s == 4.8
        assert updated[0].end_s == 10.2
        assert updated[0].is_refined is True
        assert "frame-boundary refined" in updated[0].evidence

    @patch("forge_trace.temporal_refiner.refine_boundary_with_frames")
    def test_refinement_failed_status_on_error(self, mock_refine_bound):
        mock_refine_bound.side_effect = RuntimeError("OpenCV frame decode failed")

        cfg = ForgeTraceConfig(
            gemini_api_key="mock",
            enable_frame_refinement=True,
        )
        actions = [_make_refined_action(start=5.0, end=10.0)]
        client = MagicMock()

        updated, status = refine_boundaries_with_frames(
            client, "dummy_path.mp4", actions, cfg
        )

        assert status == "failed"
        # Original timestamps preserved on failure
        assert updated[0].start_s == 5.0
        assert updated[0].end_s == 10.0

    @patch("forge_trace.temporal_refiner.refine_boundary_with_frames")
    def test_refinement_partial_status(self, mock_refine_bound):
        def side_effect(client, path, act, b_type, cfg):
            if act.verb == "succeed":
                return 4.5 if b_type == "start" else 9.5
            raise RuntimeError("Failed for fail action")

        mock_refine_bound.side_effect = side_effect

        cfg = ForgeTraceConfig(
            gemini_api_key="mock",
            enable_frame_refinement=True,
            max_frame_refinement_actions=5,
        )
        actions = [
            _make_refined_action(verb="succeed", start=5.0, end=10.0),
            _make_refined_action(verb="fail", start=12.0, end=15.0),
        ]
        client = MagicMock()

        updated, status = refine_boundaries_with_frames(
            client, "dummy_path.mp4", actions, cfg
        )
        assert status == "partial"
        assert updated[0].start_s == 4.5
        assert updated[1].start_s == 12.0


class TestPass2ModelConfidenceMetadata:
    def test_pass2_preserves_model_confidence_score(self):
        client = MagicMock()
        client.analyze_video.return_value = """[
            {
                "verb": "fasten",
                "object": "panel",
                "hand": "right",
                "start_s": 5.2,
                "end_s": 9.8,
                "model_confidence_score": 0.88,
                "description": "Fastens panel"
            }
        ]"""
        client.parse_json_response.return_value = [
            {
                "verb": "fasten",
                "object": "panel",
                "hand": "right",
                "start_s": 5.2,
                "end_s": 9.8,
                "model_confidence_score": 0.88,
                "description": "Fastens panel",
            }
        ]

        coarse = [
            ActionPrediction(
                verb="fasten",
                object="panel",
                hand="right",
                start_s=5.0,
                end_s=10.0,
                model_confidence_score=0.80,
            )
        ]
        cfg = ForgeTraceConfig(gemini_api_key="mock")

        refined = refine_actions_pass2(client, "dummy_video", coarse, cfg)
        assert len(refined) == 1
        assert refined[0].model_confidence_score == 0.88
        assert refined[0].confidence == 0.88
        assert refined[0].start_s == 5.2
        assert refined[0].end_s == 9.8
