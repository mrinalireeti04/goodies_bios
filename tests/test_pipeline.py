"""End-to-end pipeline tests with mocked Gemini API."""

from unittest.mock import MagicMock, patch
import pytest

from forge_trace.config import ForgeTraceConfig
from forge_trace.pipeline import process_video_to_timeline


@pytest.fixture
def mock_gemini_client():
    client = MagicMock()
    # Mock video upload
    video_file = MagicMock()
    video_file.name = "files/mock123"
    video_file.uri = "https://generativelanguage.googleapis.com/v1beta/files/mock123"
    video_file.mime_type = "video/mp4"
    client.upload_video.return_value = video_file

    # Mock action recognition response (coarse)
    coarse_json = """[
        {
            "verb": "fasten",
            "object": "bolt",
            "hand": "right",
            "start_s": 1.0,
            "end_s": 3.5,
            "model_confidence_score": 0.88,
            "description": "Worker fastens bolt"
        }
    ]"""
    client.analyze_video.return_value = coarse_json
    client.parse_json_response.return_value = [
        {
            "verb": "fasten",
            "object": "bolt",
            "hand": "right",
            "start_s": 1.0,
            "end_s": 3.5,
            "model_confidence_score": 0.88,
            "description": "Worker fastens bolt",
        }
    ]
    return client


class TestPipelineIntegration:
    @patch("forge_trace.pipeline.GeminiClient")
    def test_pipeline_single_request_mode(self, mock_client_cls, mock_gemini_client, synthetic_video):
        mock_client_cls.return_value = mock_gemini_client

        cfg = ForgeTraceConfig(
            gemini_api_key="mock",
            single_request_mode=True,
            enable_refinement=False,
            enable_task_grouping=False,
        )

        timeline = process_video_to_timeline(
            synthetic_video,
            clip_id="test_clip",
            config=cfg,
        )

        assert timeline.total_tasks == 1
        assert timeline.tasks[0].verb == "fasten"
        assert timeline.processing_metadata.single_request_mode is True
        assert timeline.processing_metadata.segments_analyzed == 1
        assert timeline.processing_metadata.timestamp_refinement_status == "skipped"
        assert timeline.processing_metadata.timestamp_refinement_applied is False

    @patch("forge_trace.pipeline.GeminiClient")
    def test_pipeline_ps2_tool_integration(self, mock_client_cls, mock_gemini_client, synthetic_video):
        mock_client_cls.return_value = mock_gemini_client

        cfg = ForgeTraceConfig(
            gemini_api_key="mock",
            enable_refinement=False,
            enable_task_grouping=False,
        )

        tool_events = [
            {
                "tool": "pneumatic wrench",
                "hand": "right",
                "start_s": 0.5,
                "end_s": 4.0,
                "confidence": 0.95,
            }
        ]

        timeline = process_video_to_timeline(
            synthetic_video,
            clip_id="test_tool_clip",
            tool_events=tool_events,
            config=cfg,
        )

        assert timeline.total_tasks == 1
        task = timeline.tasks[0]
        assert task.tool == "pneumatic wrench"
        assert task.tool_hand == "right"
        assert task.tool_association_status == "associated"
        assert task.tool_match_confidence is not None
        assert "PS2_Tools" in timeline.processing_metadata.enrichment_sources

    @patch("forge_trace.pipeline.GeminiClient")
    def test_pipeline_refinement_reporting(self, mock_client_cls, mock_gemini_client, synthetic_video):
        mock_client_cls.return_value = mock_gemini_client

        # With semantic pass 2 refinement enabled
        cfg = ForgeTraceConfig(
            gemini_api_key="mock",
            enable_refinement=True,
            enable_frame_refinement=False,
            enable_task_grouping=False,
        )

        timeline = process_video_to_timeline(
            synthetic_video,
            clip_id="test_refine_clip",
            config=cfg,
        )

        assert timeline.processing_metadata.timestamp_refinement_status == "applied"
        assert timeline.processing_metadata.timestamp_refinement_applied is True
