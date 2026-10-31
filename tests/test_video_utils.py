"""Tests for video utilities."""

import pytest
import numpy as np

from forge_trace.schemas import VideoSegment
from forge_trace.exceptions import VideoReadError, VideoValidationError
from forge_trace.video_utils import (
    extract_frames_around_boundary,
    extract_frames_at_timestamps,
    extract_video_segment,
    frames_to_jpeg_bytes,
    get_video_metadata,
    plan_video_segments,
)


class TestGetVideoMetadata:
    def test_valid_video(self, synthetic_video):
        meta = get_video_metadata(synthetic_video)
        assert meta.duration_s > 0
        assert meta.fps > 0
        assert meta.width == 160
        assert meta.height == 120
        assert meta.frame_count > 0

    def test_missing_file(self):
        with pytest.raises(VideoReadError):
            get_video_metadata("/no/such/video.mp4")


class TestPlanVideoSegments:
    def test_short_video(self, config):
        segments = plan_video_segments(30.0, config)
        assert len(segments) == 1
        assert segments[0].start_s == 0.0
        assert segments[0].end_s == 30.0

    def test_exact_one_segment(self, config):
        config.segment_duration_s = 60.0
        segments = plan_video_segments(60.0, config)
        assert len(segments) == 1

    def test_single_request_mode_for_3min_clip(self, config):
        config.single_request_mode = True
        config.segment_duration_s = 60.0  # Even with smaller default segment duration
        # 180s (3-minute) clip
        segments = plan_video_segments(180.0, config)
        assert len(segments) == 1
        assert segments[0].start_s == 0.0
        assert segments[0].end_s == 180.0
        assert segments[0].duration_s == 180.0

    def test_overlapping_segments_step_and_boundaries(self, config):
        config.single_request_mode = False
        config.segment_duration_s = 120.0
        config.segment_overlap_s = 15.0
        segments = plan_video_segments(180.0, config)
        assert len(segments) == 2
        assert segments[0].start_s == 0.0
        assert segments[0].end_s == 120.0
        # Segment 1 starts at step = 120 - 15 = 105.0s
        assert segments[1].start_s == 105.0
        assert segments[1].end_s == 180.0

    def test_long_video(self, config):
        config.segment_duration_s = 60.0
        config.segment_overlap_s = 10.0
        segments = plan_video_segments(200.0, config)
        assert len(segments) >= 3

        # Check overlap: each segment except the first should start
        # within the previous segment's end
        for i in range(1, len(segments)):
            assert segments[i].start_s < segments[i - 1].end_s

        # Last segment should end at the video duration
        assert segments[-1].end_s == 200.0

    def test_all_segments_have_positive_duration(self, config):
        config.segment_duration_s = 30.0
        config.segment_overlap_s = 5.0
        segments = plan_video_segments(150.0, config)
        for seg in segments:
            assert seg.duration_s > 0

    def test_indices_are_sequential(self, config):
        segments = plan_video_segments(300.0, config)
        for i, seg in enumerate(segments):
            assert seg.index == i


class TestExtractVideoSegment:
    def test_extract_segment(self, synthetic_video, tmp_path):
        out = extract_video_segment(
            synthetic_video, 1.0, 3.0, str(tmp_path)
        )
        meta = get_video_metadata(out)
        assert meta.duration_s > 0


class TestExtractFrames:
    def test_extract_at_timestamps(self, synthetic_video):
        frames = extract_frames_at_timestamps(
            synthetic_video, [0.0, 1.0, 2.0]
        )
        assert len(frames) == 3
        for ts, frame in frames:
            assert isinstance(frame, np.ndarray)
            assert frame.shape[0] == 120
            assert frame.shape[1] == 160

    def test_out_of_range_skipped(self, synthetic_video):
        frames = extract_frames_at_timestamps(
            synthetic_video, [0.0, 100.0]
        )
        assert len(frames) == 1

    def test_extract_around_boundary(self, synthetic_video):
        frames = extract_frames_around_boundary(
            synthetic_video, 2.0, window_s=1.0, step_s=0.5,
        )
        assert len(frames) >= 3

    def test_missing_file_raises(self):
        with pytest.raises(VideoReadError):
            extract_frames_at_timestamps("/no/file.mp4", [0.0])


class TestFramesToJpeg:
    def test_encoding(self):
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        result = frames_to_jpeg_bytes(frames)
        assert len(result) == 1
        assert isinstance(result[0], bytes)
        assert len(result[0]) > 0
