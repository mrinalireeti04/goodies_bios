"""Video file utilities for FORGE TRACE.

Handles metadata extraction, segment planning, and frame sampling using
OpenCV.  All timestamps are relative to the *original* video.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np

from forge_trace.config import ForgeTraceConfig
from forge_trace.exceptions import VideoReadError, VideoValidationError
from forge_trace.logging_config import get_logger
from forge_trace.schemas import VideoMetadata, VideoSegment

logger = get_logger("video_utils")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def get_video_metadata(video_path: str) -> VideoMetadata:
    """Extract metadata from a video file using OpenCV.

    Raises :class:`VideoReadError` if the file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoReadError(f"Cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = (
            "".join(chr((codec_int >> (8 * i)) & 0xFF) for i in range(4))
            if codec_int > 0
            else "unknown"
        )
        duration_s = frame_count / fps if fps > 0 else 0.0
        file_size = Path(video_path).stat().st_size

        if duration_s <= 0:
            raise VideoValidationError(
                f"Computed video duration is non-positive ({duration_s:.2f}s). "
                "The file may be corrupt."
            )

        meta = VideoMetadata(
            path=str(Path(video_path).resolve()),
            duration_s=round(duration_s, 4),
            fps=round(fps, 4),
            width=width,
            height=height,
            frame_count=frame_count,
            codec=codec,
            file_size_bytes=file_size,
        )
        logger.info(
            "Video metadata: %.1fs, %.1f fps, %dx%d, %s",
            meta.duration_s, meta.fps, meta.width, meta.height, meta.codec,
        )
        return meta

    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Segment planning
# ---------------------------------------------------------------------------

def plan_video_segments(
    duration_s: float,
    config: ForgeTraceConfig,
) -> List[VideoSegment]:
    """Divide a video timeline into overlapping analysis windows.

    If the video is shorter than one segment, a single segment covering the
    entire video is returned.
    """
    seg_dur = config.segment_duration_s
    overlap = config.segment_overlap_s
    step = seg_dur - overlap

    if duration_s <= seg_dur:
        return [
            VideoSegment(
                index=0,
                start_s=0.0,
                end_s=round(duration_s, 4),
                duration_s=round(duration_s, 4),
            )
        ]

    segments: List[VideoSegment] = []
    start = 0.0
    idx = 0
    while start < duration_s:
        end = min(start + seg_dur, duration_s)
        # Avoid tiny trailing segments (< 5 s)
        if duration_s - end < 5.0 and end < duration_s:
            end = duration_s
        segments.append(
            VideoSegment(
                index=idx,
                start_s=round(start, 4),
                end_s=round(end, 4),
                duration_s=round(end - start, 4),
            )
        )
        if end >= duration_s:
            break
        start += step
        idx += 1

    logger.info(
        "Planned %d segments (%.0fs window, %.0fs overlap) for %.1fs video",
        len(segments), seg_dur, overlap, duration_s,
    )
    return segments


# ---------------------------------------------------------------------------
# Segment extraction (write to temporary file)
# ---------------------------------------------------------------------------

def extract_video_segment(
    video_path: str,
    start_s: float,
    end_s: float,
    output_dir: str | None = None,
) -> str:
    """Extract a time-bounded segment from *video_path* to a temp MP4 file.

    Returns the path of the created file.  The caller is responsible for
    cleanup.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoReadError(f"Cannot open video for segment extraction: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="forge_trace_seg_")
        out_path = str(
            Path(output_dir) / f"seg_{start_s:.1f}_{end_s:.1f}.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        start_frame = int(start_s * fps)
        end_frame = int(end_s * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames_written = 0
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            frames_written += 1

        writer.release()
        logger.debug(
            "Extracted segment [%.1f–%.1f]s → %s (%d frames)",
            start_s, end_s, out_path, frames_written,
        )
        return out_path

    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def extract_frames_at_timestamps(
    video_path: str,
    timestamps: List[float],
) -> List[tuple[float, np.ndarray]]:
    """Extract frames at specific timestamps.

    Returns a list of ``(timestamp, frame)`` tuples.  Timestamps that fall
    outside the video range are silently skipped.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoReadError(f"Cannot open video for frame extraction: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results: List[tuple[float, np.ndarray]] = []

        for ts in sorted(timestamps):
            frame_idx = int(ts * fps)
            if frame_idx < 0 or frame_idx >= total_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                results.append((ts, frame))

        return results

    finally:
        cap.release()


def extract_frames_around_boundary(
    video_path: str,
    boundary_s: float,
    window_s: float = 2.0,
    step_s: float = 0.5,
) -> List[tuple[float, np.ndarray]]:
    """Extract frames in a window around *boundary_s*.

    Useful for temporal refinement: sample frames at *step_s* intervals
    within *[boundary_s − window_s, boundary_s + window_s]*.
    """
    timestamps = []
    t = max(0.0, boundary_s - window_s)
    end = boundary_s + window_s
    while t <= end:
        timestamps.append(round(t, 4))
        t += step_s
    return extract_frames_at_timestamps(video_path, timestamps)


def frames_to_jpeg_bytes(frames: List[np.ndarray]) -> List[bytes]:
    """Encode frames as JPEG byte strings for API submission."""
    result: List[bytes] = []
    for frame in frames:
        success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if success:
            result.append(buf.tobytes())
    return result
