"""Shared pytest fixtures for FORGE TRACE tests."""

from __future__ import annotations

import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from forge_trace.config import ForgeTraceConfig
from forge_trace.schemas import (
    ActionPrediction,
    RefinedAction,
    TaskRecord,
    TimelineInfo,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> ForgeTraceConfig:
    """Return a test-safe config (no real API key)."""
    return ForgeTraceConfig(
        gemini_api_key="test-key-not-real",
        gemini_model="gemini-2.0-flash",
        segment_duration_s=60.0,
        segment_overlap_s=10.0,
        confidence_threshold=0.15,
        duplicate_iou_threshold=0.6,
        enable_refinement=False,
        enable_task_grouping=False,
        max_retries=1,
    )


# ---------------------------------------------------------------------------
# Synthetic video
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_video(tmp_path: Path) -> str:
    """Create a tiny synthetic MP4 video (5 seconds, 10 fps)."""
    video_path = str(tmp_path / "test_video.mp4")
    fps = 10
    duration_s = 5
    width, height = 160, 120
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for i in range(fps * duration_s):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Create changing colours so the video isn't static
        colour = ((i * 5) % 256, (i * 3) % 256, (i * 7) % 256)
        frame[:] = colour
        writer.write(frame)

    writer.release()
    return video_path


@pytest.fixture
def empty_video(tmp_path: Path) -> str:
    """Path to an empty (0-byte) file with .mp4 extension."""
    p = tmp_path / "empty.mp4"
    p.touch()
    return str(p)


@pytest.fixture
def non_video_file(tmp_path: Path) -> str:
    """Path to a text file disguised with .txt extension."""
    p = tmp_path / "notes.txt"
    p.write_text("not a video")
    return str(p)


# ---------------------------------------------------------------------------
# Sample action data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_coarse_actions() -> List[ActionPrediction]:
    """A small list of coarse action predictions."""
    return [
        ActionPrediction(
            verb="pick up", object="bolt", hand="right",
            start_s=2.0, end_s=4.0, confidence=0.85,
            description="Worker picks up a bolt",
        ),
        ActionPrediction(
            verb="position", object="bolt", hand="right",
            start_s=4.5, end_s=6.0, confidence=0.80,
            description="Worker positions the bolt",
        ),
        ActionPrediction(
            verb="fasten", object="bolt", hand="both",
            start_s=6.5, end_s=10.0, confidence=0.90,
            description="Worker fastens the bolt",
        ),
        ActionPrediction(
            verb="inspect", object="assembly", hand=None,
            start_s=11.0, end_s=13.0, confidence=0.70,
            description="Worker inspects the assembly",
        ),
    ]


@pytest.fixture
def sample_refined_actions() -> List[RefinedAction]:
    """A small list of refined actions."""
    return [
        RefinedAction(
            verb="pick up", object="bolt", hand="right",
            start_s=2.0, end_s=4.0, duration_s=2.0,
            confidence=0.85, description="Worker picks up a bolt",
            is_refined=True,
        ),
        RefinedAction(
            verb="position", object="bolt", hand="right",
            start_s=4.5, end_s=6.0, duration_s=1.5,
            confidence=0.80, description="Worker positions the bolt",
            is_refined=True,
        ),
        RefinedAction(
            verb="fasten", object="bolt", hand="both",
            start_s=6.5, end_s=10.0, duration_s=3.5,
            confidence=0.90, description="Worker fastens the bolt",
            is_refined=True,
        ),
        RefinedAction(
            verb="inspect", object="assembly", hand=None,
            start_s=11.0, end_s=13.0, duration_s=2.0,
            confidence=0.70, description="Worker inspects the assembly",
            is_refined=True,
        ),
    ]


@pytest.fixture
def sample_task_records() -> List[TaskRecord]:
    """A list of final task records for testing."""
    return [
        TaskRecord(
            task_id="T001", task_name="Pick up bolt", verb="pick up",
            object="bolt", hand="right",
            timeline=TimelineInfo(start_s=2.0, end_s=4.0, duration_s=2.0),
            confidence=0.85, description="Worker picks up a bolt",
        ),
        TaskRecord(
            task_id="T002", task_name="Position bolt", verb="position",
            object="bolt", hand="right",
            timeline=TimelineInfo(start_s=4.5, end_s=6.0, duration_s=1.5),
            confidence=0.80, description="Worker positions the bolt",
        ),
        TaskRecord(
            task_id="T003", task_name="Fasten bolt", verb="fasten",
            object="bolt", hand="both",
            timeline=TimelineInfo(start_s=6.5, end_s=10.0, duration_s=3.5),
            confidence=0.90, description="Worker fastens the bolt",
        ),
        TaskRecord(
            task_id="T004", task_name="Inspect assembly", verb="inspect",
            object="assembly", hand="unknown",
            timeline=TimelineInfo(start_s=11.0, end_s=13.0, duration_s=2.0),
            confidence=0.70, description="Worker inspects the assembly",
        ),
    ]


@pytest.fixture
def sample_ground_truth() -> List[Dict[str, Any]]:
    """Ground-truth annotations for evaluation tests."""
    return [
        {"verb": "pick up", "object": "bolt", "start_s": 1.8, "end_s": 4.2},
        {"verb": "position", "object": "bolt", "start_s": 4.3, "end_s": 6.2},
        {"verb": "fasten", "object": "bolt", "start_s": 6.4, "end_s": 10.2},
        {"verb": "inspect", "object": "assembly", "start_s": 10.8, "end_s": 13.2},
    ]


# ---------------------------------------------------------------------------
# Enrichment fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_object_events() -> List[Dict[str, Any]]:
    return [
        {"label": "bolt", "timestamp": 3.0, "confidence": 0.92},
        {"label": "wrench", "timestamp": 7.0, "confidence": 0.88},
        {"label": "panel", "timestamp": 12.0, "confidence": 0.75},
    ]


@pytest.fixture
def sample_hand_tracks() -> List[Dict[str, Any]]:
    return [
        {"hand": "right", "timestamp": 2.5},
        {"hand": "right", "timestamp": 3.5},
        {"hand": "left", "timestamp": 7.0},
        {"hand": "right", "timestamp": 7.5},
        {"hand": "right", "timestamp": 12.0},
    ]


@pytest.fixture
def sample_interaction_events() -> List[Dict[str, Any]]:
    return [
        {"object": "bolt", "hand": "right", "timestamp": 3.0, "type": "grasp"},
        {"object": "bolt", "hand": "both", "start_s": 7.0, "end_s": 9.0, "type": "manipulation"},
    ]
