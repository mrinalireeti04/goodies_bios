"""FORGE TRACE — Task Intelligence and Timeline Generation Module.

Public API
----------

.. autofunction:: process_video_to_timeline

Schemas are importable from :mod:`forge_trace.schemas`.
"""

from forge_trace.pipeline import process_video_to_timeline
from forge_trace.schemas import (
    ActionPrediction,
    ProcessingMetadata,
    RefinedAction,
    TaskGroup,
    TaskRecord,
    TaskTimeline,
    TimelineInfo,
    VideoMetadata,
    VideoSegment,
)

__all__ = [
    "process_video_to_timeline",
    "ActionPrediction",
    "ProcessingMetadata",
    "RefinedAction",
    "TaskGroup",
    "TaskRecord",
    "TaskTimeline",
    "TimelineInfo",
    "VideoMetadata",
    "VideoSegment",
]
