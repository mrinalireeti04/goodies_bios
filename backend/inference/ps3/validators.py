"""Input and output validation for FORGE TRACE.

All validation logic is collected here so that the pipeline modules can focus
on business logic rather than guard clauses.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.inference.ps3.config import ForgeTraceConfig
from backend.inference.ps3.exceptions import (
    TimelineValidationError,
    VideoValidationError,
)
from backend.inference.ps3.logging_config import get_logger
from backend.inference.ps3.schemas import TaskRecord, TaskTimeline

logger = get_logger("validators")


# ---------------------------------------------------------------------------
# Video input validation
# ---------------------------------------------------------------------------

def validate_video_path(
    video_path: str,
    config: ForgeTraceConfig,
) -> Path:
    """Validate that *video_path* points to a readable video file.

    Returns a resolved :class:`Path` on success.
    """
    path = Path(video_path)

    if not path.exists():
        raise VideoValidationError(f"Video file does not exist: {video_path}")

    if not path.is_file():
        raise VideoValidationError(f"Path is not a file: {video_path}")

    ext = path.suffix.lower()
    if ext not in config.supported_extensions:
        raise VideoValidationError(
            f"Unsupported video format '{ext}'. "
            f"Supported: {config.supported_extensions}"
        )

    file_size = path.stat().st_size
    if file_size == 0:
        raise VideoValidationError("Video file is empty (0 bytes)")

    logger.debug(
        "Video path validated: %s (%.1f MB)", path, file_size / (1024 * 1024)
    )
    return path.resolve()


def validate_video_duration(
    duration_s: float,
    config: ForgeTraceConfig,
) -> None:
    """Raise if *duration_s* is invalid or exceeds the configured maximum."""
    if duration_s <= 0:
        raise VideoValidationError(
            f"Video duration must be positive, got {duration_s}"
        )
    if duration_s > config.max_video_duration_s:
        raise VideoValidationError(
            f"Video duration ({duration_s:.1f}s) exceeds maximum "
            f"({config.max_video_duration_s:.1f}s)"
        )


# ---------------------------------------------------------------------------
# Enrichment-input validation
# ---------------------------------------------------------------------------

def validate_enrichment_events(
    events: Optional[List[Dict[str, Any]]],
    label: str,
    video_duration: float,
) -> List[Dict[str, Any]]:
    """Best-effort validation and cleaning of enrichment event lists.

    Invalid entries are logged and dropped rather than causing a hard failure.
    """
    if events is None:
        return []

    cleaned: List[Dict[str, Any]] = []
    for idx, evt in enumerate(events):
        if not isinstance(evt, dict):
            logger.warning(
                "%s event %d is not a dict — skipping", label, idx
            )
            continue

        # Check for a timestamp or time range
        has_time = any(
            k in evt for k in ("timestamp", "start_s", "time", "frame")
        )
        if not has_time:
            logger.warning(
                "%s event %d has no temporal field — skipping", label, idx
            )
            continue

        # Clamp timestamps to video bounds
        for key in ("timestamp", "start_s", "end_s", "time"):
            if key in evt and isinstance(evt[key], (int, float)):
                evt[key] = max(0.0, min(float(evt[key]), video_duration))

        cleaned.append(evt)

    logger.info(
        "%s: %d / %d events passed validation", label, len(cleaned), len(events)
    )
    return cleaned


# ---------------------------------------------------------------------------
# Timeline output validation
# ---------------------------------------------------------------------------

def validate_task_record(task: TaskRecord, video_duration: float) -> List[str]:
    """Return a list of warnings (empty if the record is valid)."""
    warnings: List[str] = []

    if task.timeline.start_s > video_duration:
        warnings.append(
            f"{task.task_id}: start_s ({task.timeline.start_s}) > "
            f"video duration ({video_duration})"
        )
    if task.timeline.end_s > video_duration + 0.5:
        warnings.append(
            f"{task.task_id}: end_s ({task.timeline.end_s}) > "
            f"video duration ({video_duration})"
        )
    if task.timeline.start_s > task.timeline.end_s:
        warnings.append(
            f"{task.task_id}: start_s > end_s "
            f"({task.timeline.start_s} > {task.timeline.end_s})"
        )

    expected_dur = round(task.timeline.end_s - task.timeline.start_s, 4)
    if abs(task.timeline.duration_s - expected_dur) > 0.05:
        warnings.append(
            f"{task.task_id}: duration mismatch "
            f"(expected {expected_dur}, got {task.timeline.duration_s})"
        )
    return warnings


def validate_task_timeline(
    timeline: TaskTimeline,
) -> TaskTimeline:
    """Validate the full :class:`TaskTimeline` and raise on critical issues.

    Non-critical issues are logged as warnings but do not block output.
    """
    issues: List[str] = []

    # Check each task
    seen_ids: set[str] = set()
    for task in timeline.tasks:
        if task.task_id in seen_ids:
            issues.append(f"Duplicate task_id: {task.task_id}")
        seen_ids.add(task.task_id)

        record_warnings = validate_task_record(task, timeline.video_duration)
        issues.extend(record_warnings)

    # Check chronological ordering
    for i in range(1, len(timeline.tasks)):
        prev = timeline.tasks[i - 1]
        curr = timeline.tasks[i]
        if curr.timeline.start_s < prev.timeline.start_s:
            logger.warning(
                "Tasks not perfectly chronological: %s (%.2f) before %s (%.2f)",
                curr.task_id, curr.timeline.start_s,
                prev.task_id, prev.timeline.start_s,
            )

    if issues:
        for issue in issues:
            logger.warning("Timeline validation: %s", issue)

    # Critical: counts
    if timeline.total_tasks != len(timeline.tasks):
        raise TimelineValidationError(
            f"total_tasks ({timeline.total_tasks}) != "
            f"len(tasks) ({len(timeline.tasks)})"
        )

    return timeline
