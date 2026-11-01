"""Pydantic data models for FORGE TRACE Task Intelligence module.

Every data structure that flows through the pipeline is declared here so that
validation, serialisation and documentation stay in one place.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------

class VideoMetadata(BaseModel):
    """Metadata extracted from a video file."""

    path: str
    duration_s: float = Field(..., gt=0, description="Video duration in seconds")
    fps: float = Field(..., gt=0, description="Frames per second")
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    frame_count: int = Field(..., ge=0)
    codec: str = ""
    file_size_bytes: int = Field(0, ge=0)


class VideoSegment(BaseModel):
    """A time window within the source video."""

    index: int = Field(..., ge=0)
    start_s: float = Field(..., ge=0)
    end_s: float = Field(..., gt=0)
    duration_s: float = Field(..., gt=0)

    @model_validator(mode="after")
    def _check_ordering(self) -> "VideoSegment":
        if self.end_s <= self.start_s:
            raise ValueError("end_s must be greater than start_s")
        return self


# ---------------------------------------------------------------------------
# Action predictions (raw output from Gemini)
# ---------------------------------------------------------------------------

class ActionPrediction(BaseModel):
    """A single action prediction returned by the video-understanding model."""

    verb: str = Field(..., min_length=1)
    object: Optional[str] = None
    hand: Optional[str] = None
    start_s: float = Field(..., ge=0)
    end_s: float = Field(..., ge=0)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    description: str = ""
    evidence: str = "Visual observation"
    segment_index: Optional[int] = None

    @model_validator(mode="after")
    def _check_times(self) -> "ActionPrediction":
        if self.end_s < self.start_s:
            raise ValueError(
                f"end_s ({self.end_s}) must be >= start_s ({self.start_s})"
            )
        return self

    @field_validator("hand")
    @classmethod
    def _normalise_hand(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().lower()
        if v in ("left", "right", "both"):
            return v
        return None

    @property
    def duration_s(self) -> float:
        return round(self.end_s - self.start_s, 4)


# ---------------------------------------------------------------------------
# Refined action (after temporal refinement)
# ---------------------------------------------------------------------------

class RefinedAction(BaseModel):
    """An action whose temporal boundaries have been refined."""

    verb: str
    object: Optional[str] = None
    hand: Optional[str] = None
    start_s: float = Field(..., ge=0)
    end_s: float = Field(..., ge=0)
    duration_s: float = Field(..., ge=0)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    description: str = ""
    evidence: str = "Visual observation"
    start_uncertainty_s: float = Field(0.0, ge=0)
    end_uncertainty_s: float = Field(0.0, ge=0)
    is_refined: bool = False

    @model_validator(mode="after")
    def _check_times(self) -> "RefinedAction":
        if self.end_s < self.start_s:
            raise ValueError("end_s must be >= start_s")
        return self


# ---------------------------------------------------------------------------
# Final task record
# ---------------------------------------------------------------------------

class TimelineInfo(BaseModel):
    """Start, end and duration of a task."""

    start_s: float = Field(..., ge=0)
    end_s: float = Field(..., ge=0)
    duration_s: float = Field(..., ge=0)

    @model_validator(mode="after")
    def _validate(self) -> "TimelineInfo":
        if self.end_s < self.start_s:
            raise ValueError("end_s must be >= start_s")
        expected = round(self.end_s - self.start_s, 4)
        if abs(self.duration_s - expected) > 0.05:
            raise ValueError(
                f"duration_s ({self.duration_s}) != end_s - start_s ({expected})"
            )
        return self


class TaskRecord(BaseModel):
    """One task in the final timeline."""

    task_id: str = Field(..., min_length=1)
    task_name: str = Field(..., min_length=1)
    verb: str = Field(..., min_length=1)
    object: str = Field(default="unknown")
    hand: str = Field(default="unknown")
    timeline: TimelineInfo
    confidence: float = Field(..., ge=0.0, le=1.0)
    description: str = ""
    evidence: str = "Visual observation"


# ---------------------------------------------------------------------------
# Task grouping (optional higher-level activities)
# ---------------------------------------------------------------------------

class TaskGroup(BaseModel):
    """An optional higher-level activity grouping related tasks."""

    group_id: str
    group_name: str
    task_ids: List[str] = Field(default_factory=list)
    description: str = ""
    start_s: float = Field(..., ge=0)
    end_s: float = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Processing metadata
# ---------------------------------------------------------------------------

class ProcessingMetadata(BaseModel):
    """Metadata about how the pipeline processed the video."""

    model_used: str = ""
    segments_analyzed: int = 0
    processing_time_s: float = 0.0
    timestamp_refinement_applied: bool = False
    enrichment_sources: List[str] = Field(default_factory=list)
    failed_segments: List[int] = Field(default_factory=list)
    stage_timings_s: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Top-level timeline output
# ---------------------------------------------------------------------------

class TaskTimeline(BaseModel):
    """Complete structured output of the Task Intelligence module."""

    clip_id: str
    video_path: str
    video_duration: float = Field(..., gt=0)
    total_tasks: int = Field(..., ge=0)
    unique_task_types: int = Field(..., ge=0)
    unique_task_list: List[str] = Field(default_factory=list)
    tasks: List[TaskRecord] = Field(default_factory=list)
    groups: List[TaskGroup] = Field(default_factory=list)
    processing_metadata: ProcessingMetadata = Field(
        default_factory=ProcessingMetadata
    )

    @model_validator(mode="after")
    def _check_counts(self) -> "TaskTimeline":
        if self.total_tasks != len(self.tasks):
            raise ValueError(
                f"total_tasks ({self.total_tasks}) != len(tasks) ({len(self.tasks)})"
            )
        actual_unique = len({t.task_name for t in self.tasks})
        if self.tasks and self.unique_task_types != actual_unique:
            raise ValueError(
                f"unique_task_types ({self.unique_task_types}) != "
                f"actual unique names ({actual_unique})"
            )
        return self
