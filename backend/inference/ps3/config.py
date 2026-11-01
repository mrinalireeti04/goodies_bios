"""Runtime configuration for FORGE TRACE Task Intelligence module.

Settings are read from environment variables with sensible defaults.  Every
knob that affects pipeline behaviour is centralised here so that nothing is
hard-coded in business-logic modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from backend.inference.ps3.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "gemini-3.6-flash"
_DEFAULT_SEGMENT_DURATION = 120.0       # seconds
_DEFAULT_SEGMENT_OVERLAP = 15.0         # seconds
_DEFAULT_CONFIDENCE_THRESHOLD = 0.15    # keep predictions above this
_DEFAULT_DUPLICATE_IOU_THRESHOLD = 0.60 # merge if temporal IoU > this
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BASE_WAIT = 2.0          # seconds (exponential back-off base)
_DEFAULT_REFINEMENT_WINDOW = 2.0        # seconds each side of boundary
_DEFAULT_REFINEMENT_STEP = 0.5          # seconds between sampled frames

_SUPPORTED_VIDEO_EXTENSIONS: List[str] = [
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv",
]


@dataclass
class ForgeTraceConfig:
    """Centralised configuration for the Task Intelligence pipeline."""

    # --- Gemini API --------------------------------------------------------
    gemini_api_key: str = ""
    gemini_model: str = _DEFAULT_MODEL

    # --- Video segmentation ------------------------------------------------
    segment_duration_s: float = _DEFAULT_SEGMENT_DURATION
    segment_overlap_s: float = _DEFAULT_SEGMENT_OVERLAP

    # --- Action filtering --------------------------------------------------
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD
    duplicate_iou_threshold: float = _DEFAULT_DUPLICATE_IOU_THRESHOLD

    # --- Temporal refinement -----------------------------------------------
    enable_refinement: bool = True
    refinement_window_s: float = _DEFAULT_REFINEMENT_WINDOW
    refinement_step_s: float = _DEFAULT_REFINEMENT_STEP

    # --- Retry / rate-limit ------------------------------------------------
    max_retries: int = _DEFAULT_MAX_RETRIES
    retry_base_wait_s: float = _DEFAULT_RETRY_BASE_WAIT

    # --- File validation ---------------------------------------------------
    supported_extensions: List[str] = field(
        default_factory=lambda: list(_SUPPORTED_VIDEO_EXTENSIONS)
    )
    max_video_duration_s: float = 3600.0    # 1 hour

    # --- Task grouping -----------------------------------------------------
    enable_task_grouping: bool = True

    def validate(self) -> None:
        """Raise :class:`ConfigurationError` if the config is invalid."""
        if not self.gemini_api_key:
            raise ConfigurationError(
                "gemini_api_key is required. Set GEMINI_API_KEY env var "
                "or pass it directly."
            )
        if self.segment_duration_s <= 0:
            raise ConfigurationError("segment_duration_s must be positive")
        if self.segment_overlap_s < 0:
            raise ConfigurationError("segment_overlap_s must be >= 0")
        if self.segment_overlap_s >= self.segment_duration_s:
            raise ConfigurationError(
                "segment_overlap_s must be less than segment_duration_s"
            )


def load_config(**overrides) -> ForgeTraceConfig:
    """Build a :class:`ForgeTraceConfig` from environment variables and
    optional *overrides*.

    Environment variables (all optional except ``GEMINI_API_KEY``):

    * ``GEMINI_API_KEY``
    * ``FORGE_TRACE_MODEL``
    * ``FORGE_TRACE_SEGMENT_DURATION``
    * ``FORGE_TRACE_SEGMENT_OVERLAP``
    * ``FORGE_TRACE_CONFIDENCE_THRESHOLD``
    * ``FORGE_TRACE_MAX_RETRIES``
    * ``FORGE_TRACE_ENABLE_REFINEMENT``
    """
    cfg = ForgeTraceConfig(
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        gemini_model=os.environ.get("FORGE_TRACE_MODEL", _DEFAULT_MODEL),
        enable_task_grouping=os.environ.get("FORGE_TRACE_ENABLE_GROUPING", "true").lower() in ("true", "1", "yes"),
        segment_duration_s=float(
            os.environ.get("FORGE_TRACE_SEGMENT_DURATION", _DEFAULT_SEGMENT_DURATION)
        ),
        segment_overlap_s=float(
            os.environ.get("FORGE_TRACE_SEGMENT_OVERLAP", _DEFAULT_SEGMENT_OVERLAP)
        ),
        confidence_threshold=float(
            os.environ.get("FORGE_TRACE_CONFIDENCE_THRESHOLD", _DEFAULT_CONFIDENCE_THRESHOLD)
        ),
        max_retries=int(
            os.environ.get("FORGE_TRACE_MAX_RETRIES", _DEFAULT_MAX_RETRIES)
        ),
        enable_refinement=os.environ.get(
            "FORGE_TRACE_ENABLE_REFINEMENT", "true"
        ).lower() in ("true", "1", "yes"),
    )

    # Apply explicit overrides
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            raise ConfigurationError(f"Unknown configuration key: {key}")

    return cfg
