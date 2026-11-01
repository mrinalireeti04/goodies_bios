"""Main orchestration pipeline for FORGE TRACE Task Intelligence.

This is the single entry-point that ties together every stage:

1. Validate inputs.
2. Extract video metadata and plan segments.
3. Upload video and run coarse action recognition.
4. Refine action timestamps (Pass 2).
5. Normalise, deduplicate and merge.
6. Enrich with YOLO / MediaPipe / interaction data.
7. Assign task IDs and build summary.
8. Optionally group into higher-level activities.
9. Validate and return the structured :class:`TaskTimeline`.
"""

from __future__ import annotations

import time
import tempfile
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.inference.ps3.action_normalizer import normalize_and_deduplicate
from backend.inference.ps3.action_recognizer import recognize_actions_full_video
from backend.inference.ps3.config import ForgeTraceConfig, load_config
from backend.inference.ps3.enrichment import enrich_actions
from backend.inference.ps3.exceptions import ForgeTraceError
from backend.inference.ps3.gemini_client import GeminiClient
from backend.inference.ps3.logging_config import get_logger
from backend.inference.ps3.schemas import (
    ProcessingMetadata,
    TaskTimeline,
)
from backend.inference.ps3.task_builder import (
    assign_task_ids,
    generate_task_summary,
    group_tasks_into_activities,
)
from backend.inference.ps3.temporal_refiner import refine_actions_pass2
from backend.inference.ps3.validators import (
    validate_enrichment_events,
    validate_task_timeline,
    validate_video_duration,
    validate_video_path,
)
from backend.inference.ps3.video_utils import extract_video_segment, get_video_metadata, plan_video_segments

logger = get_logger("pipeline")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_video_to_timeline(
    video_path: str,
    clip_id: str,
    object_events: Optional[List[Dict[str, Any]]] = None,
    hand_tracks: Optional[List[Dict[str, Any]]] = None,
    interaction_events: Optional[List[Dict[str, Any]]] = None,
    *,
    config: Optional[ForgeTraceConfig] = None,
    interaction_events_provider=None,
) -> TaskTimeline:
    """Convert a factory video into a structured task timeline.

    Parameters
    ----------
    video_path : str
        Absolute or relative path to the video file.
    clip_id : str
        A unique identifier for this video clip (e.g. ``"clip_001"``).
    object_events : list of dict, optional
        YOLO object-detection events (label, timestamp, bbox …).
    hand_tracks : list of dict, optional
        MediaPipe hand-track events (hand, timestamp, landmarks …).
    interaction_events : list of dict, optional
        Hand–object interaction events from the interaction module.
    config : ForgeTraceConfig, optional
        Pipeline configuration.  Built from env vars when omitted.

    Returns
    -------
    TaskTimeline
        A validated, JSON-serialisable timeline object.

    Raises
    ------
    ForgeTraceError
        On unrecoverable failures (invalid video, API auth, etc.).
    """
    t0 = time.time()
    timings = {}

    # ---- 0. Configuration ------------------------------------------------
    if config is None:
        config = load_config()
    config.validate()
    logger.info("Pipeline started for clip_id=%s", clip_id)

    # ---- 1. Validate video -----------------------------------------------
    video_path_resolved = str(validate_video_path(video_path, config))

    meta = get_video_metadata(video_path_resolved)
    validate_video_duration(meta.duration_s, config)

    # ---- 2. Validate enrichment inputs -----------------------------------
    object_events = validate_enrichment_events(
        object_events, "object_events", meta.duration_s,
    )
    hand_tracks = validate_enrichment_events(
        hand_tracks, "hand_tracks", meta.duration_s,
    )
    interaction_events = validate_enrichment_events(
        interaction_events, "interaction_events", meta.duration_s,
    )

    # ---- 3. Plan segments ------------------------------------------------
    segments = plan_video_segments(meta.duration_s, config)
    logger.info("%d analysis segment(s) planned", len(segments))

    # ---- 4. Upload video & coarse recognition ----------------------------
    client = GeminiClient(config)
    uploaded_files = []
    segment_files = {}
    video_file = None

    try:
        stage_started = time.perf_counter()
        # Segment the actual media before asking Gemini to analyze each window.
        # The recognizer then adds each segment's start offset exactly once.
        with tempfile.TemporaryDirectory(prefix="forge_ps3_segments_") as temp_dir:
            for segment in segments:
                segment_path = extract_video_segment(
                    video_path_resolved, segment.start_s, segment.end_s, temp_dir,
                )
                try:
                    segment_file = client.upload_video(segment_path, mime_type="video/mp4")
                finally:
                    Path(segment_path).unlink(missing_ok=True)
                uploaded_files.append(segment_file)
                segment_files[segment.index] = segment_file

        # The full clip is used only for whole-video temporal refinement/grouping.
        if config.enable_refinement or config.enable_task_grouping:
            video_mime = mimetypes.guess_type(video_path_resolved)[0] or "video/mp4"
            video_file = client.upload_video(video_path_resolved, mime_type=video_mime)
            uploaded_files.append(video_file)

        timings['video_preparation_and_upload'] = round(time.perf_counter()-stage_started,3)

        failed_segments: List[int] = []
        stage_started = time.perf_counter()
        if interaction_events_provider is not None:
            interaction_events = interaction_events_provider()
        timings['wait_for_object_context'] = round(time.perf_counter()-stage_started,3)
        stage_started = time.perf_counter()
        coarse_actions = recognize_actions_full_video(
            client, segment_files, segments, failed_segments=failed_segments, object_context=interaction_events,
        )
        logger.info(
            "Coarse recognition complete: %d raw actions", len(coarse_actions)
        )
        timings['gemini_action_recognition'] = round(time.perf_counter()-stage_started,3)

        # ---- 5. Temporal refinement (Pass 2) -----------------------------
        stage_started = time.perf_counter()
        if config.enable_refinement and coarse_actions:
            refined_actions = refine_actions_pass2(
                client, video_file, coarse_actions, config,
            )
        else:
            from backend.inference.ps3.temporal_refiner import _coarse_to_refined
            refined_actions = _coarse_to_refined(coarse_actions)
        timings['timestamp_refinement'] = round(time.perf_counter()-stage_started,3)

        # ---- 6. Normalise & deduplicate ----------------------------------
        stage_started = time.perf_counter()
        cleaned_actions = normalize_and_deduplicate(refined_actions, config)
        logger.info(
            "After normalisation/dedup: %d actions (from %d refined)",
            len(cleaned_actions), len(refined_actions),
        )

        # ---- 7. Enrich with external modules -----------------------------
        enriched_actions = enrich_actions(
            cleaned_actions,
            meta.duration_s,
            object_events=object_events or None,
            hand_tracks=hand_tracks or None,
            interaction_events=interaction_events or None,
        )

        # ---- 8. Clamp timestamps to video duration -----------------------
        enriched_actions = _clamp_to_duration(enriched_actions, meta.duration_s)

        # ---- 9. Build task records and summary ---------------------------
        tasks = assign_task_ids(enriched_actions)
        summary = generate_task_summary(tasks)
        timings['normalization_and_enrichment'] = round(time.perf_counter()-stage_started,3)

        # ---- 10. Optional grouping ----------------------------------------
        groups = []
        stage_started = time.perf_counter()
        if config.enable_task_grouping and len(tasks) >= 2:
            groups = group_tasks_into_activities(
                tasks, client, video_file, config,
            )
        timings['task_grouping'] = round(time.perf_counter()-stage_started,3)

        # ---- 11. Enrichment source tracking ------------------------------
        enrichment_sources: List[str] = []
        if object_events:
            enrichment_sources.append("YOLO")
        if hand_tracks:
            enrichment_sources.append("MediaPipe")
        if interaction_events:
            enrichment_sources.append("Interaction")

        # ---- 12. Build timeline ------------------------------------------
        elapsed = round(time.time() - t0, 2)

        timeline = TaskTimeline(
            clip_id=clip_id,
            video_path=video_path_resolved,
            video_duration=meta.duration_s,
            total_tasks=summary["total_tasks"],
            unique_task_types=summary["unique_task_types"],
            unique_task_list=summary["unique_task_list"],
            tasks=tasks,
            groups=groups,
            processing_metadata=ProcessingMetadata(
                model_used=config.gemini_model,
                segments_analyzed=len(segments) - len(failed_segments),
                processing_time_s=elapsed,
                timestamp_refinement_applied=config.enable_refinement,
                enrichment_sources=enrichment_sources,
                failed_segments=failed_segments,
                stage_timings_s=timings,
            ),
        )

        # ---- 13. Final validation ----------------------------------------
        timeline = validate_task_timeline(timeline)
        logger.info(
            "Pipeline complete: %d tasks, %d groups, %.1fs processing",
            len(tasks), len(groups), elapsed,
        )
        return timeline

    finally:
        # Delete all provider-side uploads, including partial segment failures.
        for uploaded in uploaded_files:
            client.delete_file(uploaded)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp_to_duration(actions, duration_s):
    """Ensure all timestamps fall within [0, duration_s]."""
    from backend.inference.ps3.schemas import RefinedAction

    clamped = []
    for a in actions:
        start = max(0.0, min(a.start_s, duration_s))
        end = max(start, min(a.end_s, duration_s))
        if end == start and a.end_s > a.start_s:
            # Don't collapse a valid action to zero duration
            end = min(start + 0.1, duration_s)
        clamped.append(a.model_copy(update={
            "start_s": round(start, 2),
            "end_s": round(end, 2),
            "duration_s": round(end - start, 4),
        }))
    return clamped
