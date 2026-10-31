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
from typing import Any, Dict, List, Optional

from forge_trace.action_normalizer import normalize_and_deduplicate
from forge_trace.action_recognizer import recognize_actions_full_video
from forge_trace.config import ForgeTraceConfig, load_config
from forge_trace.enrichment import enrich_actions
from forge_trace.exceptions import ForgeTraceError
from forge_trace.gemini_client import GeminiClient
from forge_trace.logging_config import get_logger
from forge_trace.schemas import (
    ProcessingMetadata,
    TaskTimeline,
)
from forge_trace.task_builder import (
    assign_task_ids,
    generate_task_summary,
    group_tasks_into_activities,
)
from forge_trace.temporal_refiner import refine_actions_pass2
from forge_trace.validators import (
    validate_enrichment_events,
    validate_task_timeline,
    validate_video_duration,
    validate_video_path,
)
from forge_trace.video_utils import get_video_metadata, plan_video_segments

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

    # ---- 0. Configuration ------------------------------------------------
    if config is None:
        config = load_config()
    config.validate()
    logger.info("Pipeline started for clip_id=%s, video=%s", clip_id, video_path)

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
    video_file = client.upload_video(video_path_resolved)

    try:
        coarse_actions = recognize_actions_full_video(
            client, video_file, segments,
        )
        logger.info(
            "Coarse recognition complete: %d raw actions", len(coarse_actions)
        )

        # ---- 5. Temporal refinement (Pass 2) -----------------------------
        if config.enable_refinement and coarse_actions:
            refined_actions = refine_actions_pass2(
                client, video_file, coarse_actions, config,
            )
        else:
            from forge_trace.temporal_refiner import _coarse_to_refined
            refined_actions = _coarse_to_refined(coarse_actions)

        # ---- 6. Normalise & deduplicate ----------------------------------
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

        # ---- 10. Optional grouping ----------------------------------------
        groups = []
        if config.enable_task_grouping and len(tasks) >= 2:
            groups = group_tasks_into_activities(
                tasks, client, video_file, config,
            )

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
                segments_analyzed=len(segments),
                processing_time_s=elapsed,
                timestamp_refinement_applied=config.enable_refinement,
                enrichment_sources=enrichment_sources,
                failed_segments=[],
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
        # Clean up uploaded file
        client.delete_file(video_file)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp_to_duration(actions, duration_s):
    """Ensure all timestamps fall within [0, duration_s]."""
    from forge_trace.schemas import RefinedAction

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
