"""Temporal refinement of action boundaries.

After the coarse recognition pass produces approximate start/end timestamps,
this module performs a second analysis pass asking the model to verify and
sharpen those boundaries.  It also supports optional frame-level verification
around individual transition points.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

import numpy as np

from forge_trace.config import ForgeTraceConfig
from forge_trace.exceptions import TemporalRefinementError
from forge_trace.gemini_client import GeminiClient
from forge_trace.logging_config import get_logger
from forge_trace.schemas import ActionPrediction, RefinedAction
from forge_trace.video_utils import (
    extract_frames_around_boundary,
    frames_to_jpeg_bytes,
)

logger = get_logger("temporal_refiner")


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_REFINEMENT_PROMPT = """\
You are an expert industrial-video analyst performing **timestamp refinement**.

The following actions were identified in a previous analysis pass of this
factory video.  Your job is to:

1. **Verify** each action actually occurs in the video.  If an action is NOT
   visible, set its confidence to 0.0.
2. **Refine** the start_s and end_s timestamps to be as precise as possible:
   - start_s = the moment the specific action begins (not just hand movement
     toward an object).
   - end_s   = the moment the action is complete (not the start of the next
     action).
3. If you notice an action that was **missed** in the original list, add it.
4. Estimate your confidence honestly.

Previously identified actions:
{action_list_json}

Return a JSON array with the same fields:
  verb, object, hand, start_s, end_s, confidence, description

Return ONLY the JSON array.
"""

_FRAME_BOUNDARY_PROMPT = """\
These {n_frames} images are sequential frames sampled every {step:.1f}s \
from a factory video, covering the time range {t_start:.2f}s to {t_end:.2f}s.

An action described as "{action_desc}" was detected near this time range.
Its approximate {boundary_type} is at {approx_ts:.2f}s.

Examine the frames carefully.  Identify the most likely precise timestamp \
(in seconds) at which the action {boundary_verb}.

Return a JSON object: {{"refined_ts": <float>, "confidence": <float 0-1>}}
"""


# ---------------------------------------------------------------------------
# Pass-2: full-video refinement
# ---------------------------------------------------------------------------

def refine_actions_pass2(
    client: GeminiClient,
    video_file: Any,
    coarse_actions: List[ActionPrediction],
    config: ForgeTraceConfig,
) -> List[RefinedAction]:
    """Send the coarse action list back to Gemini for timestamp refinement.

    This is the primary refinement path (Pass 2).  It reuses the already-
    uploaded video file and asks the model to tighten boundaries.
    """
    if not coarse_actions:
        return []

    # Build a summary of actions for the prompt
    action_dicts = [
        {
            "verb": a.verb,
            "object": a.object,
            "hand": a.hand,
            "start_s": a.start_s,
            "end_s": a.end_s,
            "confidence": a.confidence,
            "description": a.description,
        }
        for a in coarse_actions
    ]
    import json
    action_json = json.dumps(action_dicts, indent=2)
    prompt = _REFINEMENT_PROMPT.format(action_list_json=action_json)

    try:
        raw_text = client.analyze_video(video_file, prompt, json_mode=True)
        parsed = client.parse_json_response(raw_text)
    except Exception as exc:
        logger.warning(
            "Pass-2 refinement failed (%s); falling back to coarse timestamps",
            exc,
        )
        return _coarse_to_refined(coarse_actions, refined=False)

    return _parse_refined_list(parsed, coarse_actions)


def _parse_refined_list(
    raw: Any,
    coarse_actions: List[ActionPrediction],
) -> List[RefinedAction]:
    """Parse the refinement response into :class:`RefinedAction` objects.

    Computes ``start_uncertainty_s`` and ``end_uncertainty_s`` as the absolute
    shift from the coarse prediction.
    """
    if not isinstance(raw, list):
        if isinstance(raw, dict):
            for key in ("actions", "tasks", "results"):
                if key in raw and isinstance(raw[key], list):
                    raw = raw[key]
                    break
            else:
                logger.warning("Refinement returned unexpected dict; using coarse")
                return _coarse_to_refined(coarse_actions, refined=False)
        else:
            logger.warning("Refinement returned non-list; using coarse")
            return _coarse_to_refined(coarse_actions, refined=False)

    # Build a lookup of coarse actions by approximate timestamp for
    # uncertainty computation
    coarse_by_approx: Dict[str, ActionPrediction] = {}
    for a in coarse_actions:
        key = f"{a.verb}_{a.start_s:.1f}"
        coarse_by_approx[key] = a

    refined: List[RefinedAction] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            continue
        try:
            verb = str(entry.get("verb", "")).strip().lower()
            if not verb:
                continue

            start = round(float(entry.get("start_s", 0)), 2)
            end = round(float(entry.get("end_s", 0)), 2)
            if end < start:
                end = start
            confidence = max(0.0, min(1.0, float(entry.get("confidence", 0.5))))

            # Compute uncertainty relative to closest coarse match
            start_unc = 0.0
            end_unc = 0.0
            key = f"{verb}_{start:.1f}"
            if key in coarse_by_approx:
                ca = coarse_by_approx[key]
                start_unc = abs(start - ca.start_s)
                end_unc = abs(end - ca.end_s)
            else:
                # Try to find closest coarse action with same verb
                for ca in coarse_actions:
                    if ca.verb == verb:
                        start_unc = abs(start - ca.start_s)
                        end_unc = abs(end - ca.end_s)
                        break

            obj_val = entry.get("object")
            hand_val = entry.get("hand")

            ra = RefinedAction(
                verb=verb,
                object=_clean(obj_val),
                hand=_normalise_hand(hand_val),
                start_s=start,
                end_s=end,
                duration_s=round(end - start, 4),
                confidence=confidence,
                description=str(entry.get("description", "")),
                evidence="Gemini video analysis (refined)",
                start_uncertainty_s=round(start_unc, 2),
                end_uncertainty_s=round(end_unc, 2),
                is_refined=True,
            )
            refined.append(ra)

        except (ValueError, TypeError) as exc:
            logger.warning("Refinement entry %d skipped: %s", idx, exc)

    if not refined:
        logger.warning("Refinement produced 0 valid actions; using coarse")
        return _coarse_to_refined(coarse_actions, refined=False)

    logger.info(
        "Pass-2 refinement: %d refined actions (from %d coarse)",
        len(refined), len(coarse_actions),
    )
    return refined


# ---------------------------------------------------------------------------
# Frame-level boundary refinement (optional, expensive)
# ---------------------------------------------------------------------------

def refine_boundary_with_frames(
    client: GeminiClient,
    video_path: str,
    action: RefinedAction,
    boundary_type: str,  # "start" or "end"
    config: ForgeTraceConfig,
) -> float:
    """Use sampled frames to refine a single boundary timestamp.

    Returns the refined timestamp.  Falls back to the original on error.
    """
    approx_ts = action.start_s if boundary_type == "start" else action.end_s
    boundary_verb = "begins" if boundary_type == "start" else "ends"

    try:
        frame_data = extract_frames_around_boundary(
            video_path,
            approx_ts,
            window_s=config.refinement_window_s,
            step_s=config.refinement_step_s,
        )
        if len(frame_data) < 3:
            return approx_ts

        timestamps = [t for t, _ in frame_data]
        frames = [f for _, f in frame_data]
        jpeg_bytes_list = frames_to_jpeg_bytes(frames)

        # Build multimodal parts
        from google.genai import types
        parts = [
            types.Part.from_bytes(data=jpg, mime_type="image/jpeg")
            for jpg in jpeg_bytes_list
        ]

        prompt = _FRAME_BOUNDARY_PROMPT.format(
            n_frames=len(frames),
            step=config.refinement_step_s,
            t_start=timestamps[0],
            t_end=timestamps[-1],
            action_desc=f"{action.verb} {action.object or ''}".strip(),
            boundary_type=boundary_type,
            approx_ts=approx_ts,
            boundary_verb=boundary_verb,
        )

        raw = client.analyze_frames(parts, prompt, json_mode=True)
        parsed = client.parse_json_response(raw)

        if isinstance(parsed, dict) and "refined_ts" in parsed:
            refined = float(parsed["refined_ts"])
            # Sanity: must be within the sampled window
            if timestamps[0] <= refined <= timestamps[-1]:
                logger.debug(
                    "Frame refinement %s: %.2f → %.2f",
                    boundary_type, approx_ts, refined,
                )
                return round(refined, 2)

    except Exception as exc:
        logger.debug("Frame-level refinement failed (%s): %s", boundary_type, exc)

    return approx_ts


# ---------------------------------------------------------------------------
# Convenience: coarse → refined passthrough
# ---------------------------------------------------------------------------

def _coarse_to_refined(
    actions: List[ActionPrediction],
    refined: bool = False,
) -> List[RefinedAction]:
    """Convert coarse predictions to :class:`RefinedAction` without changes."""
    return [
        RefinedAction(
            verb=a.verb,
            object=a.object,
            hand=a.hand,
            start_s=a.start_s,
            end_s=a.end_s,
            duration_s=a.duration_s,
            confidence=a.confidence,
            description=a.description,
            evidence=a.evidence,
            start_uncertainty_s=0.0,
            end_uncertainty_s=0.0,
            is_refined=refined,
        )
        for a in actions
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("", "null", "none", "n/a", "unknown"):
        return None
    return s


def _normalise_hand(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip().lower()
    return s if s in ("left", "right", "both") else None
