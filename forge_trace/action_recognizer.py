"""Semantic action recognition via Gemini video understanding.

This module is responsible for the *coarse* analysis pass: it sends a video
(or video segment) to Gemini and receives a list of candidate action
predictions with approximate timestamps.
"""

from __future__ import annotations

from typing import Any, List, Optional

from forge_trace.config import ForgeTraceConfig
from forge_trace.exceptions import (
    GeminiResponseParseError,
    SegmentAnalysisError,
)
from forge_trace.gemini_client import GeminiClient
from forge_trace.logging_config import get_logger
from forge_trace.schemas import ActionPrediction, VideoSegment

logger = get_logger("action_recognizer")


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_COARSE_ANALYSIS_PROMPT = """\
You are an expert industrial-video analyst.  Carefully watch this factory \
video and identify every **distinct action** the worker performs.

RULES — read carefully:
1. Report each individual action instance separately, even if the same type \
of action is repeated.
2. Do NOT merge several steps into one high-level summary.
3. A single continuous motion (e.g. sliding a component along a rail) is ONE \
action — do not split it into micro-movements.
4. If the worker holds an object while performing a different action, the \
holding and the other action are separate only when they are distinct tasks.
5. Do NOT invent actions that are not clearly visible in the video.
6. Provide approximate start and end timestamps in **seconds** from the \
beginning of the video.{time_range_note}
7. If you cannot determine the object or hand, set the field to null.
8. Estimate your confidence honestly — use lower values when the view is \
obstructed or the action is ambiguous.

For each action return a JSON object with these fields:
  verb        – the action verb phrase (e.g. "pick up", "fasten", "align")
  object      – the object involved, or null
  hand        – "left", "right", "both", or null
  start_s     – start time in seconds
  end_s       – end time in seconds
  confidence  – 0.0–1.0
  description – one-sentence description

Return ONLY a JSON array of action objects.  No other text.
"""

_TIME_RANGE_NOTE = """
   NOTE: This video clip corresponds to the time range \
{start:.1f}s – {end:.1f}s of the full recording.  Report all timestamps \
relative to the **clip start** (i.e. 0 s = the first frame of this clip).
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_coarse_prompt(segment: Optional[VideoSegment] = None) -> str:
    """Build the system prompt for coarse action recognition.

    If *segment* is provided the model is told the clip offset so that
    it can label timestamps relative to the clip.
    """
    time_note = ""
    if segment is not None and segment.start_s > 0:
        time_note = _TIME_RANGE_NOTE.format(
            start=segment.start_s, end=segment.end_s,
        )
    return _COARSE_ANALYSIS_PROMPT.format(time_range_note=time_note)


def parse_action_list(
    raw: Any,
    segment: Optional[VideoSegment] = None,
) -> List[ActionPrediction]:
    """Convert a parsed JSON structure into validated :class:`ActionPrediction` objects.

    If *segment* is given, timestamps are offset to the full-video timeline.
    Invalid entries are logged and skipped.
    """
    if not isinstance(raw, list):
        # Some models wrap the array inside a dict
        if isinstance(raw, dict):
            for key in ("actions", "tasks", "results"):
                if key in raw and isinstance(raw[key], list):
                    raw = raw[key]
                    break
            else:
                raise GeminiResponseParseError(
                    f"Expected a JSON array of actions, got dict with keys {list(raw.keys())}"
                )
        else:
            raise GeminiResponseParseError(
                f"Expected a JSON array, got {type(raw).__name__}"
            )

    offset = segment.start_s if segment is not None else 0.0
    seg_idx = segment.index if segment is not None else None
    seg_end = segment.end_s if segment is not None else None

    actions: List[ActionPrediction] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            logger.warning("Action entry %d is not a dict — skipping", idx)
            continue
        try:
            start = float(entry.get("start_s", 0)) + offset
            end = float(entry.get("end_s", 0)) + offset
            # Clamp end to segment boundary if applicable
            if seg_end is not None:
                end = min(end, seg_end)
            if end < start:
                end = start  # degenerate; will be filtered later

            verb = str(entry.get("verb", "")).strip()
            if not verb:
                logger.warning("Action entry %d has empty verb — skipping", idx)
                continue

            ap = ActionPrediction(
                verb=verb.lower(),
                object=_clean_optional(entry.get("object")),
                hand=_clean_optional(entry.get("hand")),
                start_s=round(start, 2),
                end_s=round(end, 2),
                confidence=_clamp(float(entry.get("confidence", 0.5)), 0, 1),
                description=str(entry.get("description", "")),
                evidence="Gemini video analysis",
                segment_index=seg_idx,
            )
            actions.append(ap)

        except (ValueError, TypeError) as exc:
            logger.warning(
                "Could not parse action entry %d: %s — skipping", idx, exc
            )

    logger.info(
        "Parsed %d / %d action predictions (offset=%.1fs)",
        len(actions), len(raw), offset,
    )
    return actions


def analyze_segment(
    client: GeminiClient,
    video_file: Any,
    segment: Optional[VideoSegment] = None,
) -> List[ActionPrediction]:
    """Run coarse action recognition on *video_file* for one segment.

    Returns a list of :class:`ActionPrediction` objects with timestamps
    already mapped to the full-video timeline.
    """
    prompt = build_coarse_prompt(segment)

    try:
        raw_text = client.analyze_video(video_file, prompt, json_mode=True)
    except Exception as exc:
        seg_idx = segment.index if segment else 0
        raise SegmentAnalysisError(
            f"Gemini analysis failed for segment {seg_idx}: {exc}",
            segment_index=seg_idx,
        ) from exc

    parsed = client.parse_json_response(raw_text)
    return parse_action_list(parsed, segment)


def recognize_actions_full_video(
    client: GeminiClient,
    video_file: Any,
    segments: List[VideoSegment],
) -> List[ActionPrediction]:
    """Analyse all *segments* and collect action predictions.

    Failed segments are logged and skipped so that partial results are still
    returned.
    """
    all_actions: List[ActionPrediction] = []
    failed_segments: List[int] = []

    for seg in segments:
        try:
            actions = analyze_segment(client, video_file, seg)
            all_actions.extend(actions)
            logger.info(
                "Segment %d [%.1f–%.1fs]: %d actions found",
                seg.index, seg.start_s, seg.end_s, len(actions),
            )
        except SegmentAnalysisError as exc:
            logger.error("Segment %d failed: %s", seg.index, exc)
            failed_segments.append(seg.index)

    if failed_segments:
        logger.warning(
            "%d / %d segments failed: %s",
            len(failed_segments), len(segments), failed_segments,
        )

    return all_actions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_optional(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("", "null", "none", "n/a", "unknown", "unidentified"):
        return None
    return s


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
