"""Enrichment of recognised actions with external module outputs.

This module integrates data from other FORGE TRACE modules:
- **YOLO**: object detections and bounding boxes.
- **MediaPipe**: hand landmarks and left/right identification.
- **Interaction module**: probable hand–object interaction events.

All enrichment inputs are **optional**.  When unavailable, the action
records retain ``"unknown"`` for missing fields.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.inference.ps3.logging_config import get_logger
from backend.inference.ps3.schemas import RefinedAction

logger = get_logger("enrichment")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich_actions(
    actions: List[RefinedAction],
    video_duration: float,
    object_events: Optional[List[Dict[str, Any]]] = None,
    hand_tracks: Optional[List[Dict[str, Any]]] = None,
    interaction_events: Optional[List[Dict[str, Any]]] = None,
) -> List[RefinedAction]:
    """Enrich *actions* with data from YOLO, MediaPipe and interaction modules.

    Each enrichment source is applied independently; failures in one source
    do not block the others.
    """
    sources_used: List[str] = []

    if object_events:
        actions = _enrich_with_objects(actions, object_events, video_duration)
        sources_used.append("YOLO")

    if hand_tracks:
        actions = _enrich_with_hands(actions, hand_tracks, video_duration)
        sources_used.append("MediaPipe")

    if interaction_events:
        actions = _enrich_with_interactions(
            actions, interaction_events, video_duration,
        )
        sources_used.append("Interaction")

    if sources_used:
        logger.info("Enrichment sources applied: %s", ", ".join(sources_used))
    else:
        logger.info("No enrichment sources provided; actions unchanged")

    return actions


# ---------------------------------------------------------------------------
# Object enrichment (YOLO)
# ---------------------------------------------------------------------------

def _enrich_with_objects(
    actions: List[RefinedAction],
    object_events: List[Dict[str, Any]],
    video_duration: float,
) -> List[RefinedAction]:
    """Associate YOLO-detected objects with actions based on temporal overlap.

    Each object event is expected to have at least:
    - ``label`` or ``class_name``: the detected object class
    - ``timestamp`` or ``start_s`` / ``end_s``: when the detection occurs

    When an action has no identified object and a YOLO detection falls within
    its time span, the detection's label is used.
    """
    enriched: List[RefinedAction] = []
    for action in actions:
        if action.object is not None:
            # Object already identified by Gemini — keep it
            enriched.append(action)
            continue

        best_label = _find_best_object(action, object_events)
        if best_label:
            action = action.model_copy(update={
                "object": best_label,
                "evidence": f"{action.evidence}; object from YOLO",
            })
        enriched.append(action)

    return enriched


def _find_best_object(
    action: RefinedAction,
    events: List[Dict[str, Any]],
) -> Optional[str]:
    """Return the most relevant YOLO-detected object label for *action*."""
    candidates: List[tuple[str, float]] = []

    for evt in events:
        label = str(evt.get("label", evt.get("class_name", ""))).strip().lower()
        if not label:
            continue

        ts = _event_timestamp(evt)
        if ts is None:
            continue

        if action.start_s <= ts <= action.end_s:
            conf = float(evt.get("confidence", evt.get("score", 0.5)))
            candidates.append((label, conf))

    if not candidates:
        return None

    # Return label with highest confidence
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


# ---------------------------------------------------------------------------
# Hand enrichment (MediaPipe)
# ---------------------------------------------------------------------------

def _enrich_with_hands(
    actions: List[RefinedAction],
    hand_tracks: List[Dict[str, Any]],
    video_duration: float,
) -> List[RefinedAction]:
    """Associate MediaPipe hand tracks with actions.

    Each hand track event is expected to have:
    - ``hand`` or ``handedness``: "left" or "right"
    - ``timestamp`` or ``start_s``: when the hand was detected

    When an action has no identified hand and a MediaPipe detection falls
    within its time span, the hand label is used.
    """
    enriched: List[RefinedAction] = []
    for action in actions:
        if action.hand is not None:
            enriched.append(action)
            continue

        best_hand = _find_dominant_hand(action, hand_tracks)
        if best_hand:
            action = action.model_copy(update={
                "hand": best_hand,
                "evidence": f"{action.evidence}; hand from MediaPipe",
            })
        enriched.append(action)

    return enriched


def _find_dominant_hand(
    action: RefinedAction,
    tracks: List[Dict[str, Any]],
) -> Optional[str]:
    """Determine which hand (left/right/both) is dominant during *action*."""
    hands_seen: Dict[str, int] = {"left": 0, "right": 0}

    for evt in tracks:
        hand = str(evt.get("hand", evt.get("handedness", ""))).strip().lower()
        if hand not in ("left", "right"):
            continue

        ts = _event_timestamp(evt)
        if ts is None:
            continue

        if action.start_s <= ts <= action.end_s:
            hands_seen[hand] += 1

    left = hands_seen["left"]
    right = hands_seen["right"]

    if left == 0 and right == 0:
        return None
    if left > 0 and right > 0:
        # Both hands active — call it "both" if within 2× of each other
        ratio = min(left, right) / max(left, right)
        if ratio > 0.3:
            return "both"
    return "left" if left > right else "right"


# ---------------------------------------------------------------------------
# Interaction enrichment
# ---------------------------------------------------------------------------

def _enrich_with_interactions(
    actions: List[RefinedAction],
    interaction_events: List[Dict[str, Any]],
    video_duration: float,
) -> List[RefinedAction]:
    """Use interaction events as supporting evidence for actions.

    Interaction events can:
    - Confirm an action's existence (boost confidence).
    - Fill in missing object / hand info from the interaction record.
    """
    enriched: List[RefinedAction] = []
    for action in actions:
        matched = _find_matching_interactions(action, interaction_events)
        if not matched:
            enriched.append(action)
            continue

        updates: Dict[str, Any] = {}
        evidence_parts = [action.evidence]

        # Confidence boost: clamp at 1.0
        boost = min(0.1, 0.02 * len(matched))
        updates["confidence"] = min(1.0, round(action.confidence + boost, 4))

        # Fill missing object
        if action.object is None:
            for m in matched:
                obj = str(m.get("object", m.get("label", ""))).strip().lower()
                if obj and obj not in ("", "unknown"):
                    updates["object"] = obj
                    evidence_parts.append("object from interaction module")
                    break

        # Fill missing hand
        if action.hand is None:
            for m in matched:
                hand = str(m.get("hand", "")).strip().lower()
                if hand in ("left", "right", "both"):
                    updates["hand"] = hand
                    evidence_parts.append("hand from interaction module")
                    break

        updates["evidence"] = "; ".join(evidence_parts)
        enriched.append(action.model_copy(update=updates))

    return enriched


def _find_matching_interactions(
    action: RefinedAction,
    events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Find interaction events that overlap with *action*."""
    matches: List[Dict[str, Any]] = []
    for evt in events:
        ts = _event_timestamp(evt)
        if ts is not None and action.start_s <= ts <= action.end_s:
            matches.append(evt)
            continue
        # Check range-based events
        es = evt.get("start_s")
        ee = evt.get("end_s")
        if es is not None and ee is not None:
            if float(ee) >= action.start_s and float(es) <= action.end_s:
                matches.append(evt)
    return matches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _event_timestamp(evt: Dict[str, Any]) -> Optional[float]:
    """Extract a point timestamp from an event dict."""
    for key in ("timestamp", "time", "start_s", "frame_time"):
        if key in evt:
            try:
                return float(evt[key])
            except (ValueError, TypeError):
                continue
    return None
