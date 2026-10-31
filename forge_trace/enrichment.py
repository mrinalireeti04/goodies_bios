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

from forge_trace.config import ForgeTraceConfig
from forge_trace.logging_config import get_logger
from forge_trace.schemas import RefinedAction, TaskRecord

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


def _clean_hand_str(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip().lower()
    return s if s in ("left", "right", "both") else None


# ---------------------------------------------------------------------------
# PS2 tool integration
# ---------------------------------------------------------------------------

def associate_ps2_tools(
    tasks: List[TaskRecord],
    tool_events: Optional[List[Dict[str, Any]]] = None,
    config: Optional[ForgeTraceConfig] = None,
) -> List[TaskRecord]:
    """Associate PS2 tool-hold events with PS3 tasks using temporal overlap and hand identity.

    Important: A tool being held during a task does not prove it was actively used
    throughout that task. This represents tool association, not verified tool usage.

    Returns updated TaskRecord objects with populated tool fields.
    """
    if not tasks:
        return []

    if not tool_events:
        return [
            t.model_copy(update={
                "tool": None,
                "tool_hand": None,
                "tool_match_confidence": None,
                "tool_association_status": "none",
                "tool_evidence": "No PS2 tool-hold events provided",
            })
            for t in tasks
        ]

    # Pre-parse tool events
    parsed_events: List[Dict[str, Any]] = []
    for idx, evt in enumerate(tool_events):
        if not isinstance(evt, dict):
            continue
        tool_name = str(
            evt.get("tool") or evt.get("tool_name") or evt.get("label") or evt.get("name") or ""
        ).strip().lower()
        if not tool_name or tool_name in ("null", "none", "unknown"):
            continue

        hand = evt.get("hand") or evt.get("tool_hand") or evt.get("hold_hand") or evt.get("handedness")
        hand_clean = _clean_hand_str(hand)

        start_s = evt.get("start_s")
        end_s = evt.get("end_s")
        if start_s is None and "timestamp" in evt:
            start_s = float(evt["timestamp"])
            end_s = start_s + 0.5
        elif start_s is not None and end_s is None:
            end_s = float(start_s) + 0.5
        elif start_s is not None and end_s is not None:
            start_s = float(start_s)
            end_s = float(end_s)
        else:
            continue

        if end_s < start_s:
            start_s, end_s = end_s, start_s

        conf = float(evt.get("confidence") or evt.get("score") or 0.8)
        conf = max(0.0, min(1.0, conf))

        parsed_events.append({
            "tool": tool_name,
            "hand": hand_clean,
            "start_s": start_s,
            "end_s": end_s,
            "confidence": conf,
        })

    enriched_tasks: List[TaskRecord] = []
    for task in tasks:
        t_start = task.timeline.start_s
        t_end = task.timeline.end_s
        t_duration = max(0.1, task.timeline.duration_s)
        t_hand = _clean_hand_str(task.hand)

        # Find all overlapping events
        candidate_overlaps: Dict[str, Dict[str, Any]] = {}
        hand_mismatched_tools: List[str] = []

        for evt in parsed_events:
            e_start = evt["start_s"]
            e_end = evt["end_s"]
            overlap = max(0.0, min(t_end, e_end) - max(t_start, e_start))
            if overlap <= 0.05:  # ignore negligible overlap (<50ms)
                continue

            tool_name = evt["tool"]
            evt_hand = evt["hand"]

            # Hand compatibility check
            hand_compatible = True
            if t_hand in ("left", "right") and evt_hand in ("left", "right"):
                if t_hand != evt_hand:
                    hand_compatible = False
                    if tool_name not in hand_mismatched_tools:
                        hand_mismatched_tools.append(tool_name)

            if not hand_compatible:
                continue

            if tool_name not in candidate_overlaps:
                candidate_overlaps[tool_name] = {
                    "total_overlap": 0.0,
                    "intervals": [],
                    "hands": [],
                    "confidences": [],
                }
            candidate_overlaps[tool_name]["total_overlap"] += overlap
            candidate_overlaps[tool_name]["intervals"].append((max(t_start, e_start), min(t_end, e_end)))
            if evt_hand:
                candidate_overlaps[tool_name]["hands"].append(evt_hand)
            candidate_overlaps[tool_name]["confidences"].append(evt["confidence"])

        # Determine association outcome
        if not candidate_overlaps:
            if hand_mismatched_tools:
                enriched_tasks.append(
                    task.model_copy(update={
                        "tool": None,
                        "tool_hand": None,
                        "tool_match_confidence": None,
                        "tool_association_status": "hand_mismatch",
                        "tool_evidence": (
                            f"Hand mismatch: tool(s) {', '.join(hand_mismatched_tools)} "
                            f"held in opposing hand to task hand ({t_hand})"
                        ),
                    })
                )
            else:
                enriched_tasks.append(
                    task.model_copy(update={
                        "tool": None,
                        "tool_hand": None,
                        "tool_match_confidence": None,
                        "tool_association_status": "none",
                        "tool_evidence": "No overlapping PS2 tool-hold events found",
                    })
                )
            continue

        # Sort candidates by total overlap descending
        sorted_candidates = sorted(
            candidate_overlaps.items(),
            key=lambda x: x[1]["total_overlap"],
            reverse=True,
        )

        top_tool, top_data = sorted_candidates[0]
        top_overlap = top_data["total_overlap"]

        # Check for ambiguity if multiple tools overlap
        is_ambiguous = False
        if len(sorted_candidates) > 1:
            second_tool, second_data = sorted_candidates[1]
            second_overlap = second_data["total_overlap"]

            # If second tool overlap is within 30% of top tool, or both have significant overlap
            # and neither clearly dominates:
            if top_overlap > 0 and (second_overlap / top_overlap) >= 0.70:
                is_ambiguous = True
            elif (top_overlap - second_overlap) < (0.3 * t_duration) and second_overlap >= (0.25 * t_duration):
                is_ambiguous = True

        if is_ambiguous:
            tools_str = ", ".join(
                f"{name} ({data['total_overlap']:.1f}s overlap)"
                for name, data in sorted_candidates[:3]
            )
            enriched_tasks.append(
                task.model_copy(update={
                    "tool": None,
                    "tool_hand": None,
                    "tool_match_confidence": None,
                    "tool_association_status": "ambiguous",
                    "tool_evidence": f"Ambiguous association: multiple overlapping candidate tools: {tools_str}",
                })
            )
            continue

        # Single clear dominant tool
        dominant_hand = None
        if top_data["hands"]:
            dominant_hand = max(set(top_data["hands"]), key=top_data["hands"].count)
        else:
            dominant_hand = t_hand if t_hand != "unknown" else None

        avg_conf = sum(top_data["confidences"]) / len(top_data["confidences"])
        overlap_frac = min(1.0, top_overlap / t_duration)
        match_confidence = round(0.5 * overlap_frac + 0.5 * avg_conf, 2)

        evidence = (
            f"Associated tool: '{top_tool}' held in {dominant_hand or 'unspecified'} hand "
            f"for {top_overlap:.1f}s during task ({overlap_frac * 100:.0f}% temporal overlap). "
            f"Note: represents tool association during task interval, not verified continuous tool usage."
        )

        enriched_tasks.append(
            task.model_copy(update={
                "tool": top_tool,
                "tool_hand": dominant_hand,
                "tool_match_confidence": match_confidence,
                "tool_association_status": "associated",
                "tool_evidence": evidence,
            })
        )

    return enriched_tasks
