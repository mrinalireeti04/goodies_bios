"""Action normalisation, deduplication and cross-segment merging.

After temporal refinement, this module cleans the action list:

1. Normalise verb / object strings for consistency.
2. Remove duplicate predictions (same action with overlapping timestamps).
3. Merge actions that span segment boundaries.
4. Filter out low-confidence predictions.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from forge_trace.config import ForgeTraceConfig
from forge_trace.logging_config import get_logger
from forge_trace.schemas import RefinedAction

logger = get_logger("action_normalizer")


# ---------------------------------------------------------------------------
# Verb normalisation map
# ---------------------------------------------------------------------------

_VERB_ALIASES: Dict[str, str] = {
    "picks up": "pick up",
    "picked up": "pick up",
    "picking up": "pick up",
    "grab": "pick up",
    "grabs": "pick up",
    "grasp": "pick up",
    "grasps": "pick up",
    "places": "place",
    "placed": "place",
    "placing": "place",
    "put down": "place",
    "puts down": "place",
    "set down": "place",
    "sets down": "place",
    "moves": "move",
    "moved": "move",
    "moving": "move",
    "slide": "move",
    "slides": "move",
    "positions": "position",
    "positioned": "position",
    "positioning": "position",
    "aligns": "align",
    "aligned": "align",
    "aligning": "align",
    "holds": "hold",
    "held": "hold",
    "holding": "hold",
    "fastens": "fasten",
    "fastened": "fasten",
    "fastening": "fasten",
    "tighten": "fasten",
    "tightens": "fasten",
    "tightened": "fasten",
    "tightening": "fasten",
    "screw": "fasten",
    "screws": "fasten",
    "screwed": "fasten",
    "screwing": "fasten",
    "bolt": "fasten",
    "bolts": "fasten",
    "bolting": "fasten",
    "presses": "press",
    "pressed": "press",
    "pressing": "press",
    "push": "press",
    "pushes": "press",
    "opens": "open",
    "opened": "open",
    "opening": "open",
    "closes": "close",
    "closed": "close",
    "closing": "close",
    "inspects": "inspect",
    "inspected": "inspect",
    "inspecting": "inspect",
    "examine": "inspect",
    "examines": "inspect",
    "check": "inspect",
    "checks": "inspect",
    "releases": "release",
    "released": "release",
    "releasing": "release",
    "let go": "release",
    "lets go": "release",
    "rotate": "rotate",
    "rotates": "rotate",
    "rotated": "rotate",
    "rotating": "rotate",
    "turn": "rotate",
    "turns": "rotate",
    "flip": "flip",
    "flips": "flip",
    "flipped": "flip",
    "flipping": "flip",
    "insert": "insert",
    "inserts": "insert",
    "inserted": "insert",
    "inserting": "insert",
    "remove": "remove",
    "removes": "remove",
    "removed": "remove",
    "removing": "remove",
    "attach": "attach",
    "attaches": "attach",
    "attached": "attach",
    "attaching": "attach",
    "connect": "attach",
    "connects": "attach",
    "wipe": "wipe",
    "wipes": "wipe",
    "wiped": "wipe",
    "wiping": "wipe",
    "clean": "wipe",
    "cleans": "wipe",
    "apply": "apply",
    "applies": "apply",
    "applied": "apply",
    "applying": "apply",
    "reach": "reach",
    "reaches": "reach",
    "reached": "reach",
    "reaching": "reach",
    "adjust": "adjust",
    "adjusts": "adjust",
    "adjusted": "adjust",
    "adjusting": "adjust",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_actions(
    actions: List[RefinedAction],
) -> List[RefinedAction]:
    """Normalise verb strings across all actions for consistency."""
    normalised: List[RefinedAction] = []
    for a in actions:
        verb = _normalise_verb(a.verb)
        normalised.append(a.model_copy(update={"verb": verb}))
    return normalised


def filter_low_confidence(
    actions: List[RefinedAction],
    config: ForgeTraceConfig,
) -> List[RefinedAction]:
    """Drop actions below the configured confidence threshold."""
    kept = [a for a in actions if a.confidence >= config.confidence_threshold]
    dropped = len(actions) - len(kept)
    if dropped:
        logger.info(
            "Filtered %d low-confidence actions (threshold=%.2f)",
            dropped, config.confidence_threshold,
        )
    return kept


def merge_duplicate_actions(
    actions: List[RefinedAction],
    config: ForgeTraceConfig,
) -> List[RefinedAction]:
    """Remove duplicate predictions that refer to the same action.

    Two predictions are considered duplicates when:
    - They have the same (normalised) verb.
    - Their temporal IoU exceeds ``config.duplicate_iou_threshold``.

    The prediction with higher confidence is kept; the other is merged into it
    (extending its time span to the union and averaging confidence).
    """
    if len(actions) <= 1:
        return list(actions)

    # Sort by start time
    sorted_actions = sorted(actions, key=lambda a: a.start_s)
    merged_indices: Set[int] = set()
    result: List[RefinedAction] = []

    for i, a in enumerate(sorted_actions):
        if i in merged_indices:
            continue
        current = a
        for j in range(i + 1, len(sorted_actions)):
            if j in merged_indices:
                continue
            b = sorted_actions[j]
            # Quick exit: if b starts well after current ends, no more overlaps
            if b.start_s > current.end_s + 2.0:
                break
            if _same_action_type(current, b):
                iou = _temporal_iou(
                    current.start_s, current.end_s,
                    b.start_s, b.end_s,
                )
                if iou >= config.duplicate_iou_threshold:
                    current = _merge_pair(current, b)
                    merged_indices.add(j)

        result.append(current)

    removed = len(actions) - len(result)
    if removed:
        logger.info("Merged %d duplicate action predictions", removed)
    return result


def merge_cross_segment_actions(
    actions: List[RefinedAction],
    overlap_s: float,
) -> List[RefinedAction]:
    """Merge actions that were split across overlapping segment boundaries.

    Two actions are candidates for cross-boundary merge when:
    - They have the same verb (and optionally same object).
    - One action ends within the overlap region and the other starts there.
    - Their combined span is continuous.
    """
    if len(actions) <= 1 or overlap_s <= 0:
        return list(actions)

    sorted_actions = sorted(actions, key=lambda a: a.start_s)
    merged_indices: Set[int] = set()
    result: List[RefinedAction] = []

    for i, a in enumerate(sorted_actions):
        if i in merged_indices:
            continue
        current = a
        for j in range(i + 1, len(sorted_actions)):
            if j in merged_indices:
                continue
            b = sorted_actions[j]
            if b.start_s > current.end_s + overlap_s:
                break
            if _same_action_type(current, b):
                overlap = current.end_s - b.start_s
                # Must overlap in time (or be immediately contiguous within 1.0s)
                if overlap >= -1.0 and overlap <= overlap_s + 1.0:
                    current = _merge_pair(current, b)
                    merged_indices.add(j)

        result.append(current)

    removed = len(actions) - len(result)
    if removed:
        logger.info(
            "Merged %d cross-segment duplicate actions", removed
        )
    return result


def normalize_and_deduplicate(
    actions: List[RefinedAction],
    config: ForgeTraceConfig,
) -> List[RefinedAction]:
    """Full normalisation + deduplication pipeline."""
    actions = normalize_actions(actions)
    actions = merge_cross_segment_actions(actions, config.segment_overlap_s)
    actions = merge_duplicate_actions(actions, config)
    actions = filter_low_confidence(actions, config)
    return sorted(actions, key=lambda a: a.start_s)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_verb(verb: str) -> str:
    v = verb.strip().lower()
    return _VERB_ALIASES.get(v, v)


def _same_action_type(a: RefinedAction, b: RefinedAction) -> bool:
    """Check if two actions represent the same type of action."""
    if a.verb != b.verb:
        return False
    # Objects must match if both are known
    if a.object and b.object and a.object != b.object:
        return False
    return True


def _temporal_iou(
    s1: float, e1: float,
    s2: float, e2: float,
) -> float:
    """Compute temporal Intersection-over-Union between two intervals."""
    inter_start = max(s1, s2)
    inter_end = min(e1, e2)
    intersection = max(0.0, inter_end - inter_start)
    union = max(e1, e2) - min(s1, s2)
    if union <= 0:
        return 0.0
    return intersection / union


def _merge_pair(a: RefinedAction, b: RefinedAction) -> RefinedAction:
    """Merge two overlapping/adjacent duplicate actions into one."""
    start = min(a.start_s, b.start_s)
    end = max(a.end_s, b.end_s)
    # Keep the higher-confidence description
    primary = a if a.confidence >= b.confidence else b
    avg_conf = round((a.confidence + b.confidence) / 2, 4)
    return RefinedAction(
        verb=primary.verb,
        object=primary.object or a.object or b.object,
        hand=primary.hand or a.hand or b.hand,
        start_s=round(start, 2),
        end_s=round(end, 2),
        duration_s=round(end - start, 4),
        confidence=avg_conf,
        description=primary.description,
        evidence=primary.evidence,
        start_uncertainty_s=max(a.start_uncertainty_s, b.start_uncertainty_s),
        end_uncertainty_s=max(a.end_uncertainty_s, b.end_uncertainty_s),
        is_refined=a.is_refined or b.is_refined,
    )
