"""Task building: ID assignment, task-name generation, and optional grouping.

Converts a cleaned list of :class:`RefinedAction` objects into the final
:class:`TaskRecord` list, assigns chronological IDs, computes summary
statistics, and optionally groups related actions into higher-level
activities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from forge_trace.config import ForgeTraceConfig
from forge_trace.gemini_client import GeminiClient
from forge_trace.logging_config import get_logger
from forge_trace.schemas import (
    RefinedAction,
    TaskGroup,
    TaskRecord,
    TimelineInfo,
)

logger = get_logger("task_builder")


# ---------------------------------------------------------------------------
# ID assignment
# ---------------------------------------------------------------------------

def assign_task_ids(
    actions: List[RefinedAction],
) -> List[TaskRecord]:
    """Convert refined actions into :class:`TaskRecord` objects with
    chronologically assigned IDs.

    Actions are sorted by start time; IDs are ``T001``, ``T002``, …
    """
    sorted_actions = sorted(actions, key=lambda a: (a.start_s, a.end_s))
    records: List[TaskRecord] = []

    for idx, action in enumerate(sorted_actions, start=1):
        task_id = f"T{idx:03d}"
        task_name = _build_task_name(action)
        duration = round(action.end_s - action.start_s, 4)

        # Ensure verb contains only the normalized action verb
        clean_verb = action.verb.split(" with ")[0].strip()
        if action.object and action.object.lower() in clean_verb.lower():
            import re
            clean_verb = re.sub(rf"\b{re.escape(action.object)}\b", "", clean_verb, flags=re.IGNORECASE).strip()
        if not clean_verb:
            clean_verb = action.verb

        record = TaskRecord(
            task_id=task_id,
            task_name=task_name,
            verb=clean_verb.lower(),
            object=action.object if action.object else "unknown",
            hand=action.hand if action.hand else "unknown",
            timeline=TimelineInfo(
                start_s=action.start_s,
                end_s=action.end_s,
                duration_s=duration,
            ),
            model_confidence_score=getattr(action, "model_confidence_score", action.confidence),
            confidence=getattr(action, "model_confidence_score", action.confidence),
            description=action.description or task_name,
            evidence=action.evidence,
        )
        records.append(record)

    logger.info("Assigned IDs to %d tasks (T001–T%03d)", len(records), len(records))
    return records


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def generate_task_summary(
    tasks: List[TaskRecord],
) -> Dict[str, Any]:
    """Compute summary statistics for the task timeline."""
    if not tasks:
        return {
            "total_tasks": 0,
            "unique_task_types": 0,
            "unique_task_list": [],
        }

    unique_names = sorted({t.task_name for t in tasks})
    return {
        "total_tasks": len(tasks),
        "unique_task_types": len(unique_names),
        "unique_task_list": unique_names,
    }


# ---------------------------------------------------------------------------
# Task grouping (optional higher-level activities)
# ---------------------------------------------------------------------------

_GROUPING_PROMPT = """\
You are analysing a sequence of individual worker actions from a factory video.

Here is the chronological list of identified tasks:
{task_list_json}

Determine if any of these actions form logical **higher-level activities** \
(e.g. "Component Assembly", "Quality Inspection", "Part Preparation").

Rules:
1. Only group actions when the visual sequence clearly supports it.
2. A group must contain at least 2 tasks.
3. Not every task needs to belong to a group.
4. Do not invent groups that are not supported by the task list.

Return a JSON array of groups:
[
  {{
    "group_name": "Activity name",
    "task_ids": ["T001", "T002", "T003"],
    "description": "Brief description of the higher-level activity"
  }}
]

If no meaningful groups exist, return an empty array: []
"""


def group_tasks_into_activities(
    tasks: List[TaskRecord],
    client: Optional[GeminiClient] = None,
    video_file: Any = None,
    config: Optional[ForgeTraceConfig] = None,
) -> List[TaskGroup]:
    """Optionally group tasks into higher-level activities.

    If a :class:`GeminiClient` is provided, uses the model to identify
    logical groupings.  Otherwise returns an empty list.
    """
    if not tasks or len(tasks) < 2:
        return []

    if client is None or video_file is None:
        return _heuristic_grouping(tasks)

    try:
        return _model_based_grouping(tasks, client, video_file)
    except Exception as exc:
        logger.warning("Model-based grouping failed (%s); using heuristic", exc)
        return _heuristic_grouping(tasks)


def _model_based_grouping(
    tasks: List[TaskRecord],
    client: GeminiClient,
    video_file: Any,
) -> List[TaskGroup]:
    """Use Gemini to identify logical task groupings."""
    import json

    task_summary = [
        {
            "task_id": t.task_id,
            "task_name": t.task_name,
            "start_s": t.timeline.start_s,
            "end_s": t.timeline.end_s,
        }
        for t in tasks
    ]
    prompt = _GROUPING_PROMPT.format(
        task_list_json=json.dumps(task_summary, indent=2)
    )

    raw_text = client.analyze_video(video_file, prompt, json_mode=True)
    parsed = client.parse_json_response(raw_text)

    if not isinstance(parsed, list):
        return []

    valid_ids = {t.task_id for t in tasks}
    task_lookup = {t.task_id: t for t in tasks}
    groups: List[TaskGroup] = []

    for idx, entry in enumerate(parsed):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("group_name", f"Activity {idx + 1}"))
        task_ids = entry.get("task_ids", [])
        task_ids = [tid for tid in task_ids if tid in valid_ids]
        if len(task_ids) < 2:
            continue

        group_tasks = [task_lookup[tid] for tid in task_ids]
        groups.append(
            TaskGroup(
                group_id=f"G{idx + 1:03d}",
                group_name=name,
                task_ids=task_ids,
                description=str(entry.get("description", "")),
                start_s=min(t.timeline.start_s for t in group_tasks),
                end_s=max(t.timeline.end_s for t in group_tasks),
            )
        )

    logger.info("Model-based grouping: %d groups identified", len(groups))
    return groups


def _heuristic_grouping(tasks: List[TaskRecord]) -> List[TaskGroup]:
    """Simple proximity-based grouping as a fallback.

    Groups consecutive tasks that are close in time (gap < 3 s) into
    activities when there are at least 3 consecutive tasks.
    """
    if len(tasks) < 3:
        return []

    MAX_GAP = 3.0  # seconds between consecutive tasks
    MIN_GROUP_SIZE = 3

    groups: List[TaskGroup] = []
    current_group: List[TaskRecord] = [tasks[0]]

    for i in range(1, len(tasks)):
        gap = tasks[i].timeline.start_s - tasks[i - 1].timeline.end_s
        if gap <= MAX_GAP:
            current_group.append(tasks[i])
        else:
            if len(current_group) >= MIN_GROUP_SIZE:
                groups.append(_make_group(current_group, len(groups) + 1))
            current_group = [tasks[i]]

    if len(current_group) >= MIN_GROUP_SIZE:
        groups.append(_make_group(current_group, len(groups) + 1))

    if groups:
        logger.info("Heuristic grouping: %d groups identified", len(groups))
    return groups


def _make_group(tasks: List[TaskRecord], group_num: int) -> TaskGroup:
    verbs = list(dict.fromkeys(t.verb for t in tasks))
    desc = f"Sequence of {len(tasks)} actions: {', '.join(verbs)}"
    return TaskGroup(
        group_id=f"G{group_num:03d}",
        group_name=f"Activity {group_num}",
        task_ids=[t.task_id for t in tasks],
        description=desc,
        start_s=tasks[0].timeline.start_s,
        end_s=tasks[-1].timeline.end_s,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_task_name(action: RefinedAction) -> str:
    """Create a human-readable task name consistently from verb + object (+ tool if present)."""
    import re
    raw_verb = action.verb.strip()
    obj = action.object.strip() if action.object else "object"

    # Strip any " with <tool>" from verb
    clean_verb = raw_verb
    if " with " in clean_verb:
        clean_verb = clean_verb.split(" with ")[0].strip()

    # If the object was duplicated in the verb string (e.g. "align wood piece"), strip it
    if obj.lower() != "object" and obj.lower() in clean_verb.lower():
        pattern = re.compile(rf"\b{re.escape(obj)}\b", re.IGNORECASE)
        clean_verb = pattern.sub("", clean_verb).strip()

    if not clean_verb:
        clean_verb = raw_verb

    verb_cap = clean_verb.capitalize()

    # Determine tool (from action.tool or action.tool_hint or 'with' in raw_verb)
    tool = getattr(action, "tool", None)
    tool_hint = getattr(action, "tool_hint", None)
    if not tool and not tool_hint and " with " in raw_verb:
        tool_hint = raw_verb.split(" with ", 1)[1].strip()

    tool_str = tool or tool_hint
    if tool_str:
        clean_tool = re.sub(r"^(a|an|the|pneumatic)\s+", "", tool_str.strip(), flags=re.IGNORECASE)
        return f"{verb_cap} {obj} with {clean_tool}"

    return f"{verb_cap} {obj}"
