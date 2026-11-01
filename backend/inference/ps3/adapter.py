"""BIOS job adapter for PS3 task timeline generation."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from concurrent.futures import Future

from backend.inference.ps3.config import load_config
from backend.inference.ps3.pipeline import process_video_to_timeline


def _read_hands(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            frame = json.loads(line)
            timestamp = frame.get("t_ms")
            if not isinstance(timestamp, (int, float)):
                continue
            for hand in frame.get("hands", []):
                side = hand.get("side")
                if side not in {"left", "right"}:
                    continue
                events.append({
                    "timestamp": timestamp / 1000.0,
                    "hand": side,
                    "hand_id": hand.get("hand_id"),
                })
    return events


def _source_clip_id(path: Path, fallback: str) -> str:
    """Use the PS1 source clip identity so separately produced results merge."""
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line).get("clip_id")
                if isinstance(value, str) and value:
                    return value
    return fallback


def _to_studio_actions(timeline, hands: list[dict[str, Any]]) -> dict[str, Any]:
    actions = []
    groups_by_task: dict[str, list[str]] = {}
    for group in timeline.groups:
        for task_id in group.task_ids:
            groups_by_task.setdefault(task_id, []).append(group.group_id)

    for task in timeline.tasks:
        start_ms = max(0, round(task.timeline.start_s * 1000))
        end_ms = min(round(timeline.video_duration * 1000), round(task.timeline.end_s * 1000))
        if end_ms <= start_ms:
            continue
        observed_ids = {
            str(event["hand_id"])
            for event in hands
            if task.timeline.start_s <= event["timestamp"] <= task.timeline.end_s
            and event.get("hand_id")
            and (task.hand == "both" or event["hand"] == task.hand)
        }
        actions.append({
            "action_id": task.task_id,
            "clip_id": timeline.clip_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "verb": task.verb,
            "object_label": None if task.object == "unknown" else task.object,
            "hand": task.hand if task.hand in {"left", "right", "both"} else "unknown",
            "hand_ids": sorted(observed_ids),
            "confidence": task.confidence,
            "source": "ps3",
            "task_name": task.task_name,
            "description": task.description,
            "evidence": task.evidence,
            "group_ids": groups_by_task.get(task.task_id, []),
        })

    return {
        "schema_version": "1.0",
        "clip_id": timeline.clip_id,
        "actions": actions,
        "groups": [group.model_dump() for group in timeline.groups],
        "unique_task_types": timeline.unique_task_types,
        "unique_task_list": timeline.unique_task_list,
        "processing_metadata": timeline.processing_metadata.model_dump(),
    }


def run_ps3(
    video_path: str | Path,
    hands_path: str | Path,
    contacts_path: str | Path | None,
    output_dir: str | Path,
    clip_id: str,
    *,
    contacts_provider=None,
    contacts_future: Future | None = None,
) -> Path:
    """Generate Studio's actions.json and a path-free PS3 timeline artifact."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("PS3 requires GEMINI_API_KEY; video was not sent to Gemini")

    config = load_config(gemini_api_key=api_key)
    hand_events = _read_hands(Path(hands_path))
    source_clip_id = _source_clip_id(Path(hands_path), clip_id)
    side_by_id = defaultdict(Counter)
    for event in hand_events:
        if event.get("hand_id"):
            side_by_id[str(event["hand_id"])][event["hand"]] += 1
    def resolve_contacts():
        if contacts_future is not None:
            contacts_future.result()
        if contacts_provider is not None:
            raw = contacts_provider()
        elif contacts_path and Path(contacts_path).is_file():
            raw = json.loads(Path(contacts_path).read_text(encoding="utf-8")).get("contacts", [])
        else:
            return []
        return [{
            "start_s": float(item.get("start_ms", 0)) / 1000,
            "end_s": float(item.get("end_ms", 0)) / 1000,
            "object": item.get("object_label") or "unknown",
            "hand": (side_by_id.get(str(item.get("hand_id")), Counter()).most_common(1)[0][0]
                     if side_by_id.get(str(item.get("hand_id"))) else "unknown"),
            "interaction": item.get("interaction", "contact"),
            "confidence": item.get("confidence") or 0.0,
        } for item in raw if isinstance(item, dict)]

    timeline = process_video_to_timeline(
        video_path=str(video_path),
        clip_id=source_clip_id,
        hand_tracks=hand_events,
        config=config,
        interaction_events_provider=resolve_contacts if contacts_future is not None or contacts_provider is not None else None,
    )

    result_dir = Path(output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    actions_path = result_dir / "actions.json"
    actions = _to_studio_actions(timeline, hand_events)
    temporary = actions_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(actions, allow_nan=False), encoding="utf-8")
    temporary.replace(actions_path)

    # Preserve the detailed PS3 output shape used by the sample, without its
    # machine-specific video_path field.
    detailed = timeline.model_dump(mode="json")
    detailed.pop("video_path", None)
    detailed["schema_version"] = "1.0"
    detailed_path = result_dir / "ps3_timeline.json"
    timeline_tmp = detailed_path.with_suffix(".json.tmp")
    timeline_tmp.write_text(json.dumps(detailed, allow_nan=False), encoding="utf-8")
    timeline_tmp.replace(detailed_path)
    return actions_path
