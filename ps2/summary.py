"""tools[] (one row per physical object) and review_queue[] (what a human should check)."""
from collections import defaultdict


INFO_ONLY = {"label_from_vlm"}  # reason codes that describe how a label was chosen, not a problem to review


def object_rows(frames: list[dict]) -> dict[str, dict]:
    """object_id -> label, type, hazard, confidences and the frame indices where it was seen."""
    rows: dict[str, dict] = {}
    for fr in frames:
        for o in fr["objects"]:
            row = rows.setdefault(o["object_id"], {"label": o["label"], "type": o["type"], "hazard_frames": 0,
                                                   "confs": [], "idx": []})
            row["idx"].append(fr["frame_idx"])
            row["hazard_frames"] += bool(o.get("is_hazard", False))
            if o["confidence"] is not None:
                row["confs"].append(o["confidence"])
    return rows


def _appearances(idx: list[int], times: list[int], frame_ms: float, gap_ms: float) -> list[dict]:
    """Runs of frames; a gap longer than gap_ms starts a new appearance."""
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if (i - prev) * frame_ms > gap_ms:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return [{"start_ms": times[a], "end_ms": int(round(times[b] + frame_ms))} for a, b in runs]


def _union_ms(intervals: list[tuple[int, int]]) -> int:
    total, end = 0, None
    for s, e in sorted(intervals):
        if end is None or s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def merged_uses(holds: list[tuple[int, int, str]], gap_ms: float) -> list[dict]:
    """Hold intervals (start, end, hand_id) with gaps shorter than gap_ms merged into single uses."""
    uses: list[dict] = []
    for s, e, hand in sorted(holds):
        if uses and s - uses[-1]["end_ms"] < gap_ms:
            uses[-1]["end_ms"] = max(uses[-1]["end_ms"], e)
        else:
            uses.append({"start_ms": s, "end_ms": e, "hand_id": hand, "interaction": "hold"})
    return uses


def build_tools(frames: list[dict], contacts: list[dict], generic: set[str], frame_ms: float, cfg: dict,
                force_status: dict[str, str] | None = None, tool_info: dict[str, dict] | None = None) -> list[dict]:
    """One row per object that was visible for at least min_tool_ms or took part in a contact.

    `force_status` (object_id -> label_status) overrides the usual rule, for names decided by the VLM step.
    `tool_info` (object_id -> {tool_key, id_confidence, evidence}) comes from the fusion with the tool ledger; when it
    is given every row also gets tool_key, id_confidence and evidence.
    A tool is a hazard if its box was flagged as one in more than half of its frames.
    """
    times = [fr["t_ms"] for fr in frames]
    in_contact = {c["object_id"] for c in contacts if c["object_id"]}
    holds = defaultdict(list)
    for c in contacts:
        if c["interaction"] == "hold" and c["object_id"]:
            holds[c["object_id"]].append((c["start_ms"], c["end_ms"], c["hand_id"]))

    tools = []
    for oid, row in object_rows(frames).items():
        visible_ms = int(round(len(row["idx"]) * frame_ms))
        if visible_ms < cfg["min_tool_ms"] and oid not in in_contact:
            continue
        mean_conf = sum(row["confs"]) / len(row["confs"]) if row["confs"] else 0.0
        uses = merged_uses(holds[oid], cfg["use_merge_gap_ms"])
        held_ms = _union_ms([(s, e) for s, e, _ in holds[oid]])
        status = (force_status or {}).get(oid) or (
            "unidentified" if row["label"] in generic or mean_conf < cfg["label_min_conf"] else "identified")
        tools.append({
            "object_id": oid,
            "label": row["label"],
            "label_status": status,
            "type": row["type"],
            "is_hazard": row["hazard_frames"] > len(row["idx"]) / 2,
            "first_seen_ms": times[row["idx"][0]],
            "last_seen_ms": times[row["idx"][-1]],
            "visible_ms": visible_ms,
            "held_ms": held_ms,
            "use_count": len(uses),
            "appearances": _appearances(row["idx"], times, frame_ms, cfg["appearance_gap_ms"]),
            "idle_visible_ms": max(visible_ms - held_ms, 0),
            "uses": uses,
        })
        if tool_info is not None:
            info = tool_info.get(oid, {})
            tools[-1].update(tool_key=info.get("tool_key"), id_confidence=info.get("id_confidence", "low"),
                             evidence=list(info.get("evidence", [])))
    tools.sort(key=lambda t: int(t["object_id"].split("_")[1]))
    return tools


def long_unknown_items(frames: list[dict], frame_ms: float, cfg: dict) -> list[dict]:
    """Stretches where one hand's state stayed unknown for longer than long_unknown_ms."""
    items = []
    for hand_id in sorted({hs["hand_id"] for fr in frames for hs in fr["hand_states"]}):
        run = []  # (t_ms, object_id)

        def close():
            if run and run[-1][0] + frame_ms - run[0][0] > cfg["long_unknown_ms"]:
                objs = {o for _, o in run}
                items.append({"start_ms": run[0][0], "end_ms": int(round(run[-1][0] + frame_ms)), "hand_id": hand_id,
                              "object_id": objs.pop() if len(objs) == 1 else None, "reason": "long_unknown"})
            run.clear()

        for fr in frames:
            hs = next((h for h in fr["hand_states"] if h["hand_id"] == hand_id), None)
            if hs is not None and hs["state"] == "unknown":
                run.append((fr["t_ms"], hs["object_id"]))
            else:
                close()
        close()
    return items


def build_review_queue(tools: list[dict], contacts: list[dict], frames: list[dict], identity_items: list[dict],
                       suggestions: dict[str, str | None], frame_ms: float, cfg: dict,
                       extra_items: list[dict] | None = None) -> list[dict]:
    """Unidentified objects, contacts needing review, uncertain re-entry matches and long unknown stretches.

    Items about objects that are not in tools[] (too brief to matter) are left out. `extra_items` (from the fusion with
    the tool ledger) are added; an object that already has one is not also reported as a plain unidentified_object.
    """
    shown = {t["object_id"] for t in tools}
    extra = [i for i in (extra_items or []) if i["object_id"] is None or i["object_id"] in shown]
    explained = {i["object_id"] for i in extra if i["object_id"] and i["reason"] in ("tool_single_evidence", "tool_evidence_conflict")}
    contacted = {c["object_id"] for c in contacts if c["object_id"]}
    items = []
    for t in tools:
        if t["label_status"] == "unidentified" and t["object_id"] not in explained:
            items.append({"start_ms": t["first_seen_ms"], "end_ms": int(round(t["last_seen_ms"] + frame_ms)),
                          "hand_id": None, "object_id": t["object_id"], "reason": "unidentified_object",
                          "suggested_label": suggestions.get(t["object_id"])})
    for c in contacts:
        if c.get("review_needed"):
            items.append({"start_ms": c["start_ms"], "end_ms": c["end_ms"], "hand_id": c["hand_id"],
                          "object_id": c["object_id"], "reason": next((r for r in c.get("reasons", []) if r not in INFO_ONLY), "low_contact_confidence"),
                          "suggested_label": None})
    # an uncertain re-entry match only matters for objects that are in tools[] and took part in a contact
    items += [dict(i) for i in identity_items if i["object_id"] in shown and i["object_id"] in contacted]
    for i in long_unknown_items(frames, frame_ms, cfg):
        items.append({**i, "suggested_label": None})
    items += [dict(i) for i in extra]

    return number_items(items)


def number_items(items: list[dict]) -> list[dict]:
    """Sort by time and give item ids r_1, r_2, ... (in place; also returned)."""
    items.sort(key=lambda i: (i["start_ms"], i["reason"]))
    for n, item in enumerate(items, 1):
        item["item_id"] = f"r_{n}"
    items[:] = [{"item_id": i["item_id"], **{k: v for k, v in i.items() if k != "item_id"}} for i in items]
    return items
