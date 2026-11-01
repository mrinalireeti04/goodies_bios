"""Hazard proximity warnings (experimental): a fingertip close to a hazard that is running.

Everything here is measured in the image plane, so depth is unknown ("2d_distance_only"). "Running" comes from two
sources: the tool ledger (Gemini saw the machine running during an interval: "hazard_running_from_ledger") and video
change inside the hazard's box ("hazard_motion_estimated"). Either one is enough; the event says which.
"""
from collections import Counter

from ps2.contact import track_key
from ps2.hands import hand_scale, tip_distances

BASE_REASON = "2d_distance_only"
LEDGER_REASON = "hazard_running_from_ledger"
MOTION_REASON = "hazard_motion_estimated"


def moving_scores(dets_by_frame: list[list[dict]], fps: float, cfg: dict, times: list[int] | None = None,
                  running_ms: dict[str, list[tuple[float, float]]] | None = None) -> list[dict[str, tuple[float, set[str]]]]:
    """Per frame: object_id -> (score, sources) for hazards judged moving.

    Local motion: aligned frame difference above hazard_motion_thresh in more than hazard_moving_frac of the frames the
    hazard was seen in within hazard_window_ms (seen at least twice). Ledger: the frame time lies inside a running
    interval of that object (running_ms: object_id -> [(start_ms, end_ms)]).
    """
    half = max(1, round(cfg["hazard_window_ms"] / 2 * fps / 1000))
    series: dict[str, dict[int, float]] = {}
    for i, dets in enumerate(dets_by_frame):
        for d in dets:
            if d["is_hazard"]:
                series.setdefault(d["object_id"], {})[i] = d["diff"]
    out: list[dict[str, tuple[float, set[str]]]] = [{} for _ in dets_by_frame]
    for oid, by_frame in series.items():
        for i in by_frame:
            sources, score = set(), 0.0
            window = [by_frame[j] for j in range(i - half, i + half + 1) if j in by_frame]
            share = sum(v > cfg["hazard_motion_thresh"] for v in window) / len(window)
            if len(window) >= 2 and share > cfg["hazard_moving_frac"]:
                sources.add(MOTION_REASON)
                score = share
            if running_ms and times and any(a <= times[i] < b for a, b in running_ms.get(oid, [])):
                sources.add(LEDGER_REASON)
                score = 1.0
            if sources:
                out[i][oid] = (score, sources)
    return out


def find_events(hands_by_frame: dict, dets_by_frame: list[list[dict]], times: list[int], fps: float, cfg: dict,
                running_ms: dict[str, list[tuple[float, float]]] | None = None) -> list[dict]:
    """Warning intervals: a fingertip within safety_near_norm hand sizes of a moving hazard for >= safety_min_ms."""
    frame_ms = 1000 / fps
    moving = moving_scores(dets_by_frame, fps, cfg, times, running_ms)
    near: dict[tuple[str, str], list[tuple]] = {}  # (hand track, hazard) -> (frame, dist, conf, score, sources, PS1 hand_id)
    for i, dets in enumerate(dets_by_frame):
        hazards = [d for d in dets if d["object_id"] in moving[i]]
        for hand in hands_by_frame.get(i, []):
            scale = hand_scale(hand.kp)
            if scale <= 0:
                continue
            for d in hazards:
                dist = min(tip_distances(hand.kp, d["bbox"], scale))
                if dist <= cfg["safety_near_norm"]:
                    score, sources = moving[i][d["object_id"]]
                    near.setdefault((track_key(hand), d["object_id"]), []).append((i, dist, d["conf"], score, sources, hand.hand_id))

    events = []
    for (_, oid), rows in near.items():
        run = [rows[0]]
        for row in rows[1:] + [None]:
            if row is not None and (row[0] - run[-1][0]) * frame_ms < cfg["merge_gap_ms"]:
                run.append(row)
                continue
            start, end = times[run[0][0]], int(round(times[run[-1][0]] + frame_ms))
            if end - start >= cfg["safety_min_ms"]:
                sources = set().union(*(r[4] for r in run))
                events.append({
                    "start_ms": start, "end_ms": end, "hand_id": Counter(r[5] for r in run).most_common(1)[0][0],
                    "hazard_object_id": oid, "min_distance_norm": round(min(r[1] for r in run), 3),
                    "hazard_moving": True,
                    "confidence": round(sum(r[2] for r in run) / len(run) * sum(r[3] for r in run) / len(run), 3),
                    "reasons": [BASE_REASON] + sorted(sources),
                })
            run = [row] if row is not None else []
    events.sort(key=lambda e: (e["start_ms"], e["hand_id"], e["hazard_object_id"]))
    for n, e in enumerate(events, 1):
        e["event_id"] = f"s_{n}"
    return [{"event_id": e["event_id"], **{k: v for k, v in e.items() if k != "event_id"}} for e in events]
