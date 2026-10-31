"""Hand-object contact: per-frame decision, touch vs hold, smoothing into intervals.

Contact comes from hand geometry (PS1 keypoints) and motion, never from the detector alone.
"""
import math
from collections import Counter

import numpy as np

from ps2.hands import FINGERTIPS, Hand, finger_curl, hand_scale, tip_distances

TYPE_RANK = {"portable": 0, "unknown": 1, "fixed": 2}  # portable preferred when candidates tie


def track_key(hand: Hand) -> str:
    """Contacts are decided per internal hand track (PS1 hand_ids are not persistent); without tracks, per hand_id."""
    return hand.track or hand.hand_id


def _area(box) -> float:
    return max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)


# ---------------------------------------------------------------- per frame, per hand

def observe(hand: Hand, dets: list[dict], cfg: dict) -> dict:
    """Which object (if any) is this hand on, using fingertip distances. No motion yet."""
    scale = hand_scale(hand.kp)
    obs = {"curl": finger_curl(hand.kp), "interpolated": hand.source == "interpolated",
           "obj": None, "n_tips": 0, "ambiguous": False}
    if scale <= 0:
        return obs
    cands = []
    for d in dets:
        n = sum(x <= cfg["tip_near_norm"] for x in tip_distances(hand.kp, d.get("polygon") or d["bbox"], scale))
        if n >= cfg["min_tips"]:
            cands.append((n, d))
    if not cands:
        return obs

    def rank(c):  # most fingertips, then portable before fixed, then the smaller box
        return -c[0], TYPE_RANK[c[1]["type"]], _area(c[1]["bbox"])

    cands.sort(key=rank)
    obs["n_tips"], obs["obj"] = cands[0][0], cands[0][1]
    if len(cands) > 1 and rank(cands[1])[:2] == rank(cands[0])[:2]:
        small, big = sorted((_area(cands[0][1]["bbox"]), _area(cands[1][1]["bbox"])))
        obs["ambiguous"] = big <= cfg["ambiguous_area_ratio"] * max(small, 1e-9)
    return obs


def interaction_from_motion(hand_move: float, hand_vec, obj_vec, curl: float, cfg: dict):
    """Touch or hold? Returns (interaction, object_moves_with_hand, score 0-1 of how clear the motion evidence is).

    hand_move: wrist movement over the window in hand sizes. hand_vec/obj_vec: camera-free movement over
    the window (obj_vec is None when the object was not tracked with the hand).
    """
    if hand_move >= cfg["min_hand_move_norm"]:
        if obj_vec is None:
            return "unknown", False, 0.0
        h = float(np.linalg.norm(hand_vec))
        if h == 0:
            return "unknown", False, 0.0
        differs = float(np.linalg.norm(obj_vec - hand_vec)) / h   # 0 = object moved exactly like the hand
        static = float(np.linalg.norm(obj_vec)) / h               # 0 = object did not move
        if differs <= cfg["comove_tol"]:
            return "hold", True, 1 - min(differs, 1)
        if static <= cfg["comove_tol"]:
            return "touch", False, 1 - min(static, 1)
        return "unknown", False, 0.0
    # hand (nearly) still: decide from the pose
    pose = curl_class(curl, cfg)
    return {"grip": "hold", "flat": "touch"}.get(pose, "unknown"), False, 0.5


def curl_class(curl: float, cfg: dict) -> str:
    if curl >= cfg["grip_curl_min"]:
        return "grip"
    return "flat" if curl <= cfg["flat_curl_max"] else "between"


def decide(obs: dict, hand_move: float, hand_vec, obj_vec, frame_area: float, cfg: dict) -> dict:
    """One hand in one frame -> state, object, confidence, reasons.

    kind: "contact" (can start/continue an interval), "weak" (object known but state unknown:
    can only continue an interval), "none".
    """
    reasons = ["hand_interpolated"] if obs["interpolated"] else []
    d = {"object_id": None, "label": None, "kind": "none", "state": "none",
         "interaction": "unknown", "confidence": None, "reasons": reasons}
    obj = obs["obj"]
    if obj is None:
        if curl_class(obs["curl"], cfg) == "grip":
            d.update(state="unknown", reasons=reasons + ["grasp_pose_no_object"])
        return d
    if obs["ambiguous"]:
        d.update(state="unknown", reasons=reasons + ["ambiguous_candidates"])
        return d

    interaction, comoving, score = interaction_from_motion(hand_move, hand_vec, obj_vec, obs["curl"], cfg)
    d.update(object_id=obj["object_id"], label=obj["label"], interaction=interaction)
    typ = obj["type"]
    if typ == "fixed" and comoving:
        typ, reasons = "portable", reasons + ["type_overridden_by_motion"]

    weak_reason = None
    if obj["conf"] < cfg["contact_min_obj_conf"]:
        weak_reason = "low_object_confidence"
    elif typ not in ("portable", "fixed"):
        weak_reason = "unidentified_object"
    elif typ == "fixed" and _area(obj["bbox"]) >= cfg["big_box_frac"] * frame_area \
            and hand_move >= cfg["min_hand_move_norm"] and not comoving:
        weak_reason = "fixed_contact_unverified"
    if weak_reason:
        d.update(kind="weak", state="unknown", reasons=reasons + [weak_reason])
        return d

    curl_support = obs["curl"] if interaction == "hold" else 1 - obs["curl"] if interaction == "touch" else 0.5
    conf = (cfg["w_tips"] * obs["n_tips"] / len(FINGERTIPS) + cfg["w_det"] * obj["conf"]
            + cfg["w_curl"] * curl_support + cfg["w_comotion"] * score)
    if obs["interpolated"]:
        conf *= cfg["interpolated_factor"]
    d.update(kind="contact", state=typ, confidence=round(min(conf, 1.0), 3), reasons=reasons)
    return d


def window_motion(aux: list[dict], i: int, hand_id: str, object_id: str | None, half: int):
    """Sum camera-free wrist and object movement over frames i-half..i+half.

    Returns (wrist movement over all frames, (wrist, object) movement over frames where both are known or None).
    """
    hand_all = np.zeros(2)
    pair_h, pair_o, n_pair = np.zeros(2), np.zeros(2), 0
    for j in range(max(0, i - half), min(len(aux), i + half + 1)):
        w = aux[j]["wrist_res"].get(hand_id)
        if w is None:
            continue
        hand_all += w
        o = aux[j]["obj_res"].get(object_id) if object_id else None
        if o is not None:
            pair_h, pair_o, n_pair = pair_h + w, pair_o + o, n_pair + 1
    return hand_all, ((pair_h, pair_o) if n_pair >= 2 else None)


def decide_all(hands_by_frame: dict, dets_by_frame: list[list[dict]], aux: list[dict],
               fps: float, frame_area: float, cfg: dict) -> list[dict]:
    """decisions[frame_idx][hand_id] for every frame."""
    half = max(1, round(cfg["comotion_window_ms"] / 2 * fps / 1000))
    decisions = []
    for i, dets in enumerate(dets_by_frame):
        frame = {}
        for hand in hands_by_frame.get(i, []):
            obs = observe(hand, dets, cfg)
            oid = obs["obj"]["object_id"] if obs["obj"] else None
            hand_all, pair = window_motion(aux, i, track_key(hand), oid, half)
            scale = hand_scale(hand.kp)
            hand_move = float(np.linalg.norm(hand_all)) / scale if scale > 0 else 0.0
            hand_vec, obj_vec = pair if pair else (hand_all, None)
            frame[track_key(hand)] = decide(obs, hand_move, hand_vec, obj_vec, frame_area, cfg)
        decisions.append(frame)
    return decisions


# ---------------------------------------------------------------- smoothing into intervals

def _spans(decisions, present, hand_id, max_gap: int, cfg) -> list[dict]:
    """Runs of frames on one object. Starts after on_frames contact frames; ends after off_frames without it
    (or max_gap frames if the hand itself vanished)."""
    spans, cand_obj, cand_start, cand_n, open_, miss = [], None, 0, 0, None, 0
    for i in range(len(decisions)):
        if hand_id in present[i]:
            d = decisions[i][hand_id]
            oid, kind = d["object_id"], d["kind"]
        else:
            oid, kind = None, "missing"
        if open_ is not None:
            if oid == open_["object"] and kind in ("contact", "weak"):
                open_["last"], miss = i, 0
                continue
            miss += 1
            if miss <= (max_gap if kind == "missing" else cfg["off_frames"] - 1):  # frames we may still skip
                continue
            spans.append(open_)
            open_, miss, cand_obj, cand_n = None, 0, None, 0
        if kind == "contact":
            if oid == cand_obj:
                cand_n += 1
            else:
                cand_obj, cand_start, cand_n = oid, i, 1
            if cand_n >= cfg["on_frames"]:
                open_, miss, cand_obj, cand_n = {"object": cand_obj, "first": cand_start, "last": i}, 0, None, 0
        else:
            cand_obj, cand_n = None, 0
    if open_ is not None:
        spans.append(open_)
    return spans


def _merge(spans: list[dict], times, frame_ms: float, merge_gap_ms: float) -> list[dict]:
    merged: list[dict] = []
    for sp in sorted(spans, key=lambda s: s["first"]):
        prev = merged[-1] if merged else None
        if prev and prev["object"] == sp["object"] and times[sp["first"]] - (times[prev["last"]] + frame_ms) < merge_gap_ms:
            prev["last"] = sp["last"]
        else:
            merged.append(dict(sp))
    return merged


def _finish(sp: dict, hand_id: str, decisions, present, times, frame_ms: float, cfg: dict, hand_info=None) -> dict:
    span = range(sp["first"], sp["last"] + 1)
    mine = [decisions[i][hand_id] for i in span if hand_id in present[i] and decisions[i][hand_id]["object_id"] == sp["object"]]
    contact_frames = [d for d in mine if d["kind"] == "contact"] or mine
    states = Counter(d["state"] for d in contact_frames if d["state"] in ("portable", "fixed"))
    interactions = Counter(d["interaction"] for d in contact_frames if d["interaction"] != "unknown")
    confs = [d["confidence"] for d in contact_frames if d["confidence"] is not None]
    n_missing = sum(hand_id not in present[i] for i in span)

    codes = Counter(r for d in mine for r in d["reasons"])
    codes["hand_not_detected"] += n_missing
    n = len(span)
    reasons = sorted(c for c, k in codes.items() if k / n > cfg["review_reason_frac"])
    frames_with_reasons = sum(bool(d["reasons"]) for d in mine) + n_missing
    confidence = round(sum(confs) / len(confs), 3) if confs else None
    low_confidence = confidence is None or confidence < cfg["review_threshold"]
    if low_confidence:
        reasons = sorted(set(reasons) | {"low_contact_confidence"})
    ps1_id, side = hand_id, "unknown"
    if hand_info is not None:   # the PS1 hand_id / side that appears most often in the interval
        seen = [hand_info[i][hand_id] for i in span if hand_id in present[i]]
        if seen:
            ps1_id = Counter(h for h, _ in seen).most_common(1)[0][0]
            side = Counter(s for _, s in seen).most_common(1)[0][0]
    return {
        "event_id": "",
        "start_ms": times[sp["first"]],
        "end_ms": int(round(times[sp["last"]] + frame_ms)),
        "hand_id": ps1_id,
        "object_id": sp["object"],
        "object_label": Counter(d["label"] for d in mine).most_common(1)[0][0],
        "state": states.most_common(1)[0][0],
        "interaction": interactions.most_common(1)[0][0] if interactions else "unknown",
        "confidence": confidence,
        "reasons": reasons,
        "review_needed": low_confidence or frames_with_reasons / n > cfg["review_reason_frac"],
        "_first": sp["first"],
        "_last": sp["last"],
        "_track": hand_id,
        "_side": side,
    }


def find_contacts(decisions, present, times, fps: float, cfg: dict, hand_info=None) -> list[dict]:
    """Contact intervals per hand track: smooth, merge short gaps, drop very short ones.

    hand_info[i][track] = (PS1 hand_id, side) lets contacts carry the PS1 hand_id seen most often in the interval."""
    frame_ms = 1000 / fps
    max_gap = max(1, math.ceil(cfg["max_hand_gap_ms"] / frame_ms))
    contacts = []
    for hand_id in sorted({h for p in present for h in p}):
        spans = _merge(_spans(decisions, present, hand_id, max_gap, cfg), times, frame_ms, cfg["merge_gap_ms"])
        for sp in spans:
            c = _finish(sp, hand_id, decisions, present, times, frame_ms, cfg, hand_info)
            if c["end_ms"] - c["start_ms"] >= cfg["min_interval_ms"]:
                contacts.append(c)
    contacts.sort(key=lambda c: (c["start_ms"], c["hand_id"]))
    for n, c in enumerate(contacts, 1):
        c["event_id"] = f"c_{n}"
    return contacts


# ---------------------------------------------------------------- write back to frames

def apply_to_frames(frames: list[dict], present, decisions, contacts: list[dict], hand_info=None) -> None:
    """Fill every frame's hand_states from the decisions and the final intervals.

    Each frame keeps that frame's own PS1 hand_id (via hand_info); decisions and intervals are per hand track."""
    inside = {}                                # (frame_idx, track) -> contact
    vanished = [dict() for _ in frames]        # tracks missing from a frame in the middle of a contact -> contact
    for c in contacts:
        track = c.get("_track", c["hand_id"])
        for i in range(c["_first"], c["_last"] + 1):
            inside[(i, track)] = c
            if track not in present[i]:
                vanished[i][track] = c

    for i, frame in enumerate(frames):
        states = []
        for hand_id in sorted(set(present[i]) | set(vanished[i])):
            c = inside.get((i, hand_id))
            if hand_id not in present[i]:  # hand vanished during a contact
                states.append({"hand_id": c["hand_id"], "state": "unknown", "object_id": c["object_id"],
                               "confidence": None, "reasons": ["hand_not_detected"]})
                continue
            d = decisions[i][hand_id]
            if c is None:
                state, oid, conf, reasons = d["state"], None, None, d["reasons"]
                if state == "unknown":
                    oid = d["object_id"]
            elif d["object_id"] == c["object_id"]:
                state = d["state"] if d["kind"] == "contact" else "unknown"
                oid, conf, reasons = c["object_id"], d["confidence"], d["reasons"]
            else:  # short bridged gap inside an interval
                state, oid, conf, reasons = c["state"], c["object_id"], c["confidence"], []
            shown_id = hand_info[i][hand_id][0] if hand_info is not None else hand_id
            hs = {"hand_id": shown_id, "state": state, "object_id": oid, "confidence": conf}
            if reasons:
                hs["reasons"] = list(reasons)
            states.append(hs)
        frame["hand_states"] = states
