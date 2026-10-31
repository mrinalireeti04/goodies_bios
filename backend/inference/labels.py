"""Clip-wide hand-side correction using short-gap spatial tracks.

MediaPipe handedness is a frame-level estimate. This post-pass preserves those
measurements in hands.raw.jsonl, then uses consistent track votes to repair
unknown or contradictory labels in the final result.
"""
import copy
import itertools
import math


def _wrist(hand):
    points = hand.get("kp2d_px") or []
    if len(points) == 21:
        return points[0]
    box = hand.get("bbox_px") or [0, 0, 0, 0]
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def correct_hand_sides(records, window=60):
    """Resolve labels over the complete clip, linking detections over <=window frames."""
    result = copy.deepcopy(records)
    tracks = []
    frame_diagonal = max((math.hypot(r.get("frame_width", 1), r.get("frame_height", 1)) for r in records), default=1)
    for frame_index, frame in enumerate(result):
        hands = frame.get("hands", [])
        active = [t for t in tracks if frame_index - t["last_frame"] <= window]
        # Enumerate assignments; there are at most two physical hands in PS1.
        options = [None] + active
        best = None
        for assignment in itertools.product(options, repeat=len(hands)):
            selected = [t for t in assignment if t is not None]
            if len({id(t) for t in selected}) != len(selected):
                continue
            costs = []
            valid = True
            for hand, track in zip(hands, assignment):
                if track is None:
                    costs.append(0.45)  # prefer a viable recent track over a new identity
                    continue
                current = _wrist(hand)
                prev = track["observations"][-1]
                predicted = prev["wrist"]
                if len(track["observations"]) > 1:
                    older = track["observations"][-2]
                    gap = max(1, prev["frame"] - older["frame"])
                    step = (frame_index - prev["frame"]) / gap
                    predicted = [prev["wrist"][d] + (prev["wrist"][d] - older["wrist"][d]) * step for d in (0, 1)]
                distance = math.hypot(current[0] - predicted[0], current[1] - predicted[1]) / frame_diagonal
                if distance > 0.34:
                    valid = False
                    break
                costs.append(distance)
            if valid:
                score = sum(costs)
                if best is None or score < best[0]:
                    best = (score, assignment)
        assignment = best[1] if best else (None,) * len(hands)
        for hand, track in zip(hands, assignment):
            if track is None:
                track = {"observations": [], "votes": {"left": 0.0, "right": 0.0}}
                tracks.append(track)
            track["observations"].append({"frame": frame_index, "wrist": list(_wrist(hand)), "hand": hand})
            track["last_frame"] = frame_index
            side = hand.get("side")
            confidence = hand.get("handedness_score")
            if side in ("left", "right"):
                track["votes"][side] += max(0.05, float(confidence) if isinstance(confidence, (int, float)) else 0.5)

    for track in tracks:
        left, right = track["votes"]["left"], track["votes"]["right"]
        total = left + right
        if total <= 0:
            continue
        winner = "left" if left > right else "right"
        margin = max(left, right) / max(min(left, right), 0.05)
        # Require repeated agreement or one very confident observation, and
        # a meaningful vote margin to avoid manufacturing certainty.
        winning_observations = sum(1 for o in track["observations"] if o["hand"].get("side") == winner)
        if margin < 1.4 or (winning_observations < 2 and max(left, right) < 0.9):
            continue
        for observation in track["observations"]:
            hand = observation["hand"]
            if hand.get("side") != winner:
                hand["side"] = winner
                hand["hand_id"] = "L" if winner == "left" else "R"
                hand["side_source"] = "clip_track_consensus"

    # Keep each anatomical side unique per frame; ambiguous collisions remain unknown.
    for frame in result:
        for side in ("left", "right"):
            same = [h for h in frame.get("hands", []) if h.get("side") == side]
            same.sort(key=lambda h: h.get("handedness_score") if isinstance(h.get("handedness_score"), (int, float)) else -1, reverse=True)
            for hand in same[1:]:
                hand["side"] = "unknown"
                hand["hand_id"] = "U1"
                hand["side_source"] = "ambiguous"
    return result
