"""Fill short hand dropouts by interpolating between the frames before and after.

    python tools/fill_gaps.py out_rtm2/<clip>/hands.jsonl            # writes hands_filled.jsonl next to it
    python tools/fill_gaps.py out/<clip>/hands.jsonl --max-gap 3
    python tools/fill_gaps.py --evaluate out_rtm2/<clip>/hands.jsonl  # how good are the guesses?

A hand is filled only when ALL of these hold:
  * the same side (left or right) was detected before AND after the gap, and the gap is at most
    --max-gap processed frames long;
  * the hand did not move implausibly far, and did not change size a lot, across the gap;
  * neither end of the gap touches the edge of the frame (the hand may have left the view);
  * no hand already in the gap frame overlaps the guessed position (avoids duplicates).

Filled hands have source "interpolated", handedness_score null, and keypoints interpolated
linearly in time. The original file is never changed. Frames with no hand on either side of
a long gap stay empty: nothing in the data says whether the hand is out of view or just missed.
"""
import argparse
import copy
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

SIDES = ("left", "right")
MAX_GAP = 3            # processed frames (with --step 2 at 30 fps: 3 frames = 0.2 s)
SPEED_LIMIT = 1.0      # hand sizes the hand may move per processed frame step
SIZE_RATIO = (0.5, 2.0)
EDGE_MARGIN = 0.02     # fraction of the frame
DUP_IOU = 0.3


# ---------------------------------------------------------------- helpers
def bbox_of(kp):
    a = np.asarray(kp, float)
    return [float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def size_of(h):
    b = h["bbox_px"]
    return max(b[2] - b[0], b[3] - b[1], 1.0)


def centre(h):
    b = h["bbox_px"]
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def touches_edge(h, w, hgt, margin=EDGE_MARGIN):
    b = h["bbox_px"]
    return b[0] <= margin * w or b[1] <= margin * hgt or b[2] >= (1 - margin) * w or b[3] >= (1 - margin) * hgt


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def side_hand(rec, side):
    for h in rec["hands"]:
        if h["side"] == side:
            return h
    return None


# ----------------------------------------------------------- interpolation
def interpolate_hand(h0, h1, frac, side):
    """Linear interpolation of the 21 points. frac = 0 gives h0, 1 gives h1."""
    a, b = np.asarray(h0["kp2d_px"], float), np.asarray(h1["kp2d_px"], float)
    kp = (1 - frac) * a + frac * b
    kp3d = None
    if h0.get("kp3d_hand_m") is not None and h1.get("kp3d_hand_m") is not None:
        a3, b3 = np.asarray(h0["kp3d_hand_m"], float), np.asarray(h1["kp3d_hand_m"], float)
        kp3d = [[round(float(x), 5) for x in p] for p in ((1 - frac) * a3 + frac * b3)]
    kp_r = [[round(float(x), 2), round(float(y), 2)] for x, y in kp]
    return {
        "hand_id": "L" if side == "left" else "R",
        "side": side,
        "handedness_score": None,
        "bbox_px": [round(v, 2) for v in bbox_of(kp_r)],
        "kp2d_px": kp_r,
        "kp3d_hand_m": kp3d,
        "source": "interpolated",
    }


def plausible(h0, h1, steps, w, hgt, speed_limit=SPEED_LIMIT):
    if touches_edge(h0, w, hgt) or touches_edge(h1, w, hgt):
        return False
    s0, s1 = size_of(h0), size_of(h1)
    if not (SIZE_RATIO[0] <= s1 / s0 <= SIZE_RATIO[1]):
        return False
    dist = float(np.linalg.norm(centre(h0) - centre(h1)))
    return dist <= speed_limit * steps * (s0 + s1) / 2


def find_gaps(recs, side, max_gap):
    """Yield (i0, i1) where recs[i0] and recs[i1] have this side and everything between does not."""
    last = None
    for i, r in enumerate(recs):
        if side_hand(r, side) is not None:
            if last is not None and 1 < i - last <= max_gap + 1:
                yield last, i
            last = i


def fill(recs, max_gap=MAX_GAP, speed_limit=SPEED_LIMIT):
    """Return (new_records, stats). The input list is not modified."""
    out = copy.deepcopy(recs)
    stats = {"gaps_found": 0, "gaps_filled": 0, "frames_filled": 0, "hands_added": 0,
             "skipped_implausible": 0, "skipped_duplicate": 0, "frames_gained": 0}
    order = {"left": 0, "right": 1, "unknown": 2}
    for side in SIDES:
        for i0, i1 in find_gaps(recs, side, max_gap):
            stats["gaps_found"] += 1
            h0, h1 = side_hand(recs[i0], side), side_hand(recs[i1], side)
            w, hgt = recs[i0]["frame_width"], recs[i0]["frame_height"]
            if not plausible(h0, h1, i1 - i0, w, hgt, speed_limit):
                stats["skipped_implausible"] += 1
                continue
            t0, t1 = recs[i0]["frame_idx"], recs[i1]["frame_idx"]
            new = []
            for i in range(i0 + 1, i1):
                frac = (recs[i]["frame_idx"] - t0) / (t1 - t0)
                g = interpolate_hand(h0, h1, frac, side)
                if any(iou(g["bbox_px"], h["bbox_px"]) > DUP_IOU for h in out[i]["hands"]):
                    new = None
                    break
                new.append((i, g))
            if not new:
                stats["skipped_duplicate"] += 1
                continue
            stats["gaps_filled"] += 1
            for i, g in new:
                was_empty = not out[i]["hands"]
                out[i]["hands"].append(g)
                out[i]["hands"].sort(key=lambda h: order[h["side"]])
                stats["frames_filled"] += 1
                stats["hands_added"] += 1
                stats["frames_gained"] += was_empty
    return out, stats


# -------------------------------------------------------------- evaluation
def kp_error(h_true, kp_guess):
    a, b = np.asarray(h_true["kp2d_px"], float), np.asarray(kp_guess, float)
    return float(np.linalg.norm(a - b, axis=1).mean() / size_of(h_true))


def evaluate(recs, gaps=(1, 2, 3, 4), speed_limit=SPEED_LIMIT):
    """Hide detected hands, guess them from their neighbours, compare with the real detection.

    For each side and each gap length g: find places where the hand was detected on both
    sides of g consecutive frames AND in those g frames, pretend the middle g frames were
    missing, interpolate, and measure the keypoint error as a fraction of hand size.
    'hold' is the do-nothing baseline: copy the last detection forward.
    """
    res = {}
    for g in gaps:
        lin, hold, n_all, n_ok = [], [], 0, 0
        for side in SIDES:
            seq = [side_hand(r, side) for r in recs]
            for i0 in range(len(recs) - g - 1):
                i1 = i0 + g + 1
                mid = seq[i0 + 1:i1]
                if seq[i0] is None or seq[i1] is None or any(m is None for m in mid):
                    continue
                n_all += 1
                w, hgt = recs[i0]["frame_width"], recs[i0]["frame_height"]
                if not plausible(seq[i0], seq[i1], i1 - i0, w, hgt, speed_limit):
                    continue
                n_ok += 1
                t0, t1 = recs[i0]["frame_idx"], recs[i1]["frame_idx"]
                for k, m in enumerate(mid, start=1):
                    frac = (recs[i0 + k]["frame_idx"] - t0) / (t1 - t0)
                    guess = interpolate_hand(seq[i0], seq[i1], frac, side)
                    lin.append(kp_error(m, guess["kp2d_px"]))
                    hold.append(kp_error(m, seq[i0]["kp2d_px"]))
        res[g] = {"candidates": n_all, "passed_plausibility": n_ok,
                  "linear": summarize(lin), "hold_last": summarize(hold)}
    return res


def summarize(v):
    if not v:
        return None
    s = sorted(v)
    return {"n": len(s), "median": round(st.median(s), 3), "p90": round(s[min(len(s) - 1, int(0.9 * len(s)))], 3),
            "share_over_0.25": round(sum(x > 0.25 for x in s) / len(s), 3)}


def step_speeds(recs):
    """Distribution of how far real hands move between consecutive processed frames (in hand sizes)."""
    v = []
    for side in SIDES:
        prev = None
        for r in recs:
            h = side_hand(r, side)
            if h is not None and prev is not None:
                v.append(float(np.linalg.norm(centre(h) - centre(prev))) / ((size_of(h) + size_of(prev)) / 2))
            prev = h
    return summarize(v), (sorted(v)[int(0.99 * len(v))] if v else None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hands_jsonl")
    ap.add_argument("--max-gap", type=int, default=MAX_GAP)
    ap.add_argument("--speed-limit", type=float, default=SPEED_LIMIT)
    ap.add_argument("--out", help="output path (default: hands_filled.jsonl next to the input)")
    ap.add_argument("--evaluate", action="store_true", help="measure guess quality instead of writing a file")
    args = ap.parse_args(argv)

    recs = load(args.hands_jsonl)
    if args.evaluate:
        print(json.dumps({"step_speed": step_speeds(recs)[0], "p99_speed": step_speeds(recs)[1],
                          "by_gap_length": evaluate(recs, speed_limit=args.speed_limit)}, indent=2))
        return 0
    filled, stats = fill(recs, args.max_gap, args.speed_limit)
    dest = Path(args.out) if args.out else Path(args.hands_jsonl).with_name("hands_filled.jsonl")
    with open(dest, "w", encoding="utf-8") as f:
        for r in filled:
            f.write(json.dumps(r, allow_nan=False, separators=(",", ":")) + "\n")
    n = len(recs)
    before = sum(1 for r in recs if r["hands"])
    after = sum(1 for r in filled if r["hands"])
    print(json.dumps({"input": args.hands_jsonl, "output": str(dest), "frames": n,
                      "frames_with_hand_before": before, "frames_with_hand_after": after, **stats}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
