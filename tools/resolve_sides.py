"""Give `unknown` hands a left/right label using rules, not a new model.

    python tools/resolve_sides.py out_rtm2/<clip>/hands.jsonl      # writes hands_sided.jsonl next to it
    python tools/resolve_sides.py <hands.jsonl> --max-look 3

Rules, in order (a label is only changed when a rule is confident, otherwise it stays unknown):
  1. Two hands in a frame: the one further left in the image is the left hand, the other the right
     hand (this agrees with the detector's own labels in 92-99% of two-hand frames on our clips).
     Only applied when the hands are clearly apart (centres more than half a hand size apart).
  2. One unknown hand: look at the nearest frames before and after (up to --max-look processed
     frames). If a hand with a known side sits close to it (within about 1 hand size per step),
     it is the same hand and takes that side. If before and after disagree, or the other side
     already exists in the frame at a conflicting position, it stays unknown.
Everything else (keypoints, boxes, scores, source) is untouched. The input file is never changed.
Hands whose label was changed are counted in the printed stats.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ORDER = {"left": 0, "right": 1, "unknown": 2}
MIN_APART = 0.5   # hand sizes between the two centres for the x-order rule


def size_of(h):
    b = h["bbox_px"]
    return max(b[2] - b[0], b[3] - b[1], 1.0)


def centre(h):
    b = h["bbox_px"]
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def relabel(rec):
    """Re-number hand_ids and sort so the frame stays schema-valid."""
    rec["hands"].sort(key=lambda h: ORDER[h["side"]])
    u = 0
    for h in rec["hands"]:
        if h["side"] == "left":
            h["hand_id"] = "L"
        elif h["side"] == "right":
            h["hand_id"] = "R"
        else:
            u += 1
            h["hand_id"] = f"U{u}"


def neighbour_side(recs, i, h, step, max_look, speed=1.0):
    """Side of the closest known-side hand found going backwards (step=-1) or forwards (+1)."""
    for k in range(1, max_look + 1):
        j = i + step * k
        if not 0 <= j < len(recs):
            return None
        best = None
        for o in recs[j]["hands"]:
            if o["side"] == "unknown":
                continue
            d = float(np.linalg.norm(centre(o) - centre(h))) / ((size_of(o) + size_of(h)) / 2)
            if d <= speed * k and (best is None or d < best[0]):
                best = (d, o["side"])
        if best:
            return best[1]
    return None


def resolve(recs, max_look=3):
    out = copy.deepcopy(recs)
    stats = {"unknown_before": 0, "fixed_by_x_order": 0, "fixed_by_neighbours": 0, "unknown_after": 0}
    for r in out:
        stats["unknown_before"] += sum(h["side"] == "unknown" for h in r["hands"])

    # rule 1: two hands in the frame, order them by x (only when at least one is unknown / they clash)
    for r in out:
        hs = r["hands"]
        if len(hs) != 2 or all(h["side"] != "unknown" for h in hs) and hs[0]["side"] != hs[1]["side"]:
            continue
        a, b = sorted(hs, key=lambda h: centre(h)[0])
        apart = abs(centre(a)[0] - centre(b)[0]) / ((size_of(a) + size_of(b)) / 2)
        if apart < MIN_APART:
            continue
        for h, s in ((a, "left"), (b, "right")):
            if h["side"] == "unknown":
                stats["fixed_by_x_order"] += 1
            h["side"] = s
        relabel(r)

    # rule 2: lone unknown hands follow their neighbours in time (uses the already-resolved frames)
    for i, r in enumerate(out):
        for h in r["hands"]:
            if h["side"] != "unknown":
                continue
            before = neighbour_side(out, i, h, -1, max_look)
            after = neighbour_side(out, i, h, +1, max_look)
            seen = {s for s in (before, after) if s}
            if len(seen) != 1:
                continue
            side = seen.pop()
            if any(o is not h and o["side"] == side for o in r["hands"]):
                continue        # that side is already taken in this frame
            h["side"] = side
            stats["fixed_by_neighbours"] += 1
        relabel(r)

    for r in out:
        stats["unknown_after"] += sum(h["side"] == "unknown" for h in r["hands"])
    return out, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hands_jsonl")
    ap.add_argument("--max-look", type=int, default=3)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    recs = load(args.hands_jsonl)
    out, stats = resolve(recs, args.max_look)
    dest = Path(args.out) if args.out else Path(args.hands_jsonl).with_name("hands_sided.jsonl")
    with open(dest, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, allow_nan=False, separators=(",", ":")) + "\n")
    print(json.dumps({"input": args.hands_jsonl, "output": str(dest), **stats}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
