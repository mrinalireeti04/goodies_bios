"""Quality gate: how usable is a clip for hand-based annotation and training?

    python tools/quality_gate.py out_rtm2/<clip>/hands_final.jsonl
    python tools/quality_gate.py out_rtm2 out                      # every clip in these folders
    python tools/quality_gate.py out_rtm2 --video data             # also look for faces (privacy flag)

Reads the hands file only (no footage needed) and writes quality.json next to it.

For each processed frame:
  good    at least one hand, every hand is a real detection with a known side, a confident label,
          fully inside the image and not tiny
  check   a hand is there but something is weak (guessed/interpolated, unknown side, low label
          score, cut off by the image edge, tiny, or it jumped implausibly far since last frame)
  empty   no hand

Each 1-second window is then graded: empty (hand in under 20% of frames), good (60% or more of
frames good) or check. Outputs per clip:
  usable_time_share   share of 1-second windows graded good
  grade               A/B/C/D from usable_time_share (cut-offs below, tune them to taste)
  review_queue        windows a human should look at first, with the reasons
  privacy             with --video: seconds where a face may be visible (timestamps only, no images)

Nothing here proves a keypoint is on the right joint; that still needs hand-labelled frames.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

MIN_SCORE = 0.6          # MediaPipe label score below this = weak label
MIN_SIZE_PX = 40         # hand box shorter than this = tiny
EDGE_MARGIN = 0.01       # keypoints within this fraction of the border count as cut off
MAX_JUMP = 1.5           # hand sizes the hand may move between processed frames
WINDOW_MS = 1000
GRADES = ((0.70, "A"), (0.40, "B"), (0.15, "C"), (0.0, "D"))
REASONS = ("interpolated", "unknown_side", "low_score", "cut_off", "tiny", "jump")


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def size_of(h):
    b = h["bbox_px"]
    return max(b[2] - b[0], b[3] - b[1], 1.0)


def centre(h):
    b = h["bbox_px"]
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def hand_problems(h, rec, prev_same_side):
    """List of reasons this hand is weak (empty list = fine)."""
    w, hgt = rec["frame_width"], rec["frame_height"]
    out = []
    if h["source"] == "interpolated":
        out.append("interpolated")
    if h["side"] == "unknown":
        out.append("unknown_side")
    if h["handedness_score"] is not None and h["handedness_score"] < MIN_SCORE:
        out.append("low_score")
    kp = np.asarray(h["kp2d_px"], float)
    mx, my = EDGE_MARGIN * w, EDGE_MARGIN * hgt
    outside = ((kp[:, 0] < mx) | (kp[:, 0] > w - mx) | (kp[:, 1] < my) | (kp[:, 1] > hgt - my)).mean()
    if outside > 0.1:
        out.append("cut_off")
    if size_of(h) < MIN_SIZE_PX:
        out.append("tiny")
    if prev_same_side is not None:
        d = float(np.linalg.norm(centre(h) - centre(prev_same_side))) / ((size_of(h) + size_of(prev_same_side)) / 2)
        if d > MAX_JUMP:
            out.append("jump")
    return out


def classify_frames(recs):
    """-> list of (state, reasons) per record."""
    out, prev = [], {}
    for rec in recs:
        if not rec["hands"]:
            out.append(("empty", []))
            prev = {}
            continue
        reasons = []
        for h in rec["hands"]:
            reasons += hand_problems(h, rec, prev.get(h["side"]) if h["side"] != "unknown" else None)
        prev = {h["side"]: h for h in rec["hands"] if h["side"] != "unknown"}
        out.append(("check", sorted(set(reasons))) if reasons else ("good", []))
    return out


def grade_of(share):
    return next(g for cut, g in GRADES if share >= cut)


def analyse(recs):
    states = classify_frames(recs)
    windows = {}
    for rec, (state, reasons) in zip(recs, states):
        win = windows.setdefault(rec["t_ms"] // WINDOW_MS, {"n": 0, "hand": 0, "good": 0, "reasons": {}})
        win["n"] += 1
        win["hand"] += state != "empty"
        win["good"] += state == "good"
        for r in reasons:
            win["reasons"][r] = win["reasons"].get(r, 0) + 1

    wins, queue = [], []
    for sec in sorted(windows):
        w = windows[sec]
        if w["hand"] / w["n"] < 0.2:
            state = "empty"
        elif w["good"] / w["n"] >= 0.6:
            state = "good"
        else:
            state = "check"
        wins.append({"second": sec, "state": state, "frames": w["n"], "hand_frames": w["hand"],
                     "good_frames": w["good"]})
        if state == "check":
            top = sorted(w["reasons"], key=w["reasons"].get, reverse=True)
            queue.append({"start_ms": sec * WINDOW_MS, "end_ms": (sec + 1) * WINDOW_MS,
                          "hand_share": round(w["hand"] / w["n"], 2), "reasons": top[:3]})
    queue.sort(key=lambda q: (-q["hand_share"], q["start_ms"]))   # busiest unreliable seconds first

    n = len(recs)
    hand_frames = sum(s != "empty" for s, _ in states)
    hands = [h for r in recs for h in r["hands"]]
    usable = sum(w["state"] == "good" for w in wins) / len(wins) if wins else 0.0
    return {
        "clip_id": recs[0]["clip_id"] if recs else None,
        "frames": n,
        "seconds": len(wins),
        "usable_time_share": round(usable, 3),
        "grade": grade_of(usable),
        "hand_visible_share": round(hand_frames / n, 3) if n else 0.0,
        "good_frame_share": round(sum(s == "good" for s, _ in states) / n, 3) if n else 0.0,
        "interpolated_hand_share": round(sum(h["source"] == "interpolated" for h in hands) / len(hands), 3) if hands else 0.0,
        "unknown_side_hand_share": round(sum(h["side"] == "unknown" for h in hands) / len(hands), 3) if hands else 0.0,
        "windows": {"good": sum(w["state"] == "good" for w in wins),
                    "check": sum(w["state"] == "check" for w in wins),
                    "empty": sum(w["state"] == "empty" for w in wins)},
        "review_queue": queue[:50],
        "review_queue_total": len(queue),
        "privacy": {"checked": False},
    }


# ------------------------------------------------------------- face flag
def face_cascade():
    import cv2
    path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml" if hasattr(cv2, "data") else None
    if path is None or not path.exists():
        return None
    c = cv2.CascadeClassifier(str(path))
    return None if c.empty() else c


def face_seconds(video, cascade=None, every_s=1.0):
    """Seconds (sampled once per `every_s`) in which a frontal face may be visible. No pixels are kept."""
    import cv2
    cascade = cascade or face_cascade()
    if cascade is None:
        return None
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    hop, idx, found = max(1, int(round(fps * every_s))), 0, []
    try:
        while cap.grab():
            if idx % hop == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                small = cv2.resize(frame, (480, int(frame.shape[0] * 480 / frame.shape[1])))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, 1.1, 6, minSize=(30, 30))
                if len(faces):
                    found.append(int(idx / fps))
            idx += 1
    finally:
        cap.release()
    return sorted(set(found))


def add_privacy(report, video, cascade=None):
    secs = face_seconds(video, cascade)
    if secs is None:
        report["privacy"] = {"checked": False, "note": "face check unavailable (video or OpenCV face model missing)"}
    else:
        report["privacy"] = {"checked": True, "flagged_seconds": secs, "review_needed": bool(secs),
                             "note": "simple frontal-face check once per second; side views and small faces can be missed"}
    return report


# ------------------------------------------------------------------ CLI
PREFERENCE = ("hands_final.jsonl", "hands_sided.jsonl", "hands.jsonl")


def find_inputs(paths):
    found = []
    for p in map(Path, paths):
        if p.is_file():
            found.append(p)
            continue
        for clip in sorted(d for d in p.iterdir() if d.is_dir()):
            f = next((clip / n for n in PREFERENCE if (clip / n).exists()), None)
            if f:
                found.append(f)
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="hands .jsonl files or result folders (uses hands_final > hands_sided > hands)")
    ap.add_argument("--video", help="folder with the source .mp4 files, enables the face flag")
    args = ap.parse_args(argv)

    files = find_inputs(args.paths)
    if not files:
        print("no hands files found")
        return 1
    cascade = face_cascade() if args.video else None
    rows = []
    for f in files:
        recs = load(f)
        if not recs:
            continue
        rep = analyse(recs)
        if args.video:
            v = Path(args.video) / f"{rep['clip_id']}.mp4"
            add_privacy(rep, v, cascade) if v.exists() else rep["privacy"].update(note="source video not found")
        with open(f.with_name("quality.json"), "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=2, allow_nan=False)
        pr = rep["privacy"]
        face = "-" if not pr.get("checked") else (f"{len(pr['flagged_seconds'])} s" if pr["flagged_seconds"] else "none")
        rows.append((f"{f.parent.parent.name}/{rep['clip_id']}", rep, face))

    head = f"{'clip':<40}{'grade':>6}{'usable':>8}{'hand vis.':>11}{'good fr.':>10}{'to review':>11}{'faces':>8}"
    print(head)
    print("-" * len(head))
    for name, r, face in rows:
        print(f"{name:<40}{r['grade']:>6}{r['usable_time_share']:>8.0%}{r['hand_visible_share']:>11.0%}"
              f"{r['good_frame_share']:>10.0%}{r['review_queue_total']:>8} s{face:>8}")
    print("\nusable = share of 1-second windows graded good; to review = windows graded check; faces: - = not checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
