"""Compare out (MediaPipe), out_rtm (old RTMPose) and out_rtm2 (new RTMPose). Read-only."""
import json
import sys
from pathlib import Path

CLIPS = ["MTL-SKL00089-011", "mytronlabs-000060-014", "MTL-SGF00106-029"]
SETS = {"MP": "out", "RTM-old": "out_rtm", "RTM-new": "out_rtm2"}


def load(folder, clip):
    rows = [json.loads(l) for l in open(Path(folder) / clip / "hands.jsonl", encoding="utf-8")]
    summ = json.load(open(Path(folder) / clip / "summary.json", encoding="utf-8"))
    return rows, summ


def stats(rows):
    with_hands = {r["frame_idx"] for r in rows if r["hands"]}
    unknown = sum(1 for r in rows for h in r["hands"] if h["side"] == "unknown")
    two = ordered = 0
    for r in rows:
        sides = {h["side"]: h for h in r["hands"]}
        if "left" in sides and "right" in sides and len(r["hands"]) == 2:
            two += 1
            lx = sum(sides["left"]["bbox_px"][::2]) / 2
            rx = sum(sides["right"]["bbox_px"][::2]) / 2
            ordered += lx < rx
    gt2 = sum(1 for r in rows if len(r["hands"]) > 2)
    return with_hands, unknown, two, ordered, gt2


for clip in CLIPS:
    print(f"\n=== {clip}")
    data = {k: load(v, clip) for k, v in SETS.items()}
    mp_frames = stats(data["MP"][0])[0]
    for k, (rows, summ) in data.items():
        wh, unk, two, ordered, gt2 = stats(rows)
        lost = len(mp_frames - wh)
        share = f"{ordered}/{two} ({100 * ordered / two:.1f}%)" if two else "n/a"
        extra = summ.get("rtmpose_stats", {})
        print(f"{k:8} frames_with_hand={len(wh):5} rate={summ['detection_rate']:.4f} "
              f"MP_frames_lost={lost:5} unknown_hands={unk:5} "
              f"two_hand_left_of_right={share:18} >2hands_frames={gt2} "
              f"fps={summ['processing_fps']:6} stats={extra}")
