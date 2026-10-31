"""The Phase 1 contact rule, re-applied to a finished PS2 result, as the "before" in the before/after table.

Rule (brief): contact if >= min_tips fingertips are inside the object box padded by v0_pad_norm hand sizes; the object
with the most fingertips wins; an interval starts after on_frames and ends after off_frames. It runs on the SAME
detections and object ids as the final system, so the table compares the contact rule only (not the detector).

Usage: python dev/baseline_v0.py <clip_stem> [source_variant=base]   -> writes outputs/<clip>/v0.json (contacts only)
"""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ps2.hands import hand_scale, load_hands, tip_distances  # noqa: E402


def v0_contacts(result: dict, hands_by_frame: dict, cfg: dict) -> list[dict]:
    frames = result["frames"]
    frame_ms = frames[1]["t_ms"] - frames[0]["t_ms"] if len(frames) > 1 else 33
    picks: dict[str, list] = {}  # hand_id -> per frame (object, state) or None
    for hand_id in {h.hand_id for hs in hands_by_frame.values() for h in hs}:
        picks[hand_id] = [None] * len(frames)
    for i, fr in enumerate(frames):
        for hand in hands_by_frame.get(i, []):
            scale = hand_scale(hand.kp)
            best = None
            for o in fr["objects"]:
                if o["type"] not in ("portable", "fixed") or scale <= 0:
                    continue
                n = sum(d <= cfg["v0_pad_norm"] for d in tip_distances(hand.kp, o["bbox_px"], scale))
                area = (o["bbox_px"][2] - o["bbox_px"][0]) * (o["bbox_px"][3] - o["bbox_px"][1])
                if n >= cfg["min_tips"] and (best is None or (n, -area) > best[0]):
                    best = ((n, -area), o)
            picks[hand.hand_id][i] = None if best is None else (best[1]["object_id"], best[1]["label"], best[1]["type"])

    contacts = []
    for hand_id, seq in picks.items():
        cand, run, cur, miss = None, 0, None, 0
        for i, p in enumerate(seq + [None] * cfg["off_frames"]):
            oid = p[0] if p else None
            if cur is not None:
                if oid == cur["object_id"]:
                    cur["last"], miss = i, 0
                    continue
                miss += 1
                if miss < cfg["off_frames"]:
                    continue
                contacts.append(cur)
                cur, miss, cand, run = None, 0, None, 0
            if oid is not None and oid == cand:
                run += 1
            else:
                cand, run, start, first = oid, int(oid is not None), i, p
            if oid is not None and run >= cfg["on_frames"]:
                cur = {"hand_id": hand_id, "object_id": oid, "object_label": first[1], "state": first[2],
                       "interaction": "unknown", "first": start, "last": i, "confidence": None}
                cand, run, miss = None, 0, 0
        if cur is not None:
            contacts.append(cur)
    out = []
    for c in sorted(contacts, key=lambda c: (c["first"], c["hand_id"])):
        last = min(c["last"], len(frames) - 1)
        out.append({k: v for k, v in c.items() if k not in ("first", "last")}
                   | {"start_ms": frames[c["first"]]["t_ms"], "end_ms": int(frames[last]["t_ms"] + frame_ms)})
    return out


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    clip, source = sys.argv[1], (sys.argv[2] if len(sys.argv) == 3 else "base")
    out_dir = ROOT / "outputs" / clip
    result = json.loads((out_dir / f"{source}.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((ROOT / "ps2" / "config" / "ps2.yaml").read_text(encoding="utf-8"))
    ps1 = json.loads(next(iter((out_dir / "ps1_stub.jsonl").open(encoding="utf-8"))))
    hands = load_hands(out_dir / "ps1_stub.jsonl", ps1["frame_width"], ps1["frame_height"])
    contacts = v0_contacts(result, hands, cfg)
    (out_dir / "v0.json").write_text(json.dumps({"clip_id": result["clip_id"], "contacts": contacts}), encoding="utf-8")
    print(f"{len(contacts)} contacts -> {out_dir / 'v0.json'}")
