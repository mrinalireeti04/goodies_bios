"""Overlay videos for checking and for the team (dev only).

Usage: python dev/render.py <video_path> [--variant ledger_sam] [--hands PATH] [--clip START END]
  no --clip : writes outputs/<clip>/highlight_<clip>.mp4 (the 60 s window with the most holds) and
              outputs/<clip>/weak_<clip>.mp4 (the 20 s window with the most unknown hand-frames)
  --clip    : writes outputs/<clip>/clip_<START>_<END>.mp4 (seconds)
All are 1280x720, H.264 if ffmpeg is installed. Reads outputs/<clip>/<variant>.json (default ledger_sam, else ps2) and the hand data
(default outputs/<clip>/ps1_stub.jsonl; the footer says "hands: stub" if the file name contains "stub", else "hands: PS1").

What is drawn: each hand's 21-point skeleton (one colour per hand_id); only objects within 2 hand-sizes of a hand or in a
contact (held object: thick box, plus its mask polygon if present); a badge at each wrist (NONE / TOUCH / HOLD + label,
or UNKNOWN + first reason code); a timeline strip per hand; a panel with the tools held now and the clip totals;
safety events in red marked EXPERIMENTAL (only if safety_enabled); a footer with clip id, models, vocab source,
hand source and processing fps.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ps2.hands import hand_scale, load_hands, tip_distances  # noqa: E402

W, H = 1280, 720
STRIP_ROW, FOOTER_H = 24, 24
NEAR_HAND_SIZES = 2.0          # objects farther than this from every hand are not drawn (unless in a contact)
HIGHLIGHT_S, WEAK_S = 60, 20
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
HAND_COLORS = [(255, 170, 0), (200, 0, 255), (0, 220, 120), (60, 80, 255)]
CLASS_COLORS = {"none": (110, 110, 110), "touch": (255, 110, 0), "hold": (0, 200, 0), "fixed": (0, 140, 255),
                "unknown": (0, 220, 255)}   # BGR: grey, blue, green, orange, yellow
RED = (0, 0, 255)


def color_for(object_id: str):
    n = int(object_id.split("_")[-1])
    return tuple(int(v) for v in np.random.default_rng(n).integers(60, 255, 3))


def text(img, s, org, color=(255, 255, 255), scale=0.5, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_skeleton(img, kp, scale, color, offset=(0, 0)) -> None:
    pts = [(int(x * scale) + offset[0], int(y * scale) + offset[1]) for x, y in kp]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 3, (255, 255, 255), -1, cv2.LINE_AA)


def active_contact(contacts: list[dict], hand_id: str, t_ms: float) -> dict | None:
    return next((c for c in contacts if c["hand_id"] == hand_id and c["start_ms"] <= t_ms < c["end_ms"]), None)


def hand_class(hs: dict | None, contact: dict | None) -> str:
    """none / touch / hold / fixed / unknown: the colour class of a hand in one frame."""
    if hs is None:
        return "none"
    if hs["state"] == "unknown":
        return "unknown"
    if hs["state"] in ("portable", "fixed") and contact:
        if contact["state"] == "fixed":
            return "fixed"
        return {"hold": "hold", "touch": "touch"}.get(contact["interaction"], "unknown")
    return "none"


def badge(hs: dict | None, contact: dict | None) -> tuple[str, str]:
    """(text, class) for the label next to a wrist."""
    cls = hand_class(hs, contact)
    if cls == "none":
        return "NONE", cls
    if cls == "unknown":
        reason = (hs or {}).get("reasons", [None])[0] or ("touch_or_hold" if contact else "unknown")
        return f"UNKNOWN {reason}", cls
    label = contact["object_label"] if contact else ""
    return f"{'TOUCH' if cls == 'touch' else 'HOLD' if cls == 'hold' else 'FIXED'} {label}", cls


def pick_windows(result: dict, fps: float) -> dict[str, tuple[float, float]]:
    """{'highlight': (start_s, end_s), 'weak': (start_s, end_s)} from the result alone."""
    frames, contacts = result["frames"], result["contacts"]
    n = len(frames)
    hold = np.zeros(n)
    unknown = np.zeros(n)
    for i, fr in enumerate(frames):
        hold[i] = any(c["interaction"] == "hold" and c["start_ms"] <= fr["t_ms"] < c["end_ms"] for c in contacts)
        unknown[i] = sum(hs["state"] == "unknown" for hs in fr["hand_states"])

    def best(series, seconds):
        size = min(int(seconds * fps), n)
        sums = np.convolve(series, np.ones(size), mode="valid")
        start = int(np.argmax(sums))
        return start / fps, (start + size) / fps

    return {"highlight": best(hold, HIGHLIGHT_S), "weak": best(unknown, WEAK_S)}


def render_clip(video: Path, result: dict, hands: dict, hands_kind: str, start_s: float, end_s: float,
                out_path: Path, safety_on: bool) -> None:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames, contacts, tools = result["frames"], result["contacts"], result.get("tools", [])
    meta = result.get("meta", {})
    hand_ids = sorted({hs["hand_id"] for f in frames for hs in f["hand_states"]})
    colors = {h: HAND_COLORS[i % len(HAND_COLORS)] for i, h in enumerate(hand_ids)}
    unidentified = {t["object_id"] for t in tools if t["label_status"] == "unidentified"}
    named_by_ai = {c["object_id"] for c in contacts if "label_from_vlm" in c.get("reasons", [])}
    n_identified = sum(t["label_status"] == "identified" for t in tools)
    totals = f"contacts {len(contacts)}  holds {sum(c['interaction'] == 'hold' for c in contacts)}  " \
             f"tools {n_identified} identified / {len(tools) - n_identified} unidentified"
    events = (result.get("safety_events") or []) if safety_on else []

    shown = sorted((x for x in tools if x["held_ms"] > 0 or x["use_count"] > 0), key=lambda x: -x["held_ms"])[:6]
    panel_tools = [f"{x['label'][:16]:<16} {'[?]' if x['label_status'] == 'unidentified' else 'ok '} held {x['held_ms'] / 1000:4.1f}s  uses {x['use_count']}"
                   for x in shown]
    strip_h = STRIP_ROW * max(len(hand_ids), 1) + 6
    area_h = H - strip_h - FOOTER_H
    scale = min(W / vw, area_h / vh)
    pic_w, pic_h = int(vw * scale), int(vh * scale)
    x_off = (W - pic_w) // 2
    i0, i1 = max(int(start_s * fps), 0), min(int(end_s * fps), len(frames))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    t0, t1 = frames[i0]["t_ms"], frames[i1 - 1]["t_ms"] + 1000 / fps
    footer = f"{result['clip_id']}  |  models: {', '.join(meta.get('models', []))}  |  vocab: {meta.get('vocab_source', '?')}" \
             f"  |  hands: {hands_kind}  |  {meta.get('processing_fps', '?')} fps"

    # timeline classes for the window, per hand and frame
    classes = {h: [] for h in hand_ids}
    for fr in frames[i0:i1]:
        states = {hs["hand_id"]: hs for hs in fr["hand_states"]}
        for h in hand_ids:
            classes[h].append(hand_class(states.get(h), active_contact(contacts, h, fr["t_ms"])))

    tmp = out_path.with_name(out_path.stem + ".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for fr in frames[i0:i1]:
        ok, bgr = cap.read()
        if not ok:
            break
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[:pic_h, x_off:x_off + pic_w] = cv2.resize(bgr, (pic_w, pic_h))
        view = canvas[:pic_h, x_off:x_off + pic_w]
        t = fr["t_ms"]
        states = {hs["hand_id"]: hs for hs in fr["hand_states"]}
        hands_now = hands.get(fr["frame_idx"], [])
        active = {h: active_contact(contacts, h, t) for h in hand_ids}
        held = {hs["object_id"] for hs in fr["hand_states"] if hs["state"] in ("portable", "fixed") and hs["object_id"]}
        in_contact = {c["object_id"] for c in active.values() if c}
        alarms = {e["hazard_object_id"] for e in events if e["start_ms"] <= t < e["end_ms"]}

        for o in fr["objects"]:
            oid = o["object_id"]
            near = any(min(tip_distances(h.kp, o["bbox_px"], hand_scale(h.kp))) <= NEAR_HAND_SIZES
                       for h in hands_now if hand_scale(h.kp) > 0)
            if not (near or oid in in_contact or oid in held or oid in alarms):
                continue
            x1, y1, x2, y2 = (int(v * scale) for v in o["bbox_px"])
            col = RED if oid in alarms else color_for(oid)
            cv2.rectangle(view, (x1, y1), (x2, y2), col, 4 if oid in held else 1)
            if o.get("mask_polygon_px") and oid in held:
                cv2.polylines(view, [np.array([(int(x * scale), int(y * scale)) for x, y in o["mask_polygon_px"]])], True, col, 2)
            label = f"{oid} {o['label']} {o['confidence']:.2f}" + (" [?]" if oid in unidentified else "") + \
                    (" (AI)" if oid in named_by_ai else "")
            text(view, label, (x1, max(y1 - 5, 14)), col, 0.45)
        for hand in hands_now:
            draw_skeleton(view, hand.kp, scale, colors.get(hand.hand_id, (255, 255, 255)))
            label, cls = badge(states.get(hand.hand_id), active.get(hand.hand_id))
            wx, wy = int(hand.kp[0][0] * scale), int(hand.kp[0][1] * scale)
            text(view, f"{hand.hand_id}: {label}", (max(wx - 40, 2), min(wy + 22, pic_h - 4)), CLASS_COLORS[cls], 0.55, 2)
        if alarms:
            text(canvas, "SAFETY - EXPERIMENTAL: hand near moving hazard", (x_off + 12, 26), RED, 0.7, 2)

        # panel: tools held now + clip totals
        held_now = sorted({c["object_label"] for c in active.values() if c and c["interaction"] == "hold"})
        panel_h = 96 + 18 * len(panel_tools)
        canvas[8:panel_h, W - 400:W - 8] = (canvas[8:panel_h, W - 400:W - 8] * 0.35).astype(np.uint8)
        text(canvas, "Held now: " + (", ".join(held_now[:3]) if held_now else "-"), (W - 392, 30), (255, 255, 255), 0.5)
        text(canvas, totals, (W - 392, 56), (200, 200, 200), 0.4)
        text(canvas, f"t = {t / 1000:6.1f} s", (W - 392, 80), (200, 200, 200), 0.45)
        for k, line in enumerate(panel_tools):   # tools of the clip: status, held time, uses
            text(canvas, line, (W - 392, 100 + 18 * k), (220, 220, 220), 0.4)

        # timeline strip (per hand) + cursor
        y0 = pic_h + 3
        for r, h in enumerate(hand_ids):
            y = y0 + r * STRIP_ROW
            text(canvas, h, (4, y + 15), colors[h], 0.4)
            cv2.rectangle(canvas, (90, y + 2), (W - 4, y + STRIP_ROW - 3), (45, 45, 45), -1)
            run_start = 0
            for j in range(1, len(classes[h]) + 1):
                if j == len(classes[h]) or classes[h][j] != classes[h][run_start]:
                    xa = 90 + int(run_start / len(classes[h]) * (W - 94))
                    xb = 90 + int(j / len(classes[h]) * (W - 94))
                    cv2.rectangle(canvas, (xa, y + 4), (max(xb, xa + 1), y + STRIP_ROW - 5), CLASS_COLORS[classes[h][run_start]], -1)
                    run_start = j
        cx = 90 + int((t - t0) / (t1 - t0) * (W - 94))
        cv2.line(canvas, (cx, y0), (cx, y0 + strip_h - 6), (255, 255, 255), 2)
        text(canvas, footer, (6, H - 8), (210, 210, 210), 0.45)
        writer.write(canvas)
    writer.release()
    if shutil.which("ffmpeg"):  # OpenCV's mp4v files are large; ffmpeg (if installed) makes a small H.264 copy
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-crf", "28",
                        "-preset", "fast", "-pix_fmt", "yuv420p", str(out_path)], check=True)
        tmp.unlink()
    else:
        tmp.replace(out_path)
        print("ffmpeg not found: the clip is not compressed")


def summarise(result: dict, name: str, window: tuple[float, float], path: Path) -> str:
    t0, t1 = window[0] * 1000, window[1] * 1000
    inside = [f for f in result["frames"] if t0 <= f["t_ms"] < t1]
    holds = [c for c in result["contacts"] if c["interaction"] == "hold" and c["end_ms"] > t0 and c["start_ms"] < t1]
    unknown = sum(hs["state"] == "unknown" for f in inside for hs in f["hand_states"])
    reasons: dict[str, int] = {}
    for f in inside:
        for hs in f["hand_states"]:
            if hs["state"] == "unknown":
                reason = (hs.get("reasons") or ["?"])[0]
                reasons[reason] = reasons.get(reason, 0) + 1
    top = ", ".join(f"{r} {n}" for r, n in sorted(reasons.items(), key=lambda kv: -kv[1])[:3]) or "-"
    return (f"  {name}: {window[0]:.0f}-{window[1]:.0f} s, {path.stat().st_size / 1e6:.1f} MB; {len(holds)} hold contacts, "
            f"{unknown} unknown hand-frames (top reasons: {top})")


def main(video_path: str, hands_path: str | None = None, clip: tuple[float, float] | None = None, variant: str | None = None) -> None:
    import yaml
    video = Path(video_path)
    out_dir = ROOT / "outputs" / video.stem
    name = variant or ("ledger_sam" if (out_dir / "ledger_sam.json").exists() else "ps2")
    result = json.loads((out_dir / f"{name}.json").read_text(encoding="utf-8"))
    cap = cv2.VideoCapture(str(video))
    fps, vw, vh = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    hands_file = Path(hands_path) if hands_path else out_dir / "ps1_stub.jsonl"
    hands = load_hands(hands_file, vw, vh)
    kind = "stub" if "stub" in hands_file.name.lower() else "PS1"
    safety_on = bool(yaml.safe_load((ROOT / "ps2" / "config" / "ps2.yaml").read_text(encoding="utf-8")).get("safety_enabled", True))
    clip_id = result["clip_id"]
    if clip:
        out = out_dir / f"clip_{clip[0]:g}_{clip[1]:g}.mp4"
        render_clip(video, result, hands, kind, clip[0], clip[1], out, safety_on)
        print(f"{clip_id}: wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return
    windows = pick_windows(result, fps)
    print(f"{clip_id}:")
    for name in ("highlight", "weak"):
        out = out_dir / f"{name}_{clip_id}.mp4"
        render_clip(video, result, hands, kind, *windows[name], out, safety_on)
        print(summarise(result, name, windows[name], out))


if __name__ == "__main__":
    args, hands_arg, clip_arg, variant_arg, rest = [], None, None, None, sys.argv[1:]
    while rest:
        a = rest.pop(0)
        if a == "--hands":
            hands_arg = rest.pop(0)
        elif a == "--variant":
            variant_arg = rest.pop(0)
        elif a == "--clip":
            clip_arg = (float(rest.pop(0)), float(rest.pop(0)))
        else:
            args.append(a)
    if len(args) != 1:
        sys.exit(__doc__)
    main(args[0], hands_arg, clip_arg, variant_arg)
