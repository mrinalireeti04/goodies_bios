"""Function 1: hand detection on first-person factory video.

Engines: MediaPipe (default, fast) or RTMPose (--engine rtmpose, higher accuracy).

Writes one PS1Frame record per processed frame to hands.jsonl.
All processing is local; no frames are uploaded or printed.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

SWAP_LR = False  # checked on the demo clip: MediaPipe's own labels are already correct for this
                 # head-cam footage (swapping put the "left" hand on the right in ~99% of frames).
                 # Use --swap to force swapping, --no-swap to force no swapping.
MODEL_PATH = "hand_landmarker.task"
PREPROCESS_WIDTH = 480
SCHEMA_VERSION = "1.0"

TOP_KEYS = ["schema_version", "clip_id", "frame_idx", "t_ms",
            "frame_width", "frame_height", "hands"]
HAND_KEYS = ["hand_id", "side", "handedness_score", "bbox_px",
             "kp2d_px", "kp3d_hand_m", "source"]

# BGR colours per side
COLORS = {"left": (0, 200, 0), "right": (0, 140, 255), "unknown": (150, 150, 150)}
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


# ---------- video input ----------
def video_info(path):
    """Return fps, frame count, width, height."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


# ---------- find frame ----------
def read_frames(path, step=2):
    """Yield (frame_idx, BGR image) for every step-th frame."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()


# ---------- preprocess ----------
def preprocess(img, width=PREPROCESS_WIDTH, clahe=False):
    """Resize to `width` (keep aspect); optional CLAHE on L channel."""
    h, w = img.shape[:2]
    if w != width:
        img = cv2.resize(img, (width, max(1, round(h * width / w))),
                         interpolation=cv2.INTER_AREA)
    if clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return img


# ---------- MediaPipe ----------
def make_landmarker(model_path=MODEL_PATH):
    """HandLandmarker in VIDEO mode, 2 hands, confidence 0.4."""
    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"model file not found: {model_path} (place hand_landmarker.task "
            "in the project folder)")
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return vision.HandLandmarker.create_from_options(opts)


# ---------- hand detect ----------
def _flip(side):
    return {"left": "right", "right": "left"}.get(side, side)


def detect_hands(landmarker, bgr, t_ms, orig_w, orig_h, swap_lr=SWAP_LR):
    """Run detection; return `hands` entries exactly as the schema defines."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect_for_video(mp_img, int(t_ms))

    raw = []
    for i, lms in enumerate(res.hand_landmarks):
        side, score = "unknown", None
        if i < len(res.handedness) and res.handedness[i]:
            cat = res.handedness[i][0]
            name = (cat.category_name or "").lower()
            if name in ("left", "right"):
                side = _flip(name) if swap_lr else name
            score = round(float(cat.score), 3)
        kp2d = [[round(p.x * orig_w, 2), round(p.y * orig_h, 2)] for p in lms]
        xs = [p[0] for p in kp2d]
        ys = [p[1] for p in kp2d]
        bbox = [round(min(xs), 2), round(min(ys), 2),
                round(max(xs), 2), round(max(ys), 2)]
        kp3d = None
        if i < len(res.hand_world_landmarks) and len(res.hand_world_landmarks[i]) == 21:
            kp3d = [[round(p.x, 5), round(p.y, 5), round(p.z, 5)]
                    for p in res.hand_world_landmarks[i]]
        raw.append({"side": side, "score": score, "bbox": bbox,
                    "kp2d": kp2d, "kp3d": kp3d})

    # duplicate side: higher score keeps it, other becomes unknown
    for side in ("left", "right"):
        same = [h for h in raw if h["side"] == side]
        if len(same) > 1:
            same.sort(key=lambda h: h["score"] if h["score"] is not None else -1,
                      reverse=True)
            for h in same[1:]:
                h["side"] = "unknown"

    order = {"left": 0, "right": 1, "unknown": 2}
    raw.sort(key=lambda h: order[h["side"]])  # stable

    hands, u = [], 0
    for h in raw:
        if h["side"] == "left":
            hid = "L"
        elif h["side"] == "right":
            hid = "R"
        else:
            u += 1
            hid = f"U{u}"
        hands.append({
            "hand_id": hid,
            "side": h["side"],
            "handedness_score": h["score"],
            "bbox_px": h["bbox"],
            "kp2d_px": h["kp2d"],
            "kp3d_hand_m": h["kp3d"],
            "source": "detected",
        })
    return hands


# ---------- overlay ----------
def draw_overlay(bgr, hands):
    """Draw landmarks, connections and labels on a copy of the frame."""
    out = bgr.copy()
    for h in hands:
        color = COLORS[h["side"]]
        pts = [(int(round(x)), int(round(y))) for x, y in h["kp2d_px"]]
        bx1, by1, bx2, by2 = [int(round(v)) for v in h["bbox_px"]]
        pad = 12  # the box is the min/max of the 21 points, so pad it to enclose the hand
        cv2.rectangle(out, (bx1 - pad, by1 - pad), (bx2 + pad, by2 + pad), color, 3, cv2.LINE_AA)
        for a, b in CONNECTIONS:
            cv2.line(out, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(out, p, 4, color, -1, cv2.LINE_AA)
        score = h["handedness_score"]
        label = h["side"] + (f" {score:.2f}" if score is not None else "")
        x1, y1 = int(h["bbox_px"][0]) - pad, int(h["bbox_px"][1]) - pad
        cv2.putText(out, label, (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


# ---------- per frame ----------
def process_frame(landmarker, frame, frame_idx, fps, clip_id,
                  width=PREPROCESS_WIDTH, clahe=False, swap_lr=SWAP_LR, engine=None):
    """Chain preprocess + detection; return one PS1Frame record.

    With an RTMPose `engine`, MediaPipe still supplies proposals and the
    left/right label; the 21 points come from RTMPose on the full-res frame.
    """
    h, w = frame.shape[:2]
    t_ms = round(frame_idx * 1000 / fps)
    small = preprocess(frame, width, clahe)
    hands = detect_hands(landmarker, small, t_ms, w, h, swap_lr)
    if engine is not None:
        import rtm_hand
        hands = rtm_hand.to_schema_hands(engine.estimate(frame, hands))
    return {
        "schema_version": SCHEMA_VERSION,
        "clip_id": clip_id,
        "frame_idx": int(frame_idx),
        "t_ms": int(t_ms),
        "frame_width": int(w),
        "frame_height": int(h),
        "hands": hands,
    }


# ---------- check ----------
def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _check_points(pts, n_dims, name):
    if not isinstance(pts, list) or len(pts) != 21:
        raise ValueError(f"{name} must be a list of exactly 21 points")
    for p in pts:
        if not isinstance(p, list) or len(p) != n_dims or not all(_is_num(v) for v in p):
            raise ValueError(f"{name} points must be {n_dims} finite numbers")


def validate_record(rec):
    """Check one record against the PS1Frame schema; raise ValueError if bad."""
    if not isinstance(rec, dict) or list(rec.keys()) != TOP_KEYS:
        raise ValueError(f"top-level keys must be exactly {TOP_KEYS} in order")
    if rec["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema_version must be '1.0'")
    if not isinstance(rec["clip_id"], str) or not rec["clip_id"]:
        raise ValueError("clip_id must be a non-empty string")
    for k in ("frame_idx", "t_ms", "frame_width", "frame_height"):
        if not _is_int(rec[k]):
            raise ValueError(f"{k} must be an integer")
    if rec["frame_width"] <= 0 or rec["frame_height"] <= 0:
        raise ValueError("frame_width and frame_height must be positive")
    if not isinstance(rec["hands"], list):
        raise ValueError("hands must be a list")
    ids = set()
    for h in rec["hands"]:
        if not isinstance(h, dict) or list(h.keys()) != HAND_KEYS:
            raise ValueError(f"hand keys must be exactly {HAND_KEYS} in order")
        if h["side"] not in ("left", "right", "unknown"):
            raise ValueError("side must be left, right or unknown")
        if h["source"] not in ("detected", "interpolated"):
            raise ValueError("source must be detected or interpolated")
        s = h["handedness_score"]
        if s is not None and not (_is_num(s) and 0 <= s <= 1):
            raise ValueError("handedness_score must be a number in 0..1 or null")
        b = h["bbox_px"]
        if not isinstance(b, list) or len(b) != 4 or not all(_is_num(v) for v in b):
            raise ValueError("bbox_px must have 4 finite numbers")
        if b[0] > b[2] or b[1] > b[3]:
            raise ValueError("bbox_px must satisfy x1<=x2 and y1<=y2")
        _check_points(h["kp2d_px"], 2, "kp2d_px")
        if h["kp3d_hand_m"] is not None:
            _check_points(h["kp3d_hand_m"], 3, "kp3d_hand_m")
        if not isinstance(h["hand_id"], str) or h["hand_id"] in ids:
            raise ValueError("hand_id must be a string unique within the frame")
        ids.add(h["hand_id"])


def validate_file(path):
    """Validate every line of a hands.jsonl; return (lines_checked, error)."""
    prev_idx = prev_t = None
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            try:
                rec = json.loads(line)
                validate_record(rec)
                if prev_idx is not None and (rec["frame_idx"] <= prev_idx
                                             or rec["t_ms"] <= prev_t):
                    raise ValueError("frame_idx and t_ms must strictly increase")
                prev_idx, prev_t = rec["frame_idx"], rec["t_ms"]
            except (ValueError, json.JSONDecodeError) as e:
                return n, f"line {n}: {e}"
    return n, None


# ---------- per clip ----------
def run_clip(video, out_root="out", step=2, save_video=True,
             clahe=False, swap_lr=SWAP_LR, engine_name="mediapipe", engine=None,
             rtm_min_score=None):
    """Process one clip, write outputs, return a summary dict."""
    video = Path(video)
    clip_id = video.stem
    info = video_info(video)
    fps = info["fps"]
    out_dir = Path(out_root) / clip_id
    out_dir.mkdir(parents=True, exist_ok=True)

    landmarker = make_landmarker(MODEL_PATH)
    if engine is None and engine_name == "rtmpose":
        import rtm_hand
        engine = (rtm_hand.RTMHandEngine() if rtm_min_score is None
                  else rtm_hand.RTMHandEngine(min_kp_score=rtm_min_score))
    elif engine is not None:
        engine_name = "rtmpose"
    if engine is not None:
        engine.prev_boxes = []
        engine.prev_hands = []
    writer = None
    processed = with_hands = 0
    score_sum, score_n = 0.0, 0
    start = time.time()
    try:
        with open(out_dir / "hands.jsonl", "w", encoding="utf-8") as f:
            for idx, frame in read_frames(video, step):
                rec = process_frame(landmarker, frame, idx, fps, clip_id,
                                    PREPROCESS_WIDTH, clahe, swap_lr, engine)
                validate_record(rec)
                f.write(json.dumps(rec, allow_nan=False, separators=(",", ":")) + "\n")

                processed += 1
                if rec["hands"]:
                    with_hands += 1
                for h in rec["hands"]:
                    if h["handedness_score"] is not None:
                        score_sum += h["handedness_score"]
                        score_n += 1

                need_overlay = processed == 1 or save_video
                if need_overlay:
                    ov = draw_overlay(frame, rec["hands"])
                if processed == 1:
                    cv2.imwrite(str(out_dir / "first_frame.png"), frame)
                    cv2.imwrite(str(out_dir / "first_overlay.png"), ov)
                if save_video:
                    if writer is None:
                        h_, w_ = frame.shape[:2]
                        writer = cv2.VideoWriter(
                            str(out_dir / "overlay.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps / step, (w_, h_))
                    writer.write(ov)
    finally:
        landmarker.close()
        if writer is not None:
            writer.release()

    elapsed = max(time.time() - start, 1e-9)
    summary = {
        "clip_id": clip_id,
        "frames_processed": processed,
        "frames_with_hands": with_hands,
        "detection_rate": round(with_hands / processed, 4) if processed else 0.0,
        "mean_handedness_score_when_detected":
            round(score_sum / score_n, 3) if score_n else None,
        "processing_fps": round(processed / elapsed, 2),
        "run_config": {"step": step, "fps": fps, "swap_lr": swap_lr,
                       "clahe": clahe, "preprocess_width": PREPROCESS_WIDTH,
                       "engine": engine_name},
    }
    if engine is not None:
        summary["run_config"]["rtm_min_score"] = engine.min_kp_score
        summary["rtmpose_stats"] = dict(engine.stats)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    return summary


# ---------- CLI ----------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Hand detection on factory video")
    ap.add_argument("path", nargs="?", help="video file or folder")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--out", default="out")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--swap", action="store_true", help="swap MediaPipe's left/right labels")
    ap.add_argument("--no-swap", action="store_true", help="do not swap left/right labels")
    ap.add_argument("--engine", choices=["mediapipe", "rtmpose"], default="mediapipe",
                    help="rtmpose = higher accuracy, slower (needs rtmlib + onnxruntime)")
    ap.add_argument("--rtm-min-score", type=float, default=None,
                    help="rtmpose: drop hands whose mean keypoint score is below this "
                         "(default 0.2)")
    ap.add_argument("--validate", metavar="HANDS_JSONL")
    args = ap.parse_args(argv)

    if args.validate:
        n, err = validate_file(args.validate)
        print(f"{n} lines checked")
        print(err if err else "OK")
        return 1 if err else 0

    if not args.path:
        ap.error("provide a video file/folder or --validate")
    if args.step < 1:
        ap.error("--step must be >= 1")

    p = Path(args.path)
    clips = sorted(p.glob("*.mp4")) if p.is_dir() else [p]
    if not clips:
        print("no .mp4 files found")
        return 1
    for clip in clips:
        s = run_clip(clip, args.out, args.step, not args.no_video,
                     args.clahe, (SWAP_LR or args.swap) and not args.no_swap,
                     args.engine, rtm_min_score=args.rtm_min_score)
        print(json.dumps(s, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())