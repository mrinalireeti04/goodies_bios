"""SAM 2.1 tiny masks for the objects in hold contacts (portable objects only).

For every such contact: the object's box at the first frame of the contact is the prompt, the mask is tracked through the
contact plus sam2_tail_ms. The polygon (<= 40 points) is stored on the detection (d["polygon"]), used for fingertip
distances and written as mask_polygon_px. Longest contacts go first and the step stops after sam2_budget_s, so a clip
never takes more than about that long. A short temporary clip per contact is written to outputs/tmp and deleted.
Nothing here raises: any failure leaves the box-based result as it was.
"""
import gc
import logging
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("ps2")

MAX_POINTS = 40


def simplify(points: np.ndarray, max_points: int = MAX_POINTS) -> list[list[float]]:
    """Polygon with at most max_points vertices (Douglas-Peucker, tolerance raised until it fits)."""
    pts = np.asarray(points, np.float32).reshape(-1, 1, 2)
    eps = 0.5
    approx = pts
    while len(approx) > max_points:
        approx = cv2.approxPolyDP(pts, eps, True)
        eps *= 1.5
    return [[round(float(x), 1), round(float(y), 1)] for x, y in approx.reshape(-1, 2)]


def hold_targets(contacts: list[dict], dets_by_frame: list[list[dict]], n_frames: int, tail_frames: int) -> list[dict]:
    """Hold contacts of portable objects, longest first: {object_id, first (prompt frame), last, box, duration_ms}."""
    targets = []
    for c in contacts:
        if c["interaction"] != "hold" or not c["object_id"]:
            continue
        for i in range(c["_first"], min(c["_last"] + 1, n_frames)):
            det = next((d for d in dets_by_frame[i] if d["object_id"] == c["object_id"]), None)
            if det is not None:
                if det["type"] == "portable":
                    targets.append({"object_id": c["object_id"], "first": i, "last": min(c["_last"] + tail_frames, n_frames - 1),
                                    "box": det["bbox"], "duration_ms": c["end_ms"] - c["start_ms"]})
                break
    return sorted(targets, key=lambda t: -t["duration_ms"])


def run_masks(video_path: Path, targets: list[dict], dets_by_frame: list[list[dict]], cfg: dict, tmp_dir: Path,
              device: str) -> dict:
    """Fill d["polygon"] on detections. Returns stats {contacts, done, frames, seconds, peak_vram_gb, skipped_for_time}."""
    import torch
    stats = {"contacts": len(targets), "done": 0, "frames": 0, "seconds": 0.0, "peak_vram_gb": None, "skipped_for_time": 0}
    if not targets:
        return stats
    started = time.time()
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    try:
        from ultralytics.models.sam import SAM2VideoPredictor
        overrides = dict(model=cfg["sam2_model"], task="segment", mode="predict", imgsz=cfg["sam2_imgsz"], conf=0.25,
                         save=False, verbose=False, device=device, half=device != "cpu")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = min(cfg["sam2_max_side"] / max(w, h), 1.0)
        size = (int(w * scale), int(h * scale))
        for k, t in enumerate(targets):
            if time.time() - started > cfg["sam2_budget_s"]:
                stats["skipped_for_time"] = len(targets) - k
                break
            clip = tmp_dir / f"{video_path.stem}_sam_{k}.mp4"
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, t["first"])
                writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                for _ in range(t["first"], t["last"] + 1):
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    writer.write(cv2.resize(bgr, size))
                writer.release()
                predictor = SAM2VideoPredictor(overrides=overrides)
                box = [v * scale for v in t["box"]]
                results = predictor(source=str(clip), bboxes=[box], stream=False)
                for offset, res in enumerate(results):
                    if res.masks is None or not len(res.masks.xy) or len(res.masks.xy[0]) < 3:
                        continue
                    poly = simplify(res.masks.xy[0] / scale)
                    for d in dets_by_frame[t["first"] + offset]:
                        if d["object_id"] == t["object_id"]:
                            d["polygon"] = poly
                            stats["frames"] += 1
                stats["done"] += 1
                del predictor
            except Exception as e:   # one contact failing must not stop the others
                log.info("SAM 2: contact of %s skipped (%s: %s)", t["object_id"], type(e).__name__, str(e)[:100])
            finally:
                clip.unlink(missing_ok=True)
        cap.release()
    except Exception as e:
        log.info("SAM 2 skipped (%s: %s)", type(e).__name__, str(e)[:120])
    finally:
        gc.collect()
        if device != "cpu":
            stats["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            torch.cuda.empty_cache()
        stats["seconds"] = round(time.time() - started, 1)
    return stats
