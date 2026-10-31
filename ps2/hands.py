"""PS1 hand data: loading, rescaling, and geometry helpers (fingertip distance, hand size, finger curl)."""
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("ps2")

FINGERTIPS = (4, 8, 12, 16, 20)
# (mcp, pip, dip) joints of index, middle, ring, pinky; the thumb has no comparable middle joint
FINGER_JOINTS = ((5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19))
FULL_CURL_DEG = 90.0  # a bend of this many degrees at the middle joint counts as fully curled


@dataclass
class Hand:
    hand_id: str
    kp: np.ndarray    # (21, 2) video pixels
    bbox: np.ndarray  # (4,) x1, y1, x2, y2 video pixels
    source: str       # "detected" | "interpolated"
    side: str = "unknown"   # PS1 side: left | right | unknown
    track: str = ""         # internal hand track (set by assign_tracks); PS1 hand_ids are not persistent


def read_ps1(hands) -> list[dict]:
    """Accept a .jsonl path, a .json path, or a list of PS1Frame dicts; return the list."""
    if isinstance(hands, (str, Path)):
        path = Path(hands)
        if not path.is_file():
            raise ValueError(f"PS1 hands file not found: {path}")
        text = path.read_text(encoding="utf-8")
        try:
            if path.suffix.lower() == ".jsonl":
                frames = [json.loads(line) for line in text.splitlines() if line.strip()]
            else:
                frames = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"PS1 hands file is not valid JSON: {path} ({e})") from e
    elif isinstance(hands, list):
        frames = hands
    else:
        raise ValueError("hands must be a .jsonl/.json path or a list of PS1Frame dicts")
    if not isinstance(frames, list) or not all(isinstance(f, dict) for f in frames):
        raise ValueError("PS1 data must be a list of PS1Frame objects")
    return frames


MAX_HAND_IDS = 4        # more distinct hand_ids than this suggests ids are not stable (or other people's hands)
EDGE_MARGIN = 0.10      # hand landmarks may lie a little outside the frame; beyond 10% of its size is a violation
ORDER_WARN_FRAC = 0.30  # one foreshortened hand can fail the order test; warn only if this share of hands does


def check_ps1(frames: list[dict], video_w: int, video_h: int, n_frames: int, fps: float) -> dict:
    """Check the assumptions in INTEGRATION.md ("Assumptions about PS1 data") that can be checked. Never raises:
    each violation is logged as a warning and counted in the returned dict (written to meta.ps1_checks)."""
    frame_ms = 1000 / fps
    seen: set[int] = set()
    beyond = bad_time = size_mismatch = out_of_bounds = order_suspect = 0
    hand_ids: set[str] = set()
    n_hands = 0
    for f in frames:
        try:
            idx = int(f["frame_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        if idx < 0 or idx >= n_frames:
            beyond += 1
            continue
        seen.add(idx)
        if "t_ms" in f and abs(float(f["t_ms"]) - round(idx * 1000 / fps)) > frame_ms:
            bad_time += 1
        w, h = f.get("frame_width", video_w), f.get("frame_height", video_h)
        if (w, h) != (video_w, video_h):
            size_mismatch += 1
        for hand in f.get("hands", []):
            hand_ids.add(str(hand.get("hand_id")))
            n_hands += 1
            try:
                kp = np.asarray(hand["kp2d_px"], dtype=float)
                if kp.shape != (21, 2):
                    continue
                mx, my = EDGE_MARGIN * w, EDGE_MARGIN * h
                if (kp[:, 0] < -mx).any() or (kp[:, 0] > w + mx).any() or (kp[:, 1] < -my).any() or (kp[:, 1] > h + my).any():
                    out_of_bounds += 1
                if np.linalg.norm(kp[12] - kp[0]) <= np.linalg.norm(kp[9] - kp[0]):  # middle fingertip nearer than its knuckle
                    order_suspect += 1
            except (KeyError, TypeError, ValueError):
                continue
    missing = max(n_frames - len(seen), 0)
    problems = {
        "missing_frames": (missing, f"{missing} of {n_frames} video frames have no PS1 record (treated as 'no hand')"),
        "t_ms_mismatch": (bad_time, f"{bad_time} records have t_ms more than one frame away from round(frame_idx*1000/fps)"),
        "frame_idx_beyond_video": (beyond, f"{beyond} records have a frame_idx outside the video (0..{n_frames - 1}); ignored"),
        "coords_out_of_bounds": (out_of_bounds, f"{out_of_bounds} hands have keypoints more than {EDGE_MARGIN:.0%} outside the frame"),
        "frame_size_mismatch": (size_mismatch, f"{size_mismatch} records have a frame size different from the video ({video_w}x{video_h}); coordinates are rescaled"),
        "keypoint_order_suspect": (order_suspect, f"{order_suspect} of {n_hands} hands look like their keypoints are not in MediaPipe order"),
    }
    if order_suspect <= ORDER_WARN_FRAC * n_hands:  # counted in the result, but not worth a warning
        problems["keypoint_order_suspect"] = (0, "")
    if len(hand_ids) > MAX_HAND_IDS:
        problems["too_many_hand_ids"] = (len(hand_ids), f"{len(hand_ids)} distinct hand_ids (more than {MAX_HAND_IDS}): ids may not be stable")
    result = {k: v[0] for k, v in problems.items()}
    result["keypoint_order_suspect"] = order_suspect
    result["distinct_hand_ids"] = len(hand_ids)
    result["warnings"] = [msg for count, msg in problems.values() if count]
    result["ok"] = not result["warnings"]
    for msg in result["warnings"]:
        log.warning("PS1 data: %s", msg)
    return result


def load_hands(hands, video_w: int, video_h: int) -> dict[int, list[Hand]]:
    """frame_idx -> hands seen in that frame. Coordinates are rescaled to the video size if needed."""
    out: dict[int, list[Hand]] = {}
    for frame in read_ps1(hands):
        try:
            idx = int(frame["frame_idx"])
            sx = video_w / frame.get("frame_width", video_w)
            sy = video_h / frame.get("frame_height", video_h)
            scale = np.array([sx, sy])
            found = []
            for h in frame.get("hands", []):
                kp = np.asarray(h["kp2d_px"], dtype=float)
                if kp.shape != (21, 2):
                    raise ValueError(f"expected 21 keypoints, got shape {kp.shape}")
                bbox = np.asarray(h["bbox_px"], dtype=float) * np.array([sx, sy, sx, sy])
                side = str(h.get("side", "unknown")).lower()
                found.append(Hand(h["hand_id"], kp * scale, bbox, h.get("source", "detected"),
                                  side if side in ("left", "right") else "unknown"))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"malformed PS1 frame {frame.get('frame_idx', '?')}: {e}") from e
        out[idx] = found
    return out


def hand_scale(kp: np.ndarray) -> float:
    """Wrist to middle-finger knuckle distance, in pixels."""
    return float(np.linalg.norm(kp[0] - kp[9]))


def tip_distances(kp: np.ndarray, shape, scale: float | None = None) -> list[float]:
    """Distance of each fingertip to a box (x1,y1,x2,y2) or polygon (N,2), in hand sizes. 0 if inside."""
    scale = scale or hand_scale(kp)
    shape = np.asarray(shape, dtype=float)
    dists = []
    for x, y in kp[list(FINGERTIPS)]:
        if shape.shape == (4,):
            x1, y1, x2, y2 = shape
            d = float(np.hypot(max(x1 - x, 0, x - x2), max(y1 - y, 0, y - y2)))
        else:
            signed = cv2.pointPolygonTest(shape.astype(np.float32).reshape(-1, 1, 2), (float(x), float(y)), True)
            d = max(-signed, 0.0)
        dists.append(d / scale)
    return dists


def finger_curl(kp: np.ndarray) -> float:
    """0 = flat hand, 1 = fist, from the bend at the middle joint of four fingers."""
    curls = []
    for mcp, pip, dip in FINGER_JOINTS:
        a, b = kp[mcp] - kp[pip], kp[dip] - kp[pip]
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))  # 180 = straight
        curls.append(np.clip((180.0 - angle) / FULL_CURL_DEG, 0, 1))
    return float(np.mean(curls))


# ---------------------------------------------------------------- wearer hands and internal hand tracks

def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def filter_bystanders(hands_by_frame: dict[int, list[Hand]], frame_h: float, cfg: dict) -> tuple[dict[int, list[Hand]], int]:
    """PS1 does not filter other people's hands. Keep at most ego_max_hands likely wearer hands per frame: drop hands
    smaller than ego_min_scale_ratio of the largest, or with the wrist in the top ego_max_wrist_top_frac of the frame;
    of the rest keep the largest / lowest ones. Returns (filtered hands, number of hands dropped)."""
    out, dropped = {}, 0
    for f, hands in hands_by_frame.items():
        scales = [hand_scale(h.kp) for h in hands]
        biggest = max(scales, default=0.0)
        ranked = []
        for h, s in zip(hands, scales):
            if biggest > 0 and (s < cfg["ego_min_scale_ratio"] * biggest or h.kp[0][1] < cfg["ego_max_wrist_top_frac"] * frame_h):
                dropped += 1
                continue
            ranked.append((s / biggest + h.kp[0][1] / frame_h if biggest > 0 else 0.0, h))
        ranked.sort(key=lambda r: -r[0])        # larger and nearer the bottom edge = more likely the wearer's
        dropped += max(len(ranked) - cfg["ego_max_hands"], 0)
        out[f] = [h for _, h in ranked[:cfg["ego_max_hands"]]]
    return out, dropped


def assign_tracks(hands_by_frame: dict[int, list[Hand]], cfg: dict) -> int:
    """Give every hand an internal track id (Hand.track), because PS1 hand_ids are not persistent.

    A hand continues a track seen at most track_max_gap_frames earlier if the sides agree (or the boxes overlap at least
    track_side_override_iou, since a side label can flip on the same hand) and the boxes overlap (IoU >= track_min_iou)
    or the wrist moved less than track_max_wrist_norm hand sizes. Closest wrist first. Returns the number of tracks."""
    last: dict[str, tuple[int, Hand]] = {}
    count = 0
    for f in sorted(hands_by_frame):
        hands = hands_by_frame[f]
        pairs = []
        for hi, h in enumerate(hands):
            for tid, (lf, prev) in last.items():
                if f - lf > cfg["track_max_gap_frames"]:
                    continue
                overlap = box_iou(h.bbox, prev.bbox)
                if h.side != prev.side and "unknown" not in (h.side, prev.side) and overlap < cfg["track_side_override_iou"]:
                    continue
                dist = float(np.linalg.norm(h.kp[0] - prev.kp[0])) / max(hand_scale(prev.kp), 1e-6)
                if overlap >= cfg["track_min_iou"] or dist <= cfg["track_max_wrist_norm"]:
                    pairs.append((dist, hi, tid))
        used_h, used_t = set(), set()
        for dist, hi, tid in sorted(pairs):
            if hi not in used_h and tid not in used_t:
                hands[hi].track = tid
                used_h.add(hi)
                used_t.add(tid)
        for hi, h in enumerate(hands):
            if hi not in used_h:
                count += 1
                h.track = f"t{count}"
            last[h.track] = (f, h)
    return count


def check_constant_frame_rate(video_path) -> bool | None:
    """True/False from ffprobe (r_frame_rate vs avg_frame_rate), or None if ffprobe is missing or fails. Warn only."""
    import shutil
    import subprocess
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=r_frame_rate,avg_frame_rate", "-of", "json", str(video_path)],
                             capture_output=True, text=True, timeout=60, check=True).stdout
        stream = json.loads(out)["streams"][0]

        def ratio(s):
            a, _, b = s.partition("/")
            return float(a) / float(b or 1)

        cfr = abs(ratio(stream["r_frame_rate"]) - ratio(stream["avg_frame_rate"])) < 0.01
    except (OSError, ValueError, KeyError, IndexError, ZeroDivisionError, subprocess.SubprocessError):
        return None
    if not cfr:
        log.warning("video does not have a constant frame rate (r_frame_rate != avg_frame_rate): PS1 times may drift")
    return cfr
