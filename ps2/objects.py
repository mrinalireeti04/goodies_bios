"""YOLOE detection + tracking, and removal of hand/glove boxes."""
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms
from ultralytics import YOLOE
from ultralytics.engine.results import Boxes
from ultralytics.trackers import BOTSORT
from ultralytics.utils import YAML, IterableSimpleNamespace

from ps2.hands import Hand, box_iou

IGNORE_PREFIX = "ignore:"
SIZE_FALLBACK_LABEL = "tool"  # generic portable label for detections that are implausibly large


def build_prompts(vocab: dict) -> tuple[list[str], dict[str, str]]:
    """Text prompts for the detector, and prompt -> vocab key (ignore names get an 'ignore:' key)."""
    prompt_to_key: dict[str, str] = {}
    for group in ("objects", "generic"):
        for key, spec in vocab[group].items():
            for p in spec["prompts"]:
                prompt_to_key.setdefault(p, key)
    for name in vocab["ignore"]:
        prompt_to_key.setdefault(name, IGNORE_PREFIX + name)
    return list(prompt_to_key), prompt_to_key


iou = box_iou


def drop_hand_like(dets: list[dict], hands: list[Hand], max_iou: float) -> list[dict]:
    """Remove detections of the body/gloves (label in ignore list, already removed) or that sit on a PS1 hand box."""
    return [d for d in dets if all(iou(d["bbox"], h.bbox) <= max_iou for h in hands)]


MIN_CROP_PX = 32  # a hand crop smaller than this (after clipping to the frame) is skipped


def hand_crop_box(bbox, scale: float, frame_w: int, frame_h: int) -> tuple[int, int, int, int] | None:
    """Square crop around a hand box, `scale` times the longer side, clipped to the frame. None if too small."""
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    side = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * scale
    x0, y0 = max(int(cx - side / 2), 0), max(int(cy - side / 2), 0)
    x1, y1 = min(int(cx + side / 2), frame_w), min(int(cy + side / 2), frame_h)
    if x1 - x0 < MIN_CROP_PX or y1 - y0 < MIN_CROP_PX:
        return None
    return x0, y0, x1, y1


def merge_detections(xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray, max_iou: float) -> np.ndarray:
    """Indices to keep after class-aware NMS: of two overlapping boxes with the same class, the more confident stays."""
    if len(xyxy) == 0:
        return np.zeros(0, int)
    keep = batched_nms(torch.as_tensor(xyxy, dtype=torch.float32), torch.as_tensor(conf, dtype=torch.float32),
                       torch.as_tensor(cls, dtype=torch.int64), max_iou)
    return np.sort(keep.numpy())


class Detector:
    """YOLOE on the whole frame and on a crop around every PS1 hand, merged, then tracked."""

    def __init__(self, cfg: dict, vocab: dict, root: Path, device: str):
        self.cfg, self.device = cfg, device
        self.half = bool(cfg["half"]) and device != "cpu"
        self.model = YOLOE(root / cfg["det_model"])
        self.set_vocab(vocab)
        tracker_cfg = IterableSimpleNamespace(**YAML.load(str(Path(__file__).parent / "config" / "botsort.yaml")))
        tracker_cfg.device = device
        self.tracker = BOTSORT(args=tracker_cfg)

    def set_vocab(self, vocab: dict) -> None:
        """Change what the detector looks for (the tracker is not touched)."""
        self.prompts, self.prompt_to_key = build_prompts(vocab)
        self.specs = {**vocab["objects"], **vocab["generic"]}
        self.model.set_classes(self.prompts, self.model.get_text_pe(self.prompts))

    def detect(self, frame: np.ndarray, hands: list[Hand]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Boxes (xyxy, frame pixels), confidences and prompt indices from the frame and the hand crops."""
        h, w = frame.shape[:2]
        images, offsets = [frame], [(0, 0)]
        if self.cfg["hand_crop_enabled"]:
            for hand in hands:
                box = hand_crop_box(hand.bbox, self.cfg["hand_crop_scale"], w, h)
                if box is not None:
                    images.append(frame[box[1]:box[3], box[0]:box[2]])
                    offsets.append((box[0], box[1]))
        results = self.model.predict(images, imgsz=self.cfg["imgsz"], conf=self.cfg["det_conf"], half=self.half,
                                     device=self.device, verbose=False)
        xyxy, conf, cls = [], [], []
        for res, (ox, oy) in zip(results, offsets):
            b = res.boxes
            if b is None or len(b) == 0:
                continue
            xyxy.append(b.xyxy.cpu().numpy() + np.array([ox, oy, ox, oy]))
            conf.append(b.conf.cpu().numpy())
            cls.append(b.cls.cpu().numpy())
        if not xyxy:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0)
        xyxy, conf, cls = np.concatenate(xyxy), np.concatenate(conf), np.concatenate(cls)
        keep = merge_detections(xyxy, conf, cls, self.cfg["nms_iou"])
        return xyxy[keep], conf[keep], cls[keep]

    def track(self, frame: np.ndarray, hands: list[Hand]) -> list[dict]:
        """Tracked detections: track_id, label (vocab key), type, is_hazard, bbox, conf."""
        xyxy, conf, cls = self.detect(frame, hands)
        data = np.concatenate([xyxy, conf[:, None], cls[:, None]], axis=1) if len(xyxy) else np.zeros((0, 6))
        tracks = self.tracker.update(Boxes(torch.as_tensor(data, dtype=torch.float32), frame.shape[:2]).numpy(), frame)
        frame_area = frame.shape[0] * frame.shape[1]
        dets = []
        for x1, y1, x2, y2, tid, score, c, _ in tracks:
            key = self.prompt_to_key[self.prompts[int(c)]]
            if key.startswith(IGNORE_PREFIX):
                continue
            spec = self.specs[key]
            if spec["type"] == "portable" and (x2 - x1) * (y2 - y1) > self.cfg["max_portable_frac"] * frame_area:
                key = SIZE_FALLBACK_LABEL  # too big to be a hand tool: do not trust a specific name
                spec = self.specs[key]
            dets.append({
                "track_id": int(tid), "label": key, "type": spec["type"],
                "is_hazard": bool(spec.get("hazard", False)),
                "bbox": [float(x1), float(y1), float(x2), float(y2)], "conf": float(score),
            })
        return dets


# ---------------------------------------------------------------- same object after leaving and re-entering view

HIST_BINS = (16, 8)   # hue x saturation
CROP_SIZE = 32        # crops are shrunk to this before the histogram
BEST_CROP_MAX = 224   # the clearest crop of each track is kept in memory at this size (longest side) for name suggestions


def hand_overlap(box, hands: list[Hand]) -> float:
    """Fraction of `box` covered by hand boxes (summed, capped at 1)."""
    area = (box[2] - box[0]) * (box[3] - box[1])
    if area <= 0:
        return 1.0
    covered = 0.0
    for h in hands:
        ix = max(0.0, min(box[2], h.bbox[2]) - max(box[0], h.bbox[0]))
        iy = max(0.0, min(box[3], h.bbox[3]) - max(box[1], h.bbox[1]))
        covered += ix * iy
    return min(covered / area, 1.0)


class TrackBook:
    """Collects, per tracker ID, the label votes, the frames seen and a colour fingerprint.

    The fingerprint (HSV histogram) only uses frames where hands cover little of the object.
    Crops are never kept: each one is turned into a histogram and added up.
    """

    def __init__(self, cfg: dict):
        self.max_overlap = cfg["clean_crop_max_overlap"]
        self.pieces: dict[int, dict] = {}

    def add(self, frame_idx: int, det: dict, bgr: np.ndarray, hands: list[Hand]) -> None:
        piece = self.pieces.setdefault(det["track_id"], {"frames": [], "labels": Counter(), "hist": None,
                                                         "best_score": -1.0, "best_crop": None})
        piece["frames"].append(frame_idx)
        piece["labels"][det["label"]] += det["conf"]
        overlap = hand_overlap(det["bbox"], hands)
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        crop = bgr[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
        if crop.shape[0] < 4 or crop.shape[1] < 4:
            return
        score = det["conf"] * (1 - overlap)  # the clearest view: confident and not covered by hands
        if score > piece["best_score"]:
            longest = max(crop.shape[:2])
            if longest > BEST_CROP_MAX:
                f = BEST_CROP_MAX / longest
                crop_small = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            else:
                crop_small = crop.copy()
            piece["best_score"], piece["best_crop"] = score, crop_small
        if overlap < self.max_overlap:
            hsv = cv2.cvtColor(cv2.resize(crop, (CROP_SIZE, CROP_SIZE)), cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, list(HIST_BINS), [0, 180, 0, 256]).ravel()
            hist /= max(hist.sum(), 1e-9)
            piece["hist"] = hist if piece["hist"] is None else piece["hist"] + hist


def group_label(labels: Counter, generic: set[str]) -> str:
    """Most-voted specific label, or the most-voted generic one if there is no specific label."""
    specific = {k: v for k, v in labels.items() if k not in generic}
    pool = specific or labels
    return max(pool, key=pool.get)


def _closest_frames(a, b) -> tuple[int, int]:
    """The pair of frames (one from each collection) that are nearest in time."""
    a, b = np.sort(list(a)), np.sort(list(b))
    pos = np.searchsorted(a, b)
    best = (np.inf, 0, 0)
    for fb, p in zip(b, pos):
        for fa in a[max(p - 1, 0):p + 1]:
            if abs(fa - fb) < best[0]:
                best = (abs(fa - fb), int(fa), int(fb))
    return best[1], best[2]


def merge_pieces(pieces: dict[int, dict], generic: set[str], cfg: dict, frame_ms: float) -> tuple[dict[int, int], list[dict]]:
    """Decide which tracker IDs are the same physical object.

    Two groups may merge only if they are never visible in the same frame, their labels match (or one is
    generic) and their colour fingerprints are similar (>= reid_min_sim). Most similar pairs merge first.
    Groups with no fingerprint merge only with a group of the same label that is close in time.
    Returns (tracker id -> final object number from 0 by first appearance,
             uncertain merges [{"tracker_id", "frame_a", "frame_b"}]: the closest pair of frames of the two appearances).
    """
    if not pieces:
        return {}, []
    tids = sorted(pieces, key=lambda t: min(pieces[t]["frames"]))
    n = len(tids)
    groups = [{"ids": [t], "frames": set(pieces[t]["frames"]), "labels": Counter(pieces[t]["labels"]),
               "hist": None if pieces[t]["hist"] is None else np.array(pieces[t]["hist"], float)} for t in tids]
    uncertain: list[dict] = []
    max_gap = cfg["reid_max_gap_ms"] / frame_ms
    min_frames = cfg["min_tool_ms"] / frame_ms

    names = [group_label(g["labels"], generic) for g in groups]
    code = {name: i for i, name in enumerate(sorted(set(names)))}
    lab = np.array([code[x] for x in names])
    is_gen = np.array([x in generic for x in names])
    has = np.array([g["hist"] is not None for g in groups])
    root = np.zeros((n, HIST_BINS[0] * HIST_BINS[1]))   # sqrt of the normalised histograms
    for i, g in enumerate(groups):
        if has[i]:
            root[i] = np.sqrt(g["hist"] / g["hist"].sum())
    alive = np.ones(n, bool)
    blocked = np.zeros((n, n), bool)                    # pairs known to be visible together

    def sim_row(i):  # 1 - Bhattacharyya distance between group i and every group
        if not has[i]:
            return np.full(n, -1.0)
        bc = np.clip(root @ root[i], 0, 1)
        return np.where(has & alive, 1 - np.sqrt(1 - bc), -1.0)

    def ok_row(i):   # labels allow a merge
        return (lab == lab[i]) | is_gen | is_gen[i]

    sim = np.array([sim_row(i) for i in range(n)])
    ok = np.array([ok_row(i) for i in range(n)])

    def disjoint(a, b):
        return groups[a]["frames"].isdisjoint(groups[b]["frames"])

    def merge(a, b, is_uncertain):  # is_uncertain: False, or a short reason ("tie", "no_fingerprint")
        ga, gb = groups[a], groups[b]
        if is_uncertain:
            fa, fb = _closest_frames(ga["frames"], gb["frames"])
            uncertain.append({"tracker_id": ga["ids"][0], "frame_a": fa, "frame_b": fb, "why": is_uncertain})
        ga["ids"] += gb["ids"]
        ga["frames"] |= gb["frames"]
        ga["labels"] += gb["labels"]
        if gb["hist"] is not None:
            ga["hist"] = gb["hist"] if ga["hist"] is None else ga["hist"] + gb["hist"]
        alive[b] = has[b] = False
        sim[b, :] = sim[:, b] = -1.0
        has[a] = ga["hist"] is not None
        if has[a]:
            root[a] = np.sqrt(ga["hist"] / ga["hist"].sum())
        top = group_label(ga["labels"], generic)
        lab[a], is_gen[a] = code.setdefault(top, len(code)), top in generic
        blocked[a] |= blocked[b]
        blocked[:, a] = blocked[a]
        ok[a] = ok_row(a)
        ok[:, a] = ok[a]
        sim[a] = sim_row(a)
        sim[:, a] = sim[a]
        sim[a, a] = -1.0

    # 1) colour fingerprint merges, most similar first
    while True:
        cand = np.where(alive[:, None] & alive[None, :] & ~blocked & ok & (sim >= cfg["reid_min_sim"]), sim, -1.0)
        np.fill_diagonal(cand, -1.0)
        a, b = np.unravel_index(np.argmax(cand), cand.shape)
        best = cand[a, b]
        if best < 0:
            break
        if not disjoint(a, b):
            blocked[a, b] = blocked[b, a] = True
            continue
        # a near-equal rival only counts if it, and the pair being merged, were seen long enough to matter
        matters = lambda g: len(groups[g]["frames"]) >= min_frames
        rivals = sum(
            k != y and disjoint(x, k) and matters(k) and matters(x) and matters(y)
            for x, y in ((a, b), (b, a))
            for k in np.where(cand[x] >= max(best - cfg["reid_tie_margin"], 0))[0]
        )
        merge(min(a, b), max(a, b), "tie" if rivals > 0 else False)

    # 2) no fingerprint: same label, closest in time, always uncertain
    for i in range(n):
        if not alive[i] or has[i]:
            continue
        best = None
        for j in np.where(alive)[0]:
            if j == i or lab[j] != lab[i] or not disjoint(i, j):
                continue
            fa, fb = _closest_frames(groups[i]["frames"], groups[j]["frames"])
            if best is None or abs(fa - fb) < best[0]:
                best = (abs(fa - fb), j)
        if best is not None and best[0] <= max_gap:
            merge(min(i, best[1]), max(i, best[1]), "no_fingerprint")

    order = sorted(np.where(alive)[0], key=lambda k: min(groups[k]["frames"]))
    mapping = {t: num for num, k in enumerate(order) for t in groups[k]["ids"]}
    return mapping, uncertain


def _is_hazard(spec: dict, bbox, frame_area: float, cfg: dict) -> bool:
    """A hazard needs the vocabulary flag AND a box smaller than big_box_frac of the frame (a huge box is a surface)."""
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return bool(spec.get("hazard", False)) and area < cfg["big_box_frac"] * frame_area


def apply_identities(dets_by_frame: list[list[dict]], aux: list[dict], pieces: dict[int, dict],
                     mapping: dict[int, int], vocab: dict, frame_area: float, cfg: dict) -> None:
    """Give every detection its final object_id (obj_1, obj_2, ... by first appearance) and one label per object.

    The label is the most-voted specific label over all appearances; type and hazard follow that label.
    `aux[i]["obj_res"]` (keyed by tracker id) is re-keyed to the final ids.
    """
    generic = set(vocab["generic"])
    specs = {**vocab["objects"], **vocab["generic"]}
    votes: dict[int, Counter] = {}
    for tid, num in mapping.items():
        votes.setdefault(num, Counter()).update(pieces[tid]["labels"])
    final = {num: group_label(v, generic) for num, v in votes.items()}
    for dets in dets_by_frame:
        for d in dets:
            num = mapping[d["track_id"]]
            spec = specs[final[num]]
            d.update(object_id=f"obj_{num + 1}", label=final[num], type=spec["type"],
                     is_hazard=_is_hazard(spec, d["bbox"], frame_area, cfg))
    for a in aux:
        a["obj_res"] = {f"obj_{mapping[tid] + 1}": v for tid, v in a["obj_res"].items()}


def set_object_label(dets_by_frame: list[list[dict]], object_id: str, label: str, vocab: dict,
                     frame_area: float, cfg: dict) -> None:
    """Give one object a new label (type and hazard follow the vocabulary entry) in every frame."""
    spec = {**vocab["objects"], **vocab["generic"]}[label]
    for dets in dets_by_frame:
        for d in dets:
            if d["object_id"] == object_id:
                d.update(label=label, type=spec["type"], is_hazard=_is_hazard(spec, d["bbox"], frame_area, cfg))


def best_crops(pieces: dict[int, dict], mapping: dict[int, int]) -> dict[str, np.ndarray]:
    """object_id -> the clearest crop over all appearances of that object."""
    best: dict[str, tuple[float, np.ndarray]] = {}
    for tid, num in mapping.items():
        piece = pieces[tid]
        oid = f"obj_{num + 1}"
        if piece["best_crop"] is not None and piece["best_score"] > best.get(oid, (-1.0, None))[0]:
            best[oid] = (piece["best_score"], piece["best_crop"])
    return {oid: crop for oid, (_, crop) in best.items()}


def identity_review_items(uncertain: list[dict], mapping: dict[int, int], times: list[int], frame_ms: float) -> list[dict]:
    """review_queue items (reason identity_uncertain) spanning the two appearances that were merged."""
    items = []
    for n, u in enumerate(sorted(uncertain, key=lambda u: min(u["frame_a"], u["frame_b"])), 1):
        lo, hi = sorted((u["frame_a"], u["frame_b"]))
        items.append({
            "item_id": f"r_{n}", "start_ms": times[lo], "end_ms": int(round(times[hi] + frame_ms)),
            "hand_id": None, "object_id": f"obj_{mapping[u['tracker_id']] + 1}",
            "reason": "identity_uncertain", "suggested_label": None,
        })
    return items
