"""RTMPose hand engine: higher-accuracy 21-point hands, same PS1Frame output.

Pipeline per frame (all local; CoreML on Apple Silicon when compatible, CPU fallback):

    MediaPipe (480 px copy)  -> hand boxes + left/right label
    RTMDet-nano hand detector -> more hand boxes
    previous-frame boxes      -> boxes for hands the detectors miss this frame
         |
         v  merge boxes (NMS)
    RTMPose-m hand, run on a crop of the ORIGINAL full-resolution frame
         |
         v  drop hands with a low keypoint score, assign left/right
    PS1Frame hands[]

Why it should be more accurate than MediaPipe alone: MediaPipe sees a 480 px
copy of a 1920 px frame, so a hand is only ~30 px wide. RTMPose reads a 256x256
crop cut from the full-resolution frame, so fingers keep their detail.

The RTMPose weights are downloaded by `rtmlib` the first time it runs
(about 100 MB, cached in your user folder).
"""
import os
import platform
from pathlib import Path

import numpy as np

# ---------- tunables (not tuned on real footage yet) ----------
MIN_KP_SCORE = 0.2     # mean RTMPose keypoint score below this: hand is dropped
MIN_BOX_SIDE = 40      # px: RTMPose skeleton whose longest box side is below this is squashed, dropped
BOX_PAD = 1.0          # RTMPose pads the box itself (1.25x); no extra padding here
NMS_IOU = 0.5          # boxes overlapping more than this are merged
MATCH_IOU = 0.1        # RTMPose hand <-> MediaPipe hand match for left/right
FALLBACK_IOU = 0.3     # MediaPipe hand with IoU below this to every RTMPose hand is kept as is
PREV_SIDE_IOU = 0.3    # lone hand takes the side of a previous-frame hand overlapping more than this
MAX_HANDS = 2


# ---------- box helpers ----------
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def expand(box, factor, w, h):
    """Grow a box about its centre and clip it to the frame."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    bw, bh = (box[2] - box[0]) * factor, (box[3] - box[1]) * factor
    return [max(0.0, cx - bw / 2), max(0.0, cy - bh / 2),
            min(float(w), cx + bw / 2), min(float(h), cy + bh / 2)]


def merge_boxes(boxes, iou_thr=NMS_IOU):
    """Keep one box per overlapping group; the earlier box in the list wins."""
    kept = []
    for b in boxes:
        if b[2] - b[0] < 4 or b[3] - b[1] < 4:
            continue
        if all(iou(b, k) < iou_thr for k in kept):
            kept.append(list(map(float, b)))
    return kept


def kp_box(kp):
    xs, ys = kp[:, 0], kp[:, 1]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


class _CoreMLSafeSession:
    """Retry a CoreML-incompatible model on CPU after a runtime execution error."""
    def __init__(self, session, model_path, label):
        self.session = session
        self.model_path = model_path
        self.label = label

    def run(self, output_names, input_feed, run_options=None):
        try:
            return self.session.run(output_names, input_feed, run_options)
        except Exception as exc:
            if "CoreMLExecutionProvider" not in self.session.get_providers():
                raise
            import onnxruntime as ort
            print(f"[inference] {self.label} CoreML execution failed ({exc}); retrying this model on CPU", flush=True)
            self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            return self.session.run(output_names, input_feed, run_options)

    def get_inputs(self):
        return self.session.get_inputs()

    def get_outputs(self):
        return self.session.get_outputs()

    def get_providers(self):
        return self.session.get_providers()


# ---------- engine ----------
class RTMHandEngine:
    """Detector + RTMPose-hand. Pass stub `det`/`pose` objects in tests."""

    def __init__(self, min_kp_score=MIN_KP_SCORE, device=None, det=None, pose=None):
        if det is None or pose is None:
            try:
                from rtmlib import Hand
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "RTMPose needs extra packages: pip install rtmlib onnxruntime") from e
            requested = (device or os.environ.get("FORGE_RTM_DEVICE", "auto")).lower()
            if requested == "auto":
                requested = "mps" if platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"} else "cpu"
            if requested == "coreml":
                requested = "mps"
            if requested not in {"mps", "cpu"}:
                raise ValueError("FORGE_RTM_DEVICE must be auto, coreml, mps, or cpu")
            if requested == "mps":
                # Reuse CoreML's compiled subgraphs between uploads; compiling on
                # every job can cost minutes even when inference itself is fast.
                cache_dir = Path(os.environ.get(
                    "FORGE_COREML_CACHE",
                    Path(__file__).resolve().parents[2] / ".local_data" / "coreml_cache",
                )).expanduser()
                cache_dir.mkdir(parents=True, exist_ok=True)
                try:
                    from rtmlib.tools.base import RTMLIB_SETTINGS
                    RTMLIB_SETTINGS["onnxruntime"]["mps"] = (
                        "CoreMLExecutionProvider", {"ModelCacheDirectory": str(cache_dir)}
                    )
                except (ImportError, KeyError, TypeError):
                    pass  # Older RTMLib still runs CoreML without persistent caching.
            try:
                hand = Hand(mode="lightweight", backend="onnxruntime", device=requested)
                selected_device = requested
            except Exception as exc:
                if requested != "mps":
                    raise
                # CoreML may be present but unable to compile this model on a given
                # macOS/ORT version. Keep local inference usable with CPU fallback.
                print(f"[inference] CoreML initialization failed ({exc}); falling back to CPU", flush=True)
                hand = Hand(mode="lightweight", backend="onnxruntime", device="cpu")
                selected_device = "cpu"
            det, pose = hand.det_model, hand.pose_model
            if selected_device == "mps":
                hand.det_model.session = _CoreMLSafeSession(hand.det_model.session, hand.det_model.onnx_model, "RTMDet")
                hand.pose_model.session = _CoreMLSafeSession(hand.pose_model.session, hand.pose_model.onnx_model, "RTMPose")
            det_providers = hand.det_model.session.get_providers()
            pose_providers = hand.pose_model.session.get_providers()
            providers = sorted(set(det_providers + pose_providers))
            uses_coreml = "CoreMLExecutionProvider" in providers
            uses_cpu = "CPUExecutionProvider" in providers
            actual_device = "coreml+cpu" if uses_coreml and uses_cpu else "coreml" if uses_coreml else "cpu"
            self._det_model, self._pose_model = hand.det_model, hand.pose_model
            self.runtime = {"requested_device": requested, "selected_device": actual_device,
                            "providers": providers,
                            "model_providers": {"rtmdet": det_providers, "rtmpose": pose_providers}}
            print(f"[inference] RTMDet={det_providers} RTMPose={pose_providers}", flush=True)
        else:
            self.runtime = {"requested_device": device or "injected", "selected_device": "injected", "providers": []}
        self.det, self.pose = det, pose
        self.min_kp_score = min_kp_score
        self.prev_boxes = []   # boxes of the hands kept in the previous frame
        self.prev_hands = []   # (bbox, side) of every hand output in the previous frame
        self.stats = {"proposals": 0, "kept": 0, "dropped_low_score": 0,
                      "mediapipe_fallback": 0}

    def detect_boxes(self, frame):
        """RTMDet hand boxes in original-frame pixels."""
        out = self.det(frame)
        out = np.asarray(out, dtype=float).reshape(-1, 4) if len(out) else np.zeros((0, 4))
        return [list(b) for b in out]

    def estimate(self, frame, mp_hands, swap_lr=False):
        """Return raw hand dicts: side, score, bbox, kp2d (list of 21 [x, y]).

        `mp_hands` are the hands MediaPipe found in this frame (schema dicts,
        pixels of the original frame). They supply proposals and left/right.
        """
        h, w = frame.shape[:2]
        proposals = [expand(m["bbox_px"], 1.3, w, h) for m in mp_hands]   # best first
        proposals += self.detect_boxes(frame)
        proposals += [expand(b, 1.2, w, h) for b in self.prev_boxes]
        boxes = merge_boxes(proposals)[:MAX_HANDS * 2]
        self.stats["proposals"] += len(boxes)

        found = []
        if boxes:
            kps, scores = self.pose(frame, bboxes=boxes)
            kps = np.asarray(kps).reshape(len(boxes), 21, 2)
            scores = np.asarray(scores).reshape(len(boxes), 21)
            for kp, sc in zip(kps, scores):
                box = kp_box(kp)
                squashed = max(box[2] - box[0], box[3] - box[1]) < MIN_BOX_SIDE
                if float(sc.mean()) < self.min_kp_score or squashed:
                    self.stats["dropped_low_score"] += 1
                    continue
                found.append({"kp": kp, "kp_score": float(sc.mean())})

        # same hand found twice (e.g. two boxes -> two overlapping skeletons)
        found.sort(key=lambda f: f["kp_score"], reverse=True)
        uniq = []
        for f in found:
            f["bbox"] = kp_box(f["kp"])
            if all(iou(f["bbox"], u["bbox"]) < NMS_IOU for u in uniq):
                uniq.append(f)
        uniq = uniq[:MAX_HANDS]

        self._assign_sides(uniq, mp_hands, w)
        self._sides_from_previous_frame(uniq)
        self.prev_boxes = [u["bbox"] for u in uniq]
        self.stats["kept"] += len(uniq)

        # MediaPipe hands with no matching RTMPose hand are kept unchanged
        for m in mp_hands:
            if all(iou(m["bbox_px"], u["bbox"]) < FALLBACK_IOU for u in uniq):
                uniq.append({"kp": np.array(m["kp2d_px"], dtype=float), "kp_score": 0.0,
                             "bbox": list(m["bbox_px"]), "side": m["side"],
                             "score": m["handedness_score"], "kp3d": m["kp3d_hand_m"]})
                self.stats["mediapipe_fallback"] += 1

        _dedupe_sides(uniq)
        self.prev_hands = [(list(u["bbox"]), u["side"]) for u in uniq]
        if hasattr(self, "_det_model"):
            det_providers = self._det_model.session.get_providers()
            pose_providers = self._pose_model.session.get_providers()
            providers = set(det_providers + pose_providers)
            uses_coreml = "CoreMLExecutionProvider" in providers
            uses_cpu = "CPUExecutionProvider" in providers
            self.runtime["providers"] = sorted(providers)
            self.runtime["model_providers"] = {"rtmdet": det_providers, "rtmpose": pose_providers}
            self.runtime["selected_device"] = "coreml+cpu" if uses_coreml and uses_cpu else "coreml" if uses_coreml else "cpu"
        return uniq

    def _sides_from_previous_frame(self, found):
        """An unlabelled hand with no MediaPipe match takes the side of the
        best-overlapping previous-frame hand (IoU > PREV_SIDE_IOU) if that side is free."""
        for f in found:
            if f["side"] != "unknown" or f.get("matched"):
                continue
            taken = {g["side"] for g in found if g is not f}
            best, best_iou = None, PREV_SIDE_IOU
            for box, side in self.prev_hands:
                v = iou(f["bbox"], box)
                if side in ("left", "right") and side not in taken and v > best_iou:
                    best, best_iou = side, v
            if best:
                f["side"] = best

    @staticmethod
    def _assign_sides(found, mp_hands, frame_w):
        """Left/right from the best-overlapping MediaPipe hand; else x-order."""
        for f in found:
            best, best_iou = None, MATCH_IOU
            for m in mp_hands:
                v = iou(f["bbox"], m["bbox_px"])
                if v > best_iou:
                    best, best_iou = m, v
            f["side"] = best["side"] if best else "unknown"
            f["score"] = best["handedness_score"] if best else None
            f["matched"] = best is not None
        # two hands, neither labelled: leftmost in the image is the left hand
        unl = [f for f in found if f["side"] == "unknown"]
        if len(found) == 2 and len(unl) == 2:
            a, b = sorted(found, key=lambda f: (f["bbox"][0] + f["bbox"][2]) / 2)
            a["side"], b["side"] = "left", "right"
        # two hands, one labelled: the other gets the opposite side if it is free
        elif len(found) == 2 and len(unl) == 1:
            known = next(f for f in found if f["side"] != "unknown")
            other = unl[0]
            opposite = {"left": "right", "right": "left"}[known["side"]]
            k_cx = (known["bbox"][0] + known["bbox"][2]) / 2
            o_cx = (other["bbox"][0] + other["bbox"][2]) / 2
            if (opposite == "right") == (o_cx > k_cx):
                other["side"] = opposite


def _dedupe_sides(found):
    """A side may appear once per frame; the higher-scoring hand keeps it."""
    for side in ("left", "right"):
        same = [f for f in found if f["side"] == side]
        same.sort(key=lambda f: (f["score"] if f["score"] is not None else -1,
                                 f["kp_score"]), reverse=True)
        for f in same[1:]:
            f["side"] = "unknown"


def to_schema_hands(found):
    """Raw hand dicts -> schema `hands` list (same rules as hand_detect.py)."""
    _dedupe_sides(found)
    order = {"left": 0, "right": 1, "unknown": 2}
    found = sorted(found, key=lambda f: order[f["side"]])
    hands, u = [], 0
    for f in found:
        if f["side"] == "left":
            hid = "L"
        elif f["side"] == "right":
            hid = "R"
        else:
            u += 1
            hid = f"U{u}"
        kp = [[round(float(x), 2), round(float(y), 2)] for x, y in f["kp"]]
        xs, ys = [p[0] for p in kp], [p[1] for p in kp]
        hands.append({
            "hand_id": hid,
            "side": f["side"],
            "handedness_score": f["score"],
            "bbox_px": [min(xs), min(ys), max(xs), max(ys)],
            "kp2d_px": kp,
            # RTMPose is 2D only (None); a MediaPipe fallback hand keeps its own 3D points
            "kp3d_hand_m": f.get("kp3d"),
            "source": "detected",
        })
    return hands
