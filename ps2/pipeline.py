"""run_ps2: the whole flow, top to bottom."""
import gc
import copy
import logging
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from ps2 import contact, fuse, ledger, masks, safety, schema, summary, vlm, vocab_auto
from ps2.contact import track_key
from ps2.hands import (assign_tracks, check_constant_frame_rate, check_ps1, filter_bystanders, load_hands,
                       read_ps1)
from ps2.motion import Motion
from ps2.objects import Detector, TrackBook, apply_identities, best_crops, drop_hand_like, identity_review_items, merge_pieces

PKG = Path(__file__).parent
ROOT = PKG.parent  # repo root: model weights are downloaded/stored here
log = logging.getLogger("ps2")


def propagate_detections(previous_gray, gray, detections):
    """Move boxes by sparse optical flow; drop boxes without reliable points."""
    propagated = []
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        points = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2],
                           [(x1+x2)/2, (y1+y2)/2]], dtype=np.float32).reshape(-1, 1, 2)
        moved, valid, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, points, None)
        if moved is None or valid is None or int(valid.sum()) < 3:
            continue
        delta = np.median((moved - points).reshape(-1, 2)[valid.ravel() == 1], axis=0)
        item = copy.deepcopy(detection)
        item['bbox'] = [x1+float(delta[0]), y1+float(delta[1]), x2+float(delta[0]), y2+float(delta[1])]
        propagated.append(item)
    return propagated


def load_config(overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load((PKG / "config" / "ps2.yaml").read_text(encoding="utf-8"))
    cfg.update(overrides or {})
    return cfg


def running_intervals(ledger_data: dict, dets_by_frame: list, tool_info: dict) -> dict[str, list[tuple[float, float]]]:
    """object_id -> [(start_ms, end_ms)] when the ledger says it was running: objects identified as that tool, and
    fixed machines (objects carrying the label of a ledger tool that has running intervals)."""
    names = {t["tool_key"]: ledger.key_of(t["name"]) for t in ledger_data["tools"]}
    label_of: dict[str, str] = {}
    for dets in dets_by_frame:
        for d in dets:
            label_of.setdefault(d["object_id"], d["label"])
    out: dict[str, list[tuple[float, float]]] = {}
    for r in ledger_data["running"]:
        span = (r["start_s"] * 1000, r["end_s"] * 1000)
        for oid, label in label_of.items():
            if tool_info.get(oid, {}).get("tool_key") == r["tool_key"] or label == names.get(r["tool_key"]):
                out.setdefault(oid, []).append(span)
    return out


def run_ps2(video_path, hands, *, config_overrides: dict | None = None, work_dir: str | Path | None = None) -> dict:
    """Detect objects, decide hand-object contact, return a dict matching INTEGRATION.md (PS2Output)."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise ValueError(f"video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps > 0:
        raise ValueError(f"video has no frame rate: {video_path}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cfg = load_config(config_overrides)
    work_dir = Path(work_dir) if work_dir else ROOT / "outputs" / "tmp"
    cache_dir, tmp_dir = work_dir / "cache", work_dir / "tmp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    base_vocab = yaml.safe_load((PKG / "config" / "vocab.yaml").read_text(encoding="utf-8"))
    vocab = base_vocab
    ps1_frames = read_ps1(hands)
    clip_id = next((f["clip_id"] for f in ps1_frames if f.get("clip_id")), video_path.stem)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ps1_checks = check_ps1(ps1_frames, width, height, n_frames, fps)
    ps1_checks["constant_frame_rate"] = check_constant_frame_rate(video_path)
    hands_by_frame = load_hands(ps1_frames, width, height)
    hands_by_frame, dropped = filter_bystanders(hands_by_frame, height, cfg)   # PS1 does not filter other people's hands
    ps1_checks["bystander_hands_dropped"] = dropped
    ps1_checks["hand_tracks"] = assign_tracks(hands_by_frame, cfg)             # PS1 hand_ids are not persistent
    hand_info = [{track_key(h): (h.hand_id, h.side) for h in hands_by_frame.get(i, [])} for i in range(n_frames)]
    log.info("PS1: %d bystander hands dropped, %d hand tracks", dropped, ps1_checks["hand_tracks"])

    # Gemini watches the whole clip in the background while the detector loads
    gem = vlm.Gemini(cfg)
    job = None
    if cfg["use_ledger"] and gem.available:
        job = ledger.LedgerJob(video_path, n_frames / fps, gem, cfg, clip_id, cache_dir, tmp_dir)

    use_cuda = cfg["device"] in ("auto", "cuda", "cuda:0") and torch.cuda.is_available()
    use_mps = cfg["device"] == "auto" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    device = "cuda:0" if use_cuda else "mps" if use_mps else "cpu"
    device_name = f"cuda:0 {torch.cuda.get_device_name(0)}" if use_cuda else device
    detector = Detector(cfg, base_vocab, ROOT, device)
    log.info("device: %s", device_name)

    ledger_data, ledger_info = (job.result() if job else (None, None))   # the vocabulary needs it, so wait here
    vocab_info = {"source": "base", "entries": [], "unsure": []}
    if ledger_data:
        vocab = ledger.ledger_vocab(ledger_data, base_vocab)
        detector.set_vocab(vocab)
        vocab_info["source"] = "ledger"
    elif cfg["vocab_auto_enabled"]:
        vocab_info = vocab_auto.build(video_path, hands_by_frame, n_frames, fps, clip_id, cfg, base_vocab, detector, gem,
                                      cache_dir)
        vocab = vocab_info["vocab"]
    log.info("vocabulary: %s (%d objects)", vocab_info["source"], len(vocab["objects"]))

    motion = Motion(cfg)
    book = TrackBook(cfg)
    dets_by_frame, aux, times = [], [], []
    prev_wrist, prev_center, unreliable = {}, {}, 0
    started = time.time()
    idx = 0
    detection_interval = max(1, int(cfg.get('det_interval_frames', 1)))
    previous_gray = None
    previous_detections = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        t_ms = round(idx * 1000 / fps)
        hands_here = hands_by_frame.get(idx, [])

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if idx % detection_interval == 0:
            dets = detector.track(bgr, hands_here)
        else:
            dets = propagate_detections(previous_gray, gray, previous_detections)
        dets = drop_hand_like(dets, hands_here, cfg["hand_overlap_iou"])
        previous_gray, previous_detections = gray, copy.deepcopy(dets)
        for d in dets:
            book.add(idx, d, bgr, hands_here)

        # camera motion, ignoring hands and portable objects (they move on their own)
        skip = [h.bbox for h in hands_here] + [d["bbox"] for d in dets if d["type"] != "fixed"]
        motion.step(bgr, skip)
        unreliable += not motion.reliable
        hand_boxes = [h.bbox for h in hands_here]
        for d in dets:  # how much the inside of each box changed once camera motion is removed (hands ignored)
            d["diff"] = motion.aligned_diff(d["bbox"], hand_boxes)
        wrist = {track_key(h): h.kp[0] for h in hands_here}
        center = {d["track_id"]: ((d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2) for d in dets}
        aux.append({
            "wrist_res": {k: motion.residual(prev_wrist[k], v) for k, v in wrist.items() if k in prev_wrist},
            "obj_res": {k: motion.residual(prev_center[k], v) for k, v in center.items() if k in prev_center},
        })
        prev_wrist, prev_center = wrist, center

        dets_by_frame.append(dets)
        times.append(t_ms)
        idx += 1
    cap.release()
    detection_seconds = round(time.time() - started, 3)
    del detector                      # free the GPU for SAM 2
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
    log.info("camera motion unreliable in %d of %d frames", unreliable, idx)

    # one object_id per physical object, also after it left and re-entered the view
    contact_started = time.time()
    frame_ms = 1000 / fps
    mapping, uncertain = merge_pieces(book.pieces, set(vocab["generic"]), cfg, frame_ms)
    log.info("tracker ids: %d -> objects: %d; uncertain merges by cause: %s", len(mapping), len(set(mapping.values())),
             dict(Counter(u["why"] for u in uncertain)))
    frame_area = width * height
    apply_identities(dets_by_frame, aux, book.pieces, mapping, vocab, frame_area, cfg)
    identity_items = identity_review_items(uncertain, mapping, times, frame_ms)
    crops = best_crops(book.pieces, mapping)

    present = [set(hand_info[i]) for i in range(idx)]
    decisions = contact.decide_all(hands_by_frame, dets_by_frame, aux, fps, frame_area, cfg)
    contacts = contact.find_contacts(decisions, present, times, fps, cfg, hand_info)
    contact_seconds = round(time.time() - contact_started, 3)

    # masks for the objects in hold contacts; their polygons then replace the boxes in the fingertip distances
    sam_stats = None
    if cfg["sam2_enabled"]:
        tail = round(cfg["sam2_tail_ms"] * fps / 1000)
        targets = masks.hold_targets(contacts, dets_by_frame, idx, tail)
        sam_stats = masks.run_masks(video_path, targets, dets_by_frame, cfg, tmp_dir, device)
        log.info("SAM 2: %s", sam_stats)
        if sam_stats["frames"]:
            decisions = contact.decide_all(hands_by_frame, dets_by_frame, aux, fps, frame_area, cfg)
            contacts = contact.find_contacts(decisions, present, times, fps, cfg, hand_info)

    fusion_started = time.time()
    side_ids = {}
    for side in ("left", "right"):
        seen = Counter(h for info in hand_info for h, s in info.values() if s == side)
        if seen:
            side_ids[side] = seen.most_common(1)[0][0]

    force_status, suggestions, tool_info, extra_items = {}, {}, None, []
    if ledger_data:                   # time evidence + crop evidence must agree before a tool is "identified"
        fused = fuse.fuse(ledger_data, contacts, dets_by_frame, decisions, crops, gem, vocab, times, frame_ms,
                          frame_area, side_ids, cfg)
        force_status, suggestions, tool_info, extra_items = (fused["force_status"], fused["suggestions"],
                                                             fused["tool_info"], fused["review_items"])
    elif cfg["vlm_naming_enabled"]:   # no ledger: name the held objects from their crops, as before
        contact_ms: Counter = Counter()
        for c in contacts:
            if c["interaction"] in ("hold", "touch") and c["object_id"] in crops:
                contact_ms[c["object_id"]] += c["end_ms"] - c["start_ms"]
        asked = [oid for oid, _ in contact_ms.most_common()]  # longest contacts first: the call limit may cut the rest
        answers = vlm.name_objects(gem, {oid: crops[oid] for oid in asked}, list(vocab["objects"]), clip_id, cfg, cache_dir)
        force_status, suggestions, relabelled = vlm.apply_answers(answers, dets_by_frame, contacts, vocab, frame_area, cfg)
        log.info("VLM naming: %d objects named, %d labels changed", len(answers), len(relabelled))
    log.info("Gemini requests this clip: %d", gem.calls)

    frames = []
    for i, dets in enumerate(dets_by_frame):
        objects = []
        for d in dets:
            o = {"object_id": d["object_id"], "label": d["label"], "type": d["type"],
                 "bbox_px": [round(v, 1) for v in d["bbox"]], "confidence": round(d["conf"], 3),
                 "is_hazard": d["is_hazard"]}
            if d.get("polygon"):
                o["mask_polygon_px"] = d["polygon"]
            objects.append(o)
        frames.append({"frame_idx": i, "t_ms": times[i], "objects": objects, "hand_states": []})
    contact.apply_to_frames(frames, present, decisions, contacts, hand_info)

    tools = summary.build_tools(frames, contacts, set(vocab["generic"]), frame_ms, cfg, force_status, tool_info)
    safety_events = None
    if cfg["safety_enabled"]:
        running = running_intervals(ledger_data, dets_by_frame, tool_info or {}) if ledger_data else None
        safety_events = safety.find_events(hands_by_frame, dets_by_frame, times, fps, cfg, running)
    review_queue = summary.build_review_queue(tools, contacts, frames, identity_items, suggestions, frame_ms, cfg, extra_items)
    if vocab_info["source"] == "gemini":
        review_queue += vocab_auto.review_items(vocab_info, dets_by_frame, times, frame_ms, set(vocab["generic"]))
        summary.number_items(review_queue)
    log.info("hazard objects: %d; safety events: %s; review items: %d", sum(t["is_hazard"] for t in tools),
             "off" if safety_events is None else len(safety_events), len(review_queue))

    meta = {
        "models": [cfg["det_model"]] + ([cfg["sam2_model"]] if sam_stats and sam_stats["done"] else [])
                  + ([cfg["vlm_model"]] if gem.calls else []),
        "device": device_name,
        "frames_processed": idx,
        "detection_interval_frames": detection_interval,
        "processing_fps": round(idx / (time.time() - started), 2) if idx else 0.0,
        "vocab_size": len(vocab["objects"]) + len(vocab["generic"]),
        "vocab_source": vocab_info["source"],
        "ps1_checks": ps1_checks,
        "gemini_requests": gem.calls,
        "stage_timings_s": {"detection_and_tracking": detection_seconds,
                            "identity_and_initial_contact": contact_seconds,
                            "fusion_and_summary": round(time.time() - fusion_started, 3)},
    }
    if vocab_info["source"] == "gemini":
        meta["vocab"] = vocab_auto.meta_vocab(vocab_info, times)
    if ledger_data:
        meta["tool_ledger"] = ledger_data
    if ledger_info:
        meta["ledger_run"] = ledger_info
    if sam_stats:
        meta["sam2"] = sam_stats
    result = schema.build_output(clip_id, frames, contacts, meta, tools=tools, review_queue=review_queue,
                                 safety_events=safety_events)
    schema.validate(result)
    return result
