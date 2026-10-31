# PS2 â€” "What is the hand touching?" â€” Build plan for Claude Code (v3, final)

Read `CLAUDE.md` (rules) and `INTEGRATION.md` (exact input/output contract) first. This file says **what to build, in what order, and how to know each step is done**.

Scope: objects + handâ€“object contact only. PS1 (hands) and PS3 (task steps) belong to teammates. The frontend handles review, renaming and merging.

---

## 1. Facts this plan is built on

- **Machine:** Windows 11, NVIDIA RTX 3050 Ti Laptop (4 GB VRAM). Every frame is processed. Use `cuda:0` + FP16; CPU fallback.
- **Folders:** `D:\Bios hack_ForgeStudio\` holds `Hack-data\` (the videos, never in git) and `ps2\` (this repo; `git init` only inside `ps2\`). The code only ever receives a video path. The folder name has a space, so use `pathlib.Path` and quote shell paths.
- **Content:** carpentry, electrical and marble work. Many tools leave and re-enter the view. No gloves in the samples, but judges will run unseen videos that may have gloves.
- **Detector:** YOLOE (Ultralytics, `ultralytics>=8.4.0`). Text prompts â†’ any object by name, no training; returns boxes, masks and supports tracking. Weak on rare categories, and prompts can't express "the one being held" â€” **contact must come from hand geometry and motion, never from the detector.**
- **PS1 input:** `PS1Frame` per frame with `hand_id`, 21 `kp2d_px`, `bbox_px`, `source`, `frame_width/height` (see `INTEGRATION.md`).
- **Moving machine parts exist** in the clips â†’ safety warnings are in scope (experimental).

---

## 2. Final repo layout

```
D:\Bios hack_ForgeStudio\ps2\      <- repo root (the package lives one level deeper, in ps2\ps2\)
(videos are in D:\Bios hack_ForgeStudio\Hack-data\, outside the repo)
  CLAUDE.md
  INTEGRATION.md
  PS2_CLAUDE_CODE_PLAN.md
  README_PS2.md
  requirements.txt
  .gitignore
  ps2/
    __init__.py           # exports run_ps2
    pipeline.py           # run_ps2(): the full flow, top to bottom
    hands.py              # load PS1 data, rescale, fingertip distances, hand size, finger curl
    objects.py            # YOLOE detect + track, drop hand/glove boxes, re-entry ID merge, name suggestion
    motion.py             # camera-motion compensation; residual motion of points and boxes
    contact.py            # per-frame hand state + smoothing into contact intervals (touch/hold)
    safety.py             # hazard proximity + "is it moving" (experimental)
    summary.py            # tools[] and review_queue[]
    schema.py             # build + validate the output dict
    config/ps2.yaml       # thresholds, model names, device
    config/vocab.yaml     # object names, portable/fixed, hazard flags, ignore list
  dev/                    # never imported by ps2/; for us only
    hands_stub.py         # MediaPipe -> PS1Frame list, until real PS1 output exists
    probe_vocab.py        # first test: does YOLOE find our tools?
    render.py             # overlay video into outputs/ for checking + demo
    label_tool.py         # mark contact intervals by hand for evaluation
    evaluate.py           # compare output vs labels; prints counts
    run_one.py            # convenience: run_ps2 on one video, save JSON to outputs/
  tests/                  # synthetic data only
```

`requirements.txt`: `ultralytics>=8.4.0`, `opencv-python`, `numpy`, `pyyaml`, `mediapipe` (dev stub only), `pytest`. PyTorch with CUDA is installed separately (T0.1).

---

## 3. Task checklist

Rule: finish a phase, run `pytest`, report results with numbers, then stop for review.

### Phase 0 â€” Setup and first test (~1 hour)

- [x] **T0.1 Environment.** Python 3.11 venv. Run `nvidia-smi` to read the CUDA version, then install the matching CUDA build of PyTorch using the command from pytorch.org's selector (Windows, pip, CUDA). Then `pip install -r requirements.txt`.
  *Done when:* `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` prints `True` and the RTX 3050 Ti.
- [x] **T0.2 Repo skeleton.** Create the layout above. Copy the provided `vocab.yaml` into `ps2/config/`. Confirm `.gitignore` is in place.
  *Done when:* `git status` shows only code and config.
- [x] **T0.3 Weights.** From repo root, run one text-prompted prediction with `yoloe-26s-seg.pt` so the weights and the ~254 MB text encoder download into the repo root (both gitignored). Also download `yoloe-26s-seg-pf.pt` (prompt-free, used for name suggestions).
- [x] **T0.4 `dev/probe_vocab.py <video_path>`.** Sample 50 frames evenly across the video. Print, per vocab label: detection count and mean confidence. Also print the top 30 labels the prompt-free model sees. Save crops to `outputs/probe/` for me to eyeball.
  *Done when:* I can tell which of our tools are found. Run it on all 3 sample videos.
  **Decision:** prompts that never fire â†’ reword them in `vocab.yaml`. Tools that are clearly visible but still missed â†’ they will fall to the generic "tool" label and get a name suggestion (T4.3).
- [x] **T0.5 Speed check.** Time YOLOE-26s with tracking at `imgsz` 640 on 300 consecutive frames on the GPU, FP16. Watch VRAM in `nvidia-smi`. If it's too slow or VRAM is over ~3.5 GB, try 26n.
  *Done when:* we record frames/second and pick the model. Target: a 3-minute clip in under ~10 minutes.

### Phase 1 â€” Minimum version end to end (~2 hours)

- [x] **T1.1 `dev/hands_stub.py`.** MediaPipe Hand Landmarker (VIDEO mode, 2 hands) over every frame â†’ list of `PS1Frame` dicts exactly as in `INTEGRATION.md` (`hand_id` = `"h_left"`/`"h_right"` from side, `source:"detected"`, `kp3d_hand_m` from world landmarks or null). Save to `outputs/<clip>/ps1_stub.jsonl`.
- [x] **T1.2 `hands.py`.**
  - `load_hands(hands) -> dict[int, list[Hand]]`: accepts `.jsonl` path, `.json` path, or `list[dict]`. Validates 21 keypoints. Rescales coordinates if `frame_width/height` differ from the video.
  - `hand_scale(kp) = distance(kp[0], kp[9])` (wrist to middle knuckle).
  - `tip_distances(kp, box_or_polygon) -> list[float]` for fingertips 4, 8, 12, 16, 20; 0 if inside; divided by hand scale.
  - `finger_curl(kp) -> float` in [0,1] from the bend at the middle joints (high = gripping).
- [x] **T1.3 `objects.py` (detection + tracking).** Build prompts from `vocab.yaml` (`objects` + `generic`), `set_classes` once. Each frame: `model.track(frame, persist=True, tracker="botsort.yaml", half=True, device=0, imgsz=640, conf=det_conf, verbose=False)`. Map the matched prompt back to its vocab key, `type`, `is_hazard`. Remove detections whose label is in `ignore`, or that overlap a PS1 hand box with IoU > `hand_overlap_iou`. `object_id = "obj_<track id>"` for now.
- [x] **T1.4 `contact.py` v0.** Brief's rule: contact if â‰¥ 2 fingertips are inside the object box padded by 15% of hand size; pick the object with the most fingertips inside. Smoothing: start an interval after `on_frames` consecutive contact frames, end it after `off_frames` non-contact frames.
- [x] **T1.5 `schema.py`.** Build the dict exactly per `INTEGRATION.md`. `validate(result)` checks: enums, `"none"` â†’ `object_id` null, `start_ms < end_ms`, every referenced `object_id` exists in frames, required fields present. `run_ps2` calls it before returning.
- [x] **T1.6 `pipeline.py` + `__init__.py`.** `run_ps2(video_path, hands)`: read config, load hands, loop over every frame, collect detections and hand states, build intervals, validate, return.
- [x] **T1.7 `dev/run_one.py` and `dev/render.py`.** Run on one video; draw boxes with id + label, hand state near the wrist, and a timeline strip; write `outputs/<clip>/overlay.mp4`.
  *Done when:* one sample video gives a valid dict and a watchable overlay. This meets the brief's minimum for PS2.

### Phase 2 â€” Reliable contact decisions (~4 hours)

- [x] **T2.1 `motion.py`.** Per frame pair: downscale to 480 px wide grayscale, track background corners (`goodFeaturesToTrack` + `calcOpticalFlowPyrLK`) excluding hand and object boxes, fit `cv2.estimateAffinePartial2D` with RANSAC. Provide `residual_motion(prev_point, point)` = movement with head motion removed, and `aligned_diff(box)` = how much the inside of a box changed after aligning the previous frame (used by safety).
- [x] **T2.2 Contact score per hand per frame.**
  - Candidates: objects with â‰¥ `min_tips` fingertips within `tip_near_norm` hand-sizes (use the mask polygon when available, else the box).
  - Several candidates: most fingertips near, then portable before fixed, then smaller box.
  - Confidence (weights in config): fingertips near, detector confidence, finger curl, co-motion (T2.3); Ã— `interpolated_factor` if PS1 `source` is `"interpolated"` (+ reason `hand_interpolated`).
  - `"unknown"` with reason when: `hand_not_detected` (hand vanished during an ongoing contact, kept open up to `max_hand_gap_ms`), `grasp_pose_no_object`, `low_object_confidence`, `ambiguous_candidates`, `fixed_contact_unverified` (hand over a big fixed box but not still relative to the scene).
- [x] **T2.3 Touch vs hold.** Over `comotion_window_ms`, compare residual motion of the object centre and the wrist: object moves with hand â†’ `hold`; hand moves, object doesn't â†’ `touch`; hand nearly still â†’ decide from curl + fingertips, else `interaction:"unknown"`. A fixed-type object that moves with the hand â†’ treat as portable, reason `type_overridden_by_motion`.
- [x] **T2.4 Smoothing v1.** Merge same-hand, same-object gaps < `merge_gap_ms`; drop intervals < `min_interval_ms`; interval confidence = mean of frames; `review_needed` = confidence < `review_threshold` or reasons on > 30% of its frames.
- [x] **T2.5 Glove robustness (samples have no gloves, judges' videos may).** Contact logic uses PS1 keypoints only, so it doesn't depend on skin colour. Make sure glove/hand detections are removed near hands (`ignore` list + IoU). Unit-test with a synthetic "glove" box on the hand.
  *Done when:* overlay shows far less flicker than v0, every `"unknown"` frame has a reason.

### Phase 3 â€” Same object after leaving and re-entering view (~2 hours)

Done once, after all frames are processed.

- [x] **T3.1 Track pieces.** Per tracker ID: label, frames seen, and a colour fingerprint (HSV histogram) built only from frames where a hand box covers less than `clean_crop_max_overlap` of the object box. Crops stay in memory. A piece with no clean frame has no fingerprint.
- [x] **T3.2 Merge.** Two pieces may be the same object only if: they are never visible in the same frame; labels match (or one is generic); fingerprint similarity â‰¥ `reid_min_sim`. Merge most-similar pairs first. Pieces without a fingerprint may merge on label + time order only, always with a `review_queue` item `identity_uncertain`. Two near-equal candidates â†’ merge with the best, add `identity_uncertain`.
- [x] **T3.3 Rewrite IDs.** Replace tracker IDs with final IDs (`obj_1`, `obj_2`, â€¦ in order of first appearance) everywhere: frames, contacts.
- [x] **T3.4 Tests:** re-entry merges; pieces visible together never merge; near-tie creates a review item.
  *Done when:* on the sample videos, re-entering tools keep one ID (checked on the overlay).

### Phase 4 â€” Tools summary, review list, name suggestions (~2 hours)

- [x] **T4.1 `summary.py` â†’ `tools[]`** exactly per `INTEGRATION.md`: visible/held time, `use_count` (hold intervals with gaps < `use_merge_gap_ms` merged), `appearances`, `label_status` (`unidentified` for generic labels or mean confidence < `label_min_conf`).
- [x] **T4.2 `review_queue[]`:** one item per unidentified object, `review_needed` contact, `identity_uncertain` merge, and unknown stretch > 1 s (`long_unknown`).
- [x] **T4.3 Name suggestion for unidentified objects.** For each unidentified object, take its single clearest crop and run the prompt-free YOLOE model on it once; put the top name in `suggested_label` (never as the label). Load the prompt-free model only if unidentified objects exist.

### Phase 5 â€” Safety warnings (experimental, ~2 hours)

- [x] **T5.1 `safety.py`.** For objects with `is_hazard`: "moving" = `aligned_diff` inside its box above `hazard_motion_thresh` for most of a short window (catches spinning blades and running grinders). Warning interval when any fingertip is within `safety_near_norm` hand-sizes of a moving hazard for â‰¥ `safety_min_ms`.
- [x] **T5.2 Output** `safety_events` per `INTEGRATION.md`; reasons always include `2d_distance_only` and `hazard_motion_estimated`.
  *Done when:* on clips with moving machine parts, warnings appear where a human agrees the hand is close to a running part. Report false alarms honestly.

### Phase 6 â€” Measure (~2â€“3 hours; can run alongside 3â€“5)

- [x] **T6.1 `dev/label_tool.py`.** OpenCV player: scrub with arrow keys; mark start/end of a contact, hand (`left`/`right`), state (portable/fixed), interaction (touch/hold), object name. Saves to `labels/` (gitignored).
- [ ] **T6.2 Label** ~30â€“60 contact intervals across 2 sample videos. Video A = tuning, video B = reporting only.
- [x] **T6.3 `dev/evaluate.py`.** Match predicted vs labelled contacts (same hand, time overlap â‰¥ 50%): precision, recall, median start error and end error in ms, naming accuracy, re-entry ID accuracy. Print counts like "17/22".
- [ ] **T6.4** Before/after table (Phase 1 rule vs final) into `README_PS2.md`, numbers only.

### Phase 7 â€” Masks (bonus, after Phase 6 numbers exist)

- [ ] **T7.1 YOLOE masks.** For objects in contact, convert the YOLOE mask to a simplified polygon (`cv2.approxPolyDP`, â‰¤ 40 points) â†’ `mask_polygon_px`; use it in fingertip distances. Re-run evaluation; keep it only if numbers don't get worse.
- [ ] **T7.2 SAM 2 (optional).** Try `sam2.1_t.pt` via Ultralytics on the GPU for contact intervals only (box prompt at interval start, propagate through the interval). Controlled by `sam2_enabled` in config, default `false`. Keep only if it fits in 4 GB VRAM, runs in acceptable time and improves numbers.

### Phase 8 â€” Integration and handover

- [ ] **T8.1** Run with the PS1 teammate's real output; fix mismatches inside `hands.py` only.
- [ ] **T8.2** Fill `meta` (models, device, frames, FPS, vocab size).
- [ ] **T8.3** Run all sample videos; hand the backend teammate the function + one example output (outside git).
- [ ] **T8.4** `README_PS2.md`: setup (including CUDA PyTorch), the one function, models and libraries, GPU used, metrics, known failure cases (text only).
- [ ] **T8.5** `git ls-files` shows no media, arrays, weights or JSONL.

---

## 4. Tests (synthetic only) â€” `tests/`

- `test_schema.py` â€” enums, null rules, ordering, referenced ids exist, optional fields can be absent.
- `test_hands.py` â€” loads jsonl/json/list; rescales coordinates; tip inside box â†’ 0; scale normalisation; curl flat vs fist.
- `test_contact.py` â€” 2 tips â†’ candidate; 1 tip â†’ none; portable preferred; hand missing mid-contact â†’ unknown + reason; interpolated hand lowers confidence.
- `test_smoothing.py` â€” 1-frame blip dropped; short gap merged; ms boundaries correct.
- `test_motion.py` â€” object moving with hand â†’ hold; static object â†’ touch; camera-only motion â†’ no hold.
- `test_reid.py` â€” re-entry merge; simultaneous pieces never merge; tie â†’ review item; no-fingerprint merge â†’ review item.
- `test_safety.py` â€” fingertip near moving hazard â†’ event; near still hazard â†’ no event.
- `test_filter.py` â€” glove box over the hand is removed.

---

## 5. `ps2/config/ps2.yaml` starting values (tune on video A only)

```yaml
device: auto              # cuda:0 if available, else cpu
half: true
det_model: yoloe-26s-seg.pt
pf_model: yoloe-26s-seg-pf.pt
imgsz: 640
det_conf: 0.15
hand_overlap_iou: 0.6
max_hand_gap_ms: 300
interpolated_factor: 0.7
tip_near_norm: 0.35
min_tips: 2
on_frames: 3
off_frames: 3
merge_gap_ms: 300
min_interval_ms: 200
comotion_window_ms: 500
min_hand_move_norm: 0.15
review_threshold: 0.5
label_min_conf: 0.35
use_merge_gap_ms: 1000
reid_min_sim: 0.7
clean_crop_max_overlap: 0.1
hazard_motion_thresh: 12.0
safety_near_norm: 0.5
safety_min_ms: 300
sam2_enabled: false
```
