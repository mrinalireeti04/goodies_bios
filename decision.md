# PS2 decisions and status (kept up to date by Claude; read this instead of the chat)

## Status
- Phases 0-5: done. Phase 6: tools built (label tool, evaluator, baseline); **T6.2 labelling is yours** (I do not label my own output). Phase 7 (masks): skipped, no labels to check it against. Phase 8: README, meta, 4-clip runs done; real PS1 data not available yet.
- Phase 7 Part A: items 1-9 built and run on clips A and B (see the Phase 7 log); the clip-A check list exists but is degraded (Groq daily limit). Waiting for you before Part B.
- Phase 6b: items 2-10 done; **item 1 (updated vocab.yaml) still blocked**: the file on disk is unchanged.
- Phase 6c Part 1 (Gemini check, per-video vocabulary, held-object naming, review-queue rule, PS1 assumptions): done, results below.
- Phase 6c Part 2 (glove test, verify tool, team overlay, README): done except the items marked below. Nothing is committed.
- **Open problem, needs your decision:** with the per-video vocabulary, recall drops sharply (see "Part 1 results").
- 130 tests pass.

## Phase 7 log (Part A = items 1-9, then stop; no git add/commit/push until you say "commit")
| Item | Done / result |
|---|---|
| 1 | `dev/label_tool.py` rewritten: raw video only; on-screen key map; contacts with start/end, hand (worker's own), state, interaction, object name, instance tag, unsure flag; unsure time spans; typed answers inside the window (Enter = previous answer, Tab completes names; instance tag suggested from earlier tags of the same object); +-1 frame / +-1 s / +-5 s keys with fallback letter keys; undo of the last added row; time display; autosave after every change to `labels/<clip>_contacts.csv` (written to a temp file then renamed, so a crash never leaves half a file) plus `labels/<clip>_contacts.pos` (last position) for resuming. A contact flagged unsure is scored like an unsure span (excluded). 11 tests on the file handling, the question flow and the time parser; the window and keys are untested by me. The old tool's `labels/<clip>.json` format is gone; the evaluator will read the new CSV. |
| 2 | `ps2/ledger.py` (new): Gemini Files API upload with our urllib client (resumable; no new dependency), reduced temporary copy via ffmpeg (<= 720 p, no audio, CRF 30; falls back to the original), one `generateContent` call per window with `video_metadata` fps 2 (and start/end offsets only when the clip is longer than 300 s), temperature 0, strict JSON schema, `MM:SS` times, validation (bad/outside intervals dropped, names normalised, <= 30 tools), cache in `cache/ledger_<clip>.json` keyed by clip id + file size + modification time, upload and temp copy deleted in a `finally` block and recorded in `meta.ledger_run`. Started in a background thread at the start of `run_ps2`. **The ledger gives the detector its vocabulary, so the local detection pass has to wait for it: the thread overlaps only with the detector loading and the PS1 preparation, not with the GPU pass itself.** Live test on clip A: 4 tools, 10 uses, 2 unsure, 1 request, 95 s, upload and copy deleted. The earlier `vocab_auto` is kept but off (`vocab_auto_enabled: false`). |
| 3 | `ps2/fuse.py` (new): time evidence + crop evidence (8 crops per request) -> identified only when both name the same tool; single evidence / conflict -> unidentified with `tool_single_evidence` / `tool_evidence_conflict`; fixed objects never relabelled; same tool and never visible together -> merged; ledger uses without a local contact -> `tool_use_ai_only`; ledger unsure -> `vlm_unsure_object`. New optional `tools[]` fields (`tool_key`, `id_confidence`, `evidence`, `idle_visible_ms`, `uses`) and `meta.tool_ledger`; INTEGRATION.md updated. One request budget per clip (ledger first, then crops by longest contact). Hand agreement uses the PS1 `side`, which on this footage comes from the MediaPipe stub whose left/right labels are unreliable. |
| 4 | Safety: a hazard is moving if inside a ledger `running` interval OR local motion says so; event reasons say which (`hazard_running_from_ledger`, `hazard_motion_estimated`) plus `2d_distance_only`. Hazard objects = vocabulary hazard flag (ledger: powered; fixed machines only if the ledger lists them as running) and a box under 15% of the frame. Events use hand tracks, with the PS1 id seen most often. |
| 5 | `ps2/masks.py` (new): SAM 2.1 tiny via Ultralytics `SAM2VideoPredictor`, only for hold contacts of portable objects, longest first, box prompt at the first frame of the contact, tracked through the contact plus 0.5 s on a temporary 720 p clip in `outputs/tmp` (deleted after), polygon <= 40 points written to `mask_polygon_px` and used in the fingertip distances (contacts are decided again). YOLOE is deleted and the GPU cache emptied first. Time budget `sam2_budget_s` 300 (longest contacts first, the rest skipped and counted in `meta.sam2.skipped_for_time`). Test on 3 hold contacts of clip B: 47 s, peak 0.53 GB VRAM. First use downloads `sam2.1_t.pt` (75 MB) into the repo root (gitignored). |
| 6 | PS1 contract (teammate's answers): `hand_id` not persistent -> internal hand tracks (side, box IoU, wrist distance; a track survives 10 missing frames), contacts per track, `contacts[].hand_id` = most common PS1 id; frames keep their own PS1 ids. No bystander filter in PS1 -> at most 2 likely wearer hands per frame (smaller than half the largest or wrist in the top 35% dropped), drops counted in `meta.ps1_checks.bystander_hands_dropped`. `ffprobe` constant-frame-rate check (warn only). INTEGRATION.md section 5 updated. Unit tests for tracks and the bystander filter. |
| Plan change | You replaced the manual-labelling evaluation with "AI-proposed, human-verified": `dev/independent_proposals.py` (a second model proposes contacts), `dev/groq_judge.py` (Groq `qwen/qwen3.8-27b` judges each item from up to 3 frames), `dev/checklist.py` (pool + about 40 items: tier 1 complete disagreement, tier 2 other disagreement, tier 3 random agreements, fixed seed 7), `dev/verify_contacts.py --checklist` (shows source only), `dev/evaluate.py` (stratified estimates with bootstrap 95% CIs, agreement shown separately and labelled as agreement). The label tool (item 1) stays for optional exact labelling of a window. Tiers that do not fit into the list are sampled at random so the stratified estimates stay valid. |
| 8 (runs) | One at a time, A then B, `outputs/<clip>/<variant>.json`; `v0` = the Phase 1 rule re-applied to the `base` run's detections (`dev/baseline_v0.py <clip> base`). Contacts / holds: A v0 97 / 0, base 48 / 16, ledger 15 / 7, ledger_sam 15 / 7; B v0 420 / 0, base 95 / 45, ledger 93 / 45, ledger_sam 87 / 38. Speed (fps, runs not sharing the GPU with another PS2 run except where noted): A base 12.2, ledger 12.5, ledger_sam 8.5; B base 8.8, ledger 9.8, ledger_sam 6.9 (B's `ledger` runs of clips A and B overlapped with other jobs, so fps are only indicative). Gemini requests: A ledger 3, ledger_sam 2 (ledger came from cache); B ledger 6, ledger_sam 8 (the ledger itself: 1 request, 101 s). **Upload deleted: yes** for the one live upload during these runs (B ledger_sam) and for every other upload (A's earlier live test, both independent-proposal uploads); A's ledger came from cache (no upload). SAM 2: A 1 contact 6 s; B 11 contacts, 313 frames, 81 s (about 1.3 min added), peak 0.53 GB VRAM: well under the 5-minute limit, so no contact was dropped. |
| 8 (what the runs show) | **Clip A:** the ledger named 4 tools (a workpiece, a glue bottle, a pencil, a pneumatic nail gun); the detector found few of them, so contacts fall from 48 to 15, no tool reached "identified with high confidence", no hazards, no safety events. **Clip B** (a precision-workshop clip): ledger 5 tools, 68 uses, 1 running interval; 8 objects identified with high confidence (loupe, dop stick, tang, digital gauge), hazards 85 -> 12 and safety events 69 -> 24 against the base vocabulary, review items 267 -> 440 (mostly `identity_uncertain` 329, which the base run mostly did not have to explain). Neither run says anything about accuracy: that is what the check list is for. |
| 9 | `dev/render.py` re-rendered from the ledger variant for A (highlight 113-173 s, 24.5 MB; weak 157-177 s, 8.1 MB) and B (highlight 34-94 s, 20.8 MB, 17 hold contacts; weak 114-134 s, 7.5 MB); the top-right panel lists tools with `ok` / `[?]` status, held time and use count. |
| Groq limits (measured) | `qwen/qwen3.8-27b` exists on this key. Real free-tier limits from the response headers: 1,000 requests/day and **8,000 tokens per minute**; a request with 3 images used about 2,500 tokens, so only about 3 such requests fit per minute. The first clip-A attempt judged 23 of 55 items (about 57k tokens), then got a 429 with `retry-after` 724 s; the daily token limit is not in the headers, so it is unknown. My first assumption (a wait longer than 65 s means the daily limit) was wrong for this case. Now: waits up to `groq_max_wait_s` (900 s) are slept out, after the first long wait the judge steps down to 1 frame per item (each cached answer records how many frames it saw), a longer wait ends judging with cached answers kept, and `groq_max_tokens_per_day: 0` means "unknown, rely on the API's answers". The first `checklist.json` for clip A was built with only 22 of 55 items judged, so it was NOT ready (all items fell into tier 2); it is being rebuilt. |
| Check list A (status) | Built, but **degraded**: Groq stopped after 24 requests / 59,247 tokens (a long wait of 979 s, over `groq_max_wait_s`), so only 23 of the 55 pooled contacts have a Groq vote (22 yes, 3 no among the judged). With so few votes every item falls into tier 2, so the 40-item list is a plain random sample (seed 7) of the 55 pooled contacts (28 Gemini-only, 11 system-only, 1 both in the list) rather than a disagreement-focused one. The sample is still valid for the precision estimates of the system and of the independent model, but the three-way tiering and the Groq agreement numbers are thin. Fix for tomorrow: the judge now sends 512 px frames (`groq_image_px`), plans for a 55,000-token daily cap (about 650 tokens per 1-frame item, so all 55 items fit in one day), and `dev/checklist.py` can be re-run to finish the judging; a rebuilt list will contain different items, and only answers on items that are in the list at evaluation time are counted. |
| Independent model | This key cannot use Pro models: `gemini-2.5-pro` returns 404 ("not available to new users") and `gemini-3.1-pro-preview` returns 429 (no quota). The independent proposals therefore fall back to a different full Flash model (never the pipeline's own `gemini-3.6-flash` / `gemini-3.5-flash-lite`); the model actually used is stored in `independent.json`. |

## Phase 6c Part 2 log
| Item | Done / result |
|---|---|
| 6 | `dev/glove_test.py`: paints the convex hull of each hand's 21 keypoints (flat blue + noise, sigma 6) frame by frame in memory by wrapping `cv2.VideoCapture` (so `run_ps2` and the vocabulary sampling both see gloved frames), runs `run_ps2` with the same hand data, compares with the normal run, writes only `outputs/<clip>/glove_test.json`. 2 tests. Result: see "Glove test" below. |
| 6 result | Clip A, per-video vocabulary on, same hand data: normal run 17 contacts (6 holds, 6 touches, 10 objects with a contact), gloved run 22 contacts (8 holds, 11 touches, 13 objects); detections removed by the hand filter 0 vs 8; held objects named by Gemini 9 vs 2 (the second run ran into 429 rate limits and hit the 20-request cap, so the naming difference is API noise, not gloves). Contacts did not get worse, which is expected because contact uses keypoints, not appearance. Limits: a painted hull is not a real glove, one clip, and the vocabulary was the small Gemini one, which limits what can be detected at all. |
| 7 | `dev/verify_contacts.py <video> <ps2.json> [--safety]`: loops each predicted contact/safety event from 0.5 s before to 0.5 s after (max 10 s) with hand skeleton and object box; keys y/n/o/i/s/q; `o` asks for the right name; saves to `labels/verify_<clip>.csv` at once, resumable (key = kind+hand+start+end, so renumbered ids do not matter); prints precision / naming / interaction counts. File handling and statistics tested with synthetic data (8 tests); **the window and keys are untested, you test them**. Found and fixed a bug on the way: a contact session used to overwrite the safety answers in the same file. |
| 8 | `dev/render.py` rewritten as specified: skeletons per hand_id, only objects within 2 hand-sizes or in a contact (held = thick box + mask polygon if present), label `obj_id label conf` + `[?]` / `(AI)`, wrist badges, per-hand timeline (grey/blue/green/orange/yellow) with cursor, top-right panel (held now + clip totals), red EXPERIMENTAL safety (only if `safety_enabled`), footer (clip id, models, vocab_source, hands stub/PS1, fps). Default run writes `highlight_<clip>.mp4` (60 s with the most hold time) and `weak_<clip>.mp4` (20 s with the most unknown hand-frames); `--clip START END` for any part; 1280x720, H.264 via ffmpeg. The old whole-video 960 px overlay (400 MB) is gone. Clip B: highlight 19-79 s = 20.2 MB, weak 114-134 s = 7.2 MB; one frame of the highlight checked by eye. |
| 9 | README_PS2.md rewritten: tech stack (Gemini, ffmpeg, Claude Code as coding assistant), the per-video vocabulary and its guardrails, the glove test result, the known weakness of the vocabulary step, limitations; no images, no references to missing sections. |
| 10 | `git status` and the file list shown in chat; nothing staged or committed. |

## Phase 6c log (Part 1 = items 1-5; Part 2 waits for your "continue")
| Item | Done / result |
|---|---|
| 1 | `GEMINI_API_KEY` is a Windows *user* environment variable; it was not in my shell, so my test commands read it from the user environment for that process only (never printed or written). `dev/gemini_models.py` calls ListModels: 46 models support generateContent. **My old default `gemini-2.5-flash` returns 404 ("no longer available to new users")**, so the Phase 6b naming step would have been silently skipped. Chosen: **`gemini-3.6-flash`** (answered in about 6 s). The newest flash models (`gemini-3.7-flash`, `gemini-3.8-flash`) were slow (one call took 79 s) or returned 503 "high demand"; with default thinking, tokens were also used up before the answer appeared, so I set `vlm_thinking_level: low` and generous token limits. Fallback `vlm_fallback_model: gemini-3.5-flash-lite` (4 s, no thinking), one try if the main model keeps failing. New keys: `vlm_max_calls_per_clip` 20 (counts every HTTP request, retries included, vocabulary and naming together), `vlm_retries` 3, `vlm_backoff_s` 2 (doubles), `vlm_timeout_s` 60. Retries on 429/500/502/503/504 and timeouts; 401/403/404 stop all further calls for the clip; each problem is logged once; without a key everything runs on the base vocabulary. |
| 2 | New `ps2/vocab_auto.py`, all guardrails implemented as specified (24 frames preferring hands, 3 interleaved batches of 8, inline JPEG <= 768 px, frame index labelled in the prompt, strict JSON schema, temperature 0, normalise/merge, >= 2 batches, YOLOE cross-check -> confirmed/vlm_only, hazard rule, cap 40, final = Gemini objects + base fixed non-hazard surfaces + generic + ignore, JSON-only cache keyed by clip id + frame hash). My interpretations: (a) two entries merge only if names are equal or each lists the other as a synonym (a one-sided synonym would wrongly merge e.g. a tool with the machine that is built around it); (b) hazard = powered in at least half of the mentions AND cited in >= 2 frames; (c) "base fixed surfaces" = base entries of type fixed without a hazard flag (the fixed hazard machines are specific tools, so they only come from Gemini); (d) a clip needs 2 of the 3 answers to be usable, else the base vocabulary is used. New keys `vocab_frames`, `vocab_frame_max_px`, `vocab_min_batches`, `vocab_max_objects`. Output: `meta.vocab_source`, `meta.vocab`, review reasons `vocab_unverified` and `vlm_unsure_object` (INTEGRATION.md updated). |
| 3 | Naming now goes through the same client (shared call budget). Objects with hold/touch contacts are asked longest-contact first, so the call limit cuts the least important ones. |
| 4 | `identity_uncertain` items only for objects with a `tools[]` row AND at least one contact. `safety_enabled` (default true); false = no `safety_events` field. |
| 5 | INTEGRATION.md section 5 "Assumptions about PS1 data" (your 7 points); `hands.check_ps1` checks missing frames, t_ms mismatch > 1 frame, frame_idx beyond the video, keypoints out of bounds, frame size mismatch, more than 4 hand_ids and a keypoint-order plausibility test, logs a warning for each and writes `meta.ps1_checks`. Never raises. Not checkable: "own hands only" and "not mirrored". |
| Part 1 results (A / B, base vocab -> Gemini vocab) | Contacts (holds) 48 (14) -> **17 (6)** / 126 (56) -> **90 (54)**; tools 213 -> 49 / 326 -> 116; hazard tools 96 -> **0** / 94 -> **0**; safety events 11 -> **0** / 78 -> **0**. Review items with the new "has a contact" rule: A 206 -> 83 (base vocab) -> 46 (Gemini), B 981 -> 332 -> 247. **The per-video vocabulary, as specified, cuts recall badly**: the 2-of-3-batches rule left 2 objects on A and 1 on B, and the base tool list is dropped. Held-object naming on A: 7 of 9 answered, 4 relabelled (vise->nail_gun, wall->nail_gun, workbench->machine, tool->nail_gun; some look wrong because the model must choose from a tiny label list); on B 2 of 28 answered because the 20-request cap was used up by 429/503 retries. Live Gemini was slow and overloaded during these runs (12 requests for B's vocabulary step alone, 116 s). |
| Fix found while reading the table | A review item could get the reason `label_from_vlm` (an information code, not a problem). Review items now skip it. |

## Phase 6b log (one line per numbered item, newest requirements first read from your message)
| Item | Done / result |
|---|---|
| 0 | Reported the 4 background runs in chat: after decision 38 the review queue had 177 / 678 / 222 / 57 items (A / B / SKL-005 / 000060-006); speed 6.4-11.7 fps because runs shared the GPU. Table is in "Final 4-clip run" below. |
| 1 | **Blocked.** `ps2/config/vocab.yaml` on disk was last written when I moved it in Phase 0 (17:39); no updated vocab exists anywhere I searched (repo, Downloads, Desktop, Documents, `D:\`). Re-ran `probe_vocab.py` with the vocab that IS on disk as the "before": total detections (body/PPE names excluded) 416 (SGF-021), 331 (SKL-005), 834 (000003-002), saved as `outputs/probe/<clip>/summary_before.json`. `probe_vocab.py` now takes `[tag] [vocab_path]` and no longer loads the prompt-free model. To finish: put the new file at `ps2/config/vocab.yaml` and run `dev/probe_vocab.py <video> after`. |
| 2 | Hand-anchored detection added (`Detector.detect/track` in `objects.py`): one batched YOLOE call on the frame + a square crop around every PS1 hand (`hand_crop_scale` 2.5), boxes mapped back, class-aware NMS (`nms_iou` 0.5, more confident box stays). Because `model.track` cannot take merged detections, the Ultralytics BoT-SORT tracker is now fed directly with the merged boxes (same `botsort.yaml`). Config keys `hand_crop_enabled`, `hand_crop_scale`, `nms_iou`. Numbers: see table at the end of this log. |
| 3 | Box sanity: `max_portable_frac` 0.25 (portable-type box above it gets the generic label `tool`); `is_hazard` needs vocab hazard AND box < `big_box_frac` (0.15) of the frame; a tool is a hazard if flagged in more than half of its frames; safety events use only those. |
| 4 | Prompt-free model removed: `suggest_names` and `pf_model` deleted, `suggested_label` is null unless item 5 provides a name. The `-pf` weights file is still on disk (gitignored); I did not delete it. |
| 5 | Gemini naming added (`ps2/vlm.py`, new module because it is a new feature; stdlib `urllib`, no new dependency). Model name is config key `vlm_model` (default `gemini-2.5-flash`, **not verified: there is no `GEMINI_API_KEY` in this environment, so no live call has been made**). Tested only with a fake transport (11 tests). CLAUDE.md and INTEGRATION.md updated. Cache: `cache/<clip>_<object>.json`, valid only if the crop hash matches (object ids can change between runs). |
| 6 | `low_contact_confidence` added: contacts below `review_threshold` get it in `reasons`, and the review item uses it. INTEGRATION.md updated. |
| 8 | Removed descriptions of footage content from this file. |
| 7 | README_PS2.md rewritten: removed the references to a results table and sections that did not exist; it now says no accuracy has been measured. |
| 10 | `git status` and the file list were shown in chat. Repo is initialised; nothing staged, nothing committed. 41 files would be added (code, config, tests, docs); 0 media, arrays, weights or JSONL. |
| 9 | `dev/render.py <video> --clip START END` writes a 1280x720 `clip_<start>_<end>.mp4`; if `ffmpeg` is installed it is re-encoded to H.264 (a 10 s test clip is 3.1 MB, about 37 MB for 2 minutes). |

### Phase 6b before/after on clip MTL-SGF00106-021 (one run at a time, same hand data)
| | No hand crops (A) | **Final: crops + rules (B)** | Crops, no size rules (C) |
|---|---|---|---|
| `grasp_pose_no_object` frames | 530 | **319** | n/a |
| Contacts (hold / touch / unknown) | 30 (6 / 20 / 4) | **48 (14 / 23 / 11)** | 53 |
| Unknown hand-frames | 671 | **524** | n/a |
| Objects / tools[] rows | 581 / 183 | **705 / 213** | 703 / n/a |
| Hazard tools (is_hazard) | 83 | **96** | 96 |
| Safety events | 3 | **11** | 12 |
| review_queue items | 169 | **206** | 208 |
| Speed | 13.75 fps | **13.22 fps** | 13.72 fps |
- Hand crops: fewer unexplained grasp poses (530 -> 319), more contacts and more holds (6 -> 14), 4% slower. They also create more objects and more safety events (3 -> 11); I have not checked whether the new events are real.
- Size rules (B vs C): almost no effect on this clip (hazard tools 96 in both, safety events 11 vs 12, generic-labelled objects 3 vs 2). Run C also switches off `big_box_frac` for the contact rule, so its contact count (53) is not a clean comparison. The differences are within run-to-run variation. 96 hazard objects remain because the vocabulary flags many common tool names as hazards.
- Item 1 (vocab before/after) could not be measured, see the log above.
- Totals for the 100 tests: all pass.

## Decisions made without asking (small, reversible — tell me if you disagree)
| # | Decision | Why |
|---|---|---|
| 1 | PyTorch is the **cu128** build (torch 2.11) | Driver reports CUDA 13.1; cu128 wheels run on it. |
| 2 | Model = **yoloe-26s-seg**, FP16, imgsz 640 | Measured 17.8 fps on 300 frames (about 5 min per 3-min clip). |
| 3 | MediaPipe model file `hand_landmarker.task` (7.8 MB) downloaded from Google's mediapipe-models bucket into `outputs/models/` (gitignored) | The MediaPipe Tasks API needs it; plan says the stub uses MediaPipe. Dev stub only. |
| 4 | Added `ps2/config/botsort.yaml` (copy of Ultralytics default with `track_high_thresh` and `new_track_thresh` lowered to 0.15) | Default 0.25 would never start a track for detections in the 0.15-0.25 range, and most of our detections are there (probe). Plan says `tracker="botsort.yaml"`; this is the same tracker with our thresholds. |
| 5 | Added config key `v0_pad_norm: 0.15` | T1.4 rule ("box padded by 15% of hand size") needs a threshold, and CLAUDE.md says thresholds live in `ps2.yaml`. |
| 6 | Phase 1 contacts have `interaction: "unknown"` | Touch vs hold needs motion (Phase 2). Not inventing it. |
| 7 | Frame `hand_states` are rewritten from the smoothed contact intervals | So frames, contacts and overlay always agree. Raw per-frame picks are only used to build intervals. |
| 8 | `t_ms = round(frame_idx * 1000 / fps)` everywhere (stub and PS2) | Same formula on both sides, so PS1 and PS2 times match. |
| 9 | Objects without a tracker ID are dropped | We cannot reference them by `object_id`. |
| 10 | In v0 an object of type `unknown` gives hand state `unknown`, no interval | Schema only allows portable/fixed for contacts. |

## Phase 2 decisions (added without asking)
| # | Decision | Why |
|---|---|---|
| 11 | Contact is decided after all frames are read (per-frame observe -> motion over a centred 500 ms window -> decide -> smooth) | Co-motion needs frames before and after. |
| 12 | New config keys: `grip_curl_min 0.5`, `flat_curl_max 0.25`, `comove_tol 0.5`, `contact_min_obj_conf 0.2`, `big_box_frac 0.15`, `ambiguous_area_ratio 1.5`, confidence weights `w_tips/w_det/w_curl/w_comotion` (0.35/0.25/0.2/0.2), `review_reason_frac 0.3`, `motion_width/max_corners/min_points` | Plan says thresholds live in `ps2.yaml`; these were not in its starting list. Untuned guesses. |
| 13 | `low_object_confidence` triggers below 0.2 detector confidence (not `label_min_conf` 0.35) | Most detections are 0.2-0.35; 0.35 would make almost every contact unknown. |
| 14 | `fixed_contact_unverified` only for fixed boxes covering >=15% of the frame with a moving hand; a small fixed object (vise, switch board) with a moving hand is still a touch | The plan's two rules (T2.2 and T2.3) conflict for fixed objects; this keeps both. |
| 15 | Camera motion ignores hands and non-fixed objects when picking background points | They move on their own. Falls back to "no camera motion" (flag) if < 8 points; this happened in 5 of 5400 frames. |
| 16 | A hand that vanishes mid-contact gets a `hand_states` entry (`unknown`, `hand_not_detected`) for up to 300 ms | The reason code is meaningless otherwise. |
| 17 | `interaction` of an interval = most common non-unknown frame value; reasons on an interval = codes on > 30% of its frames | Simple and explainable. |

## Final 4-clip run (after the tie-rule fix)
| Clip | Hand-frames (stub) | Contacts: final | Contacts: Phase 1 rule* | tools[] | review_queue | safety events | Speed** |
|---|---|---|---|---|---|---|---|
| MTL-SGF00106-021 (A) | 1795 | 34 | 64 | 198 | 177 | 4 | 6.4 fps |
| mytronlabs-000003-002 (B) | 5115 | 111 | 282 | 184 | 678 | 42 | 8.7 fps |
| MTL-SKL00089-005 | 5105 | 73 | 159 | 94 | 222 | 31 | 6.7 fps |
| mytronlabs-000060-006 | 411 | 6 | 11 | 61 | 57 | 2 | 11.7 fps |

\* The Phase 1 rule re-applied to the final detections (`dev/baseline_v0.py`): counts only, no accuracy.
\** Two runs shared the GPU/CPU at the same time, so these speeds are lower than single runs (12-14 fps).
All clips: 5400 frames, valid against the schema, on cuda:0 RTX 3050 Ti. Single-run speed on clip A was 12-14 fps.
- The tie-rule fix helped on clip A (544 -> 125 identity_uncertain items) but **not on clip B (568 items)** or SKL-005. It is not a general fix.
- Clip `mytronlabs-000060-006` has hands in only 411 of 5400 frames from the MediaPipe stub, so it produced just 6 contacts. Either hands are rarely visible or the stub misses them; check the overlay.
- Safety events: 42 on B and 31 on SKL-005 were NOT inspected. Given the detector's wrong names I expect many false alarms.
- `git add -n .` lists 38 files, all code, config, tests and docs. No media, weights, arrays or JSONL. (T8.5)

## Phase 6 status and decisions (updated as I go)
| # | Decision / status | Why |
|---|---|---|
| 33 | I do NOT label contacts myself. You label with `python dev/label_tool.py <video>`. | Ground truth must be human; me labelling my own output's video would make the numbers meaningless. |
| 34 | Video A (tuning) = `MTL-SGF00106-021`; video B (report only) = `mytronlabs-000003-002`. Pipeline run on B started in the background (stub + PS2). | A is the clip I tuned on; B comes from a different recording batch, so it tests generalisation. Tell me if you want other clips. |
| 35 | Matching rule: same hand and time IoU >= 0.5, one-to-one, best first. `--ignore-hand` also available. | MediaPipe left/right labels are unreliable on this footage, so I print both. |
| 36 | "Before" column = Phase 1 rule re-applied to the final detections (`dev/baseline_v0.py`), not the old code run again. | The old run only exists for clip A; re-applying the rule on the same detections compares the contact rule only. Sanity check: it gives 64 contacts on clip A, same as the real Phase 1 run. |
| 37 | Re-entry accuracy needs you to tag contacts of the same physical object with the same "instance" tag (e.g. `drill1`) in the label tool. | Otherwise there is no ground truth for IDs. |
| 38 | You said "pick yourself, do whatever you have to, I'll test at the end", so I kept clips A and B as chosen, and applied my own recommendation for the review-queue flood: a near-tie only counts if the two objects being merged and the rival were each seen for at least `min_tool_ms` (500 ms). Re-running to measure. | Brief noise pieces were creating 544 "uncertain" items. |
| 39 | Phase 7 (masks) is skipped for now. | Plan says do it only after Phase 6 numbers exist and keep it only if numbers do not get worse; there are no labels yet. |
| 40 | T8.3 "run all sample videos" is cut down to 4 clips (A, B and one each from two other companies). | One clip takes about 17 min (10 min MediaPipe stub on CPU + 7.5 min PS2); 40 clips would be about 11 hours. |
| 41 | T8.1 (real PS1 output) cannot be done: no teammate file exists yet. The stub follows the same PS1Frame format. | |
| Status | `label_tool.py`, `evaluate.py`, `baseline_v0.py` written. 78 tests pass. The label tool's window and keys were NOT tried by me (I cannot press keys); only its file handling and drawing were tested. | |
| Not done | T6.2 (labelling), T6.4 (before/after table in README_PS2.md). | Need your labels. |

## Phase 5 decisions (added without asking)
| # | Decision | Why |
|---|---|---|
| 28 | You did not answer the identity_uncertain question, so I left it as is and started Phase 5. The flood (544 items) is still there. | Your message was "start Phase 5". |
| 29 | `aligned_diff` ignores pixels inside hand boxes | Otherwise a hand moving near a machine makes the machine look "moving", and the warning would trigger itself. |
| 30 | New config keys `hazard_window_ms 500`, `hazard_moving_frac 0.5` | The plan says "most of a short window" without numbers. |
| 31 | Safety event confidence = mean detector confidence of the hazard x mean share of moving frames in the event | The plan gives no formula. Not calibrated. |
| 32 | Events are made only for hazards judged moving; `hazard_moving` is therefore always true | `null` would mean "unknown motion"; we do not warn on that. |

## Phase 5 results (same clip)
- 4 safety events in total (2.9 s over a 3-minute clip), all on `marble_cutter` objects (2 distinct objects), at about 116 s and 138-140 s. Distances are 0.0 (the fingertip is inside the box).
- I looked at the overlay for 3 of the 4. **At best 1 of 3 looked plausible**; one was a false alarm because the hazard box was very large (it covered far more than one machine). Not checked: whether running machines were missed.
- Why so few events: only objects labelled as hazards in `vocab.yaml` count, and the detector mislabels most real tools, so the system cannot warn about what it cannot name. Also, 96 hazard-labelled objects exist (`is_hazard`) but only a few look "moving".
- No labels exist, so false-alarm and miss rates are not measured. Needs Phase 6.
- Limitations (also in the output reasons): distance is 2D image only; a hand gripping a running tool will count as "near" it.
- Speed 12.1 fps on GPU (about 7.5 min per 3-min clip).

## Phase 4 decisions (added without asking)
| # | Decision | Why |
|---|---|---|
| 23 | You did not answer the noise-track question, so I took the non-destructive option: objects seen < 500 ms (`min_tool_ms`) and in no contact get no `tools[]` row and no review items. Their detections stay in `frames[]`. | Keeps 611 mostly-noise objects out of the summary without deleting detections. Easy to change. |
| 24 | New config keys `min_tool_ms 500`, `appearance_gap_ms 500`, `long_unknown_ms 1000` | Thresholds belong in `ps2.yaml`. |
| 25 | A contact with `review_needed` becomes a review item whose reason is its first reason code, or `low_object_confidence` if it has none | INTEGRATION.md has no code for "overall confidence low"; adding one would change the schema, so I used the closest. Say if you want a new code. |
| 26 | `held_ms` is the union of hold intervals (two hands on one object do not double count); `visible_ms` = frames seen x frame time | Simple and explainable. |
| 27 | The clearest crop per track (confidence x share not covered by hands, max 224 px) is kept in memory; name suggestions run once per unidentified object that has a tools row | Plan T3.1 allows crops in memory; avoids re-reading the video. Never written to disk. |

## Phase 4 results (same clip)
- `tools[]`: 198 rows (of 611 objects): 155 identified, 43 unidentified. 155 have more than one appearance. 5 have held time, 7 uses in total. 96 of 198 are flagged `is_hazard`, which is too many (the vocabulary labels many things `jigsaw` / `marble_cutter`).
- `review_queue[]`: 596 items = 544 `identity_uncertain`, 43 `unidentified_object`, 4 `low_object_confidence`, 2 `hand_not_detected`, 1 each `long_unknown`, `fixed_contact_unverified`, `type_overridden_by_motion`.
- **Name suggestions are useless.** The prompt-free model answered for all 43 but the names are things like `quarry`, `frosty`, `play pong`, `underwater`, `carousel`. They are only suggestions and never applied, but a human reviewer would learn nothing from them.
- **The queue is still flooded by `identity_uncertain` (544).** Same cause as in Phase 3 (the tie rule fires almost every time). A fix needs your call: count a rival only if both objects are long enough to matter, or raise `reid_min_sim`.
- Speed: 12.3 fps on GPU (about 7.3 min per 3-min clip; name suggestions load a second model). Output file 3.6 MB.

## Phase 3 decisions (added without asking)
| # | Decision | Why |
|---|---|---|
| 18 | Re-entry merge runs right after the frame loop and BEFORE contact decisions (plan said "after all frames"; contact is also decided after all frames, so the order is only about which comes first) | A tracker ID switch would otherwise split one contact into two before the IDs are merged. |
| 19 | Each final object gets ONE label for the whole clip: most-voted specific label (generic only if nothing specific); type and hazard follow that label | A stable `object_id` with a flickering label is hard to use. Schema is unchanged, only values. |
| 20 | New config keys `reid_tie_margin 0.05` and `reid_max_gap_ms 3000` | Tie rule and the no-fingerprint time limit needed thresholds. |
| 21 | Fingerprint = 16x8 hue/saturation histogram of a 32x32 crop, summed over frames where hands cover < 10% of the box; similarity = 1 - Bhattacharyya distance | Cheap, no extra model. Crops are never stored. |
| 22 | `identity_uncertain` items are already written to `review_queue` (optional field) | Phase 4 adds the other item types. |

## Phase 3 results (same clip)
- Tracker IDs 1371 -> **611 objects**. 138 of them are visible for 30+ frames; 366 are visible for under 10 frames (detector noise that the tracker turned into tracks).
- One label per object: 0 objects with more than one label. Contacts 34 (was 35), review_needed 8.
- 567 `identity_uncertain` review items: 557 from the tie rule, 10 from merging without a fingerprint. **That is too many to be useful.** Cause: hundreds of short noise pieces look alike in colour, so almost every merge has a near-equal rival. The merging itself works (tests pass), but the tie rule floods the queue.
- Not verified: that real tools keep one ID when they leave and re-enter (I only checked one overlay frame; many tools are not detected at all, which is a detector/vocabulary problem).
- Speed: 14.4 fps on GPU (about 6.2 min per 3-min clip).

## Phase 2 results (same clip, Phase 1 vs now)
| | Phase 1 | Phase 2 |
|---|---|---|
| Contacts | 64 | 35 |
| Median / longest contact | 0.20 s / 2.1 s | 0.37 s / 1.9 s |
| Interaction | all unknown | 22 touch, 8 hold, 5 unknown |
| Contacts flagged `review_needed` | 0 | 8 |
| Unknown hand-frames | 0 | 623, all with a reason |
| Speed (GPU) | 16.6 fps | 14.7 fps (6.1 min per 3-min clip) |
- Reason counts over frames: `grasp_pose_no_object` 521, `fixed_contact_unverified` 74, `hand_not_detected` 21, `type_overridden_by_motion` 19, `ambiguous_candidates` 7.
- Honest caveats: there are no labels yet (Phase 6), so I cannot say whether the 35 contacts are right. `grasp_pose_no_object` is the biggest "unknown" source: 34% of hand-frames have curl >= 0.5 with the stub's keypoints, so `grip_curl_min` probably needs raising. One earlier run silently used the CPU (6.5 fps) for an unknown reason; the pipeline now logs the device. Results were the same on GPU.

## Phase 1 results (clip MTL-SGF00106-021, 5400 frames)
- `run_ps2` returns a dict that passes `schema.validate()`. 24 tests pass.
- Speed: 16.6 fps for PS2 alone (5.4 min per 3-min clip). MediaPipe stub takes about 10 min for the same clip (CPU), so the full chain is slower than the 10-min target until real PS1 data replaces the stub.
- Hands seen in 1788 of 5400 frames (33%). 64 contacts, 507 frames in contact. Median contact 0.2 s, longest 2.1 s.
- 1371 distinct object ids: the tracker fragments heavily because the detector is weak and flickers. Phase 3 (re-entry merge) is where this gets fixed.
- Overlay checked on one frame: boxes, hand state and timeline draw correctly. The sampled contact was on a wrongly named object, so many contacts are with wrong labels. Contact logic works; the detector vocabulary is the weak part.
- Overlay: `outputs/MTL-SGF00106-021/overlay.mp4` (409 MB, full clip). Output JSON: `outputs/MTL-SGF00106-021/ps2.json`.

## Open questions for you
- (Phase 0) Which tools really appear in the videos? You said you updated `vocab.yaml` in Phase 6b, but the file on disk is unchanged (see Phase 6b item 1).

## Known issues / risks
- Phase 1 contact rule is the brief's v0 (2+ fingertips inside the padded box). Expect flicker and false contacts with very large fixed boxes; Phase 2 addresses this.
- MediaPipe left/right labels assume a mirrored camera; on egocentric video they can be swapped or flip between frames. Contacts are tracked per `hand_id`, so flips can split intervals. The real PS1 should fix this.

## Phase 7 Part B (no human answers yet)
- Groq worked again (rolling window, not UTC midnight); I reset my own daily counter file (cache/groq_usage.json) and rebuilt the clip A check list: 50 of 55 pooled contacts now have a Groq vote (was 23). Still 40 items, seed 7, all tier 2 (no item had all three models agree or fully disagree).
- Items 10-13 cannot be done honestly yet: tool-level F1, tuning "keep only if A improves", the run on B and safety precision all need your answers (`verify_contacts.py --checklist`, `--safety`). Without ground truth any tuning would fit noise, so the config is unchanged and `safety_enabled` stays as it was.
- `evaluate.py` was run on all four variants for A; it only gives model-vs-model agreement until answers exist. Note the pool is built from the `ledger_sam` result, so the variants share one check list.
- Groq as a proxy: it said "contact" for 4 of the system's 15 contacts on A and for 23 of 40 Gemini-only ones. Noisy, but it points at doubtful contacts and at recall on A.
- README_PS2.md has a Results (Phase 7) section stating all of this. Nothing staged or committed.
