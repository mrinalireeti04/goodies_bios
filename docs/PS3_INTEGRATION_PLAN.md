# PS3 Integration Plan for FORGE Studio

## Goal

Adapt the task intelligence and timeline generation work on `sk_branch` into the existing BIOS application. Preserve the current FastAPI job API, PS1/PS4 outputs, Studio playback, search, analytics, and annotation import formats. Do not merge `sk_branch` wholesale: it is an orphan/root commit containing a separate package and its tree removes much of the current application.

## What PS3 currently provides

The branch's `forge_trace.pipeline.process_video_to_timeline(...)` accepts a video path and clip ID, optionally accepts object, hand, and interaction event lists, and returns a validated Pydantic `TaskTimeline`. Internally it validates video metadata, segments the video (120 seconds with 15 seconds overlap by default), uploads it to Gemini for coarse action recognition, optionally refines temporal boundaries, normalizes and deduplicates actions, enriches available object/hand/interaction labels, then assigns task IDs and optionally groups related tasks.

The branch has useful separation into recognition, temporal refinement, normalization, enrichment, task building, schemas, validators, and video utilities, with unit tests for those units. Reuse/adapt these domain stages behind a BIOS PS3 adapter rather than transplanting its standalone CLI, config tree, or job runner.

## Integration findings and blockers

1. **Incompatible output contracts.** The PS3 branch returns `TaskTimeline` with `tasks`, nested `timeline.start_s/end_s`, `object`, and `hand`. `frontend/src/services/adapters/ps3Adapter.js` expects a top-level `actions` array with `action_id`, integer `start_ms`/`end_ms`, `object_label`, `hand`, and `hand_ids`. `backend/app.py` and frontend import/search paths also read `actions.json` in that Studio shape. Persisting the branch's raw JSON as `actions.json` will fail validation or silently break consumers.
2. **PS2 is incorrectly a hard gate for PS3.** The job runner currently wires PS3 to `modules.ps3.run_ps3(video, ps1_path, ps2_path, output_dir)` and requires both PS1 and PS2 outputs. The PS3 branch instead accepts optional event lists, and this repository has no installed `modules.ps2` implementation. Make PS3 depend on the uploaded video and successful PS1 job only; supply PS2 data only when available.
3. **PS1 enrichment needs an adapter.** PS1 uses frame records (`frame_idx`, `t_ms`, `hands`) with per-hand side/IDs and landmarks. The PS3 enrichment helpers expect event dictionaries with seconds-based temporal fields and model-specific names. Convert without changing the canonical PS1 file; pass only validated events and never invent object/contact tracks.
4. **External processing and privacy.** `GeminiClient` uploads the source video to Google's Gemini API. This conflicts with the current backend's local-only processing description and cannot be hidden behind an apparently local upload. PS3 needs explicit configuration, a clear Studio indication that video leaves the device, bounded upload/model timeouts, failure status, and best-effort remote file deletion. With no PS3 API key/configuration, PS1/PS4 must still complete and PS3 must be reported as skipped/unavailable, not cause the whole clip to fail.
5. **Dependency mismatch.** `forge_trace/gemini_client.py` imports `google.genai`, while `sk_branch/pyproject.toml` declares `google-generativeai`. That declared distribution does not provide the modern `google.genai` import. Resolve to the distribution that provides the imported SDK, and reconcile the PS3 Pydantic v2 APIs with backend dependency pins before integrating. Avoid installing both competing Gemini clients by default.
6. **Metadata leakage.** `TaskTimeline` includes the resolved local `video_path`. Do not return that field through the public job API or persist it in a client-visible artifact. Keep the job's opaque `clip_id` and Studio's privacy boundary.
7. **Health/status reporting needs alignment.** Current `/api/health` probes legacy `modules/ps1.py`/`ps2.py`/`ps3.py` paths even though PS1 now lives under `backend/inference`; its readiness result should be corrected as part of wiring the optional PS3 capability. Track per-module outcomes so unavailable PS3 does not incorrectly turn successful hand/pose results into a failed job.
8. **Model claims need qualification.** The branch's task labels/timestamps come from a generative video model and heuristics; output confidence is not a calibrated accuracy score. Keep evidence, confidence, and refinement metadata, and make actions editable/reviewable in Studio.
9. **Segment timestamps need correction before production.** The recognition loop passes the same uploaded full video to Gemini for every planned segment; it does not call the available segment extraction utility. The parser then offsets returned timestamps by each segment start. This can analyze the full clip repeatedly and shift timestamps incorrectly. Either extract/upload actual segment clips and add their offsets exactly once, or analyze the full clip once and remove segment offsets; verify overlap deduplication against known timestamps.

## Recommended architecture

- Add a BIOS-owned `backend/inference/ps3` (or equivalent existing backend module namespace) adapter with a stable callable such as `run_ps3(video_path, hands_path, optional_contacts_path, output_dir, config) -> Path`.
- Keep reusable recognition/refinement/normalization/task-building domain code in the backend package. Change imports to package-relative BIOS imports; do not add the standalone `run_video.py` as a production path.
- Create one explicit serializer from the branch's `TaskTimeline` into Studio's PS3 `actions.json` contract. Keep seconds internally; convert interval boundaries once with rounded `seconds * 1000`, preserve positive duration, and clamp to the uploaded video's duration. Map `object` to `object_label`, assign deterministic `action_id` values from task IDs, and set `hand_ids` only when a real stable source identity exists. PS1 side labels (`left`/`right`) are not persistent track IDs, so do not manufacture H0/H1 identities from them. Include optional evidence, description, confidence, group/task ID, and model metadata as additive fields that current consumers can ignore.
- Keep the full internal timeline in a separate versioned artifact (for example `ps3_timeline.json`) if task grouping and processing metadata are useful, but strip local filesystem paths and avoid changing the Studio-facing `actions.json` schema without a coordinated frontend migration.
- Run PS3 after PS1 is available. Pass the original uploaded video, PS1 measurements converted to the enrichment event contract, and PS2 contact/object events only if the PS2 result file exists and validates. PS3 should not block PS1/PS4 artifact publication.
- Add a PS3-specific configuration object/environment prefix (for example `FORGE_PS3_ENABLED`, `FORGE_PS3_API_KEY` or an established secret manager binding, model, segment duration/overlap, confidence threshold, refinement and grouping switches, retry/time limits). Never accept an API key in the video filename, public job object, or unauthenticated URL query.
- Extend job metadata with per-module states (`queued`, `processing`, `completed`, `skipped`, `failed`) and safe issue messages. Maintain current output-serving routes and only mark the PS3 module completed once its contract artifact validates and is atomically written.

## Implementation sequence

### 1. Contract and dependency spike

- Pin and verify backend-compatible Pydantic and the correct `google-genai` SDK distribution.
- Import the adapted PS3 package under the existing backend runtime without changing the production job flow.
- Create representative fixtures for PS1 frame records, optional PS2 events, and expected `actions.json`.
- Lock down timestamp rounding, clip ID, hand-side/identity rules, artifact version, and privacy-safe metadata.

**Exit criteria:** the PS3 package imports under the same `uv` backend environment; a mocked Gemini client can run the pipeline and produce both a validated internal timeline and valid Studio action artifact without a network call.

### 2. PS3 service adapter and serializer

- Adapt the PS3 pipeline imports and inject configuration/client dependencies so unit tests can use a fake Gemini client.
- Convert available PS1 frame-level hand observations into the PS3 enrichment event format. Keep PS1 output immutable and preserve `hand_id` semantics (side labels are not identity tracking).
- Convert optional PS2 contact/object events only after confirming their schema. Missing PS2 remains an empty enrichment input.
- Serialize to Studio `actions.json`; validate it with the existing frontend adapter rules mirrored in backend validation. Persist the raw task timeline separately only if it contains no private path.
- Ensure remote Gemini file cleanup runs on success, model failure, cancellation, and serialization failure.

**Exit criteria:** deterministic adapter/serializer tests cover empty actions, one/multiple tasks, unknown hand/object, boundary timestamps, optional enrichment, malformed model output, and no leaked local paths.

### 3. Job orchestration and status

- Add PS3 as an optional post-PS1 stage, not as a PS2-gated stage. Keep PS1/PS4 outputs accessible while PS3 runs or fails.
- Expose module states/progress in the existing job status payload. Preserve compatibility for current frontend status polling.
- Make overall job completion reflect successful core outputs; a disabled or failed PS3 should be visible as a module-level outcome and must not erase successful hand/3D results.
- Update `/api/health` capability checks to the actual backend module paths and report PS3 disabled/missing configuration distinctly.
- Keep PS3 search indexing and `/actions` serving pointed at the Studio-shaped artifact. Ensure clip filtering and action interval overlap use integer milliseconds consistently.

**Exit criteria:** API integration checks cover PS3 enabled, disabled/no key, PS3 failure after PS1 success, missing PS2, and PS3 success; PS1/PS4 URLs/results remain usable in every non-upload-failure case.

### 4. Studio disclosure and review flow

- Show that PS3 sends footage to an external Gemini service before enabling that processing path; provide a clear configuration/disabled state and non-technical error when it is unavailable.
- Load normalized actions via the existing PS3 adapter. Show actions on the synchronized timeline, in the observation inspector, search, analytics, and exports.
- Preserve model confidence/evidence and let the user edit labels/times; do not present model output as ground truth.
- Keep existing hand overlay, original video, and 3D playback behavior unchanged.

**Exit criteria:** a real short clip completes PS1/PS4 and PS3; the original and overlay remain synchronized; action intervals seek to the correct time and appear in Studio/search/export. The UI identifies external processing and gracefully reports disabled/failed PS3.

### 5. End-to-end validation and rollout

- Run backend unit and API integration checks, frontend build/lint checks, and a representative clip smoke run with a mocked Gemini service.
- Separately run an opt-in real Gemini smoke test on a short permitted video. Record latency, segment count, failures, model ID, and provider/API billing implications; do not include source footage in logs/artifacts.
- Compare action timestamps and labels against manually reviewed sample intervals before describing accuracy. Tune confidence thresholds/refinement only from reviewed examples.

**Exit criteria:** machine-readable schema validation passes; local PS1/PS4 results survive PS3 failure; PS3 is opt-in/configured; and the reviewed smoke result is visible in Studio with no synthetic fallback.

## Decisions to keep explicit during implementation

- Whether PS3 processing is opt-in per clip or enabled by project configuration. Recommended initial behavior: disabled until a key is configured and the Studio disclosure is visible.
- Whether task grouping is worth its additional Gemini call and latency. The provided sample includes two groups, so the implementation keeps grouping enabled to match that output; disable with `FORGE_TRACE_ENABLE_GROUPING=false` if latency or billing is more important.
- Which PS2 artifact schema will be the source for object/contact enrichment. Until that module exists and its schema is confirmed, run PS3 without PS2 enrichment.
- Whether to expose the internal `TaskTimeline` artifact to users. Recommended initial behavior: expose only Studio's stable `actions.json`; retain a path-free internal timeline for debugging only if needed.

## Out of scope for the first integration

- Merging the orphan `sk_branch` history or replacing the BIOS frontend/backend.
- Implementing PS2 object/contact detection.
- Claiming persistent worker identity based only on PS1 left/right sides.
- Training/calibrating an action model or treating model confidence as accuracy.
