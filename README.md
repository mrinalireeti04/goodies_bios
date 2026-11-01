# FORGE Studio

## Factory video intelligence for hand-aware workflow understanding

FORGE Studio turns first-person factory footage into an inspectable workspace for understanding **hands, objects, contacts, task actions, and 3D hand pose**. The product is designed for a human reviewer: every result is timestamped, evidence-linked, editable, and exportable.

The system is intentionally modular. Hand tracking is the highest-priority signal, object/contact reasoning is a separate stage, task understanding is optional, and the frontend can display partial results when one module is unavailable.

> **Current status:** The React/Vite frontend, local FastAPI bridge, PS1 hand pipeline, PS4 pose conversion, PS1–PS4 import flow, synthetic demo, search, annotation editing, analytics, and JSON/CSV export are implemented. The bundled demo data is synthetic. Real PS2/PS3 inference requires the optional model packages, model weights, and configuration described below. Do not describe the synthetic demo as measured factory performance.

## 1. The problem

Factory and assembly video contains valuable process knowledge, but ordinary video review is slow and difficult to search. A reviewer must answer questions such as:

- When did the right hand hold a particular tool?
- Which object was contacted, and was it actually being held?
- Where did a task step start and end?
- What evidence supports the predicted action?
- Which results are uncertain enough to require human review?

FORGE Studio addresses this by combining frame-level hand evidence, object/contact evidence, semantic task events, and a visual review interface in one local-first product.

## 2. Product workflow

The intended judged path is:

```text
Landing page
    ↓
Upload footage or enter the synthetic demo
    ↓
Select a clip in Vision Explorer
    ↓
View video/image + 2D hand landmarks + object boxes
    ↓
Follow the synchronized 3D hand-centered pose
    ↓
Inspect pick/hold/fasten/inspect task events
    ↓
Search across timestamped events
    ↓
Correct a label or interval
    ↓
Export JSON/CSV with original predictions and manual corrections
```

The same interface supports two operating modes:

1. **Frontend/demo mode:** the browser uses clearly labelled synthetic data and local object-URL previews.
2. **Local inference mode:** the browser sends selected media to a localhost FastAPI service. The service runs whichever Python modules are installed and reports `queued`, `processing`, `partial`, `completed`, or `failed` status per job.

## 3. What is implemented

### Frontend

- React 19 + Vite application written in JavaScript/JSX.
- Landing page with product explanation, synthetic product preview, privacy statement, and working CTAs.
- Routes for Vision Explorer, Library, Search, Analytics, Export, and 404 handling.
- Multi-file image/video selection with browser-local previews.
- Synthetic demo clip with simulated hands, objects, task timeline, and 3D pose.
- Import dialog for `hands.jsonl`, `ps2.json`, `actions.json`, and `ps4.json`.
- 2D hand/keypoint and object overlays.
- Video playback controls, seeking, playback speed, fullscreen, and time-linked panels.
- React Three Fiber/Three.js hand-centered 3D viewer.
- Event timeline with timestamp navigation.
- Search across imported/synthetic action events with object, action, hand, and clip filters.
- Human-in-the-loop annotation correction for action, object, hand side, and interval fields.
- Separate correction records preserve the original model output.
- Undo for the latest correction.
- Analytics view with explicit synthetic/illustrative labelling.
- JSON and CSV exports.
- TanStack Query adapter for the optional backend and Zustand stores for shared UI/playback state.

### Backend and inference

- FastAPI localhost job bridge in `backend/app.py`.
- Safe per-job private storage under `.local_data/` by default.
- Multipart upload validation for supported image/video extensions and maximum file size.
- Per-module status and issue reporting; missing optional modules do not fabricate predictions.
- PS1 hand inference with a MediaPipe + RTMPose/RTMDet hybrid path.
- PS4 conversion from PS1 hand-centered 3D points to `ps4.json` and an optional 3D visualization MP4.
- PS2 adapter contract for object/contact reasoning.
- Optional PS3 Gemini video-understanding pipeline for action recognition, temporal refinement, grouping, and evidence enrichment.
- Timestamp-windowed API responses so the frontend does not need to scan full videos per query.
- Local deletion endpoints for completed jobs and private server data.

## 4. System architecture

```text
                         ┌──────────────────────────┐
                         │ React/Vite frontend       │
                         │                          │
                         │ Landing                  │
                         │ Vision Explorer          │
                         │ Library / Search         │
                         │ Analytics / Export      │
                         └────────────┬─────────────┘
                                      │ native fetch()
                         localhost only, opt-in
                                      │
                         ┌────────────▼─────────────┐
                         │ FastAPI local job bridge  │
                         │                          │
                         │ upload / status          │
                         │ media / result windows  │
                         │ search / delete         │
                         └────────────┬─────────────┘
                                      │
                  ┌───────────────────┼───────────────────┐
                  │                   │                   │
          ┌───────▼───────┐   ┌───────▼───────┐   ┌───────▼───────┐
          │ PS1            │   │ PS2            │   │ PS3            │
          │ hands          │   │ objects +     │   │ actions +      │
          │ landmarks      │   │ contact        │   │ task timeline  │
          └───────┬────────┘   └───────┬────────┘   └───────┬────────┘
                  │                   │                   │
                  └───────────────────┼───────────────────┘
                                      │ shared contracts
                              ┌───────▼───────┐
                              │ PS4            │
                              │ hand-centered  │
                              │ 3D pose        │
                              └────────────────┘
```

### Design principles

1. **Evidence before semantics.** A visible object is not automatically a contacted or held object.
2. **Partial results are valid.** PS1/PS4 can be useful even if PS2 or PS3 is unavailable.
3. **No fabricated inference.** Synthetic values are labelled as synthetic; missing model outputs remain missing.
4. **Stable contracts.** Downstream modules consume versioned JSON/JSONL artifacts instead of importing each other's internals.
5. **Human review is first-class.** Corrections are separate overrides, not destructive edits to model output.
6. **Local-first privacy.** The default frontend does not upload media. The optional backend is localhost-only and unauthenticated.

## 5. Tech stack

| Layer | Technology | Role |
| --- | --- | --- |
| UI | React 19, JavaScript/JSX | Product interface and interaction model |
| Build/dev | Vite 7 | Frontend development server and production bundle |
| Styling | Tailwind CSS v4, Radix primitives, shadcn/ui-style components | Layout, tokens, accessible primitives |
| Icons | Lucide React | Consistent interface iconography |
| Client state | Zustand | Playback, media, annotations, inference, and search state |
| Async state | TanStack Query | Local API jobs, polling, and server-state invalidation |
| Routing | React Router | Product routes and navigation |
| 3D | Three.js, React Three Fiber, Drei | Hand-centered 3D skeleton rendering |
| Charts | Recharts | Analytics visualizations |
| API | FastAPI, Uvicorn, Pydantic | Local upload, job, result, search, and deletion APIs |
| Video/image | OpenCV, FFmpeg | Decoding, frame access, overlays, and optional video writing |
| Hand inference | MediaPipe Tasks, RTMLib, ONNX Runtime | Hand proposals, handedness, 2D landmarks, and optional 3D landmarks |
| Object inference | Ultralytics YOLOE and optional SAM 2.1 | Open-vocabulary object detection and optional masks |
| Video understanding | Google Gemini API | Optional PS2 ledger/naming and PS3 action/task understanding |
| Tests | Python `unittest`, Node verification script | Contracts, imports, interpolation, API smoke tests, and PS4 tests |

## 6. Exact model and inference inventory

The following is the actual model configuration in the repository. Model availability depends on installed packages, downloaded weights, API configuration, and hardware.

### PS1 — hand detection and tracking

#### Primary components

- **MediaPipe Hand Landmarker Tasks model:** `hand_landmarker.task`.
  - Downloaded by `backend/setup_models.py`.
  - Used for hand proposals, handedness, 21-point landmarks, and MediaPipe hand-world landmarks.
  - The original implementation uses MediaPipe VIDEO/IMAGE APIs with two-hand support in the hand-detection module.
- **RTMDet-nano hand detector:** downloaded by RTMLib on first use.
  - Used by `RTMHandEngine` to obtain additional/full-resolution hand boxes.
- **RTMPose-m hand pose model:** downloaded by RTMLib on first use.
  - Used for higher-resolution 21-point 2D hand landmarks on crops from the original frame.
- **ONNX Runtime:** execution backend.
  - Apple Silicon may request CoreML.
  - The code falls back to CPU if CoreML compilation or inference fails.
  - CUDA, Apple MPS, CoreML, and CPU selection are handled according to the configured runtime and platform.

#### PS1 processing sequence

```text
Video
  → decode every frame
  → MediaPipe hand proposals and handedness
  → RTMDet hand boxes
  → RTMPose-m landmarks on original-resolution crops
  → IoU matching between proposal and pose
  → MediaPipe 3D landmarks retained when available
  → conservative left/right correction
  → bounded interpolation for short gaps
  → hands.raw.jsonl + hands.jsonl + ps4.json + overlay.mp4
```

Important implementation choices:

- Output coordinates are in original-frame pixel space.
- Each frame keeps a record, including frames with no detections.
- At most two wearer hands are retained by the current PS1 integration.
- Short, bounded gaps may be interpolated; long or unbounded gaps remain empty.
- `source` records whether a value was detected or interpolated.
- Left/right labels are treated as anatomical sides, not guaranteed persistent identity through crossings.
- The 3D hand points are hand-centred metric coordinates, not factory/world coordinates.

### PS2 — objects, contact, state, and safety

Configured in `ps2/config/ps2.yaml`.

- **Ultralytics YOLOE:** `yoloe-26s-seg.pt`.
  - Open-vocabulary object detection.
  - Runs on full frames and optionally on crops around PS1 hands.
  - Default configured input size is `640`; detection threshold is `0.15`.
- **SAM 2.1 Tiny:** `sam2.1_t.pt`.
  - Optional segmentation/mask tracking for portable objects and contact intervals.
  - Controlled by `sam2_enabled` and related budget settings.
- **Gemini 3.6 Flash:** configured as the main PS2 visual-language model.
  - Used for the tool ledger, object vocabulary support, and object naming when enabled.
- **Gemini 3.5 Flash Lite:** configured fallback for selected PS2 calls.
- The PS2 pipeline combines detector evidence, hand proximity/fingertip evidence, hand curl, co-motion, temporal persistence, and review thresholds.

PS2 should distinguish:

```text
object visible
    ≠ possible contact
    ≠ verified handling
    ≠ semantic action
```

PS2 is optional at runtime. If its packages or weights are missing, the backend marks it unavailable and preserves PS1/PS4 output.

### PS3 — actions and task timeline

The PS3 implementation is in `backend/inference/ps3/`.

- **Gemini 3.6 Flash:** default model from `FORGE_TRACE_MODEL`.
- Video is segmented into 120-second windows with 15-second overlap by default.
- Gemini receives the real video segment for action recognition; segment-relative times are mapped back once to the original clip clock.
- Pydantic schemas validate raw predictions and normalized action intervals.
- Optional temporal refinement asks Gemini to tighten event boundaries.
- Optional grouping combines low-level events into task groups.
- Optional enrichment attaches YOLO object evidence, MediaPipe hand evidence, and PS2 interaction evidence.
- The result is written as Studio-compatible `actions.json` plus detailed `ps3_timeline.json`.
- Provider-side uploads are deleted on a best-effort basis after processing.

PS3 is deliberately opt-in because enabling it sends video segments to Google Gemini. The frontend exposes this as a separate checkbox.

### PS4 — 3D pose conversion

PS4 is implemented in `frontend/modules/ps4.py` and does not load a new ML model.

- Consumes PS1 `kp3d_hand_m` landmarks.
- Validates that each pose has 21 three-dimensional joints.
- Produces `ps4.json` with hand-centered 3D frames.
- Optionally renders `hands_3d.mp4` using OpenCV and a lightweight projection.
- The visualization is explicitly not a world-space factory trajectory.

## 7. Data contracts

### PS1 `hands.jsonl`

One JSON object per frame:

```json
{
  "schema_version": "1.0",
  "clip_id": "factory_001",
  "frame_idx": 120,
  "t_ms": 4000,
  "frame_width": 854,
  "frame_height": 480,
  "hands": [
    {
      "hand_id": "R",
      "side": "right",
      "handedness_score": 0.934,
      "bbox_px": [312.4, 201.15, 468.92, 355.7],
      "kp2d_px": [[330.1, 350.2], "... 20 more points ..."],
      "kp3d_hand_m": [[0.0, 0.0, 0.0], "... 20 more points ..."],
      "source": "detected"
    }
  ]
}
```

Rules:

- `kp2d_px` has exactly 21 points.
- `kp3d_hand_m` is `null` or exactly 21 triples.
- `t_ms` and `frame_idx` increase strictly within a file.
- Empty detections use `"hands": []`.
- Pixel coordinates refer to the original decoded frame, not the resized detector input.

### PS2 `ps2.json`

The PS2 output contains frame-level objects, hand state, contacts, intervals, and processing metadata. The adapter accepts the PS2 result and maps object boxes, interaction labels, confidence, and evidence into the frontend inference store.

### PS3 `actions.json`

The normalized action contract is interval-based:

```text
clip_id
action_id
start_ms
end_ms
verb
object_label
hand
hand_ids
confidence
evidence
```

Intervals are half-open `[start_ms, end_ms)` and are clamped to the source clip duration.

### PS4 `ps4.json`

```json
{
  "schema_version": "1.0",
  "clip_id": "factory_001",
  "coordinate_frame": "hand_centered",
  "frames": [
    {
      "frame_idx": 120,
      "t_ms": 4000,
      "hands": [
        {
          "hand_id": "R",
          "side": "right",
          "joints_3d_m": [[0.0, 0.0, 0.0], "... 20 more points ..."],
          "source": "detected"
        }
      ]
    }
  ]
}
```

### Frontend result import

The browser accepts these filenames:

| File | Meaning |
| --- | --- |
| `hands.jsonl` | PS1 frame-level hand landmarks |
| `ps2.json` | PS2 object/contact results |
| `actions.json` | PS3 action intervals |
| `ps4.json` | PS4 hand-centered 3D pose |

All imported files must share one `clip_id`. The user selects which media file the outputs belong to; automatic media-to-result identification is not implemented.

## 8. Repository layout

```text
BIOS/
├── backend/
│   ├── app.py                         # FastAPI localhost bridge
│   ├── setup_models.py                # download MediaPipe model
│   ├── inference/
│   │   ├── pipeline.py                # PS1 orchestration
│   │   ├── hand_detect.py             # MediaPipe/schema implementation
│   │   ├── rtm_hand.py                # RTMDet + RTMPose integration
│   │   ├── labels.py                  # side correction
│   │   └── ps3/                       # Gemini task/action pipeline
│   └── tests/                         # PS1 interpolation/label tests
├── frontend/
│   ├── src/
│   │   ├── app/                       # router/providers
│   │   ├── pages/                     # product routes
│   │   ├── features/                  # uploads, vision, timeline, search
│   │   ├── services/                  # API, adapters, search, jobs
│   │   ├── stores/                    # Zustand state
│   │   └── mocks/                     # synthetic demo only
│   ├── modules/ps4.py                 # PS4 conversion module
│   ├── scripts/                       # verification and synthetic fixtures
│   ├── tests/                         # API/PS4 integration tests
│   ├── package.json
│   └── tasks.md                       # implementation checklist
├── ps2/
│   ├── pipeline.py                    # object/contact pipeline
│   ├── objects.py                     # YOLOE detection
│   ├── masks.py                       # optional SAM2 masks
│   ├── ledger.py                      # Gemini tool ledger
│   ├── fuse.py                        # evidence fusion
│   └── config/ps2.yaml                # model and threshold configuration
├── docs/schema.md                     # PS1 schema details
├── examples/                          # synthetic schema examples
├── results/                           # reports; private outputs must stay ignored
├── hand_detect.py                     # standalone PS1 entry point
├── requirements.txt                   # minimal standalone PS1 dependencies
└── README.md
```

## 9. Installation and local development

### Frontend prerequisites

- Node.js 20.19+; Node.js 22 is recommended.
- `pnpm` is preferred because the repository includes `frontend/pnpm-lock.yaml`.
- `npm` can be used if the pnpm/Corepack installation is unavailable.

```bash
cd frontend
pnpm install
pnpm run dev
```

Open `http://127.0.0.1:5173`.

Production build:

```bash
pnpm run check
pnpm run build
pnpm run preview
```

If `pnpm` is unavailable but dependencies already exist:

```bash
cd frontend
npm run check
npm run build
npm run dev
```

### Python environment

The recommended backend environment uses `uv`:

```bash
cd /Volumes/Data/Hacks/BIOS
uv venv backend/.venv
uv pip install --python backend/.venv/bin/python -r backend/requirements.txt
```

Download the MediaPipe hand model:

```bash
backend/.venv/bin/python backend/setup_models.py
```

FFmpeg must be installed and available on `PATH` for PS1 overlay video output.

Start the backend:

```bash
backend/.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

The backend is intentionally bound to `127.0.0.1`. Do not change this to `0.0.0.0` for private factory footage.

Start the frontend in a second terminal:

```bash
cd frontend
VITE_USE_REAL_API=true pnpm run dev
```

### Synthetic fixture flow

This flow does not require private footage or real model inference:

```bash
cd frontend
PYTHONPATH=. python scripts/create_synthetic_test_assets.py
```

Then start the frontend, upload the generated synthetic media, and import the generated `hands.jsonl`, `ps2.json`, `actions.json`, and `ps4.json` files.

## 10. Configuration

Never commit secrets. `.env` and `.env.*` are ignored by Git.

### Frontend

| Variable | Default | Meaning |
| --- | --- | --- |
| `VITE_USE_REAL_API` | enabled unless explicitly `false` | Enable localhost FastAPI job calls |
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | Backend base URL, if overridden |

### Backend and PS1

| Variable | Meaning |
| --- | --- |
| `FORGE_PRIVATE_DIR` | Override `.local_data/` storage location |
| `FORGE_RTM_DEVICE` | `auto`, `coreml`, `mps`, or `cpu` |
| `FORGE_COREML_CACHE` | Override CoreML compilation cache |
| `GEMINI_API_KEY` | Enables Gemini-backed PS2/PS3 calls |

### PS3

| Variable | Default | Meaning |
| --- | --- | --- |
| `FORGE_TRACE_MODEL` | `gemini-3.6-flash` | Gemini model |
| `FORGE_TRACE_SEGMENT_DURATION` | `120` | Segment duration in seconds |
| `FORGE_TRACE_SEGMENT_OVERLAP` | `15` | Segment overlap in seconds |
| `FORGE_TRACE_CONFIDENCE_THRESHOLD` | `0.15` | Minimum normalized action confidence |
| `FORGE_TRACE_MAX_RETRIES` | `3` | Retry count for retryable API failures |
| `FORGE_TRACE_ENABLE_REFINEMENT` | `true` | Enable temporal refinement |
| `FORGE_TRACE_ENABLE_GROUPING` | `true` | Enable task grouping |

### PS2

The main PS2 model names and thresholds are in `ps2/config/ps2.yaml`. Important settings include `det_model`, `sam2_model`, `vlm_model`, `det_conf`, contact persistence (`on_frames`/`off_frames`), hand crop detection, safety thresholds, and the Gemini request budget.

## 11. FastAPI endpoint inventory

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Report API mode and available modules |
| `POST` | `/api/jobs` | Upload a local image/video and create a processing job |
| `GET` | `/api/jobs/{job_id}` | Read job/module status and issues |
| `POST` | `/api/jobs/{job_id}/ps3` | Retry PS3 when PS1 is complete and Gemini is configured |
| `GET` | `/api/clips` | List local job-backed clips |
| `GET` | `/api/clips/{job_id}` | Read clip metadata/status |
| `GET` | `/api/clips/{job_id}/media` | Stream original local media |
| `GET` | `/api/clips/{job_id}/overlay` | Stream generated PS1 overlay when present |
| `GET` | `/api/clips/{job_id}/hands` | Read PS1 results, optionally time-windowed |
| `GET` | `/api/clips/{job_id}/contacts` | Read PS2 contact results |
| `GET` | `/api/clips/{job_id}/actions` | Read PS3 actions, optionally time-windowed |
| `GET` | `/api/clips/{job_id}/pose3d` | Read PS4 pose results, optionally time-windowed |
| `POST` | `/api/search` | Search action/object/hand evidence across local results |
| `DELETE` | `/api/jobs/{job_id}` | Delete one completed local job and its files |
| `DELETE` | `/api/data` | Delete all completed local job data |

The API is single-user, unauthenticated, and intended only for localhost development.

## 12. Verification and tests

### Frontend checks

```bash
cd frontend
pnpm run check
pnpm run build
```

`check` validates JavaScript/JSX syntax, local imports, result contracts, adapters, timestamp matching, and stale-pose protection.

### Backend unit tests

```bash
cd /Volumes/Data/Hacks/BIOS
backend/.venv/bin/python -m unittest discover -s backend/tests -v
```

### PS4 and frontend integration tests

Run with the repository root and frontend directory on `PYTHONPATH`:

```bash
PYTHONPATH=/Volumes/Data/Hacks/BIOS:/Volumes/Data/Hacks/BIOS/frontend \
  backend/.venv/bin/python -m unittest discover -s frontend/tests -v
```

### API smoke checks

```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/clips
curl http://127.0.0.1:5173/
```

### What has been verified in the current workspace

- Frontend source check passed.
- Vite production build passed.
- Backend interpolation and label-correction tests passed.
- PS4 standalone conversion test passed.
- FastAPI health and frontend HTTP responses passed.
- Browser routes and synthetic demo interaction were exercised.
- Annotation correction updated the timeline and inspector while preserving the original prediction.

### Known verification gaps

- A full real-clip PS1 accuracy benchmark is not included.
- PS2 is unavailable in the current environment when Ultralytics/model weights are missing.
- PS3 requires a valid Gemini key and sends video segments to Google Gemini when enabled.
- Browser support for `.mov` varies; MP4 is the safest demo format.
- Production security, authentication, multi-user isolation, and cloud deployment are not implemented.
- The current backend tests have expectations around invalid video jobs returning `partial`; the implementation currently returns `failed` when no PS1/PS4 output is produced. This status contract should be reconciled before calling the integration suite fully green.

## 13. Privacy and security

Factory footage, frames, screenshots, derived annotations, and credentials are private.

- Do not commit `data/`, `out/`, `.local_data/`, media files, extracted frames, or model weights.
- Keep `.env` out of Git and rotate any credential that has been exposed or shared accidentally.
- The default frontend-only mode keeps media in the browser session.
- Opt-in backend mode copies media to local `.local_data/`; refreshing the browser does not delete it.
- Delete local job folders and any provider-side uploads after the event.
- Do not expose the FastAPI service through a public tunnel.
- Blur faces, screens, and company documents in anything shown to judges.
- Do not put private footage or derived screenshots in GitHub, slides, or social media.
- Synthetic assets are safe for the public landing page; private clips are not.

## 14. PPT / presentation content

The project should fit a 5–6 slide hackathon deck. The following is ready-to-use content, with claims phrased so the deck does not overstate the current implementation.

### Slide 1 — Title and one-line value proposition

**Title:** FORGE Studio  
**Subtitle:** Every frame has a story. Understand it.  
**One-liner:** A hand-aware factory video intelligence workspace that turns raw first-person footage into searchable, evidence-linked workflow events.

Show:

- Product logo/name.
- One clean Vision Explorer screenshot or the synthetic dashboard concept.
- Team name and hackathon name.

Say:

> Factory footage is full of process knowledge, but it is difficult to search, measure, and review. FORGE Studio connects hands, objects, actions, and human review in one workspace.

### Slide 2 — Problem and why existing review is insufficient

**Headline:** Video review is rich in information but poor in structure.

Show the gap:

```text
Raw footage → manual scrubbing → uncertain interpretation → no reusable dataset
```

Key points:

- Reviewers need timestamped evidence, not just a video thumbnail.
- Object visibility is not proof of contact or handling.
- Hand side, missing frames, blur, occlusion, and re-entry create ambiguity.
- Factory knowledge is trapped inside unsearchable video.
- Privacy rules make careless cloud upload and public sharing unacceptable.

Suggested speaker line:

> The product is built around the hard part: explaining why an event matched, showing the evidence, and giving a person a fast way to correct it.

### Slide 3 — Solution and end-to-end pipeline

**Headline:** From pixels to reviewable process evidence.

Show this pipeline:

```text
Video
  → PS1: hands and 21 landmarks
  → PS2: objects, contact, hand state, safety
  → PS3: actions and task timeline
  → PS4: synchronized hand-centered 3D pose
  → Search, correct, export
```

Explain:

- PS1 is the foundation and the highest-priority accuracy surface.
- PS2 fuses detector evidence with hand geometry and temporal persistence.
- PS3 turns video segments into normalized action intervals and evidence.
- PS4 makes hand pose easier to inspect spatially; it is hand-centred, not world-anchored.
- Partial results remain useful if optional stages are missing.

### Slide 4 — Technical architecture and exact models

**Headline:** Modular, contract-driven, locally inspectable.

Use a compact architecture diagram:

```text
React/Vite UI
    ↕ native fetch()
FastAPI localhost bridge
    ↕ versioned artifacts
PS1           PS2             PS3          PS4
MediaPipe +   YOLOE +         Gemini       PS1 3D →
RTMDet/       SAM2.1 +        3.6 Flash    validated
RTMPose       Gemini          actions      pose JSON
```

Exact model callouts:

- MediaPipe Hand Landmarker `hand_landmarker.task`.
- RTMDet-nano + RTMPose-m through RTMLib/ONNX Runtime.
- Ultralytics YOLOE `yoloe-26s-seg.pt`.
- Optional SAM 2.1 Tiny `sam2.1_t.pt`.
- Gemini 3.6 Flash, with Gemini 3.5 Flash Lite fallback in PS2.
- React Three Fiber/Three.js for the interactive 3D viewer.

Important honesty note for the deck:

- Do not claim all models run in the current demo unless the specific model and clip were executed.
- Label the bundled demo as synthetic.
- State that PS2/PS3 are optional and depend on local weights/API configuration.

### Slide 5 — Product demo and human-in-the-loop review

**Headline:** Searchable evidence, not a black-box prediction.

Demonstrate:

1. Enter the synthetic demo or load a permitted local clip.
2. Play/seek the video and show 2D hand landmarks.
3. Show the synchronized 3D hand-centered pose.
4. Select a timeline event and inspect its timestamp/evidence.
5. Search for an object/action/hand combination.
6. Correct a label or time interval.
7. Show that the timeline and inspector update.
8. Export JSON/CSV.

Call out:

- Original predictions remain immutable.
- Corrections are stored separately with source `manual`.
- Low-confidence or unavailable outputs are visible instead of silently invented.

### Slide 6 — Impact, evaluation, privacy, and next step

**Headline:** A credible path from prototype to validated factory intelligence.

Current product evidence:

- Functional frontend routes and synthetic demo.
- Local FastAPI job bridge.
- Schema-validated PS1/PS4 contracts.
- Search, corrections, and exports.
- Partial-result handling.

Evaluation plan:

- Hand recall on labelled difficult frames.
- Left/right classification error.
- ID switches and short-gap recovery.
- 2D landmark PCK where labels exist.
- Processing FPS and end-to-end latency.
- Contact precision/recall and action precision/timing error.
- Search Precision@5 on hand-aware queries.

Privacy plan:

- No private footage in public Git.
- Local-first browser and localhost processing.
- Explicit opt-in before Gemini video analysis.
- Provider-side cleanup and end-of-event deletion.

Next step:

> Validate PS1 on representative clips first, then connect measured PS2/PS3 results and report honest before/after metrics.

## 15. Demo narration script

Use this short narration while presenting:

> FORGE Studio starts with a factory clip and makes the process searchable. The hand pipeline produces 21-point landmarks and preserves timestamps and confidence. Object and contact reasoning adds evidence about what is visible and what may actually be handled. Optional task understanding turns the sequence into action intervals. The viewer keeps the video, 2D overlay, timeline, and hand-centred 3D pose synchronized. When a prediction is wrong, a reviewer edits the label without destroying the original result. The same evidence can then be searched and exported for evaluation or downstream robotics workflows.

## 16. Responsible claims

Use these phrases in the PPT and demo:

- “Synthetic demo data” when showing the bundled fixture.
- “Hand-centred 3D pose” rather than “factory-world 3D tracking.”
- “Possible contact” unless contact evidence meets the configured rule.
- “Illustrative analytics” for synthetic charts.
- “Local inference bridge” rather than “production backend.”
- “Optional Gemini analysis” and clearly disclose when video leaves the local machine.

Avoid these claims unless backed by a measured evaluation:

- “Production-ready accuracy.”
- “Real-time” without a measured FPS and hardware specification.
- “World-space hand tracking.”
- “Verified grasp” from visual proximity alone.
- “All models integrated” when PS2 packages/weights or PS3 credentials are unavailable.

## 17. License and competition handling

This repository is a hackathon project. Follow the competition rules for code ownership, model licenses, data handling, public release, and end-of-event deletion. Public repositories must contain code and safe synthetic assets only; private clips and derived artifacts belong in the approved private storage location.
