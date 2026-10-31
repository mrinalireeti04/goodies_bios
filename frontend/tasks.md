# FORGE Studio — Implementation Tasks

> **Project:** FORGE Studio | **Challenge:** Mytron Labs FORGE — PS5 Open Innovation  
> **Document:** `tasks.md` | **Status:** In progress — implementation source created; npm build and browser QA pending | **Target:** 24-hour hackathon  
> **Primary deliverable:** A polished landing page leading to a functional, hand-aware factory vision studio.  
> **Source of requirements:** Team decisions + official 17-page *Egocentric Vision Hackathon — Problem Statements* brief.


## Implementation progress (updated October 31, 2026)

- **Implemented in source:** React/Vite frontend layout, landing page, upload CTA, browser-local multi-file library, 2D annotation overlay, interactive 3D skeleton viewer, action timeline, cross-clip search, annotation edits with undo, analytics and JSON/CSV exports.
- **Integrated next:** Local **Import PS1–PS4 output** dialog accepts `hands.jsonl`, `ps2.json`, `actions.json`, `ps4.json`, associates results with the selected media, validates clip IDs, and displays results via the existing panels.
- **Implemented next:** `backend/app.py` implements a localhost FastAPI job bridge with opt-in frontend upload/poll/results loading via TanStack Query, status reporting, action search, and deletion of completed jobs. Missing PS1–PS3 modules produce partial status, not fabricated AI output. The API is not suitable for public exposure.
- **Implemented:** `modules/ps4.py` provides `run_ps4(hands_path, output_dir) -> Path`, writes `ps4.json` and an optional 3D MP4. A synthetic test fixture generator allows full import-flow testing without private video.
- **Verified:** `npm run check` (source syntax, file imports, data contract tests), `python -m unittest discover -s tests -v` (PS4 and backend endpoint tests).
- **Not verified / not complete:** `npm install` (npm registry timed out), Vite production build, screenshot/UI comparison, real PS1–PS3 inference, real hackathon clip evaluation, persistent server jobs, production security or privacy audit of a future Git repository. Do not mark these complete until tested.
- **Deployment note:** The updated local user project directory was not exposed in this tool session. Continue from the downloadable complete source folder; copy it into the authorized repo root when its path is accessible.

---

## 0. Project decisions (locked)

- **Frontend:** React with **JavaScript/JSX** using **Vite** (not TypeScript, not Next.js).
- **Styles:** Tailwind CSS v4, **shadcn/ui**, and **Lucide React** icons (not Material UI).
- **State:** **Zustand** for shared interactive client state; **TanStack Query** for asynchronous/server state, using native **`fetch()`** for HTTP (not Redux, not Axios).
- **Routing:** React Router.
- **3D:** Three.js + React Three Fiber for 21-point hand skeletons.
- **Charts:** Recharts.
- **2D annotations:** HTML5 video/image under a transparent Canvas overlay.
- **Backend later:** Python + FastAPI integrating PS1 (hands), PS2 (objects/contact), PS3 (actions), PS4 (3D).
- **Workflow:** **Frontend first**, all initially powered by **clearly labeled mock data**; integrate real model outputs afterward.
- **Uploads:** Multiple images and videos in a local clip library; image mode and video mode are distinct.
- **Edits:** Human-in-the-loop annotation corrections, preserving originals.
- **Pages:** Landing `/`, Vision Explorer `/studio`, Library `/library`, Search `/search`, Analytics `/analytics`, Export `/export`.
- **Design target:** Match the previously generated **FORGE Studio** reference dashboard, with a separate polished landing page and upload CTA. Primary desktop reference: **1672 × 941**.
- **Privacy:** Local-first; never include competition clips, extracted frames, screenshots, derived private annotations, or secrets in the public Git repository.

### Task conventions

- `[ ]` To do; `[x]` verified complete. **Never check off a task solely because a UI placeholder exists.**
- **P0:** Required for the judged end-to-end demonstration. **P1:** High-value after P0 is stable. **P2:** Stretch/post-hackathon.
- **Owners:** `INT` = you (integration, PS4, frontend); `PS1`, `PS2`, `PS3` = respective teammates. Suggestions can be reassigned.
- Each phase ends with a **gate**: a demonstrable acceptance test, not just source code.
- **Policy reminder:** Event rules require all *project code* to be written during the 24-hour competition. Study, dependency setup, and model downloads may be prepared as permitted by the organizers. Follow the event's actual start time.

## 1. Release targets and critical path

**Demo path (must work):** `/` landing → Upload/Explore Demo → `/studio` → choose clip → video + 2D hand annotations → synchronized 3D hand skeleton + action timeline → select an event → edit a label → search event → export results.

**Critical path:** App setup → design tokens + shared layout → upload/media library → playback synchronization → PS1 adapter + overlay → PS4 viewer → PS2/PS3 adapters → event search + edit/export → evaluation/demo.

**Important fallback:** If a model is delayed, the UI still renders all available modules independently. **Mock results must never be presented as real inference.**

### Stop/go gates

| Deadline (relative to event start) | Gate | Required evidence |
|---|---|---|
| **Hour 2** | G0 — Foundation | Vite boots; routes + providers; shadcn + Tailwind work |
| **Hour 5** | G1 — Visual shell | Landing and dashboard visually coherent at reference resolution |
| **Hour 8** | G2 — Interactive frontend | Multi-upload preview, playback, seek, mock overlay/timeline/3D, search |
| **Hour 12** | G3 — Real hand data | PS1 landmarks over real clip + linked PS4 skeleton |
| **Hour 16** | G4 — Integrated clip | PS1–PS4 outputs shown where available; events/search operate |
| **Hour 21** | G5 — Validated demo | Common demo clips tested; metrics, privacy review, failures documented |
| **Hour 24** | G6 — Submission | Repo, private demo ≤2 min total, 5–6 slides, rehearsal complete |

---

## 2. Phase A — Foundation and design system [P0 | INT | Hours 0–2]

- [ ] **A01** Create a new React **JavaScript** Vite app in the authorized project folder. No separate application is assumed to exist.
- [ ] **A02** Install/configure Tailwind CSS v4 and verify utilities render.
- [ ] **A03** Set up shadcn/ui in JSX mode (`tsx: false`) with `@/` import alias in Vite and `jsconfig.json`; add only needed components as development proceeds.
- [ ] **A04** Install/configure React Router, Zustand, TanStack Query, React Three Fiber/Three.js, Recharts and Lucide React; check compatible package versions.
- [ ] **A05** Add TanStack `QueryClientProvider` and global error/loading boundaries; create a thin native-`fetch()` wrapper that **throws on non-2xx responses**.
- [ ] **A06** Define global design tokens: background `#F5F7FA`, surface `#FFFFFF`, text `#202938`, secondary `#64748B`, border `#E0E5EB`, primary `#2563EB`, video background `#17212F`; spacing/radii, typography and responsive breakpoints.
- [ ] **A07** Build common shells: `LandingLayout`, `StudioLayout`, responsive `AppHeader`, navigation, main content region.
- [ ] **A08** Wire all primary routes: `/`, `/studio`, `/library`, `/search`, `/analytics`, `/export`, plus 404 page.
- [ ] **A09** Create reusable shadcn-based `Panel`, `StatCard`, `EmptyState`, `ErrorState`, `LoadingState`, `ConfirmDialog` components.
- [ ] **A10** Create `.gitignore` immediately for private footage, image extracts, generated annotations, outputs, caches, logs, secrets and environment files.

**Gate G0:** Dev server loads, navigation works, design primitives render, and **no private data** is in tracked assets.

## 3. Phase B — Landing page and entry CTA [P0 | INT | Hours 2–3]

**Goal:** A convincing industrial AI product homepage, visually distinct from but consistent with the light studio dashboard.

- [ ] **B01** Create `LandingNavbar` with FORGE Studio brand, anchor links (Features, How It Works, Use Cases) and prominent **Launch Studio** action.
- [ ] **B02** Build polished dark/industrial `HeroSection` with headline **“Every Frame Has a Story. Understand It.”**, product description and two CTAs: **Upload Footage** and **Explore Demo**.
- [ ] **B03** Implement `ProductPreview` using a synthetic/dashboard reference or a designed placeholder—**never private source footage** on the public landing page.
- [ ] **B04** Add `HowItWorks`: Upload → Track Hands → Detect Objects → Understand Actions → Explore/Export.
- [ ] **B05** Add six capability cards: Hand Tracking, Object/Interaction Detection, Task Timeline, 3D Hand Explorer, Video Search, Workflow Analytics.
- [ ] **B06** Add Use Cases (factory engineers, labeling teams, robotics researchers, training) and concise, non-overclaiming privacy statement.
- [ ] **B07** Add final CTA + footer; responsive mobile/tablet layout and keyboard-accessible CTA buttons.
- [ ] **B08** Wire **Upload Footage** to shared upload dialog; wire **Explore Demo** to `/studio` with explicit **synthetic demo** mode. CTA actions must not be inert.

**Gate:** A new visitor understands what FORGE does, can start uploading, and can enter a demo without a real file.

## 4. Phase C — Studio layout faithful to the reference [P0 | INT | Hours 3–5]

- [ ] **C01** Build a shared light-theme `StudioLayout`: header tabs (Vision Explorer, Search, Analytics, Export), left panel and main content.
- [ ] **C02** Implement `UploadSidebar`: drag/drop trigger and filters for object, hand (all/left/right), action, confidence; reset button.
- [ ] **C03** Implement four KPI cards: Clips Indexed, Events Found, Objects Detected, Average Confidence. **Label simulated values in mock mode**; do not fabricate a meaningful confidence average from missing scores.
- [ ] **C04** Create desktop grid: approximately **280–300 px sidebar**, **wide primary video viewer**, **narrower 3D viewer** (~360–390 px at the reference width).
- [ ] **C05** Add `EventTimeline` under the viewer: colored Pick / Hold / Fasten / Inspect / Unknown events, visible time scale and playhead.
- [ ] **C06** Add lower three panels: `ObservationInspector`, `SearchPanel` (compact results) and `ClipInformation`.
- [ ] **C07** Add informative first-use/empty/loading/partial-result states; no dummy button that appears functional but is not.
- [ ] **C08** Compare actual UI to the generated screenshot at **1672 × 941** and a smaller laptop viewport; fix overflow, truncation, typography, spacing and panel proportions.
- [ ] **C09** On smaller screens, collapse sidebar to a drawer and move 3D viewer/bottom details into tabs or stacked panels.

**Gate G1:** A reference-matched, responsive, navigable dashboard exists before model integration.

## 5. Phase D — Multiple uploads and media library [P0 | INT | Hours 5–6]

- [ ] **D01** Create reusable `UploadDialog` + `UploadDropzone` supporting multiple `.jpg/.jpeg/.png` and compatible video inputs (`.mp4`; attempt `.mov` if browser decoding permits).
- [ ] **D02** Validate MIME/signature where possible, file size, unsupported types and duplicate selection; show actionable errors.
- [ ] **D03** For local frontend demo, create browser `ObjectURL` previews; release/revoke URLs on removal, replacement and app cleanup. **Do not put private files in `public/`.**
- [ ] **D04** Build `ClipLibrary` with name, type, duration/resolution where available, thumbnail or icon, and status (`local`, `queued`, `processing`, `completed`, `partial`, `failed`).
- [ ] **D05** Implement add, select, remove, and batch queue actions. Preserve existing library items when additional files are uploaded.
- [ ] **D06** On image selection: show still image + labels/3D single pose; hide irrelevant video timeline and play controls.
- [ ] **D07** On video selection: show player + timeline + temporal annotations + 3D animation.
- [ ] **D08** Create a mock processing service that resolves to schema-valid **synthetic** data and visibly identifies itself as demo output.
- [ ] **D09** Reserve server job progress wiring for the integration phase; do not claim actual ML processing before backend exists.

**Gate:** Upload/select **at least 2 videos and 1 image**, switch among them, and remove a file without losing others or leaking object URLs.

## 6. Phase E — Interactive Vision Explorer [P0 | INT | Hours 6–8]

### E1. Video & image display
- [ ] **E01** Implement video element with play/pause, seek, playback speed, duration and fullscreen; image viewer supports fit/zoom as appropriate.
- [ ] **E02** Implement transparent `AnnotationCanvas` above media for 2D hand keypoints, hand connections, IDs, object boxes, labels and selection hit-testing.
- [ ] **E03** Map source-frame pixel coordinates to displayed content accurately, including `object-fit: contain`, letterboxing, ResizeObserver and fullscreen resize.
- [ ] **E04** Provide overlay toggles: boxes, 21 landmarks, IDs, confidence, objects and contacts.
- [ ] **E05** Render **no hand** if detection is absent or stale. Clearly distinguish detected and interpolated landmarks; do not blindly hold last pose over long gaps.

### E2. Playback sync and timeline
- [ ] **E06** Choose the media element as master clock; use `requestVideoFrameCallback` when supported, fallback as needed. Expose `currentTimeMs` to dependent UI at a controlled rate.
- [ ] **E07** Build nearest-frame lookup using clip-relative `t_ms`, with explicit max tolerance and source frame metadata.
- [ ] **E08** Clicking an event seeks the video, updates overlay, selected observation, 3D pose and timeline cursor to the same timestamp.
- [ ] **E09** Add timeline zoom/scroll only if P0 timeline navigation already works smoothly.

### E3. PS4 3D skeleton viewer
- [ ] **E10** Build `HandViewer3D` using React Three Fiber; render all 21 joints and known bone connections for selected hand.
- [ ] **E11** Add orbit/zoom/reset controls, grid and axis indicator; allow choosing left/right hand.
- [ ] **E12** Consume 3D data in **hand-centered meters**, keep a consistent scale and camera, and show `Pose unavailable` when data is missing.
- [ ] **E13** Synchronize 3D pose to selected media time; never claim global factory/world position from MediaPipe world landmarks.
- [ ] **E14** Use refs/memoization and sensible update throttling; do not push full per-frame arrays through Zustand.

### E4. Inspector & mock search
- [ ] **E15** Implement `ObservationInspector`: active hand ID/side, object label, interaction/action, event timestamps, source and confidence/unknown.
- [ ] **E16** Implement a basic in-panel search for object + hand + action from mock event records; clicking a hit opens its clip at the event timestamp.
- [ ] **E17** Provide a short synthetic sample with a few steps and believable missing-hand/unknown intervals to test all empty-state behavior.

**Gate G2:** An end-to-end **mock** demo works: choose a clip, seek, see synchronized 2D/3D/timeline, search an event and open it. No real inference is implied.

## 7. Phase F — Data architecture and modular PS contracts [P0 | INT + PS1/PS2/PS3 | Starts in parallel]

### F1. Freeze the shared contract **before** teammate code integration
- [ ] **F01** Approve schema `1.0` with one `clip_id` naming convention, integer millisecond timestamps, `frame_idx`, original-pixel bounding boxes and 21-landmark arrays.
- [ ] **F02** **PS1** produces `hands.jsonl`: `schema_version`, `clip_id`, `frame_idx`, `t_ms`, `frame_width`, `frame_height`, `hands[]` containing `hand_id`, `side`, `handedness_score`, `bbox_px`, `kp2d_px[21]`, optional `kp3d_hand_m[21]`, `source` (`detected`/`interpolated`).
- [ ] **F03** **PS2** produces `ps2.json`: `frames[]` (`objects[]`, `hand_states[]`) and `contacts[]` with `event_id`, `start_ms`, `end_ms`, `hand_id`, `object_id`, `object_label`, `state`, `interaction`, `confidence`.
- [ ] **F04** **PS3** produces `actions.json`: `actions[]` with `action_id`, interval, `verb`, `object_label`, `object_id`, `hand`, `hand_ids`, `confidence`, `source`, `evidence_event_ids`.
- [ ] **F05** **PS4** produces `ps4.json` and `hands_3d.mp4`: `coordinate_frame: hand_centered`, `render_fps`, `frames[]` with timestamped per-hand `joints_3d_m[21]` and matching `hand_id`.
- [ ] **F06** Standardize nullable/unknown states. Never invent confidences, contact, action, object names or landmarks when unavailable.
- [ ] **F07** Preserve consistent hand IDs from PS1 and object IDs from PS2; keep frame and event indexing separate.
- [ ] **F08** Use half-open event intervals `[start_ms, end_ms)`; allow overlapping actions, and define expected frame time tolerance.
- [ ] **F09** Create **small synthetic** valid and invalid JSON fixtures for PS1–PS4, including occlusion, missed hand, mixed-hand events and empty timelines.
- [ ] **F10** Add schema validators/adapter tests: 21 point shape, bounds, timestamps, referenced IDs, clip IDs and negative/missing cases.

### F2. App state & data-access boundaries
- [ ] **F11** `useMediaStore`: selected clip, preview URL reference, mode, local upload queue and UI selection only.
- [ ] **F12** `usePlaybackStore`: play/pause, current time, speed, selected hand and overlay visibility; avoid unnecessarily frequent global updates.
- [ ] **F13** `useAnnotationStore`: selection, local edits, edit history/draft; **do not copy full query datasets into Zustand**.
- [ ] **F14** `useSearchStore`: query text, filters, selected result; remote results remain in TanStack Query cache.
- [ ] **F15** Create `mediaService`, `annotationService`, `searchService` behind a mock/real implementation switch; mock outputs must have API-compatible shapes.
- [ ] **F16** TanStack query keys: `['clips']`, `['job', id]`, `['hands', clipId, window]`, `['contacts', clipId]`, `['actions', clipId]`, `['pose3d', clipId, window]`, `['search', query, filters]`, `['analytics', clipId]`.
- [ ] **F17** Ensure `fetch` wrapper throws on HTTP failure, propagates abort signals and has sensible request timeouts/retry behavior.

**Gate:** Every frontend view can render independently from valid mock PS files; schema failures produce explicit, useful error states.

## 8. Phase G — Real PS1 + PS4 integration [P0 | INT + PS1 | Hours 8–12]

- [ ] **G01** Accept PS1 teammate's `run_ps1(video_path, output_dir) -> Path` and validate generated `hands.jsonl` on one official clip.
- [ ] **G02** Normalize PS1 output through `ps1Adapter.js`; do **not** add MediaPipe-specific assumptions inside React components.
- [ ] **G03** Verify that handedness is correct for non-mirrored head-mounted footage, IDs remain stable, and empty frames do not show stale data.
- [ ] **G04** Render 21 landmarks, boxes and labels aligned to real video frames; test resizing and scrubbing.
- [ ] **G05** Build PS4 consumer `run_ps4(hands_path, output_dir) -> Path`; use PS1 hand-centered 3D landmarks as baseline rather than rerunning a tracker.
- [ ] **G06** Write `ps4.json` and a 3D rendered video; use the 3D JSON for interactive React Three Fiber playback.
- [ ] **G07** Check axis conventions, scale and handedness visually; display pose uncertainty and gaps.
- [ ] **G08** Record PS1 performance: processed FPS, detected-frame share, left/right consistency and a small manual frame-level accuracy evaluation on difficult cases.

**Gate G3:** Real clip shows accurate 2D hands plus synchronized 3D skeleton; missing detections visibly fail gracefully. **PS1 accuracy remains the top scoring priority (30/100 points).**

## 9. Phase H — PS2 + PS3, search and workflows [P0 | INT + PS2 + PS3 | Hours 12–16]

- [ ] **H01** Accept `run_ps2(video_path, hands_path, output_dir) -> Path`; validate PS2 objects, contact states and intervals.
- [ ] **H02** Add `ps2Adapter.js` and render object boxes, object selection, probable interaction labels and event intervals.
- [ ] **H03** Distinguish **object visible**, **possible contact/handling** and **semantic action**. Visual proximity is not proof of physical contact.
- [ ] **H04** Accept `run_ps3(video_path, hands_path, ps2_path, output_dir) -> Path`; validate action vocabulary, intervals and evidence references.
- [ ] **H05** Add `ps3Adapter.js`: action timeline segments, PS3 labels, confidence if real, and time-seeking.
- [ ] **H06** Support partial completion: PS1-only clips are explorable even if PS2, PS3 or PS4 fail.
- [ ] **H07** Build `/search` with structured filters: object label, hand side, action, contact type, clip; return **clip + timestamp** and open the relevant segment.
- [ ] **H08** Implement server-side/adapter-friendly search indexing (`clip_id`, event ID, interval, object, hand, action, source evidence); do not scan full videos in the browser per query.
- [ ] **H09** Group adjacent frame detections into interaction intervals; clearly mark unknown periods.
- [ ] **H10** Demonstrate **at least three truthful queries** on common demo clips; validate a few returned matches by visual review.

**Gate G4:** At least one official clip uses actual module outputs in the UI. Search results have verified timestamps and evidence, not invented examples.

## 10. Phase I — Editable annotations and export [P0 basic / P1 advanced | INT | Hours 16–18]

- [ ] **I01 [P0]** Enable editing of object labels, hand-side corrections and action verbs for a selected observation.
- [ ] **I02 [P0]** Enable editing action/contact start and end times with validation (`0 ≤ start_ms < end_ms ≤ duration_ms`).
- [ ] **I03 [P0]** Keep original model results immutable; store edits separately (`annotation_id`, target, old value, new value, updated time, source=`manual`).
- [ ] **I04 [P0]** Implement Save/Cancel, dirty-state indicator and Undo for at least the last edit; avoid claiming an edit is persisted before receiving a response.
- [ ] **I05 [P0]** Reflect saved edits in inspector, search results and timeline; invalidate affected TanStack Query data when server-backed.
- [ ] **I06 [P0]** Implement `/export`: normalized **JSON** export including metadata, predictions, corrections and schema version.
- [ ] **I07 [P1]** Add CSV export for contacts/actions and selectable clip scope.
- [ ] **I08 [P1]** Add draggable/resizable bounding-box correction and edit-history view only after basic edits are stable.
- [ ] **I09 [P1]** Implement privacy-safe annotated video export if time and backend capacity permit; save only to authorized private locations.

**Gate:** Select an incorrect label → correct → save → search/timeline reflects correction → exported JSON includes the correction while preserving the original prediction.

## 11. Phase J — Analytics and polish [P1 | INT | Hours 18–19]

- [ ] **J01** Implement `/analytics` with Recharts: action durations, object appearances, probable interaction counts and hand detection coverage.
- [ ] **J02** Make denominator, time range and unknown intervals explicit; distinguish detected presence from verified tool usage.
- [ ] **J03** Implement clips/jobs overview with real (not fabricated) counts; report null/unknown when confidence is unavailable.
- [ ] **J04** Add compact analytics preview to Vision Explorer and deep links to `/analytics`.
- [ ] **J05** Add accessibility checks: keyboard navigation, focus visibility, accessible names, reasonable contrast, disabled actions and live processing status.
- [ ] **J06** Test loading/error/empty/partial states for all pages, including unsupported media, no matching search result and unavailable 3D pose.
- [ ] **J07** Profile playback and 3D rendering. Use memoization, bounded frame buffers and throttling rather than full dataset re-render per frame.
- [ ] **J08** Capture comparison screenshots at the desktop reference size and a normal laptop size to check visual fidelity.

**Gate:** Analytics reflect event data honestly, the interface is stable, and the live demo has no blocking interaction bugs.

## 12. Phase K — Backend API and processing jobs [P0 for real integration | INT | Parallel/as needed]

> Frontend should depend on a **service interface**, not directly on FastAPI or the internal functions of PS modules. The backend may be minimal in the hackathon; local processing is preferred.

- [ ] **K01** Implement FastAPI multipart file intake with local private storage and safe per-clip IDs; accept multiple uploads without exposing raw files publicly.
- [ ] **K02** Connect orchestrator: `run_ps1` first → PS2/PS3 when their inputs are ready; PS4 can run independently from PS1. Preserve per-module status and failures.
- [ ] **K03** Implement `POST /api/jobs`, `GET /api/jobs/{id}` and `GET /api/clips`; return `queued`, `processing`, `partial`, `completed`, `failed` states.
- [ ] **K04** Implement `GET /api/clips/{id}` and authorized media streaming/access for the local studio.
- [ ] **K05** Implement interval-windowed endpoints for hands and 3D poses, and event-level endpoints for contacts/actions.
- [ ] **K06** Implement `POST /api/search`, `PATCH /api/annotations/{id}` and export endpoint; validate all edits server-side.
- [ ] **K07** Switch mock-to-real services through one configuration setting; clearly indicate mock mode vs real mode on screen.
- [ ] **K08** Configure TanStack Query polling only while jobs are unfinished; handle retries, errors, cancel/retry and `partial` outputs.
- [ ] **K09** Add explicit deletion of clip, associated files, temporary frames, job output and cloud copies as applicable.

**Target API shape:**

| Method | Route | Purpose |
|---|---|---|
| POST | `/api/jobs` | Create upload/processing job |
| GET | `/api/jobs/{id}` | Job and per-module statuses |
| GET | `/api/clips` | Media library |
| GET | `/api/clips/{id}` | Metadata |
| GET | `/api/clips/{id}/hands?start_ms=&end_ms=` | PS1 window |
| GET | `/api/clips/{id}/contacts` | PS2 intervals |
| GET | `/api/clips/{id}/actions` | PS3 steps |
| GET | `/api/clips/{id}/pose3d?start_ms=&end_ms=` | PS4 window |
| POST | `/api/search` | Structured event search |
| PATCH | `/api/annotations/{id}` | Save a manual edit |
| GET | `/api/clips/{id}/export` | Download annotations |

## 13. Phase L — Evaluation, privacy and submission [P0 | Whole team | Hours 19–24]

### Evaluation
- [ ] **L01** Run on the **organizer-designated common demo clips**, not only handpicked easy clips.
- [ ] **L02** Manually annotate/sample approximately 100 frames if time allows; report hand detection recall on visible hands, handedness correctness and difficult failure cases.
- [ ] **L03** Record processing throughput (FPS) and hardware used; distinguish processing FPS from source video FPS.
- [ ] **L04** Visually check object/action search top results and representative contact events; report sample precision rather than unvalidated claims.
- [ ] **L05** Show at least one known failure (gloves, occlusion, small object, blur) and how the application represents uncertainty.
- [ ] **L06** Test pipeline with a corrupted/unsupported file, no hands, a missing PS2 result, and an absent 3D pose.

### Privacy / compliance
- [ ] **L07** Confirm `.gitignore` covers `data/`, `uploads/`, `outputs/`, `tmp/`, `private_data/`, frame extracts, videos, screenshots, models' private caches and `.env*` secrets (except `.env.example`).
- [ ] **L08** Inspect Git tracked files and history before push; ensure no private footage, frames, screenshots or sensitive company information are included.
- [ ] **L09** Keep real-video dashboard access on localhost or a properly access-controlled environment; use synthetic previews in any public pages or repository.
- [ ] **L10** Delete local/cloud/API uploaded copies at event end as required by organizers. Do not use analytics/telemetry services that silently upload private footage.

### Submission and presentation
- [ ] **L11** Public GitHub contains source, setup README, license/credit notices for models and libraries, problem statement, exact run command and hardware/compute disclosure; **no private media**.
- [ ] **L12** Upload processed demo clips to the **designated private Drive folder**, totaling **2 minutes or less**.
- [ ] **L13** Create **5–6 slides**: problem, product demo, technical approach + hand tracking, measured results, failure cases, value + next steps. Include GitHub link on first slide.
- [ ] **L14** Rehearse 5-minute presentation + 3-minute Q&A; every team member can explain their component and shared schema.
- [ ] **L15** Prepare reliable fallback: an authorized private demonstration recording and locally cached result artifacts, if live inference is slow.

**Gate G5/G6:** Demonstration works on real common clips, metrics are supported, repo is privacy-safe, private demo/slides submitted before code freeze.

---

## 14. Suggested ownership and parallel execution

| Owner | Primary work | Integration contract | Must deliver first |
|---|---|---|---|
| **PS1 teammate** | Hand tracker accuracy, stable IDs, landmarks, FPS | `run_ps1(video_path, output_dir) -> hands.jsonl` | Synthetic fixture + first real valid JSONL |
| **PS2 teammate** | Objects, per-hand states, grab/release intervals | `run_ps2(video_path, hands_path, output_dir) -> ps2.json` | First objects + contacts sample |
| **PS3 teammate** | Action timeline, step labels, timing | `run_ps3(video_path, hands_path, ps2_path, output_dir) -> actions.json` | First actions sample |
| **You (INT)** | Landing, studio, upload, state, search/edit/export, PS4, backend wiring | `run_ps4(hands_path, output_dir) -> ps4.json` and `hands_3d.mp4` | Running mock-driven frontend + 3D viewer |

**Integration rules:**

- PS1 is the **only source of hand identity**; PS2/PS3/PS4 reference its `hand_id`.
- Each module returns an output **file path** and does not require importing another teammate's internal objects.
- PS3 may infer action meaning from video but normalizes IDs and timestamps before export.
- PS4's baseline uses PS1's 3D landmarks; it does **not** require SLAM or full hand meshes.
- Frontend reads normalized records via adapters/services, not raw model internals.
- Produce a small valid example immediately; **do not wait for a fully trained/working model to test integration**.

## 15. Acceptance criteria: minimum end-to-end product

### P0 — Ship these first
- [ ] Landing with live Upload Footage and Explore Demo CTAs.
- [ ] Multi-file library where user can select/remove 2+ videos and images.
- [ ] Video/image viewer with aligned PS1 hand overlays and an explicit no-hands state.
- [ ] Synchronized PS4 interactive 3D skeleton, rotating/zooming and tracking video time.
- [ ] PS2 object labels/boxes and probable contact evidence where available.
- [ ] PS3 clickable task-step timeline; seek updates all panels.
- [ ] Search by object, hand and action across indexed clips; results jump to timestamps.
- [ ] Correct an annotation, save it, reflect it in results and export JSON.
- [ ] Work with partial/missing module outputs without crashing.
- [ ] Real common clips demonstrated, hand tracking performance documented, private data not publicly exposed.

### P1 — Only after P0 is stable
- [ ] Dedicated Analytics page with truthful durations and interaction counts.
- [ ] CSV export and enhanced edit history/undo.
- [ ] Richer search filters and saved/similar events.
- [ ] Drag-to-edit bounding boxes and smoother timeline zoom.

### P2 — Do **not** jeopardize the 24-hour submission
- [ ] Free-form natural-language retrieval with embeddings / video LLM.
- [ ] Search by example image crop across videos.
- [ ] Automatic private training-video compilation.
- [ ] CCTV + synchronized multi-worker factory optimization and SLAM/world-frame 3D.
- [ ] Authentication, multi-user collaboration and production deployment.

## 16. Developer test scenarios / definition of done

- **Upload:** Select 3 files; switch selected file; remove one; library still contains remaining two.
- **Image mode:** No fake temporal timeline or playback; a single pose is shown if available.
- **Sync:** Seek to 12.4 s; video, overlay, 3D pose, timeline marker and inspector reflect ~12,400 ms within declared tolerance.
- **No hands:** A frame with `hands: []` displays no ghost skeleton; missing 3D data shows a clear unavailable state.
- **Alignment:** Resize the viewer and fullscreen; 2D boxes remain on the same visible source pixels.
- **Search:** Filter `object=screwdriver`, `hand=right`, `action=fasten`; matching results jump to correct clip and timestamp; unsupported queries return zero results honestly.
- **Correction:** Change an action to `inspect`, save, see updated timeline/search/export; original model prediction remains recoverable.
- **Partial failure:** PS3 unavailable, PS1 works → 2D/3D explorer stays usable with clear partial warning.
- **Privacy:** Public repo contains no raw or processed private footage, screenshots, frame crops or hidden cache.
- **Evaluation:** Show real detection share, FPS and manually verified failure examples for the common demo clips.

## 17. Scope-control and contingency rules

1. **Do not spend more than ~3 hours on the landing page.** The visual studio and tracking quality determine success.
2. If behind by **hour 8**, stop adding decorative features or new pages; finish synchronized Vision Explorer.
3. If behind by **hour 12**, use PS1 3D landmarks for PS4 and defer advanced rendering/meshes.
4. If behind by **hour 16**, prioritize one official clip with credible end-to-end results over processing all twenty clips poorly.
5. If inference/search is unreliable, **label uncertainty**; never substitute simulated output without indicating it.
6. If the selected browser cannot decode a MOV, show a clear compatibility error and support MP4 instead.
7. No public demonstration of faces, screens, documents or company footage; only the private authorized judging environment may display restricted material.
8. The full factory-wide pipeline/CCTV concept remains a **future product roadmap**, not a claim supported by the supplied head-mounted-video-only dataset.

---

## 18. First work session — exact next tasks

Do these in order when implementation begins:

- [ ] **NEXT-01:** Confirm the authorized repository path; initialize Vite React JSX and Git safely.
- [ ] **NEXT-02:** Establish Tailwind, shadcn/ui, routes, providers and reference palette.
- [ ] **NEXT-03:** Build landing navbar + hero + working upload dialog/Explore Demo CTA.
- [ ] **NEXT-04:** Build dashboard shell at the reference desktop size with empty panels.
- [ ] **NEXT-05:** Add synthetic clip + schema-valid PS1–PS4 fixtures and simple playback/seek.
- [ ] **NEXT-06:** Show synchronized 2D hands + 3D pose + selected event from fixture.
- [ ] **NEXT-07:** Request first real output from PS1 and perform schema acceptance test.

**No implementation code has been written by this document. Check tasks off only after the implementation and acceptance tests are complete.**
