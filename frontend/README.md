# FORGE Studio — Factory Vision Intelligence

> **Implementation status:** Frontend-first prototype with local PS1–PS4 result-file import and a standalone PS4 Python module. The bundled demo observations are **synthetic illustrations**, not results on the private hackathon clips. A **local FastAPI job bridge** is now implemented. It runs installed PS modules but PS1–PS3 teammate implementations are not bundled.

A React + Vite workspace for hand-aware factory video exploration: multi-file upload, PS1–PS4 JSON/JSONL import, 2D landmark and object overlays, a rotatable 3D hand skeleton, a clickable action timeline, cross-clip object/action search, edit-and-undo annotation corrections, basic analytics, and JSON/CSV export.

## Tech stack

- React 19 + JavaScript/JSX and Vite 7
- Tailwind CSS v4, locally implemented **shadcn/ui-style components** (Radix primitives), Lucide React
- Zustand for playback/UI state
- TanStack Query for asynchronous search/server state; native `fetch` for future API calls
- React Three Fiber + Three.js for 3D hand pose display
- Recharts for illustrative analytics
- React Router for navigation

## Getting started

Requires Node.js 20.19+ or newer (Node 22 recommended), and network access to npm.

```bash
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

To run the production build:

```bash
npm run build
npm run preview
```

Dependency-free fixture/import checks:

```bash
npm run check
```

**Environment note:** The source bundle was prepared in a tool environment without access to the npm package registry, so the dependency install, Vite build and browser rendering could **not** be verified there. `npm run check` does run without installing dependencies.

## Routes

| Route | Experience |
| --- | --- |
| `/` | Product landing page; Upload Footage / Explore Synthetic Demo CTA |
| `/studio` | Video/image viewer, 2D overlays, 3D hand explorer, event timeline, inspector |
| `/library` | Multi-file local preview library |
| `/search` | Filterable indexed synthetic action events |
| `/analytics` | Illustrative timeline analytics |
| `/export` | JSON/CSV export of corrected synthetic annotations |

## User workflow

1. On the landing page, select **Explore Synthetic Demo** to see the integrated PS1–PS4-shaped data.
2. In Vision Explorer, play/seek the 84-second simulated event timeline. Hands and objects update; a 10-second interval intentionally has no hands.
3. Rotate the Three.js 3D pose. Select either hand.
4. Click a timeline segment to jump to the action; edit the label or event interval in the inspector. Undo is available.
5. Search actions by label and hand; open matching timestamped results. Exports include the local corrections.
6. **Upload Footage** imports multiple real files as browser-only previews. Real files initially have **no fabricated AI labels**.
7. Select the uploaded file in Vision Explorer, click **Import PS1–PS4 output**, and choose the output files from the teammates. Matching annotations, tasks and 3D poses become visible for that media; the browser does not run model inference.

## Team integration contract (schema 1.0)

PS1 is the authoritative source of hand IDs and all downstream modules reference its IDs. Timestamps use integer milliseconds from clip start; events use half-open `[start_ms,end_ms)` intervals.

| Python module | Output | Frontend adapter |
| --- | --- | --- |
| `run_ps1(video_path, output_dir)` | `hands.jsonl` frame records, `kp2d_px[21]`, `kp3d_hand_m[21]` | `src/services/adapters/ps1Adapter.js` |
| `run_ps2(video_path, hands_path, output_dir)` | `ps2.json` objects, hand states, contact intervals | `ps2Adapter.js` |
| `run_ps3(video_path, hands_path, ps2_path, output_dir)` | `actions.json` action intervals | `ps3Adapter.js` |
| `run_ps4(hands_path, output_dir)` | `ps4.json` hand-centered 3D pose frames and optional MP4 | `ps4Adapter.js` |

### Local model-result import (works now)

Select an uploaded image/video and use **Import PS1–PS4 output**. Accepted filenames: `hands.jsonl`, `ps2.json`, `actions.json`, `ps4.json`. You may attach one module at a time; all attached files must have the same `clip_id`. The model output is associated with the currently selected browser media. Choose the correct video—automatic camera-to-result identification is not implemented.

Frame landmarks are associated using bounded timestamp-nearest matching; long gaps do not show a stale hand skeleton. The original media and annotations remain in browser memory for the session and disappear on refresh. Edits are applied as separate overrides and included in exports.

### PS4 Python module (works now)

```bash
# Requires Python + OpenCV for optional video output
python -m modules.ps4 private_data/factory_001/hands.jsonl private_data/factory_001/output
python -m unittest discover -s tests -v
```

This writes `ps4.json` and, if OpenCV video encoding works, `hands_3d.mp4`. It only consumes PS1 3D hand-centered landmarks; it cannot recover world-space hand trajectories.

### End-to-end synthetic integration fixture (no confidential footage)

```bash
python -m scripts.create_synthetic_test_assets
```

The command creates `outputs/synthetic_test/factory_sample.mp4`, `hands.jsonl`, `ps2.json`, `actions.json` and `ps4.json`. Open the studio, upload the MP4, then use **Import PS1–PS4 output** for the four JSON/JSONL files. The entire sample is generated locally and deliberately synthetic. `outputs/` is Git-ignored.

### Planned backend (not implemented)

### Local FastAPI job bridge (implemented, optional)

The backend lives at `backend/app.py` and uses a temporary, Git-ignored `.local_data/` directory.

```bash
# From project root, with your Python environment activated:
python -m pip install -r backend/requirements.txt
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
cp .env.example .env.local
# In .env.local, set VITE_USE_REAL_API=true
npm install
npm run dev
```

Now selecting images/videos uploads them to your **local** FastAPI service (they are not sent to a third-party cloud). The frontend continues showing local media while the backend reports `queued`, `processing`, `partial`, or `completed` status. When a teammate's `modules/ps1.py`, `modules/ps2.py`, or `modules/ps3.py` exists and obeys the agreed functions, the backend runs it, then uses PS1 output to run the included PS4 module. Missing models are reported as *partial results* and no detections are invented. Still-image inference is not yet implemented. For now, importing PS outputs manually also remains available.

Core endpoints:

- `POST /api/jobs` — upload a file (`file`, optional `client_clip_id`) and create a local job.
- `GET /api/jobs/{id}` — job status and module/errors information.
- `GET /api/clips` / `GET /api/clips/{id}` — session clip metadata.
- `GET /api/clips/{id}/media` — local-only media playback.
- `GET /api/clips/{id}/hands|contacts|actions|pose3d` — model outputs, optional `start_ms`/`end_ms` filtering.
- `POST /api/search` — simple action/label search where PS3 output exists.
- `DELETE /api/jobs/{id}` and `DELETE /api/data` — remove finished jobs and their server files. Active jobs must finish before deletion.

**Security warning:** This is a single-user localhost backend without authentication. Never run it on `0.0.0.0` or expose it through a public tunnel while using private factory footage. Uploaded file contents and outputs remain on disk under `.local_data/` until explicitly removed. Clear all local and cloud copies at the end of the event. Refreshing the browser alone does not delete backend uploads.

Backend tests:

```bash
python -m unittest discover -s tests -v
```

Not implemented yet: real PS1–PS3 models, persistence across API restarts, background job cancellation, large-scale media indexing, server-side annotation-edit storage, and a secure multi-user deployment. Those are not required to test the modular integration on localhost.


## Privacy safeguards

- **Do not commit private factory footage, extracted frames, clips, screenshots, or private derived annotations.** `.gitignore` excludes common media, inference filenames and processing directories. `project_notes/` contains team reference documents and is Git-ignored.
- The only media assets bundled in `public/assets` are the **AI-generated concept dashboard reference** and **synthetic vector workstation illustration**. Neither contains competition data.
- Uploads are represented by temporary browser object URLs; they are revoked when files are removed, and cleared on page reload.
- Do not publicly host private recordings or display faces, screens, or company documents. Handle any future API-uploaded media with restricted access and end-of-event deletion.
- In the default frontend-only mode, uploads never leave the browser. In opt-in localhost backend mode, video is copied to `.local_data/` and must be deleted after the event. Neither mode sends media to a third-party cloud service. There is no analytics/telemetry instrumentation.

## Known limitations / next implementation tasks

- The React app cannot infer hands, objects or task actions from new video yet. You can import existing PS1–PS4 module outputs today; automatic Python execution/API job orchestration remains future work.
- Edited synthetic annotations persist in the in-memory session only; reload loses changes, but exporting preserves them.
- Search covers the synthetic example and imported PS3 action events; it is not an open-ended video-language model.
- Video browser codec support varies; MOV may not decode without conversion to MP4.
- 3D is **hand-centered**, not absolute spatial trajectory in a factory/world coordinate system.
- Browser rendering, npm dependency resolution, and production build have not been verified in this isolated environment due to npm registry DNS failure. `npm run check` validates JS/JSX syntax, imports and data/lookup logic without dependencies. Python PS4 tests also pass.
- The included dashboard preview is a **concept**, not an image of the functioning product.

## Hackathon compliance

The official brief requires work on common demo clips and emphasizes hand tracking accuracy (30 points). Public GitHub must exclude competition footage. Processed demo videos (≤2 minutes total) belong only in the designated private Drive folder, and a 5–6 slide presentation is required. All project code must be written within the competition's authorized 24-hour period.
