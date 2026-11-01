# Local hand inference

Run from the BIOS project root:

```sh
uv pip install --python backend/.venv/bin/python -r backend/requirements.txt
backend/.venv/bin/python backend/setup_models.py
backend/.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

FFmpeg must be installed and on PATH. RTMLib downloads public model weights on first use; later runs use its local cache. PS1/PS4 hand and pose inference runs locally. PS3 task analysis is separate and off by default; when enabled with the Studio upload checkbox, the video is sent to Google Gemini and the returned artifacts are stored locally. Configure `GEMINI_API_KEY` before enabling PS3. Leave the checkbox off to keep all video processing local. Studio uses the local API by default; VITE_USE_REAL_API=false explicitly disables it.

On Apple Silicon, RTMPose automatically requests ONNX Runtime's CoreML provider (`FORGE_RTM_DEVICE=auto`, the default). Set `FORGE_RTM_DEVICE=cpu` to force CPU or `FORGE_RTM_DEVICE=coreml` to request CoreML explicitly. Compiled CoreML partitions are cached in `.local_data/coreml_cache`; override this with `FORGE_COREML_CACHE`. Job logs report actual providers separately for RTMDet and RTMPose. A model that fails CoreML initialization or its first inference is retried on CPU; that model will not receive CoreML acceleration for that run. CoreML graph support varies by model and ONNX Runtime/macOS version.

Each video is processed at every decoded frame. RTMPose provides original-pixel 2D landmarks; MediaPipe supplies hand-centred metric 3D landmarks. Unmatched detections get a cropped MediaPipe retry. Unavailable 3D remains null. Before filling coordinate gaps, a clip-wide post-pass links detections by wrist motion over gaps up to 60 frames and uses confidence-weighted left/right votes to repair unknown or contradictory labels. Weak or conflicting evidence stays unknown. L/R identify anatomical sides, not guaranteed persistent identities through crossings.

hands.raw.jsonl retains unsmoothed hybrid measurements. hands.jsonl fills at most 3 consecutive missing frames per side and coordinate modality, between valid endpoints only, using timestamp-weighted linear interpolation. It rejects implausible movement, large size changes, frame-edge cases and overlapping duplicate hands. Longer and unbounded gaps stay empty. Estimated coordinates have source=interpolated and per-modality provenance. No ballistic extrapolation is used.

Results include overlay.mp4 (H.264), hands.jsonl and ps4.json. The API serves the original at /api/clips/{job}/media and overlay at /api/clips/{job}/overlay. Both video panes share transport controls; the original video's clock drives the 3D viewer. 3D is hand-centred metres, not absolute factory position; the viewer magnifies these coordinates for display.

PS2/PS3 are optional and their absence does not block hand results. PS1 (including its PS4 pose output) runs first; then PS2 object/contact analysis and optional PS3 action analysis are scheduled concurrently. PS3 waits for PS2 only at the final interaction-enrichment step, so the expensive video analysis overlaps. A shared two-worker scheduler bounds concurrent PS2/PS3 work across uploads. PS2 uses Ultralytics/YOLOE and requires its model weights; CUDA is selected when available, otherwise Apple MPS when available, then CPU. Each job writes PS2 cache and temporary clips under its private job folder. When PS3 is selected, optional PS2 Gemini ledger/naming calls are disabled to avoid duplicating cloud processing and competing for quota. Install the backend requirements before enabling PS2; missing runtime packages mark PS2 unavailable without failing PS1/PS4.

PS3 uses `FORGE_TRACE_MODEL` (default `gemini-3.6-flash`), `FORGE_TRACE_SEGMENT_DURATION`, `FORGE_TRACE_SEGMENT_OVERLAP`, `FORGE_TRACE_ENABLE_REFINEMENT`, and `FORGE_TRACE_ENABLE_GROUPING`. PS3 extracts real segment files before recognition, then maps segment-relative timestamps to the original clip once. It writes Studio-compatible `actions.json` plus a detailed, path-free `ps3_timeline.json` matching the task/timeline structure shown in sample outputs. Gemini uploads are deleted after processing on a best-effort basis. Current processing retains frame records in memory; long clips may require substantial memory. Playback assumes constant-frame-rate input timing. Model accuracy and identity stability need representative real-motion validation.

Validation: python -m unittest discover -s backend/tests -v. Public-hand-image smoke test exercises real models but is not an accuracy benchmark.
