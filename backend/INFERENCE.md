# Local hand inference

Run from the BIOS project root:

```sh
uv pip install --python backend/.venv/bin/python -r backend/requirements.txt
backend/.venv/bin/python backend/setup_models.py
backend/.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

FFmpeg must be installed and on PATH. RTMLib downloads public model weights on first use; later runs use its local cache. Media files are processed locally. Studio uses the local API by default; VITE_USE_REAL_API=false explicitly disables it.

On Apple Silicon, RTMPose automatically requests ONNX Runtime's CoreML provider (`FORGE_RTM_DEVICE=auto`, the default). Set `FORGE_RTM_DEVICE=cpu` to force CPU or `FORGE_RTM_DEVICE=coreml` to request CoreML explicitly. Compiled CoreML partitions are cached in `.local_data/coreml_cache`; override this with `FORGE_COREML_CACHE`. Job logs report actual providers separately for RTMDet and RTMPose. A model that fails CoreML initialization or its first inference is retried on CPU; that model will not receive CoreML acceleration for that run. CoreML graph support varies by model and ONNX Runtime/macOS version.

Each video is processed at every decoded frame. RTMPose provides original-pixel 2D landmarks; MediaPipe supplies hand-centred metric 3D landmarks. Unmatched detections get a cropped MediaPipe retry. Unavailable 3D remains null. Before filling coordinate gaps, a clip-wide post-pass links detections by wrist motion over gaps up to 60 frames and uses confidence-weighted left/right votes to repair unknown or contradictory labels. Weak or conflicting evidence stays unknown. L/R identify anatomical sides, not guaranteed persistent identities through crossings.

hands.raw.jsonl retains unsmoothed hybrid measurements. hands.jsonl fills at most 10 consecutive missing frames per side and coordinate modality, between valid endpoints only, using timestamp-weighted linear interpolation. Longer and unbounded gaps stay empty. Estimated coordinates have source=interpolated and per-modality provenance. No ballistic extrapolation is used.

Results include overlay.mp4 (H.264), hands.jsonl and ps4.json. The API serves the original at /api/clips/{job}/media and overlay at /api/clips/{job}/overlay. Both video panes share transport controls; the original video's clock drives the 3D viewer. 3D is hand-centred metres, not absolute factory position; the viewer magnifies these coordinates for display.

PS2/PS3 are optional and their absence does not block hand results. Jobs are serialized during inference. Current processing retains frame records in memory; long clips may require substantial memory. Playback assumes constant-frame-rate input timing. Model accuracy and identity stability need representative real-motion validation.

Validation: python -m unittest discover -s backend/tests -v. Public-hand-image smoke test exercises real models but is not an accuracy benchmark.
