<!-- Specification given to the AI coding assistant that wrote hand_detect.py. Kept for reference. -->

# Task: build Function 1, hand detection on first-person factory video

Build ONLY what is described here. Do not add tracking, smoothing, object detection, contact logic, step timelines, a UI, or anything else. A second function will be specified later, so do not guess at it.

## Goal

A modular Python function that takes a video, goes through its frames, runs MediaPipe hand detection on each one, and writes one record per processed frame in the exact schema below.

## Pipeline (each arrow is its own small function)

```
video input -> find frame -> preprocess -> MediaPipe -> hand detect -> confidence
```

| Step | Function name | What it does |
|---|---|---|
| video input | `video_info(path)` | Return fps, frame count, width, height. |
| find frame | `read_frames(path, step=2)` | Generator that yields `(frame_idx, BGR image)` for every `step`-th frame. |
| preprocess | `preprocess(img, width=480, clahe=False)` | Resize to `width` for detection, keep aspect ratio. Optional CLAHE brightening on the L channel (off by default). |
| MediaPipe | `make_landmarker(model_path)` | Create a MediaPipe `HandLandmarker` in VIDEO running mode, `num_hands=2`, detection and tracking confidence 0.4. |
| hand detect | `detect_hands(landmarker, bgr, t_ms, orig_w, orig_h, swap_lr=False)` | Run detection and return a list of `hands` entries exactly as the schema defines them. |
| overlay | `draw_overlay(bgr, hands)` | Draw the 21 landmarks and connections on a copy of the original-size frame. Left = green, right = orange, unknown = grey. Label each hand with its side and `handedness_score`. |
| per frame | `process_frame(...)` | Chain the steps above and return one `PS1Frame` record. |
| per clip | `run_clip(video, out_root, step, save_video, clahe, swap_lr)` | Loop over frames, write outputs, return a summary. |
| check | `validate_record(rec)` | Check one record against the schema. Raise an error with the reason if it fails. |

## Output schema (STRICT)

Every line of `hands.jsonl` must be one JSON object that matches this type exactly. This is the source of truth.

```ts
type PS1Frame = {
  schema_version: "1.0";
  clip_id: string;
  frame_idx: number;
  t_ms: number;

  frame_width: number;
  frame_height: number;

  hands: {
    hand_id: string;
    side: "left" | "right" | "unknown";

    handedness_score: number | null;

    bbox_px: [number, number, number, number];

    kp2d_px: [number, number][]; // exactly 21

    kp3d_hand_m:
      [number, number, number][] | null; // exactly 21

    source: "detected" | "interpolated";
  }[];
};
```

### Rules for every field

- **Strictness:** no extra keys, no missing keys, no renamed keys, anywhere. Write the keys in the order shown. Write JSON with `allow_nan=False` so NaN or Infinity can never appear.
- `schema_version`: the string `"1.0"`.
- `clip_id`: the video file name without its extension, for example `"factory_001"`.
- `frame_idx`: integer index of the frame in the original video (0-based). It is the real index, not a count of processed frames. With `step=2` the values are 0, 2, 4 and so on.
- `t_ms`: integer, `round(frame_idx * 1000 / fps)`. These values are strictly increasing, and the same value is passed to MediaPipe.
- `frame_width`, `frame_height`: integers, taken from the shape of the decoded original frame (not from the file metadata, and not from the resized copy).
- **Coordinate space:** all `_px` values are in pixels of the original decoded frame (`frame_width` by `frame_height`). MediaPipe returns normalized 0 to 1 values, so multiply by `frame_width` and `frame_height`. The resize inside `preprocess` is for detection only and must not change the output coordinates.
- `hands`: always present. When no hand is found it is the empty list `[]`. Never skip a frame and never drop the record. Order the list left first, then right, then unknown.
- `hand_id`: a string that is unique within the frame. Use `"L"` for the left hand, `"R"` for the right hand, and `"U1"`, `"U2"` for unknown. This is not a persistent track ID across frames, because tracking is out of scope.
- `side`: `"left"` or `"right"` from MediaPipe's handedness, after the swap described below. If two hands in one frame end up with the same side, the higher-scoring hand keeps the side and the other gets `"unknown"`. Do not discard either hand.
- `handedness_score`: MediaPipe's handedness score as a float between 0 and 1, rounded to 3 decimals. `null` only if MediaPipe gives none.
- `bbox_px`: `[x1, y1, x2, y2]`, the min and max of the 21 `kp2d_px` points, rounded to 2 decimals. Do not clip it to the image bounds.
- `kp2d_px`: exactly 21 `[x, y]` pairs in MediaPipe landmark order, rounded to 2 decimals.
- `kp3d_hand_m`: exactly 21 `[x, y, z]` triples taken from MediaPipe `hand_world_landmarks` for the same hand, in metres, rounded to 5 decimals. These are relative to the hand's own centre, not to the camera. Do not transform them. Use `null` if they are unavailable. Never output a list that is not 21 long.
- `source`: this function only ever writes `"detected"`. The value `"interpolated"` belongs to the schema but is reserved for a later function. Do not interpolate anything here.

### Example 1: abridged, one right hand found

The `kp2d_px` and `kp3d_hand_m` lists are shortened here with `...`. In the real file they must contain all 21 points.

```json
{"schema_version":"1.0","clip_id":"factory_001","frame_idx":120,"t_ms":4000,"frame_width":854,"frame_height":480,"hands":[{"hand_id":"R","side":"right","handedness_score":0.934,"bbox_px":[312.4,201.15,468.92,355.7],"kp2d_px":[[330.1,350.2],[345.0,330.4],"... 19 more pairs ..."],"kp3d_hand_m":[[0.0,0.0,0.0],[0.01234,-0.01987,-0.00421],"... 19 more triples ..."],"source":"detected"}]}
```

### Example 2: no hands found

```json
{"schema_version":"1.0","clip_id":"factory_001","frame_idx":122,"t_ms":4067,"frame_width":854,"frame_height":480,"hands":[]}
```

## Files written per clip, inside `out/<clip_id>/`

- `hands.jsonl`: one `PS1Frame` per line, nothing else on any line.
- `first_frame.png`: the first processed frame at original size.
- `first_overlay.png`: the same frame with the overlay drawn.
- `overlay.mp4`: every processed frame with the overlay, at original size (skippable with `--no-video`). Use fps = original fps / step.
- `summary.json`: this is NOT a `PS1Frame`, so it may hold run details: `clip_id`, `frames_processed`, `frames_with_hands`, `detection_rate`, `mean_handedness_score_when_detected`, `processing_fps`, and a `run_config` object with `step`, `fps`, `swap_lr`, `clahe`, `preprocess_width`.

## Left / right labels

MediaPipe assumes a mirrored selfie image, so its labels are often reversed on head-camera footage. On the demo clips checked, they were already correct without swapping. Provide a `SWAP_LR = False` constant, a `--swap` flag to force swapping and a `--no-swap` flag to force no swapping. Do not auto-detect this. Apply the swap only to the side label. Do not change any coordinates.

## Command-line interface

```
python hand_detect.py <video file or folder>   [--step 2] [--out out] [--no-video] [--clahe] [--swap] [--no-swap]
python hand_detect.py --validate <path to hands.jsonl>
```

- If the path is a folder, process every `.mp4` in it in sorted order and print each clip's summary.
- `--validate` reads a `hands.jsonl` and runs `validate_record` on every line. It prints the number of lines checked and the first error (line number and reason), or `OK`. Also run `validate_record` on every record inside `run_clip` before writing it.

## What `validate_record` must check

- Exactly the keys of `PS1Frame`, in order, at the top level and inside each hand.
- `schema_version == "1.0"`, `clip_id` is a non-empty string.
- `frame_idx`, `t_ms`, `frame_width`, `frame_height` are integers, and the sizes are positive.
- `side` is one of `left`, `right`, `unknown`. `source` is one of `detected`, `interpolated`.
- `handedness_score` is a number from 0 to 1, or `null`.
- `bbox_px` has 4 finite numbers with `x1 <= x2` and `y1 <= y2`.
- `kp2d_px` has exactly 21 pairs of finite numbers.
- `kp3d_hand_m` is `null` or has exactly 21 triples of finite numbers.
- `hand_id` values are unique within the frame.
- `t_ms` and `frame_idx` strictly increase from line to line within a file (this one is checked only by `--validate`).

## Technical requirements

- Python 3.10+. Libraries: `mediapipe` (Tasks API, `HandLandmarker`), `opencv-python`. Nothing else beyond the standard library.
- The model file is `hand_landmarker.task` in the project folder. Do not commit it.
- MediaPipe VIDEO mode needs strictly increasing timestamps in milliseconds. `t_ms` already satisfies this.
- Never crash on a frame with no hands. Create one landmarker per clip and close it at the end.
- One file: `hand_detect.py`. Small functions, short comments, no classes needed.

## Data rules (must follow)

The videos are private company footage.
- Create `.gitignore` first, containing: `data/`, `out/`, `*.mp4`, `*.png`, `*.task`.
- Never commit, upload or print video frames.
- Do not send the footage to any cloud service or external API.
- Keep all clips in `data/` and all results in `out/`.

## Folder layout

```
project/
  data/                  # the clips, never committed
  out/                   # results, never committed
  hand_detect.py
  hand_landmarker.task
  .gitignore
```

## Acceptance checks

1. `python hand_detect.py data/factory_001.mp4` runs without errors and creates all five output files.
2. `python hand_detect.py --validate out/factory_001/hands.jsonl` prints `OK`.
3. Every line has exactly the `PS1Frame` keys. A frame with no hands has `"hands": []`.
4. Every hand has `kp2d_px` with 21 points, and `kp3d_hand_m` is null or has 21 points.
5. `first_overlay.png` shows landmarks on the hands, at the original frame size, with labels. The overlay lines up with the hands, which confirms the pixel coordinates are in the original frame space.
6. Running with `--swap` flips the `side` labels and nothing else.
7. `python hand_detect.py data/` processes every clip and prints a summary for each.

## Out of scope. Do not build any of this.

Hand tracking across frames, persistent hand IDs, smoothing, gap filling or interpolation (the `"interpolated"` value stays unused), object detection, contact or grab/release logic, task steps, Gemini or any other API, a dashboard, or a web interface.
