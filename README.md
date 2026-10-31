# Hand detection for first-person factory video

Function 1 of our Egocentric Vision Hackathon project: it turns head-camera factory footage into one structured record per frame, with left and right hands and 21 landmarks each. Hand tracking is the base that every later step builds on.

**Status:** hand detection is done, validated and tested. Stable hand IDs, smoothing, accuracy measurement on hand-labelled frames and everything after that are not built yet. See [What works and what does not](#what-works-and-what-does-not).

## Problem statement

Egocentric Vision Hackathon, **PS1: hand tracking in factory video**. For every processed frame: find each hand, label it left or right, and mark its 21 landmarks. This repository covers the detection step. Later steps are listed under [Next steps](#next-steps).

## Compute

- Runs on a **laptop CPU**. No GPU and no cloud service is needed.
- Developed on Windows with Python 3.10.
- Speed is reported per clip in `results/RESULTS.md`.

## Quick start

```
python -m venv .venv
.venv\Scripts\activate                 # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Download MediaPipe's hand model into the project folder:

```
curl.exe -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

(macOS / Linux: use `curl` instead of `curl.exe`.)

Put the organizers' clips in `data/`. That folder is ignored by git.

## The one command

```
python hand_detect.py data/ --no-video
```

This processes every `.mp4` in `data/` and writes, for each clip, `out/<clip_id>/`:

| File | What it holds |
|---|---|
| `hands.jsonl` | One record per processed frame, in the [strict schema](docs/schema.md). |
| `summary.json` | Frames processed, detection rate, speed and the run settings. |
| `first_frame.png`, `first_overlay.png` | First frame, and the same frame with the landmarks drawn. |
| `overlay.mp4` | Whole clip with landmarks drawn. Only written without `--no-video`. It is large (about 860 MB for a 3-minute 1080p clip). |

Options:

| Flag | Meaning |
|---|---|
| `--step N` | Process every N-th frame (default 2). |
| `--clahe` | Brighten dark frames before detection. |
| `--swap` | Swap MediaPipe's left/right labels. Off by default (see the schema notes). |
| `--no-video` | Do not write `overlay.mp4`. |
| `--out DIR` | Output folder (default `out`). |
| `--validate FILE` | Check a `hands.jsonl` against the schema, then exit. |

Check results and build the numbers table:

```
python hand_detect.py --validate out/<clip_id>/hands.jsonl
python tools/make_report.py            # writes results/RESULTS.md
python -m unittest discover tests      # tests need no footage and no model file
```

## How it works

```
video -> find frame -> preprocess -> MediaPipe -> hand detect -> confidence
```

1. Read every N-th frame of the original video.
2. Resize a copy to 480 px wide for detection. Optional CLAHE brightening.
3. Run MediaPipe Hand Landmarker in VIDEO mode (two hands at most).
4. Convert the result to pixels of the original frame, split into left and right, and write one record.

Two details matter for correctness:

- **Left/right labels.** MediaPipe assumes a mirrored selfie image. Whether swapping is needed has to be checked on the footage. On the clips we checked, the labels were already correct, so swapping is off.
- **Two hands with the same label.** The higher-scoring hand keeps the label. The other is kept and marked `unknown` instead of being thrown away.

## Results

See [`results/RESULTS.md`](results/RESULTS.md) for per-clip numbers and an explanation of what each number means.

## What works and what does not

Works:

- Every line of `hands.jsonl` matches the schema, and this is checked automatically when the file is written.
- Frames with no hand still get a record, with `hands: []`.
- Tests cover the schema checks, the side handling and the pixel coordinates.

Does not work yet, or is not measured yet:

- **Accuracy is not measured.** "Frames with a hand" is coverage. Whether the landmarks sit on the correct joints needs hand-labelled frames, which are not done.
- **Left/right is only partly checked.** The only available check is plausibility (the left hand is left of the right hand). A share of frames still gives both hands the same label.
- **No tracking.** There are no stable hand IDs across frames, no smoothing and no recovery after a hand leaves and returns.
- **Some clips have long hand-free stretches.** We do not yet know how much of that is hands out of view and how much is hands MediaPipe misses (for example gloves or blur).

## Next steps

1. Label about 100 hard frames by hand and report real accuracy numbers.
2. Find out why detection is low on some clips. Test `--clahe` and a crop-based second pass.
3. Add tracking: stable left/right IDs, smoothing and short-gap filling.
4. Report before/after numbers on the labelled frames.

## Repository layout

```
hand_detect.py        the detection function and command line
requirements.txt      Python dependencies
tools/make_report.py  builds results/RESULTS.md from the output files
tests/                unit tests (synthetic video, no footage)
docs/schema.md        the output schema and its rules
docs/SPEC_function1.md  the specification this was built from
examples/             one synthetic example record
results/RESULTS.md    per-clip numbers
data/  out/           private footage and outputs, ignored by git
```

## Data rules

The footage belongs to a real company and is used only for this hackathon.

- No clips, frames or screenshots are in this repository. `.gitignore` excludes `data/`, `out/` and all video and image files.
- Outputs under `out/` are derived from the footage and are also ignored.
- The code runs fully locally and needs no upload. All copies of the footage and its outputs, wherever they were processed, are deleted when the event ends.

## Libraries and models

| Name | Used for |
|---|---|
| [MediaPipe](https://ai.google.dev/edge/mediapipe) Hand Landmarker (`hand_landmarker.task`, float16) | Hand detection and 21 landmarks. |
| [OpenCV](https://opencv.org/) (`opencv-python`) | Reading video, resizing, drawing overlays. |
| Python standard library | Everything else. |

AI coding assistants were used to write and review the code, as the hackathon rules allow.

## Team

Add team name and members here.
