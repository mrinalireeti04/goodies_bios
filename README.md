# Function 1: hand detection (BIOS hack)

Video in, one `PS1Frame` JSON record per processed frame out (schema in `docs/schema.md`).
No footage, frames or screenshots are stored in this repository.

## Setup

```
pip install -r requirements.txt
```

Put MediaPipe's `hand_landmarker.task` in this folder (it is git-ignored). The RTMPose weights
download by themselves the first time `--engine rtmpose` runs.

## Run

```
python hand_detect.py data\<clip>.mp4 --out out --no-video                      # MediaPipe
python hand_detect.py data\<clip>.mp4 --engine rtmpose --out out_rtm2 --no-video  # RTMPose
python hand_detect.py --validate out\<clip>\hands.jsonl                         # schema check
```

Each run writes `<out>/<clip>/hands.jsonl` (the PS1Frame records) and `summary.json`.
Leave out `--no-video` to also write a private `overlay.mp4`.

## Post-processing (optional, never changes the original file)

```
python tools\postprocess_all.py out out_rtm2 --max-gap 2
```

- `tools/resolve_sides.py`: gives `unknown` hands a left/right side (x-order, then neighbouring frames) -> `hands_sided.jsonl`
- `tools/fill_gaps.py`: fills gaps of up to `--max-gap` frames by interpolation; filled hands have `source: "interpolated"` and a null score -> `hands_final.jsonl`

## Tests

```
python -m unittest discover tests
```

## Notes

- `kp3d_hand_m` is filled for MediaPipe hands and `null` for RTMPose hands.
- Accuracy against hand-labelled frames has not been measured yet; `results/RESULTS.md` reports coverage only.
- Footage and everything derived from it (`data/`, `out*/`, videos, images) is git-ignored. Do not commit them.
