# PS2 — what is the hand touching?

Python package `ps2` with one public function. It finds objects in a video, decides which object each hand is touching or holding, and returns one dict. The exact input and output contract is `INTEGRATION.md`. Decisions, measurements and open problems are logged in `decision.md`.

```python
from ps2 import run_ps2

result = run_ps2(video_path=r"D:\Bios hack_ForgeStudio\Hack-data\<video>.mp4",
                 hands=r"<PS1 output>.jsonl")   # or a .json path, or a list of PS1Frame dicts
```

It raises `ValueError` with a clear message if the video or the hand data is missing or malformed. Every frame is processed, and the result is checked with `schema.validate()` before it is returned. Problems with the PS1 data (see `INTEGRATION.md`, section 5) are logged as warnings and reported in `meta.ps1_checks`; they never stop the run.

## Tech stack

- **Detector:** YOLOE-26s segmentation (Ultralytics), text-prompted; BoT-SORT tracker (Ultralytics).
- **Video and geometry:** OpenCV (optical flow for camera motion, histograms), NumPy, PyYAML, PyTorch with CUDA (FP16).
- **Gemini (optional, cloud):** names the objects in each video and the objects that hands touch or hold. Used through plain HTTPS requests from the standard library; needs the environment variable `GEMINI_API_KEY`. Without it PS2 runs on the base vocabulary and sends nothing anywhere.
- **ffmpeg (optional, development only):** compresses the demo clips written by `dev/render.py`.
- **MediaPipe:** only in the development stub `dev/hands_stub.py` that makes hand data until PS1's output exists; never imported by `ps2/`.
- **Claude Code** was used as the coding assistant for this project.

## Setup (Windows 11, NVIDIA GPU)

1. Python 3.11 virtual environment in the repo folder: `py -3.11 -m venv .venv`
2. Install the CUDA build of PyTorch first. Read the driver's CUDA version with `nvidia-smi`, then use the command from pytorch.org (Windows, pip). On the development machine (driver CUDA 13.1) this was:
   `.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
3. `.venv\Scripts\python.exe -m pip install -r requirements.txt`
4. The first run downloads the detector weights into the repo root (gitignored): `yoloe-26s-seg.pt` and the text encoder `mobileclip2_b.ts` (about 250 MB).
5. Check the GPU: `.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
6. Optional: set `GEMINI_API_KEY`, and check `vlm_model` in `ps2/config/ps2.yaml` against your key with `.venv\Scripts\python.exe dev\gemini_models.py` (lists the models that support `generateContent`).
7. Tests (synthetic data only): `.venv\Scripts\python.exe -m pytest`

Without a CUDA GPU it falls back to the CPU, which is much slower. The log line `ps2: device: ...` shows which was used.

## How it works

1. **Hands:** PS1 keypoints, rescaled to the video size, with load-time checks of the PS1 assumptions (`hands.py`).
2. **Vocabulary for this video** (`vocab_auto.py`, only with a working Gemini key): see below.
3. **Objects:** YOLOE with text prompts from the vocabulary, run on the whole frame and on a crop around every hand, merged by NMS, then tracked. Boxes on the hands/gloves are dropped. A portable-type box that covers more than `max_portable_frac` of the frame gets the generic label (`objects.py`).
4. **Same object after re-entry:** tracker IDs are merged by colour fingerprint, label, and "never visible together"; each object gets one label (`objects.py`).
5. **Camera motion** is estimated from background features and removed from all motion measurements (`motion.py`).
6. **Contact** comes from fingertip distances and motion, never from the detector alone. Touch vs hold depends on whether the object moves with the hand. Weak evidence gives `"unknown"` with a reason code (`contact.py`).
7. **Naming of held objects** (`vlm.py`, only with a Gemini key): the clearest crop of every touched/held object is sent inline with the vocabulary labels; an answer from the list becomes the object's label.
8. **Summary:** `tools[]` and `review_queue[]` (`summary.py`). **Safety warnings** (experimental, `safety_enabled`): a fingertip near a hazard that seems to be moving (`safety.py`).

All thresholds are in `ps2/config/ps2.yaml`; the base object names, portable/fixed and hazard flags are in `ps2/config/vocab.yaml`.

## Per-video vocabulary and its guardrails

Judges' videos show tools the base list does not know, so with a Gemini key PS2 builds the vocabulary from the video itself:

- 24 frames spread over the video (preferring frames with a hand), shrunk to at most 768 px, sent inline as JPEG in 3 batches of 8, each image labelled with its frame index. Nothing is uploaded as a file; only JSON text is cached (`cache/`).
- Gemini answers in strict JSON at temperature 0: only objects it can clearly see being handled, plus work surfaces and machines; short name, up to 3 synonyms, portable/fixed, whether it is a powered cutting/rotating tool, and the frames where it is visible. Anything it cannot name goes to an `unsure` list. It is told not to infer from context and never to guess.
- Names are normalised and merged, entries without a valid frame index are dropped, and a name is kept only if at least 2 of the 3 batches name it.
- Each kept name is cross-checked: the detector is run with its prompts on the frames that cite it. Found = `confirmed`, otherwise `vlm_only` (kept as a prompt but flagged for review).
- A hazard needs "powered cutting or rotating" and at least 2 cited frames; at most 40 objects.
- The final vocabulary is these objects plus the base file's fixed non-hazard surfaces, the generic catch-alls and the ignore list. The base file's specific tool list is used only when Gemini is unavailable.
- Requests are limited per clip (`vlm_max_calls_per_clip`), retried with exponential backoff on rate limits and overload, and every failure is logged once; the run always completes.

**Known weakness:** on the two clips tested, the 2-of-3-batches rule kept only 1–2 objects, and recall dropped (for example 48 -> 17 contacts and no hazard tools on one clip, compared with the base vocabulary). Held-object naming was also limited by rate limits. See `decision.md`, "Part 1 results".

## Glove test

`dev/glove_test.py` paints each hand's keypoint hull with a flat glove colour plus noise, in memory, and reruns the pipeline with the same hand data. On one clip (per-video vocabulary on): normal run 17 contacts (6 holds), gloved run 22 contacts (8 holds); the hand filter removed 8 detections in the gloved run and none in the normal run. Contacts did not get worse because contact logic uses the hand keypoints, not skin colour. Caveats: the glove is a painted shape, not real footage; the numbers come from one clip; and the count of objects named by Gemini differs (9 vs 2) mainly because of API rate limits in the second run, not because of the gloves.

## Hardware and speed

Developed on an RTX 3050 Ti Laptop GPU (4 GB), FP16. For one 3-minute 1080p clip (5400 frames), `run_ps2` alone ran at about 10–13 fps without Gemini; Gemini calls add time (the vocabulary step took 10–116 s on the clips tested, mostly waiting on rate limits). The MediaPipe stub that makes the hand data for development takes about 10 minutes per clip on the CPU.

## Development tools (`dev/`, never imported by `ps2/`)

- `hands_stub.py` makes PS1-format hand data; `run_one.py <video>` runs everything for one clip and saves `outputs/<clip>/ps2.json`.
- `render.py <video>` writes `highlight_<clip>.mp4` (the 60 s with the most holds) and `weak_<clip>.mp4` (the 20 s with the most unknown frames) at 720p with skeletons, badges, timelines and totals; `--clip START END` writes any part.
- `verify_contacts.py <video> <ps2.json> [--safety]` loops each predicted contact (or safety event) for a human to mark correct / wrong; `label_tool.py` and `evaluate.py` mark and score contacts from scratch; `baseline_v0.py` applies the first, simple contact rule for comparison.
- `glove_test.py`, `probe_vocab.py`, `gemini_models.py` as described above and in their headers.

Phase 7 additions: `independent_proposals.py` (Gemini watches the clip and proposes contacts), `groq_judge.py` (a Groq vision model votes on a few frames per item), `checklist.py` (pools the system's and the models' contacts into a ~40-item check list), `verify_contacts.py <video> --checklist` (you answer it without seeing the models' answers) and `evaluate.py <clip>` (turns your answers into stratified estimates).

No accuracy has been measured against human answers yet.

## Results (Phase 7)

**Status: not human-verified.** Every number below is a count or a model-vs-model agreement. Precision, recall, tool-level F1 and safety precision need your answers on the check list (`outputs/<clip>/checklist.json`) and on `verify_contacts.py --safety`; until then none of them is known, and the system was not tuned.

Local runs (RTX 3050 Ti, FP16; speed is indicative, other jobs ran at the same time):

| Clip | Variant | Contacts (holds) | High-confidence identified tools | Hazard tools | Safety events | Review items | fps |
|---|---|---|---|---|---|---|---|
| A | v0 rule | 97 (0) | - | - | - | - | - |
| A | base vocabulary | 48 (16) | 0 | 96 | 10 | 86 | 12.2 |
| A | ledger | 15 (7) | 0 | 0 | 0 | 49 | 12.5 |
| A | ledger + SAM 2 | 15 (7) | 0 | 0 | 0 | 49 | 8.5 |
| B | v0 rule | 420 (0) | - | - | - | - | - |
| B | base vocabulary | 95 (45) | 0 | 85 | 69 | 267 | 8.8 |
| B | ledger | 93 (45) | 6 | 12 | 24 | 445 | 9.8 |
| B | ledger + SAM 2 | 87 (38) | 8 | 12 | 24 | 440 | 6.9 |

SAM 2 adds 6 s (clip A) and 81 s (clip B), peak 0.5 GB of GPU memory. The Gemini ledger took one request per clip; the upload and the temporary copy were deleted each time.

Agreement on clip A (check list built with the `ledger_sam` result; 55 pooled contacts, 50 judged by Groq; this is agreement, not accuracy):

| Pair | Same vote |
|---|---|
| system vs Gemini proposals | 2 of 55 |
| system vs Groq | 21 of 54 |
| Gemini proposals vs Groq | 34 of 54 |

Groq voted "contact" on 4 of the system's 15 contacts and on 23 of the 40 that only Gemini proposed. Gemini's independent model is `gemini-3.8-flash` because this key cannot use a Pro model. Taken at face value this suggests the system finds fewer contacts than the models do on clip A and that some of its own are doubtful, but a single-frame model vote is noisy, so treat it as a pointer to check, not as a result.

What this means: on clip A the ledger vocabulary cut contacts from 48 to 15 because the detector found few of the ledger's four tools; on clip B contacts barely changed and 8 objects were identified with both Gemini and the local pipeline agreeing. Whether the ledger variant is better than the base one is not known until the check list is answered.

## Known limitations

- The detector (YOLOE with text prompts) misses many tools and sometimes names objects wrongly. Every contact is only as good as the object name and box.
- With a Gemini key the vocabulary can become very small (see above); without one the base list is used and may not match the video.
- Gemini can be slow or overloaded; naming and vocabulary are then partial.
- Safety warnings depend on the same names and on 2D distance only; a hand gripping a running tool counts as "near" it. They are experimental and their false-alarm rate is not measured.
- Hand left/right labels from the MediaPipe stub can flip; the real PS1 output should fix this.
- Contacts with very large fixed objects are reported as `unknown` when the hand is moving, because the evidence is weak.
- The review queue can be long: re-entry matches are often uncertain.
- The label and verify tools' windows could not be tried by the assistant that wrote them.
