# CLAUDE.md — rules for working in this repo (PS2: hand–object contact)

Read this first, then `PS2_CLAUDE_CODE_PLAN.md` (what to build, in order) and `INTEGRATION.md` (exact input/output contract).

## What we are building
A Python package `ps2` with ONE public function:

```python
from ps2 import run_ps2
result: dict = run_ps2(video_path: str, hands: str | list[dict]) -> dict
```
The backend calls it and forwards the returned dict. The frontend does review, renaming, merging and display. Therefore:
- No CLI flags, no editing commands, no UI code in `ps2/`.
- Settings live in `ps2/config/ps2.yaml` and `ps2/config/vocab.yaml` with working defaults.
- Keep files few and readable. Do not add a module, class or dependency unless the plan asks for it.

## Hard data rules (breaking them disqualifies the team)
- Folder layout: `D:\Bios hack_ForgeStudio\` contains `Hack-data\` (the videos) and `ps2\` (this repo; `git init` happens inside `ps2\` only, never in the parent). The package is `ps2\ps2\`.
- Videos live OUTSIDE the repo, in `D:\Bios hack_ForgeStudio\Hack-data\`. Never copy, move or commit them into the repo. Find the exact video files by listing that folder; do not assume subfolder names.
- The folder name contains a space: always use `pathlib.Path`, and quote paths in shell commands.
- Never write frames, crops, masks images, overlay videos or any frame-derived arrays inside tracked folders. Dev outputs go to `outputs/` (gitignored).
- Never put images or frames in README, tests or docs.
- Tests use synthetic data only (made-up boxes and keypoints).
- No cloud API calls, with these approved exceptions, all using keys from the user's environment variables only (never written into code, config, logs, cache or any file, never printed):
  - `ps2/`: Gemini (`GEMINI_API_KEY`). (a) Crops and sampled frames are sent inline (naming of held objects, crop checks). (b) The tool ledger (`ps2/ledger.py`) uploads a reduced temporary copy of the clip with the Gemini Files API; the uploaded file and the copy are deleted in a `finally` block. Only JSON text is cached in `cache/` (gitignored). Without a key every step is skipped.
  - `dev/` tools only (never imported by `ps2/`): Gemini independent proposals (same upload/delete rules as the ledger) and Groq per-item judging (`GROQ_API_KEY`, image frames sent inline, JSON answers cached).
  - Nothing else may call the network.
- No API keys in code.
- Before every commit, run `git status` and confirm no media or arrays are staged.

## Engineering rules
- Python 3.11, Windows 11, NVIDIA RTX 3050 Ti Laptop (4 GB VRAM). Use `cuda:0` when available, FP16 on GPU, CPU fallback. Ignore the AMD integrated GPU.
- Process EVERY frame. No frame skipping.
- All thresholds in `ps2/config/ps2.yaml`. No magic numbers in code.
- When evidence is weak, output `"unknown"` with a reason string. Never invent a grab/release.
- Output must pass `schema.validate()` before `run_ps2` returns.
- Work phase by phase. After each phase: run `pytest`, report what works, what doesn't, and measured numbers, then STOP and wait for my go-ahead.
- Ask me before: changing the output schema, adding a dependency not in the plan, or deleting files.
