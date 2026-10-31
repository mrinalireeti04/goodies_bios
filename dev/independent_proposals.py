"""A second, independent opinion on a clip: a Gemini model that is NOT the pipeline's watches the whole video and proposes
contacts (dev only).

A Pro model is used if this key can use one; otherwise a different full Flash model (never the pipeline's own Flash models),
so the two can disagree. The video goes through the Gemini Files API with the same rules as the tool ledger: reduced
temporary copy, one upload, and the upload and the copy are deleted in a finally block. Saves outputs/<clip>/independent.json.
The proposals are only proposals: a human verifies them (dev/verify_contacts.py --checklist).

Usage: python dev/independent_proposals.py <video_path>
"""
import json
import logging
import re
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ps2 import ledger, vlm  # noqa: E402
from ps2.pipeline import load_config  # noqa: E402

log = logging.getLogger("ps2")

TOKENS = 32000
SKIP = ("tts", "image", "customtools", "robotics", "research", "transcribe", "lyria", "omni", "computer", "embedding",
        "banana", "antigravity", "gemma")
MAX_PRO, MAX_FLASH = 2, 3
_STR = {"type": "STRING"}
SCHEMA = {"type": "OBJECT", "properties": {"contacts": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "start": {"type": "STRING", "description": "MM:SS"}, "end": {"type": "STRING", "description": "MM:SS"},
    "hand": {"type": "STRING", "enum": ["left", "right", "both", "unknown"]},
    "tool": _STR, "action": _STR, "certainty": {"type": "STRING", "enum": ["high", "medium", "low"]}},
    "required": ["start", "end", "hand", "tool", "action", "certainty"]}}}, "required": ["contacts"]}
PROMPT = (
    "This is a first-person video of factory or workshop work ({length}). List every contact between the worker's own hand "
    "and a tool or object: each time a hand touches, grasps or holds something. For each give start and end (MM:SS from "
    "the start of the video you are given), the worker's hand (left, right, both or unknown), the tool or object in the "
    "hand (short common name; for a held tool, the tool itself, not the material it acts on), the action verb, and your "
    "certainty (high, medium or low). Report only what is visible; do not infer from context or typical workflows; use "
    "unknown as the tool name if you cannot name it. Return JSON only."
)


def _rank(name: str):
    version = re.search(r"(\d+(?:\.\d+)?)", name)
    return ("latest" in name, "preview" in name, -(float(version.group(1)) if version else 0.0), name)


def pro_candidates(models: list[str]) -> list[str]:
    """Pro models from a ListModels answer, best first: stable before preview, higher version first, aliases last."""
    return sorted((m for m in models if "pro" in m and not any(s in m for s in SKIP)), key=_rank)


def flash_alternatives(models: list[str], exclude: set) -> list[str]:
    """Full Flash models other than the ones the pipeline uses (used only when no Pro model works on this key)."""
    return sorted((m for m in models if "flash" in m and "lite" not in m and m not in exclude and not any(s in m for s in SKIP)),
                  key=_rank)


def parse_proposals(data: dict, duration_s: float) -> list[dict]:
    """Valid contacts as {start_ms, end_ms, hand, object, action, certainty}; bad intervals are dropped."""
    out = []
    for c in data.get("contacts", []) if isinstance(data, dict) else []:
        try:
            span = ledger._interval(c["start"], c["end"], 0.0, duration_s)
            name = " ".join(str(c["tool"]).lower().split())
            hand = str(c.get("hand", "")).lower()
            certainty = str(c.get("certainty", "")).lower()
        except (KeyError, TypeError, AttributeError):
            continue
        if span and name:
            out.append({"start_ms": int(round(span[0] * 1000)), "end_ms": int(round(span[1] * 1000)),
                        "hand": hand if hand in ("left", "right", "both", "unknown") else "unknown",
                        "object": name, "action": " ".join(str(c.get("action", "")).lower().split()),
                        "certainty": certainty if certainty in ("high", "medium", "low") else "low"})
    return sorted(out, key=lambda c: (c["start_ms"], c["end_ms"]))


def propose(video: Path, duration_s: float, cfg: dict, key_models: list[str], tmp_dir: Path, files=None,
            make_copy=ledger.reduced_copy, make_gemini=vlm.Gemini) -> dict:
    """Upload once, then ask the Pro models (at most MAX_PRO) and, if none works on this key, other Flash models (at most
    MAX_FLASH) until one answers. The upload and the temporary copy are always deleted."""
    info = {"model": None, "kind": None, "tried": [], "failed": {}, "windows": 0, "requests": 0, "seconds": 0.0,
            "upload_deleted": None, "temp_deleted": None, "error": None}
    started = time.time()
    proposals: list[dict] = []
    own = {cfg["vlm_model"], cfg.get("vlm_fallback_model")}
    candidates = [(m, "pro") for m in pro_candidates(key_models)[:MAX_PRO]] + \
                 [(m, "flash") for m in flash_alternatives(key_models, own)[:MAX_FLASH]]
    probe = make_gemini(cfg)
    if not probe.available:
        info["error"] = "no usable Gemini key"
        return {"model": None, "contacts": [], "run": info}
    files_api = files or ledger.FilesAPI(probe.key)
    temp = uploaded = None
    try:
        temp = make_copy(video, tmp_dir, cfg["ledger_max_height"]) if make_copy else None
        uploaded = files_api.upload(temp or video, "video/mp4", f"ps2-independent-{video.stem}")
        ready = files_api.wait_active(uploaded["name"])
        for model, kind in candidates:
            gem = make_gemini({**cfg, "vlm_model": model, "vlm_fallback_model": None, "vlm_retries": 2,
                               "vlm_thinking_level": None if kind == "pro" else cfg["vlm_thinking_level"],
                               "vlm_max_calls_per_clip": cfg["independent_max_calls"]})
            info["tried"].append(model)
            found_all, got_any = [], False
            for start, end in ledger.windows(duration_s, cfg["ledger_max_window_s"]):
                part = {"file_data": {"mime_type": "video/mp4", "file_uri": ready["uri"]},
                        "video_metadata": {"fps": cfg["ledger_fps"]}}
                if end - start < duration_s:
                    part["video_metadata"].update(start_offset=f"{int(start)}s", end_offset=f"{int(end)}s")
                length = f"{int(end - start) // 60}:{int(end - start) % 60:02d} long"
                text = gem.generate([part, {"text": PROMPT.format(length=length)}], TOKENS, SCHEMA,
                                    timeout=cfg["independent_timeout_s"])
                info["windows"] += 1
                try:
                    found = parse_proposals(json.loads(text), duration_s) if text else []
                except ValueError:
                    found = []
                got_any |= bool(text)
                found_all += [{**c, "start_ms": c["start_ms"] + int(start * 1000), "end_ms": c["end_ms"] + int(start * 1000)}
                              for c in found]
            info["requests"] += gem.calls
            if got_any:
                info.update(model=model, kind=kind)
                proposals = found_all
                break
            info["failed"][model] = "no answer (unavailable, rate limit or overloaded)"
            log.info("independent proposals: %s gave no answer; trying the next model", model)
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"[:200]
    finally:
        if uploaded is not None:
            info["upload_deleted"] = files_api.delete(uploaded["name"])
            log.info("independent proposals: uploaded file %s deleted: %s", uploaded["name"],
                     "yes" if info["upload_deleted"] else "NO")
        if temp is not None:
            temp.unlink(missing_ok=True)
            info["temp_deleted"] = not temp.exists()
    info["seconds"] = round(time.time() - started, 1)
    return {"model": info["model"], "contacts": sorted(proposals, key=lambda c: (c["start_ms"], c["end_ms"])), "run": info}


def main(video_path: str) -> None:
    import os
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    video = Path(video_path)
    cfg = load_config()
    cap = cv2.VideoCapture(str(video))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY is not set in this process")
    sys.path.insert(0, str(Path(__file__).parent))
    from gemini_models import list_models
    models = [m["name"].removeprefix("models/") for m in list_models(key)]
    own = {cfg["vlm_model"], cfg.get("vlm_fallback_model")}
    print("Pro candidates:", pro_candidates(models)[:MAX_PRO], "| other Flash:", flash_alternatives(models, own)[:MAX_FLASH])
    result = propose(video, duration, cfg, models, ROOT / "outputs" / "tmp")
    out = ROOT / "outputs" / video.stem / "independent.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    run = result["run"]
    print(f"model {result['model']} ({run['kind']}); tried {run['tried']}; {len(result['contacts'])} proposed contacts; "
          f"{run['requests']} requests; {run['seconds']} s; upload deleted: {run['upload_deleted']}; "
          f"copy deleted: {run['temp_deleted']}; error: {run['error']}")
    print(f"saved {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
