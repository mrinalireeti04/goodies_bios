"""Tool ledger: Gemini watches the whole clip once and writes down every tool, when it is used and when machines run.

The local pipeline gives frame-exact hand/object evidence; the ledger gives names and ~1 s timing. Fusion (fuse.py)
marks a tool "identified" only when both agree. The video is sent through the Gemini Files API (resumable upload with our
urllib client): a reduced temporary copy is uploaded, and both the uploaded file and the copy are deleted in a finally
block. Only JSON text is cached (cache/). Without a key, or if anything fails, the ledger is None and PS2 runs on the base
vocabulary; the reason is logged.
"""
import json
import logging
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from ps2 import vlm

log = logging.getLogger("ps2")

API_ROOT = "https://generativelanguage.googleapis.com"
LEDGER_TOKENS = 16000
POLL_S = 3
MIN_USE_S = 0.2     # a use shorter than this is not a real interval
EDGE_TOL_S = 2.0    # an end this far past the video end is clamped, further is dropped
_STR = {"type": "STRING"}
_TIME = {"type": "STRING", "description": "MM:SS"}
SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "tools": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "tool_key": _STR, "name": _STR, "synonyms": {"type": "ARRAY", "items": _STR}, "appearance": _STR,
            "type": {"type": "STRING", "enum": ["portable", "fixed"]},
            "powered_cutting_or_rotating": {"type": "BOOLEAN"}, "workpiece": {"type": "BOOLEAN"},
            "uses": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
                "start": _TIME, "end": _TIME,
                "hand": {"type": "STRING", "enum": ["left", "right", "both", "unknown"]},
                "action": _STR, "certainty": {"type": "STRING", "enum": ["high", "medium", "low"]}},
                "required": ["start", "end", "hand", "action", "certainty"]}}},
            "required": ["tool_key", "name", "synonyms", "appearance", "type", "powered_cutting_or_rotating",
                         "workpiece", "uses"]}},
        "running": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "tool_key": _STR, "start": _TIME, "end": _TIME}, "required": ["tool_key", "start", "end"]}},
        "unsure": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "description": _STR, "start": _TIME, "end": _TIME}, "required": ["description", "start", "end"]}},
    },
    "required": ["tools", "running", "unsure"],
}
PROMPT = (
    "This is a first-person video of factory or workshop work ({length}). Write a tool ledger.\n"
    "tools: every tool or machine the worker uses or that is clearly visible, with tool_key (short id such as hammer_1), "
    "name (short, common, lowercase), up to 3 synonyms a text-based object detector would understand, appearance "
    "(colour, shape, marks), type (portable if it can be carried, otherwise fixed), powered_cutting_or_rotating "
    "(true only for powered tools or machines with a blade, disc or bit that cuts or rotates), workpiece (true if it is "
    "material being worked on rather than an instrument), and uses: each time the worker is using it, with start and end "
    "(MM:SS counted from the start of the video you are given), the worker's hand (left, right, both or unknown), the "
    "action verb, and your certainty (high, medium or low).\n"
    "running: the intervals when a powered tool or machine is running (use its tool_key).\n"
    "unsure: objects the worker handles that you cannot name, with an interval and a short visual description.\n"
    "Rules: report only what is visible; do not infer tools from context or from typical workflows; the tool is the "
    "instrument in the hand, not the workpiece it acts on; create separate instances (hammer_1, hammer_2) only if they "
    "are visibly distinct; use unknown instead of guessing. Return JSON only."
)


# ---------------------------------------------------------------- Files API

def _http(method: str, url: str, headers: dict, data: bytes | None, timeout: float) -> tuple[dict, bytes]:
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return {k.lower(): v for k, v in resp.headers.items()}, resp.read()


class FilesAPI:
    """Resumable upload, status polling and deletion of one file. `http` is injectable for tests."""

    def __init__(self, key: str, http=_http):
        self.key, self.http = key, http

    def _h(self, **extra) -> dict:
        return {"x-goog-api-key": self.key, **extra}

    def upload(self, path: Path, mime: str, display_name: str) -> dict:
        size = path.stat().st_size
        headers, _ = self.http("POST", f"{API_ROOT}/upload/v1beta/files", self._h(**{
            "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(size), "X-Goog-Upload-Header-Content-Type": mime,
            "Content-Type": "application/json"}), json.dumps({"file": {"display_name": display_name}}).encode(), 60)
        url = headers.get("x-goog-upload-url")
        if not url:
            raise RuntimeError("the upload start gave no upload url")
        _, body = self.http("POST", url, self._h(**{"Content-Length": str(size), "X-Goog-Upload-Offset": "0",
                                                   "X-Goog-Upload-Command": "upload, finalize"}), path.read_bytes(), 600)
        return json.loads(body)["file"]

    def wait_active(self, name: str, timeout_s: float = 300, sleep=time.sleep) -> dict:
        waited = 0.0
        while True:
            _, body = self.http("GET", f"{API_ROOT}/v1beta/{name}", self._h(), None, 60)
            info = json.loads(body)
            if info.get("state") == "ACTIVE":
                return info
            if info.get("state") == "FAILED" or waited >= timeout_s:
                raise RuntimeError(f"uploaded file is not usable (state {info.get('state')})")
            sleep(POLL_S)
            waited += POLL_S

    def delete(self, name: str) -> bool:
        try:
            self.http("DELETE", f"{API_ROOT}/v1beta/{name}", self._h(), None, 60)
            return True
        except (urllib.error.URLError, OSError):
            return False


def reduced_copy(video: Path, out_dir: Path, max_height: int) -> Path | None:
    """A small temporary copy for upload: <= max_height, no audio, CRF 30. None if ffmpeg is missing or fails."""
    if not shutil.which("ffmpeg"):
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{video.stem}_ledger.mp4"
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vf", f"scale=-2:min({max_height}\\,ih)",
                        "-an", "-c:v", "libx264", "-crf", "30", "-preset", "fast", "-pix_fmt", "yuv420p", str(out)],
                       check=True, timeout=1200)
    except (OSError, subprocess.SubprocessError):
        out.unlink(missing_ok=True)
        return None
    return out


# ---------------------------------------------------------------- reading the answer

def parse_time(text) -> float | None:
    """'MM:SS', 'H:MM:SS', 'SS' or with decimals -> seconds; None if it is not a time."""
    if not isinstance(text, str) or not re.fullmatch(r"\s*\d+(:\d{1,2}){0,2}(\.\d+)?\s*", text):
        return None
    parts = [float(p) for p in text.strip().split(":")]
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def _norm(text) -> str:
    return " ".join(str(text).lower().replace("_", " ").split())


def _key(text) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _interval(start, end, offset: float, duration: float) -> tuple[float, float] | None:
    a, b = parse_time(start), parse_time(end)
    if a is None or b is None:
        return None
    a, b = a + offset, b + offset
    if a >= duration or a < 0 or b <= a:
        return None
    if b > duration + EDGE_TOL_S:
        return None
    return a, min(b, duration)


def validate_ledger(answers: list[tuple[dict, float, float]], duration_s: float, cfg: dict) -> dict:
    """Merge the (answer, window start, window length) of every window into one validated ledger.

    Dropped: malformed entries, intervals outside the video or with end <= start. Times become seconds from the start of
    the video. If an answer for a later window already uses times beyond its window it is taken as absolute."""
    tools: dict[str, dict] = {}
    running, unsure = [], []
    for data, offset, length in answers:
        if not isinstance(data, dict):
            continue
        times = [parse_time(u.get(k)) for t in data.get("tools", []) if isinstance(t, dict)
                 for u in t.get("uses", []) if isinstance(u, dict) for k in ("start", "end")]
        if offset > 0 and any(x is not None and x > length + EDGE_TOL_S for x in times):
            offset = 0.0   # the model used the original video's clock
        for t in data.get("tools", []):
            try:
                key, name = _key(t["tool_key"]), _norm(t["name"])
                kind = t["type"] if t["type"] in ("portable", "fixed") else None
            except (KeyError, TypeError, AttributeError):
                continue
            if not key or not name or kind is None:
                continue
            tool = tools.setdefault(key, {"tool_key": key, "name": name, "synonyms": [], "appearance": str(t.get("appearance", "")),
                                          "type": kind, "powered": False, "workpiece": False, "uses": []})
            tool["synonyms"] = list(dict.fromkeys(tool["synonyms"] + [s for s in (_norm(s) for s in t.get("synonyms", [])[:3]) if s and s != name]))[:3]
            tool["powered"] |= t.get("powered_cutting_or_rotating") is True
            tool["workpiece"] |= t.get("workpiece") is True
            for u in t.get("uses", []):
                if not isinstance(u, dict):
                    continue
                span = _interval(u.get("start"), u.get("end"), offset, duration_s)
                if span and span[1] - span[0] >= MIN_USE_S:
                    hand = u.get("hand") if u.get("hand") in ("left", "right", "both", "unknown") else "unknown"
                    certainty = u.get("certainty") if u.get("certainty") in ("high", "medium", "low") else "low"
                    tool["uses"].append({"start_s": span[0], "end_s": span[1], "hand": hand,
                                         "action": _norm(u.get("action", "")), "certainty": certainty})
        for r in data.get("running", []):
            span = _interval(r.get("start"), r.get("end"), offset, duration_s) if isinstance(r, dict) else None
            if span:
                running.append({"tool_key": _key(r.get("tool_key", "")), "start_s": span[0], "end_s": span[1]})
        for u in data.get("unsure", []):
            span = _interval(u.get("start"), u.get("end"), offset, duration_s) if isinstance(u, dict) else None
            description = " ".join(str(u.get("description", "")).split()) if isinstance(u, dict) else ""
            if span and description:
                unsure.append({"description": description, "start_s": span[0], "end_s": span[1]})
    for tool in tools.values():
        tool["uses"].sort(key=lambda u: u["start_s"])
    ranked = sorted(tools.values(), key=lambda t: -sum(u["end_s"] - u["start_s"] for u in t["uses"]))[:cfg["ledger_max_tools"]]
    keep = {t["tool_key"] for t in ranked}
    return {"tools": ranked, "running": [r for r in running if r["tool_key"] in keep],
            "unsure": unsure, "duration_s": round(duration_s, 2)}


def windows(duration_s: float, max_window_s: float) -> list[tuple[float, float]]:
    """(start, end) seconds of the requests: the whole video if short enough, else consecutive pieces."""
    if duration_s <= max_window_s:
        return [(0.0, duration_s)]
    out, start = [], 0.0
    while start < duration_s:
        out.append((start, min(start + max_window_s, duration_s)))
        start += max_window_s
    return out


# ---------------------------------------------------------------- building it

def _clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def build_ledger(video: Path, duration_s: float, gem: vlm.Gemini, cfg: dict, clip_id: str, cache_dir: Path,
                 tmp_dir: Path, files: FilesAPI | None = None, make_copy=reduced_copy) -> tuple[dict | None, dict]:
    """Returns (ledger or None, run info for meta). Never raises; the upload and the temp copy are always deleted."""
    started = time.time()
    info = {"model": cfg["vlm_model"], "windows": 0, "requests": 0, "from_cache": False, "seconds": 0.0,
            "upload_deleted": None, "temp_deleted": None, "error": None}
    stat = video.stat()
    cache_file = cache_dir / f"ledger_{clip_id}.json"
    cache_key = [clip_id, stat.st_size, int(stat.st_mtime)]
    try:
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("key") == cache_key:
                info.update(from_cache=True, seconds=round(time.time() - started, 1))
                log.info("tool ledger: from cache (%d tools)", len(cached["ledger"]["tools"]))
                return cached["ledger"], info
    except (OSError, ValueError, KeyError):
        pass
    if not gem.available:
        info["error"] = "no usable Gemini key"
        return None, info

    files = files or FilesAPI(gem.key)
    temp = uploaded = None
    calls_before = gem.calls
    try:
        temp = make_copy(video, tmp_dir, cfg["ledger_max_height"]) if make_copy else None
        source = temp or video
        uploaded = files.upload(source, "video/mp4", f"ps2-{clip_id}")
        ready = files.wait_active(uploaded["name"])
        answers = []
        for start, end in windows(duration_s, cfg["ledger_max_window_s"]):
            part = {"file_data": {"mime_type": "video/mp4", "file_uri": ready["uri"]},
                    "video_metadata": {"fps": cfg["ledger_fps"]}}
            if end - start < duration_s:   # real clipping of the video, never prompt text
                part["video_metadata"].update(start_offset=f"{int(start)}s", end_offset=f"{int(end)}s")
            text = gem.generate([part, {"text": PROMPT.format(length=f"{_clock(end - start)} long")}], LEDGER_TOKENS,
                                SCHEMA, timeout=cfg["ledger_timeout_s"])
            info["windows"] += 1
            if text is None:
                continue
            try:
                answers.append((json.loads(text), start, end - start))
            except ValueError:
                log.info("tool ledger: one window's answer was not valid JSON; skipped")
        ledger = validate_ledger(answers, duration_s, cfg) if answers else None
        if ledger is not None and (ledger["tools"] or ledger["unsure"]):
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"key": cache_key, "ledger": ledger}), encoding="utf-8")
        elif ledger is not None:
            ledger = None
            info["error"] = "the ledger came back empty"
        elif info["error"] is None:
            info["error"] = "no usable answer from Gemini"
        if ledger:
            log.info("tool ledger: %d tools, %d uses, %d running intervals, %d unsure", len(ledger["tools"]),
                     sum(len(t["uses"]) for t in ledger["tools"]), len(ledger["running"]), len(ledger["unsure"]))
        return ledger, info
    except Exception as e:   # the ledger is an extra; any failure falls back to the base vocabulary
        info["error"] = f"{type(e).__name__}: {e}"[:200]
        log.info("tool ledger failed (%s); using the base vocabulary", info["error"])
        return None, info
    finally:
        if uploaded is not None:
            info["upload_deleted"] = files.delete(uploaded["name"])
            log.info("tool ledger: uploaded file %s deleted: %s", uploaded["name"], "yes" if info["upload_deleted"] else "NO")
        if temp is not None:
            temp.unlink(missing_ok=True)
            info["temp_deleted"] = not temp.exists()
            log.info("tool ledger: temporary copy deleted: %s", "yes" if info["temp_deleted"] else "NO")
        info["requests"] = gem.calls - calls_before
        info["seconds"] = round(time.time() - started, 1)


class LedgerJob:
    """Runs build_ledger in a background thread, started at the beginning of run_ps2."""

    def __init__(self, *args, **kwargs):
        self.ledger, self.info = None, {"error": "not finished"}
        self._thread = threading.Thread(target=self._run, args=args, kwargs=kwargs, daemon=True)
        self._thread.start()

    def _run(self, *args, **kwargs):
        self.ledger, self.info = build_ledger(*args, **kwargs)

    def result(self) -> tuple[dict | None, dict]:
        self._thread.join()
        return self.ledger, self.info


# ---------------------------------------------------------------- what the detector should look for

def key_of(name: str) -> str:
    return _key(name)


def ledger_vocab(ledger: dict, base: dict) -> dict:
    """Detector vocabulary: ledger tools and machines (not workpieces) + the base file's fixed non-hazard surfaces,
    generic catch-alls and ignore list. A fixed machine only counts as a hazard if the ledger says it was running."""
    running = {r["tool_key"] for r in ledger["running"]}
    objects: dict[str, dict] = {}
    for tool in ledger["tools"]:
        if tool["workpiece"]:
            continue
        key = key_of(tool["name"])
        prompts = list(dict.fromkeys([tool["name"], *tool["synonyms"]]))[:4]
        spec = objects.setdefault(key, {"type": tool["type"], "prompts": []})
        spec["prompts"] = list(dict.fromkeys(spec["prompts"] + prompts))[:4]
        if tool["powered"] and (tool["type"] == "portable" or tool["tool_key"] in running):
            spec["hazard"] = True
    for key, spec in base["objects"].items():
        if spec["type"] == "fixed" and not spec.get("hazard") and key not in objects:
            objects[key] = spec
    return {"objects": objects, "generic": base["generic"], "ignore": base["ignore"]}
