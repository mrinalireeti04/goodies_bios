"""Gemini client and naming of held/touched objects. The only network use in PS2.

Crops and frames are sent INLINE (base64), never uploaded as files. The key comes from the environment variable
GEMINI_API_KEY and is never written anywhere. Without a key, or when the API keeps failing, every function here
returns "nothing" and PS2 carries on with the base vocabulary; the reason is logged once.
"""
import base64
import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("ps2")

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
PROMPT = ("This is a crop of an object that a worker's hand is touching or holding in a workshop. "
          "Answer with one label from the list, or other:<short name>, or unknown. "
          "Reply with the answer only.\nLabels: {labels}")
TRANSIENT_HTTP = {429, 500, 502, 503, 504}   # worth retrying: rate limit, overload
PERMANENT_HTTP = {401, 403, 404}             # key or model problem: stop asking for the rest of the clip
NAMING_TOKENS = 400                          # room for the model's short thinking plus a one-line answer


def _post(url: str, key: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json",
                                                                  "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def image_part(image_bgr: np.ndarray, quality: int = 85) -> dict | None:
    """Inline JPEG part for a request (None if the image cannot be encoded)."""
    ok, jpg = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpg.tobytes()).decode()}}


class Gemini:
    """One per clip: counts requests against vlm_max_calls_per_clip, retries with backoff, logs problems once."""

    def __init__(self, cfg: dict, post=_post, sleep=time.sleep):
        self.cfg, self.post, self.sleep = cfg, post, sleep
        self.key = os.environ.get("GEMINI_API_KEY")
        self.calls = 0            # HTTP requests made (retries included)
        self.failed = False       # a permanent problem was seen
        self._logged: set[str] = set()
        if not self.key:
            self._log_once("nokey", "Gemini not used: GEMINI_API_KEY is not set (base vocabulary, no VLM naming)")

    def _log_once(self, tag: str, message: str, *args) -> None:
        if tag not in self._logged:
            self._logged.add(tag)
            log.info(message, *args)

    @property
    def available(self) -> bool:
        return bool(self.key) and not self.failed

    def generate(self, parts: list[dict], max_tokens: int, schema: dict | None = None, timeout: float | None = None) -> str | None:
        """Text of the first answer, or None (no key, call budget used up, or the API failed)."""
        if not self.available:
            return None
        models = [self.cfg["vlm_model"]]
        fallback = self.cfg.get("vlm_fallback_model")
        if fallback and fallback != models[0]:
            models.append(fallback)
        for position, model in enumerate(models):
            config = {"temperature": 0, "maxOutputTokens": max_tokens}
            if position == 0 and self.cfg.get("vlm_thinking_level"):
                config["thinkingConfig"] = {"thinkingLevel": self.cfg["vlm_thinking_level"]}
            if schema is not None:
                config.update(responseMimeType="application/json", responseSchema=schema)
            body = {"contents": [{"parts": parts}], "generationConfig": config}
            tries = self.cfg["vlm_retries"] if position == 0 else 1  # the fallback model gets one try
            for attempt in range(tries):
                if self.calls >= self.cfg["vlm_max_calls_per_clip"]:
                    self._log_once("cap", "Gemini call limit reached (%d per clip); the rest is skipped",
                                   self.cfg["vlm_max_calls_per_clip"])
                    return None
                self.calls += 1
                try:
                    reply = self.post(API.format(model=model), self.key, body, timeout or self.cfg["vlm_timeout_s"])
                    candidate = reply["candidates"][0]
                    if candidate.get("finishReason") == "MAX_TOKENS":
                        self._log_once("tokens", "Gemini answer was cut off (model %s); skipped", model)
                        return None
                    return "".join(p.get("text", "") for p in candidate["content"]["parts"]).strip()
                except urllib.error.HTTPError as e:
                    if e.code in PERMANENT_HTTP:
                        self.failed = True
                        self._log_once("perm", "Gemini stopped: HTTP %s from the API (model %s)", e.code, model)
                        return None
                    if e.code not in TRANSIENT_HTTP:
                        self._log_once(f"http{e.code}", "Gemini request refused: HTTP %s (model %s)", e.code, model)
                        return None
                    problem = f"HTTP {e.code}"
                except (TimeoutError, urllib.error.URLError, OSError, KeyError, IndexError, ValueError) as e:
                    problem = type(e).__name__
                self._log_once(f"retry-{model}", "Gemini %s on model %s: retrying with backoff", problem, model)
                if attempt + 1 < tries:
                    self.sleep(self.cfg["vlm_backoff_s"] * 2 ** attempt)
        self._log_once("giveup", "Gemini gave up on one request after retries")
        return None


def parse_answer(text: str, labels: list[str]) -> tuple[str, str]:
    """('label', key) if the answer is in the list, ('other', name), or ('unknown', '')."""
    answer = text.strip().strip("`'\".").strip().lower()
    if answer.startswith("other:"):
        name = answer[len("other:"):].strip()
        return ("other", name) if name else ("unknown", "")
    answer = answer.replace(" ", "_")
    return ("label", answer) if answer in labels else ("unknown", "")


def _crop_hash(crop: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(crop).tobytes()).hexdigest()


def name_objects(gem: Gemini, crops: dict[str, np.ndarray], labels: list[str], clip_id: str, cfg: dict,
                 cache_dir: Path) -> dict[str, tuple[str, str]]:
    """object_id -> ('label', key) | ('other', name) | ('unknown', ''). `crops` is asked in its own order, so the
    caller puts the most important objects first (the call limit may cut the rest). Objects not asked are left out."""
    if not crops or not gem.available:
        return {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    answers: dict[str, tuple[str, str]] = {}
    for oid, crop in crops.items():
        cache_file = cache_dir / f"{clip_id}_{oid}.json".replace("/", "_")
        sha = _crop_hash(crop)
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("sha") == sha:  # same crop as when it was answered (object ids can change between runs)
                answers[oid] = (cached["kind"], cached["name"])
                continue
        image = image_part(crop)
        if image is None:
            continue
        if not gem.available:
            break
        text = gem.generate([{"text": PROMPT.format(labels=", ".join(labels))}, image], NAMING_TOKENS)
        if text is None:
            if not gem.available or gem.calls >= cfg["vlm_max_calls_per_clip"]:
                break
            continue
        kind, name = parse_answer(text, labels)
        answers[oid] = (kind, name)
        cache_file.write_text(json.dumps({"sha": sha, "kind": kind, "name": name}), encoding="utf-8")
    log.info("VLM naming: %d of %d objects answered", len(answers), len(crops))
    return answers


def apply_answers(answers: dict[str, tuple[str, str]], dets_by_frame: list[list[dict]], contacts: list[dict],
                  vocab: dict, frame_area: float, cfg: dict) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Use the answers. Returns (label_status overrides, suggested labels, changed {object_id: old label}).

    - a label from the list replaces the object's label everywhere (contacts get reason "label_from_vlm" if it changed);
    - "other:<name>" keeps the detector's label, marks the object unidentified and offers the name as a suggestion;
    - "unknown" changes nothing.
    """
    from ps2.objects import set_object_label  # local import: objects.py pulls in the detector libraries

    current = {}
    for dets in dets_by_frame:
        for d in dets:
            current.setdefault(d["object_id"], d["label"])
    status: dict[str, str] = {}
    suggestions: dict[str, str] = {}
    changed: dict[str, str] = {}
    for oid, (kind, name) in answers.items():
        if kind == "label" and name in {**vocab["objects"], **vocab["generic"]}:
            status[oid] = "identified"
            if name != current.get(oid):
                changed[oid] = current.get(oid, "")
                set_object_label(dets_by_frame, oid, name, vocab, frame_area, cfg)
                for c in contacts:
                    if c["object_id"] == oid:
                        c["object_label"] = name
                        c["reasons"] = sorted(set(c.get("reasons", [])) | {"label_from_vlm"})
        elif kind == "other":
            status[oid] = "unidentified"
            suggestions[oid] = name
    return status, suggestions, changed
