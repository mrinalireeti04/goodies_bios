"""An independent per-item judge: a vision model on Groq looks at a few frames of one candidate contact (dev only).

Image-only, at most 3 images per request, through Groq's OpenAI-compatible chat API with urllib (no new dependency). The key
comes from the environment variable GROQ_API_KEY and is never printed or stored. Only the JSON answers are cached
(cache/groq_<clip>.json). Free-tier limits are respected: rate-limit headers are read, 429 answers are waited out, our own
daily request/token budget is kept in cache/groq_usage.json, and when the budget will not cover everything the judge steps
down: 3 frames per item -> 1 frame per item -> only the items where the system and Gemini Pro disagree.
"""
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("ps2")

URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_IMAGES = 3
MAX_TRIES = 4
MAX_WAIT_S = 65          # waits up to this long are just slept; a longer Retry-After is a token/day window (see groq_max_wait_s)
OVERHEAD_TOKENS = 350    # prompt text + answer, for planning
ANSWER_TOKENS = 200
PROMPT = (
    "These {n} frames come from one first-person video: {when}. Is the worker's own hand touching or holding a tool or "
    "object in them? Candidate tool names (they may be wrong): {names}. Answer with JSON only: "
    '{{"contact": "yes" or "no" or "unsure", "hand": "left" or "right" or "both" or "unsure" (the worker\'s own hand), '
    '"tool": "one short common name, or unknown"}}. Do not guess: use unsure or unknown when you cannot tell.'
)


def _http(url: str, headers: dict, body: bytes, timeout: float) -> tuple[dict, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return {k.lower(): v for k, v in resp.headers.items()}, resp.read()


def parse_duration(text: str | None) -> float | None:
    """'2.5s', '250ms', '1m3.5s', '1h2m' -> seconds."""
    if not text:
        return None
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", str(text))
    if not parts:
        return None
    scale = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
    return sum(float(v) * scale[u] for v, u in parts)


def parse_answer(content: str | None) -> dict | None:
    """The JSON object in a reply (thinking text and code fences are ignored), checked and normalised, or None."""
    text = re.sub(r"<think>.*?</think>", "", content or "", flags=re.S)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    contact = str(data.get("contact", "")).lower()
    hand = str(data.get("hand", "unsure")).lower()
    if contact not in ("yes", "no", "unsure"):
        return None
    return {"contact": contact, "hand": hand if hand in ("left", "right", "both", "unsure") else "unsure",
            "tool": " ".join(str(data.get("tool", "unknown")).lower().split()) or "unknown"}


class Usage:
    """Our own per-day budget (UTC day), kept in a small JSON file."""

    def __init__(self, path: Path, today=None):
        self.path = path
        self.today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.requests, self.tokens = (data.get("requests", 0), data.get("tokens", 0)) if data.get("date") == self.today else (0, 0)

    def add(self, tokens: int) -> None:
        self.requests += 1
        self.tokens += tokens
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"date": self.today, "requests": self.requests, "tokens": self.tokens}), encoding="utf-8")


def plan(n_items: int, n_disagreements: int, requests_left: int, tokens_left: int, tokens_per_image: int) -> tuple[str, int]:
    """(mode, how many items) : '3' frames each, '1' frame each, or 'disagree' (1 frame, only the first n items asked for)."""
    cost = lambda images: images * tokens_per_image + OVERHEAD_TOKENS
    if n_items <= requests_left and n_items * cost(MAX_IMAGES) <= tokens_left:
        return "3", n_items
    if n_items <= requests_left and n_items * cost(1) <= tokens_left:
        return "1", n_items
    fit = min(n_disagreements, requests_left, tokens_left // cost(1))
    return "disagree", max(fit, 0)


class GroqJudge:
    def __init__(self, cfg: dict, cache_dir: Path, key: str | None = None, http=_http, sleep=time.sleep):
        self.cfg, self.http, self.sleep = cfg, http, sleep
        self.key = key if key is not None else os.environ.get("GROQ_API_KEY")
        self.usage = Usage(cache_dir / "groq_usage.json")
        self.exhausted = False
        self.stepped_down = False   # a long rate-limit wait happened: the caller should ask with fewer frames from now on
        self.last_tokens: int | None = None

    @property
    def available(self) -> bool:
        return bool(self.key) and not self.exhausted

    def budget(self) -> tuple[int, int]:
        """(requests, tokens) left today by OUR cap. groq_max_tokens_per_day 0 means the real daily token limit is unknown
        (it is not in the response headers): then only the API's own 429 answers stop us."""
        cap = self.cfg["groq_max_tokens_per_day"]
        return (max(self.cfg["groq_max_requests_per_day"] - self.usage.requests, 0),
                max(cap - self.usage.tokens, 0) if cap else 10 ** 9)

    def ask(self, images_b64: list[str], when: str, names: list[str]) -> dict | None:
        """One request. None if the API did not give a usable answer (or the daily limit is used up)."""
        if not self.available:
            return None
        images_b64 = images_b64[:MAX_IMAGES]
        content = [{"type": "text", "text": PROMPT.format(n=len(images_b64), when=when, names=", ".join(names) or "none")}]
        content += [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in images_b64]
        body = json.dumps({"model": self.cfg["groq_model"], "messages": [{"role": "user", "content": content}],
                           "temperature": 0, "max_tokens": ANSWER_TOKENS, "response_format": {"type": "json_object"}}).encode()
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json", "User-Agent": "ps2-dev/1.0"}
        for attempt in range(MAX_TRIES):
            try:
                reply_headers, raw = self.http(URL, headers, body, 120)
            except urllib.error.HTTPError as e:
                wait = parse_duration(e.headers.get("retry-after") + "s") if e.headers and e.headers.get("retry-after") else None
                if e.code == 429:
                    if wait is not None and wait > self.cfg["groq_max_wait_s"]:
                        self.exhausted = True
                        log.info("Groq limit reached (retry-after %.0f s is longer than groq_max_wait_s); no more judging now", wait)
                        return None
                    if wait is not None and wait > MAX_WAIT_S:   # a token window: wait it out, then ask with fewer frames
                        self.stepped_down = True
                        log.info("Groq token limit: waiting %.0f s, then 1 frame per item", wait)
                    self.sleep(wait if wait is not None else min(2 ** (attempt + 1), MAX_WAIT_S))
                    continue
                if e.code in (401, 403, 404):
                    self.exhausted = True
                    log.info("Groq stopped: HTTP %s (key or model problem)", e.code)
                    return None
                log.info("Groq HTTP %s for one item; skipped", e.code)
                return None
            except (TimeoutError, urllib.error.URLError, OSError):
                self.sleep(2 ** attempt)
                continue
            try:
                reply = json.loads(raw)
                tokens = int(reply.get("usage", {}).get("total_tokens", 0))
                answer = parse_answer(reply["choices"][0]["message"]["content"])
            except (ValueError, KeyError, IndexError, TypeError):
                return None
            self.usage.add(tokens)
            self.last_tokens = tokens
            self._respect_limits(reply_headers)
            return answer
        return None

    def _respect_limits(self, h: dict) -> None:
        """If the headers say the per-minute budget is nearly gone, wait for it to reset before the next request."""
        try:
            left_req = int(h.get("x-ratelimit-remaining-requests", "1"))
            left_tok = int(h.get("x-ratelimit-remaining-tokens", "100000"))
        except ValueError:
            return
        if left_tok < self.cfg["groq_tokens_per_image"] * MAX_IMAGES + OVERHEAD_TOKENS:
            self.sleep(min(parse_duration(h.get("x-ratelimit-reset-tokens")) or 1.0, MAX_WAIT_S))
        elif left_req <= 0:
            self.sleep(min(parse_duration(h.get("x-ratelimit-reset-requests")) or 1.0, MAX_WAIT_S))


def judge_items(judge: GroqJudge, items: list[dict], frames_for, cache_file: Path) -> dict:
    """items: [{key, names, disagree(bool), start_ms, end_ms}] in priority order. frames_for(item, n) -> n base64 JPEGs.
    Returns {key: answer}; answers already in the cache are reused. Logs how many items were judged and in which mode."""
    cached = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    answers = dict(cached.get("answers", {}))
    todo = [it for it in items if it["key"] not in answers]
    if not todo:
        log.info("Groq judge: all %d items already judged (cache)", len(items))
        return answers
    if not judge.available:
        log.info("Groq judge skipped (no GROQ_API_KEY or limit reached): %d items unjudged", len(todo))
        return answers
    requests_left, tokens_left = judge.budget()
    mode, count = plan(len(todo), sum(it["disagree"] for it in todo), requests_left, tokens_left, judge.cfg["groq_tokens_per_image"])
    chosen = todo if mode != "disagree" else [it for it in todo if it["disagree"]][:count]
    log.info("Groq judge: %d items to judge, budget %d requests / %d tokens -> mode %s (%d items)", len(todo), requests_left,
             tokens_left, mode, len(chosen))
    n_images = {"3": 3, "1": 1, "disagree": 1}[mode]
    judged = 0
    for it in chosen:
        if not judge.available:
            break
        if judge.stepped_down:
            n_images = 1
        when = "the start, middle and end of one candidate moment" if n_images == 3 else "the middle of one candidate moment"
        images = frames_for(it, n_images)
        answer = judge.ask(images, when, it["names"])
        if answer is not None:
            answers[it["key"]] = {**answer, "frames": len(images)}
            judged += 1
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"mode": mode, "answers": answers}), encoding="utf-8")
    log.info("Groq judge: %d of %d items judged now (%d in total), %d requests used today", judged, len(chosen), len(answers),
             judge.usage.requests)
    return answers
