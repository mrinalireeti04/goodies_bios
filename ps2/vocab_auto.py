"""Per-video vocabulary from Gemini, with guardrails (nothing here is specific to any clip).

Flow: sample frames (preferring ones with a hand) -> 3 batches of 8 inline JPEGs -> Gemini lists what it clearly
sees as strict JSON -> names are normalised and merged -> a name is kept only if >= vocab_min_batches batches
agree -> YOLOE is run with the name's prompts on the frames that cite it ("confirmed" or "vlm_only") ->
the final vocabulary = these objects + the base file's fixed surfaces + generic catch-alls + ignore list.
Without a usable Gemini the base vocabulary is used unchanged. Only JSON text is cached, never images.
"""
import hashlib
import json
import logging
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from ps2 import vlm

log = logging.getLogger("ps2")

BATCH_COUNT = 3
BATCH_TOKENS = 4000
MAX_SYNONYMS = 3
KEEP_PLURAL = {"pliers", "scissors", "tongs", "shears", "snips", "glasses", "goggles", "tweezers", "jeans"}
PROMPT = (
    "These are frames from a first-person video of factory or workshop work. Each image is preceded by its frame "
    "index. List ONLY objects that are clearly visible AND either being handled by a hand (tools, materials, "
    "parts) or are work surfaces or machines. Do not infer anything from context and do not guess. For each "
    "object give: name (short, common, singular, lowercase); up to 3 synonyms a text-based object detector would "
    "understand; type (portable if it can be carried, otherwise fixed); powered_cutting_or_rotating (true only for "
    "powered tools or machines with a blade, disc or bit that cuts or rotates); frames (the frame indexes from this "
    "request where it is clearly visible). If an object is visible but you cannot name it, put it in unsure with a "
    "short visual description and the frame indexes. Never guess. Return JSON only."
)
_STR, _INT = {"type": "STRING"}, {"type": "INTEGER"}
SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "objects": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "name": _STR, "synonyms": {"type": "ARRAY", "items": _STR},
            "type": {"type": "STRING", "enum": ["portable", "fixed"]},
            "powered_cutting_or_rotating": {"type": "BOOLEAN"},
            "frames": {"type": "ARRAY", "items": _INT}},
            "required": ["name", "synonyms", "type", "powered_cutting_or_rotating", "frames"]}},
        "unsure": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "description": _STR, "frames": {"type": "ARRAY", "items": _INT}},
            "required": ["description", "frames"]}},
    },
    "required": ["objects", "unsure"],
}


# ---------------------------------------------------------------- sampling

def sample_frames(video_path: Path, n_frames: int, hands_by_frame: dict, count: int, max_px: int) -> list[tuple[int, np.ndarray]]:
    """`count` frames spread over the video; in each equal segment the frame with a hand nearest the middle
    (or the middle frame if no hand is seen there). Images are shrunk to at most max_px on the long side."""
    cap = cv2.VideoCapture(str(video_path))
    out = []
    for segment in np.array_split(np.arange(n_frames), count):
        if len(segment) == 0:
            continue
        mid = int(segment[len(segment) // 2])
        with_hand = [int(f) for f in segment if hands_by_frame.get(int(f))]
        idx = min(with_hand, key=lambda f: abs(f - mid)) if with_hand else mid
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            continue
        scale = max_px / max(bgr.shape[:2])
        out.append((idx, cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr))
    cap.release()
    return out


def frames_hash(frames: list[tuple[int, np.ndarray]]) -> str:
    h = hashlib.sha1()
    for idx, img in frames:
        h.update(str(idx).encode())
        h.update(np.ascontiguousarray(img).tobytes())
    return h.hexdigest()


def split_batches(frames: list, k: int = BATCH_COUNT) -> list[list]:
    """Interleaved, so every batch spans the whole video."""
    return [frames[i::k] for i in range(k)]


# ---------------------------------------------------------------- reading the answers

def normalise(name: str) -> str:
    """lowercase, single spaces, singular."""
    words = "".join(c if c.isalnum() or c in " -" else " " for c in str(name).lower()).split()
    if not words:
        return ""
    last = words[-1]
    if last not in KEEP_PLURAL and len(last) > 3:
        if last.endswith("ies"):
            last = last[:-3] + "y"
        elif last.endswith(("sses", "xes", "ches", "shes")):
            last = last[:-2]
        elif last.endswith("s") and not last.endswith("ss"):
            last = last[:-1]
    return " ".join(words[:-1] + [last])


def parse_batch(text: str | None, shown: set[int]) -> dict | None:
    """Validated {"objects": [...], "unsure": [...]} for one batch, or None if the answer is unusable.
    Frame indexes that were not part of this batch are dropped; objects/unsure items left with none are dropped."""
    try:
        data = json.loads(text or "")
        raw_objects, raw_unsure = data["objects"], data["unsure"]
    except (ValueError, KeyError, TypeError):
        return None
    objects = []
    for o in raw_objects if isinstance(raw_objects, list) else []:
        try:
            name = normalise(o["name"])
            frames = sorted({int(f) for f in o["frames"]} & shown)
            kind = o["type"] if o["type"] in ("portable", "fixed") else None
            synonyms = [s for s in dict.fromkeys(normalise(s) for s in o.get("synonyms", [])[:MAX_SYNONYMS]) if s and s != name]
            powered = o.get("powered_cutting_or_rotating") is True
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if name and frames and kind:
            objects.append({"name": name, "synonyms": synonyms, "type": kind, "powered": powered, "frames": frames})
    unsure = []
    for u in raw_unsure if isinstance(raw_unsure, list) else []:
        try:
            description = " ".join(str(u["description"]).split())
            frames = sorted({int(f) for f in u["frames"]} & shown)
        except (KeyError, TypeError, ValueError):
            continue
        if description and frames:
            unsure.append({"description": description, "frames": frames})
    return {"objects": objects, "unsure": unsure}


def merge_batches(batches: list[dict], min_batches: int) -> tuple[list[dict], list[dict]]:
    """Merge objects across batches (same name, or each listing the other as a synonym), keep those seen in
    >= min_batches batches, and collect the unsure items (same description merged)."""
    mentions = [(b, o) for b, batch in enumerate(batches) for o in batch["objects"]]
    parent = list(range(len(mentions)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (_, a) in enumerate(mentions):
        for j in range(i + 1, len(mentions)):
            b = mentions[j][1]
            if a["name"] == b["name"] or (a["name"] in b["synonyms"] and b["name"] in a["synonyms"]):
                parent[find(j)] = find(i)
    groups: dict[int, list] = {}
    for i, m in enumerate(mentions):
        groups.setdefault(find(i), []).append(m)

    entries = []
    for members in groups.values():
        batch_ids = {b for b, _ in members}
        if len(batch_ids) < min_batches:
            continue
        names = Counter(o["name"] for _, o in members)
        name = max(names, key=lambda n: (names[n], -len(n)))
        synonyms = Counter(s for _, o in members for s in [o["name"], *o["synonyms"]] if s != name)
        kinds = Counter(o["type"] for _, o in members)
        entries.append({
            "name": name,
            "synonyms": [s for s, _ in synonyms.most_common(MAX_SYNONYMS)],
            "type": "fixed" if kinds["fixed"] > kinds["portable"] else "portable",
            "frames": sorted({f for _, o in members for f in o["frames"]}),
            "batches": len(batch_ids),
            "powered_votes": sum(o["powered"] for _, o in members), "mentions": len(members),
        })
    for e in entries:  # hazard: powered in at least half of the mentions AND seen in >= 2 frames
        e["hazard"] = e["powered_votes"] * 2 >= e["mentions"] and e["powered_votes"] > 0 and len(e["frames"]) >= 2

    unsure: dict[str, dict] = {}
    for batch in batches:
        for u in batch["unsure"]:
            item = unsure.setdefault(" ".join(u["description"].lower().split()), {"description": u["description"], "frames": set()})
            item["frames"] |= set(u["frames"])
    return entries, [{"description": u["description"], "frames": sorted(u["frames"])} for u in unsure.values()]


# ---------------------------------------------------------------- vocabulary

def key_of(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")


def build_vocab(entries: list[dict], base: dict) -> dict:
    """Gemini objects + the base file's fixed non-hazard surfaces + its generic catch-alls and ignore list."""
    objects: dict[str, dict] = {}
    for e in entries:
        spec = {"type": e["type"], "prompts": list(dict.fromkeys([e["name"], *e["synonyms"]]))[:MAX_SYNONYMS + 1]}
        if e["hazard"]:
            spec["hazard"] = True
        objects[key_of(e["name"])] = spec
    for key, spec in base["objects"].items():
        if spec["type"] == "fixed" and not spec.get("hazard") and key not in objects:
            objects[key] = spec
    return {"objects": objects, "generic": base["generic"], "ignore": base["ignore"]}


def confirm(entries: list[dict], frames: dict[int, np.ndarray], detector) -> None:
    """Set e["status"]: 'confirmed' if YOLOE finds the entry's prompts on one of the frames that cite it."""
    found: dict[int, set[str]] = {}
    for idx in sorted({f for e in entries for f in e["frames"]}):
        _, _, cls = detector.detect(frames[idx], [])
        found[idx] = {detector.prompt_to_key[detector.prompts[int(c)]] for c in cls}
    for e in entries:
        e["status"] = "confirmed" if any(key_of(e["name"]) in found.get(f, ()) for f in e["frames"]) else "vlm_only"


# ---------------------------------------------------------------- the whole step

def build(video_path: Path, hands_by_frame: dict, n_frames: int, fps: float, clip_id: str, cfg: dict, base: dict,
          detector, gem: vlm.Gemini, cache_dir: Path) -> dict:
    """Returns {"vocab", "source" ('gemini'|'base'), "entries", "unsure", "frames_ms"}. Never raises on API trouble."""
    started = time.time()
    base_result = {"vocab": base, "source": "base", "entries": [], "unsure": []}
    if not gem.available:
        return base_result
    frames = sample_frames(video_path, n_frames, hands_by_frame, cfg["vocab_frames"], cfg["vocab_frame_max_px"])
    if len(frames) < BATCH_COUNT * 2:
        log.info("per-video vocabulary skipped: only %d frames could be sampled", len(frames))
        return base_result

    digest = frames_hash(frames)
    cache_file = cache_dir / f"vocab_{clip_id}.json"
    texts = None
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("frames_hash") == digest:
            texts = cached["batches"]
    shown_per_batch = [{idx for idx, _ in batch} for batch in split_batches(frames)]
    if texts is None:
        texts = []
        for batch in split_batches(frames):
            parts = [{"text": PROMPT}]
            for idx, img in batch:
                image = vlm.image_part(img)
                if image is not None:
                    parts += [{"text": f"Frame {idx}:"}, image]
            texts.append(gem.generate(parts, BATCH_TOKENS, SCHEMA))
        if all(t is not None for t in texts):
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"frames_hash": digest, "batches": texts}), encoding="utf-8")
    parsed = [parse_batch(t, shown) for t, shown in zip(texts, shown_per_batch)]
    parsed = [p for p in parsed if p is not None]
    if len(parsed) < cfg["vocab_min_batches"]:
        log.info("per-video vocabulary skipped: only %d of %d Gemini answers were usable", len(parsed), BATCH_COUNT)
        return base_result

    entries, unsure = merge_batches(parsed, cfg["vocab_min_batches"])
    entries.sort(key=lambda e: (-e["batches"], -len(e["frames"]), e["name"]))
    entries = entries[:cfg["vocab_max_objects"]]
    if not entries:
        log.info("per-video vocabulary skipped: no object was named in %d batches", cfg["vocab_min_batches"])
        return base_result

    detector.set_vocab(build_vocab(entries, base))      # candidates + fixed surfaces, to cross-check on the cited frames
    confirm(entries, dict(frames), detector)
    vocab = build_vocab(entries, base)
    detector.set_vocab(vocab)
    log.info("per-video vocabulary: %d objects (%d confirmed, %d vlm_only, %d hazard), %d unsure, %.1f s, %d API calls so far",
             len(entries), sum(e["status"] == "confirmed" for e in entries), sum(e["status"] == "vlm_only" for e in entries),
             sum(e["hazard"] for e in entries), len(unsure), time.time() - started, gem.calls)
    return {"vocab": vocab, "source": "gemini", "entries": entries, "unsure": unsure}


# ---------------------------------------------------------------- output

def meta_vocab(info: dict, times: list[int]) -> list[dict]:
    return [{"label": key_of(e["name"]), "status": e["status"], "hazard": e["hazard"],
             "seen_ms": [times[f] for f in e["frames"] if f < len(times)]} for e in info["entries"]]


def review_items(info: dict, dets_by_frame: list[list[dict]], times: list[int], frame_ms: float, generic: set[str]) -> list[dict]:
    """'vocab_unverified' for vlm_only labels and 'vlm_unsure_object' for the unsure list. The span is the cited frames;
    object_id is a tracked object seen in them (same label for vlm_only; for unsure, preferring unnamed ones) or None."""
    def span(frames):
        usable = [f for f in frames if f < len(times)]
        return (times[min(usable)], int(round(times[max(usable)] + frame_ms))) if usable else None

    def object_in(frames, wanted):
        seen = Counter(d["object_id"] for f in frames if f < len(dets_by_frame) for d in dets_by_frame[f] if wanted(d))
        return seen.most_common(1)[0][0] if seen else None

    items = []
    for e in info["entries"]:
        if e["status"] == "vlm_only" and span(e["frames"]):
            start, end = span(e["frames"])
            items.append({"start_ms": start, "end_ms": end, "hand_id": None,
                          "object_id": object_in(e["frames"], lambda d, k=key_of(e["name"]): d["label"] == k),
                          "reason": "vocab_unverified", "suggested_label": e["name"]})
    for u in info["unsure"]:
        if span(u["frames"]):
            start, end = span(u["frames"])
            items.append({"start_ms": start, "end_ms": end, "hand_id": None,
                          "object_id": object_in(u["frames"], lambda d: d["label"] in generic) or object_in(u["frames"], lambda d: True),
                          "reason": "vlm_unsure_object", "suggested_label": u["description"]})
    return items
