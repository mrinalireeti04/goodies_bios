"""Fusion of the two kinds of evidence about a tool: the ledger (names and ~1 s timing, from Gemini watching the clip)
and the local pipeline (frame-exact hand/object contacts).

A tool is "identified" only when both agree: the ledger has a use of tool K that overlaps the object's contacts in time
AND a look at the object's crop also says K. Everything else is "unidentified" and goes to review with a reason.
Fixed objects are never relabelled (an answer only becomes a suggestion).
"""
import json
import logging
from collections import Counter, defaultdict

from ps2 import vlm
from ps2.ledger import key_of
from ps2.objects import set_object_label

log = logging.getLogger("ps2")

CROP_TOKENS = 3000
CROP_SCHEMA = {"type": "OBJECT", "properties": {"answers": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "crop": {"type": "STRING"}, "answer": {"type": "STRING"}}, "required": ["crop", "answer"]}}}, "required": ["answers"]}
CROP_PROMPT = (
    "Each image is a crop of one object that a worker's hand is touching or holding (labelled C1, C2, ...). Tools seen in "
    "this video:\n{tools}\n\nFor every crop answer with exactly one of: a tool_key from the list, other:<short name>, "
    "not_a_tool, or unknown. If a crop lists candidates, choose among those (or the other three answers); otherwise "
    "choose from the whole list. Answer unknown rather than guess. Return JSON only."
)


# ---------------------------------------------------------------- evidence from time

def hand_ok(ledger_hand: str, contact_side: str) -> bool:
    """Same hand when both are known; 'both' and 'unknown' (on either side) never rule a candidate out."""
    return ledger_hand in ("both", "unknown") or contact_side in ("unknown", "") or ledger_hand == contact_side


def time_candidates(contacts: list[dict], ledger: dict, tol_ms: float) -> list[tuple[str, float]]:
    """(tool_key, seconds of overlap) for ledger tools with a use close in time to these contacts, best first."""
    found: dict[str, float] = defaultdict(float)
    for tool in ledger["tools"]:
        if tool["workpiece"]:
            continue
        for use in tool["uses"]:
            a, b = use["start_s"] * 1000 - tol_ms, use["end_s"] * 1000 + tol_ms
            for c in contacts:
                overlap = min(b, c["end_ms"]) - max(a, c["start_ms"])
                if overlap > 0 and hand_ok(use["hand"], c.get("_side", "unknown")):
                    found[tool["tool_key"]] += overlap / 1000
    return sorted(found.items(), key=lambda kv: -kv[1])


# ---------------------------------------------------------------- evidence from the crop

def tools_text(ledger: dict, keys: set[str]) -> str:
    return "\n".join(f"- {t['tool_key']}: {t['name']} ({t['appearance'] or 'no description'}; {t['type']})"
                     for t in ledger["tools"] if t["tool_key"] in keys and not t["workpiece"])


def parse_crop_answers(text: str | None, asked: list[str], valid_keys: set[str]) -> dict[int, tuple[str, str]]:
    """{index: (kind, value)} with kind tool | other | not_a_tool | unknown. Unusable answers become unknown."""
    out: dict[int, tuple[str, str]] = {}
    try:
        items = json.loads(text or "")["answers"]
    except (ValueError, KeyError, TypeError):
        return out
    for item in items if isinstance(items, list) else []:
        try:
            n = int(str(item["crop"]).strip().upper().lstrip("C")) - 1
            answer = str(item["answer"]).strip().lower()
        except (KeyError, ValueError, TypeError):
            continue
        if not 0 <= n < len(asked):
            continue
        if answer.startswith("other:") and answer[6:].strip():
            out[n] = ("other", answer[6:].strip())
        elif key_of(answer) in valid_keys:
            out[n] = ("tool", key_of(answer))
        elif answer == "not_a_tool":
            out[n] = ("not_a_tool", "")
        else:
            out[n] = ("unknown", "")
    return out


def ask_crops(gem: vlm.Gemini, items: list[tuple[str, object, set[str]]], ledger: dict, cfg: dict) -> dict[str, tuple[str, str]]:
    """items: (object_id, crop, candidate tool keys) in priority order. Up to crop_batch_size crops per request."""
    all_keys = {t["tool_key"] for t in ledger["tools"] if not t["workpiece"]}
    answers: dict[str, tuple[str, str]] = {}
    size = cfg["crop_batch_size"]
    for start in range(0, len(items), size):
        if not gem.available:
            break
        batch = items[start:start + size]
        union = set().union(*(cands or all_keys for _, _, cands in batch)) & all_keys
        parts = [{"text": CROP_PROMPT.format(tools=tools_text(ledger, union))}]
        asked = []
        for n, (oid, crop, cands) in enumerate(batch, 1):
            image = vlm.image_part(crop)
            if image is None:
                continue
            asked.append(oid)
            parts += [{"text": f"C{n}" + (f" candidates: {', '.join(sorted(cands))}" if cands else " (no candidates: whole list)") + ":"}, image]
        if not asked:
            continue
        text = gem.generate(parts, CROP_TOKENS, CROP_SCHEMA)
        if text is None:
            if gem.calls >= cfg["vlm_max_calls_per_clip"]:
                break
            continue
        index = [oid for oid, _, _ in batch]
        for n, answer in parse_crop_answers(text, index, all_keys).items():
            answers[index[n]] = answer
    return answers


# ---------------------------------------------------------------- the decision

def decide(time_cands: list[tuple[str, float]], crop: tuple[str, str] | None, names: dict[str, str]) -> dict:
    """{status: identified|single|conflict|none, tool_key, suggested, evidence}. See the module docstring."""
    keys = [k for k, _ in time_cands]
    kind, value = crop or ("unknown", "")
    crop_tool = value if kind == "tool" else None
    if crop_tool and crop_tool in keys:
        return {"status": "identified", "tool_key": crop_tool, "suggested": names[crop_tool], "evidence": ["ledger_time", "crop_match"]}
    if keys and (crop_tool or kind == "other"):   # both point to a tool, but not the same one
        return {"status": "conflict", "tool_key": None, "suggested": names[crop_tool] if crop_tool else value,
                "evidence": ["ledger_time", "crop_" + ("other" if kind == "other" else "different")]}
    if crop_tool or kind == "other":              # only the crop points to a tool
        return {"status": "single", "tool_key": None, "suggested": names[crop_tool] if crop_tool else value, "evidence": ["crop_match" if crop_tool else "crop_other"]}
    if keys:                                      # only time points to a tool
        return {"status": "single", "tool_key": None, "suggested": names[keys[0]], "evidence": ["ledger_time"]}
    return {"status": "none", "tool_key": None, "suggested": None, "evidence": []}


# ---------------------------------------------------------------- applying it

def rename_object(old: str, new: str, dets_by_frame, decisions, contacts) -> None:
    for dets in dets_by_frame:
        for d in dets:
            if d["object_id"] == old:
                d["object_id"] = new
    for frame in decisions:
        for dec in frame.values():
            if dec["object_id"] == old:
                dec["object_id"] = new
    for c in contacts:
        if c["object_id"] == old:
            c["object_id"] = new


def fuse(ledger: dict, contacts: list[dict], dets_by_frame, decisions, crops: dict, gem: vlm.Gemini, vocab: dict,
         times: list[int], frame_ms: float, frame_area: float, side_ids: dict[str, str], cfg: dict) -> dict:
    """Returns {force_status, suggestions, tool_info, review_items, merged}. Changes dets/decisions/contacts in place
    (labels of identified portable objects, merged object ids)."""
    names = {t["tool_key"]: t["name"] for t in ledger["tools"]}
    by_object: dict[str, list[dict]] = defaultdict(list)
    for c in contacts:
        if c["object_id"]:
            by_object[c["object_id"]].append(c)
    obj_type, frames_of = {}, defaultdict(set)
    for i, dets in enumerate(dets_by_frame):
        for d in dets:
            obj_type.setdefault(d["object_id"], d["type"])
            frames_of[d["object_id"]].add(i)

    cands = {oid: time_candidates(cs, ledger, cfg["ledger_time_tol_ms"]) for oid, cs in by_object.items()}
    order = sorted(by_object, key=lambda o: -sum(c["end_ms"] - c["start_ms"] for c in by_object[o]))
    queue = [(oid, crops[oid], {k for k, _ in cands[oid]}) for oid in order if oid in crops]
    answers = ask_crops(gem, queue, ledger, cfg)
    log.info("fusion: %d objects with contacts, %d crop answers, Gemini requests so far: %d", len(by_object), len(answers), gem.calls)

    force_status, suggestions, tool_info, items = {}, {}, {}, []

    def item(oid, reason, suggested):
        cs = by_object[oid]
        items.append({"start_ms": min(c["start_ms"] for c in cs), "end_ms": max(c["end_ms"] for c in cs), "hand_id": None,
                      "object_id": oid, "reason": reason, "suggested_label": suggested})

    for oid in order:
        verdict = decide(cands[oid], answers.get(oid), names)
        evidence = verdict["evidence"]
        if obj_type.get(oid) == "fixed":     # never relabel a fixed object: the answer is only a suggestion
            if verdict["suggested"]:
                suggestions[oid] = verdict["suggested"]
            continue
        if verdict["status"] == "identified":
            key = key_of(names[verdict["tool_key"]])
            if key not in vocab["objects"]:  # not a detector label we can use: keep as a suggestion
                verdict["status"] = "single"
            else:
                changed = next((d["label"] for dets in dets_by_frame for d in dets if d["object_id"] == oid), "") != key
                if changed:
                    set_object_label(dets_by_frame, oid, key, vocab, frame_area, cfg)
                    for c in by_object[oid]:
                        c["object_label"] = key
                        c["reasons"] = sorted(set(c.get("reasons", [])) | {"label_from_vlm"})
                force_status[oid] = "identified"
                tool_info[oid] = {"tool_key": verdict["tool_key"], "id_confidence": "high", "evidence": evidence}
                continue
        force_status[oid] = "unidentified"
        if verdict["status"] == "single":
            suggestions[oid] = verdict["suggested"]
            tool_info[oid] = {"tool_key": None, "id_confidence": "low", "evidence": evidence}
            item(oid, "tool_single_evidence", verdict["suggested"])
        elif verdict["status"] == "conflict":
            suggestions[oid] = verdict["suggested"]
            tool_info[oid] = {"tool_key": None, "id_confidence": "low", "evidence": evidence}
            item(oid, "tool_evidence_conflict", verdict["suggested"])
        # status none: a held portable object without any ledger match stays unidentified (summary adds unidentified_object)

    # re-entry: objects identified as the same tool and never visible together become one object
    merged: dict[str, str] = {}
    by_tool: dict[str, list[str]] = defaultdict(list)
    for oid, info in tool_info.items():
        if info["tool_key"]:
            by_tool[info["tool_key"]].append(oid)
    for key, oids in by_tool.items():
        oids.sort(key=lambda o: min(frames_of[o]))
        keep, kept_frames = oids[0], set(frames_of[oids[0]])
        for other in oids[1:]:
            if kept_frames & frames_of[other]:   # visible together: two real objects; keep separate and ask a human
                item(other, "tool_evidence_conflict", names[key])
                continue
            rename_object(other, keep, dets_by_frame, decisions, contacts)
            kept_frames |= frames_of[other]
            merged[other] = keep
            by_object[keep] += by_object.pop(other, [])
            tool_info.pop(other, None)
            force_status.pop(other, None)
            suggestions.pop(other, None)
    for old, new in merged.items():
        log.info("fusion: %s merged into %s (same tool %s)", old, new, tool_info[new]["tool_key"])

    # ledger uses nobody saw locally, and handled objects the ledger could not name
    tol = cfg["ledger_time_tol_ms"]
    for tool in ledger["tools"]:
        if tool["workpiece"]:
            continue
        for use in tool["uses"]:
            a, b = use["start_s"] * 1000, use["end_s"] * 1000
            if not any(c["end_ms"] > a - tol and c["start_ms"] < b + tol and hand_ok(use["hand"], c.get("_side", "unknown")) for c in contacts):
                items.append({"start_ms": int(round(a)), "end_ms": int(round(b)), "hand_id": side_ids.get(use["hand"]),
                              "object_id": None, "reason": "tool_use_ai_only", "suggested_label": tool["name"]})
    for u in ledger["unsure"]:
        a, b = u["start_s"] * 1000, u["end_s"] * 1000
        near = [c["object_id"] for c in contacts if c["object_id"] and c["end_ms"] > a and c["start_ms"] < b]
        items.append({"start_ms": int(round(a)), "end_ms": int(round(b)), "hand_id": None,
                      "object_id": Counter(near).most_common(1)[0][0] if near else None,
                      "reason": "vlm_unsure_object", "suggested_label": u["description"]})
    return {"force_status": force_status, "suggestions": suggestions, "tool_info": tool_info, "review_items": items,
            "merged": merged}
