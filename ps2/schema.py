"""Build and validate the PS2 output dict (see INTEGRATION.md)."""

SCHEMA_VERSION = "1.0"
OBJECT_TYPES = {"portable", "fixed", "unknown"}
HAND_STATES = {"none", "portable", "fixed", "unknown"}
CONTACT_STATES = {"portable", "fixed"}
INTERACTIONS = {"touch", "hold", "unknown"}
LABEL_STATUS = {"identified", "unidentified"}


def build_output(clip_id: str, frames: list[dict], contacts: list[dict], meta: dict | None = None,
                 tools: list[dict] | None = None, review_queue: list[dict] | None = None,
                 safety_events: list[dict] | None = None) -> dict:
    """Assemble the output dict. Private keys (starting with '_') are stripped from contacts."""
    out = {
        "schema_version": SCHEMA_VERSION,
        "clip_id": clip_id,
        "frames": frames,
        "contacts": [{k: v for k, v in c.items() if not k.startswith("_")} for c in contacts],
    }
    for name, value in (("meta", meta), ("tools", tools), ("review_queue", review_queue),
                        ("safety_events", safety_events)):
        if value is not None:
            out[name] = value
    return out


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"invalid PS2 output: {msg}")


def _is_box(b) -> bool:
    return isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)


def _is_conf(c) -> bool:
    return c is None or (isinstance(c, (int, float)) and 0 <= c <= 1)


def validate(result: dict) -> None:
    """Raise ValueError if `result` breaks the contract in INTEGRATION.md."""
    for key in ("schema_version", "clip_id", "frames", "contacts"):
        _check(key in result, f"missing field '{key}'")
    _check(result["schema_version"] == SCHEMA_VERSION, "schema_version must be '1.0'")

    known_ids = {o["object_id"] for fr in result["frames"] for o in fr.get("objects", []) if "object_id" in o}
    for fr in result["frames"]:
        for key in ("frame_idx", "t_ms", "objects", "hand_states"):
            _check(key in fr, f"frame missing '{key}'")
        for o in fr["objects"]:
            for key in ("object_id", "label", "type", "bbox_px", "confidence"):
                _check(key in o, f"object missing '{key}' (frame {fr['frame_idx']})")
            _check(o["type"] in OBJECT_TYPES, f"bad object type '{o['type']}'")
            _check(_is_box(o["bbox_px"]), f"bad bbox for {o['object_id']}")
            _check(_is_conf(o["confidence"]), f"bad confidence for {o['object_id']}")
        for hs in fr["hand_states"]:
            for key in ("hand_id", "state", "object_id", "confidence"):
                _check(key in hs, f"hand_state missing '{key}' (frame {fr['frame_idx']})")
            _check(hs["state"] in HAND_STATES, f"bad hand state '{hs['state']}'")
            _check(hs["state"] != "none" or hs["object_id"] is None, "state 'none' needs object_id null")
            _check(hs["object_id"] is None or hs["object_id"] in known_ids,
                   f"hand_state references unknown object {hs['object_id']}")
            _check(_is_conf(hs["confidence"]), "bad hand_state confidence")

    for c in result["contacts"]:
        for key in ("event_id", "start_ms", "end_ms", "hand_id", "object_id", "object_label",
                    "state", "interaction", "confidence"):
            _check(key in c, f"contact missing '{key}'")
        _check(c["start_ms"] < c["end_ms"], f"contact {c['event_id']}: start_ms must be < end_ms")
        _check(c["state"] in CONTACT_STATES, f"contact {c['event_id']}: bad state '{c['state']}'")
        _check(c["interaction"] in INTERACTIONS, f"contact {c['event_id']}: bad interaction")
        _check(c["object_id"] is None or c["object_id"] in known_ids, f"contact references unknown object {c['object_id']}")
        _check(_is_conf(c["confidence"]), f"contact {c['event_id']}: bad confidence")

    for t in result.get("tools", []):
        for key in ("object_id", "label", "label_status", "type", "is_hazard", "first_seen_ms", "last_seen_ms",
                    "visible_ms", "held_ms", "use_count", "appearances"):
            _check(key in t, f"tool missing '{key}'")
        for a in t["appearances"]:
            _check(a["start_ms"] < a["end_ms"], f"tool {t['object_id']}: appearance start_ms must be < end_ms")
        _check(t["object_id"] in known_ids, f"tool references unknown object {t['object_id']}")
        _check(t["label_status"] in LABEL_STATUS, "bad label_status")
        _check(t["type"] in OBJECT_TYPES, "bad tool type")
    for r in result.get("review_queue", []):
        for key in ("item_id", "start_ms", "end_ms", "hand_id", "object_id", "reason", "suggested_label"):
            _check(key in r, f"review item missing '{key}'")
        _check(r["start_ms"] < r["end_ms"], f"review item {r['item_id']}: start_ms must be < end_ms")
        _check(r["object_id"] is None or r["object_id"] in known_ids, "review item references unknown object")
    for s in result.get("safety_events", []):
        for key in ("event_id", "start_ms", "end_ms", "hand_id", "hazard_object_id", "min_distance_norm",
                    "hazard_moving", "confidence", "reasons"):
            _check(key in s, f"safety event missing '{key}'")
        _check(s["start_ms"] < s["end_ms"], f"safety event {s['event_id']}: start_ms must be < end_ms")
        _check(s["hazard_object_id"] in known_ids, "safety event references unknown object")
        _check("2d_distance_only" in s["reasons"], "safety event must include '2d_distance_only'")
