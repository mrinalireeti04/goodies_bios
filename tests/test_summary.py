import numpy as np

from ps2 import schema, summary

MS = 100.0  # 10 fps keeps the numbers easy
GENERIC = {"tool", "object"}


def obj(oid, label="hammer", conf=0.8):
    return {"object_id": oid, "label": label, "type": "portable", "bbox_px": [0, 0, 10, 10],
            "confidence": conf, "is_hazard": False}


def frames_for(spans, n=200, hand_states=None):
    """spans: {object_id: [(first_frame, last_frame), ...]} or {object_id: (label, conf, [...])}."""
    frames = [{"frame_idx": i, "t_ms": int(i * MS), "objects": [], "hand_states": []} for i in range(n)]
    for oid, spec in spans.items():
        label, conf, runs = spec if isinstance(spec[0], str) else ("hammer", 0.8, spec)
        for a, b in runs:
            for i in range(a, b + 1):
                frames[i]["objects"].append(obj(oid, label, conf))
    return frames


def contact(oid, start, end, interaction="hold", **extra):
    return {"event_id": "c", "start_ms": start, "end_ms": end, "hand_id": "h_right", "object_id": oid,
            "object_label": "hammer", "state": "portable", "interaction": interaction, "confidence": 0.8, **extra}


def tool_for(frames, contacts, cfg, oid="obj_1"):
    tools = summary.build_tools(frames, contacts, GENERIC, MS, cfg)
    return next((t for t in tools if t["object_id"] == oid), None)


def test_times_and_appearances(cfg):
    frames = frames_for({"obj_1": [(10, 29), (31, 40), (100, 119)]})  # a 1-frame hole is not a re-entry
    t = tool_for(frames, [], cfg)
    assert t["first_seen_ms"] == 1000 and t["last_seen_ms"] == 11900
    assert t["visible_ms"] == (20 + 10 + 20) * MS
    assert t["appearances"] == [{"start_ms": 1000, "end_ms": 4100}, {"start_ms": 10000, "end_ms": 12000}]


def test_held_ms_and_use_count_merge_short_gaps(cfg):
    frames = frames_for({"obj_1": [(0, 190)]})
    contacts = [contact("obj_1", 1000, 2000), contact("obj_1", 2500, 3000),   # 0.5 s gap: same use
                contact("obj_1", 6000, 7000),                                  # separate use
                contact("obj_1", 8000, 9000, interaction="touch")]             # touch is not held
    t = tool_for(frames, contacts, cfg)
    assert t["held_ms"] == 1000 + 500 + 1000 and t["use_count"] == 2


def test_both_hands_holding_do_not_double_count(cfg):
    frames = frames_for({"obj_1": [(0, 190)]})
    left = dict(contact("obj_1", 1000, 3000), hand_id="h_left")
    t = tool_for(frames, [contact("obj_1", 1000, 3000), left], cfg)
    assert t["held_ms"] == 2000 and t["use_count"] == 1


def test_label_status(cfg):
    frames = frames_for({"obj_1": ("hammer", 0.8, [(0, 20)]), "obj_2": ("tool", 0.9, [(0, 20)]),
                         "obj_3": ("hammer", 0.2, [(0, 20)])})
    status = {t["object_id"]: t["label_status"] for t in summary.build_tools(frames, [], GENERIC, MS, cfg)}
    assert status == {"obj_1": "identified", "obj_2": "unidentified", "obj_3": "unidentified"}


def test_brief_objects_are_left_out_unless_in_a_contact(cfg):
    frames = frames_for({"obj_1": [(0, 2)], "obj_2": [(0, 2)]})
    tools = summary.build_tools(frames, [contact("obj_2", 0, 300)], GENERIC, MS, cfg)
    assert [t["object_id"] for t in tools] == ["obj_2"]


def hand_frames(states):
    frames = frames_for({"obj_1": [(0, 199)]})
    for fr, (state, oid) in zip(frames, states):
        fr["hand_states"] = [{"hand_id": "h_right", "state": state, "object_id": oid, "confidence": None}]
    return frames


def test_unknown_stretch_over_one_second_is_queued(cfg):
    frames = hand_frames([("none", None)] * 5 + [("unknown", "obj_1")] * 12 + [("none", None)] * 5
                         + [("unknown", None)] * 8 + [("none", None)] * 5)
    items = summary.long_unknown_items(frames, MS, cfg)
    assert [(i["start_ms"], i["end_ms"], i["object_id"]) for i in items] == [(500, 1700, "obj_1")]  # 8 frames = 0.8 s: ignored


def test_review_queue_has_each_kind_once_and_ids_are_sequential(cfg):
    frames = hand_frames([("unknown", None)] * 15)
    tools = [
        {"object_id": "obj_1", "label": "tool", "label_status": "unidentified", "type": "portable", "is_hazard": False,
         "first_seen_ms": 0, "last_seen_ms": 19900, "visible_ms": 20000, "held_ms": 0, "use_count": 0, "appearances": []},
        {"object_id": "obj_9", "label": "hammer", "label_status": "identified", "type": "portable", "is_hazard": False,
         "first_seen_ms": 0, "last_seen_ms": 900, "visible_ms": 1000, "held_ms": 0, "use_count": 0, "appearances": []},
    ]
    contacts = [contact("obj_1", 2000, 3000, review_needed=True, reasons=["ambiguous_candidates"]),
                contact("obj_1", 4000, 5000, review_needed=True),
                contact("obj_1", 6000, 7000, review_needed=False)]
    identity = [{"item_id": "x", "start_ms": 100, "end_ms": 200, "hand_id": None, "object_id": "obj_1",
                 "reason": "identity_uncertain", "suggested_label": None},
                {"item_id": "y", "start_ms": 100, "end_ms": 200, "hand_id": None, "object_id": "obj_77",
                 "reason": "identity_uncertain", "suggested_label": None}]  # obj_77 has no tools row
    q = summary.build_review_queue(tools, contacts, frames, identity, {"obj_1": "chisel"}, MS, cfg)
    reasons = [(i["reason"], i["object_id"]) for i in q]
    assert sorted(reasons) == sorted([("unidentified_object", "obj_1"), ("identity_uncertain", "obj_1"),
                                      ("ambiguous_candidates", "obj_1"), ("low_contact_confidence", "obj_1"),
                                      ("long_unknown", None)])
    assert [i["item_id"] for i in q] == [f"r_{n}" for n in range(1, 6)]
    assert next(i for i in q if i["reason"] == "unidentified_object")["suggested_label"] == "chisel"
    assert all(i["start_ms"] < i["end_ms"] for i in q)


def test_summary_output_passes_the_schema(cfg):
    frames = frames_for({"obj_1": [(0, 50)]})
    contacts = [contact("obj_1", 1000, 2000, review_needed=True, reasons=["x"])]
    tools = summary.build_tools(frames, contacts, GENERIC, MS, cfg)
    queue = summary.build_review_queue(tools, contacts, frames, [], {}, MS, cfg)
    schema.validate(schema.build_output("c", frames, contacts, tools=tools, review_queue=queue))


def test_best_crop_picks_the_clearest_view():
    from ps2.objects import best_crops
    pieces = {1: {"best_score": 0.2, "best_crop": np.zeros((4, 4, 3))},
              2: {"best_score": 0.6, "best_crop": np.ones((4, 4, 3))},
              3: {"best_score": 0.9, "best_crop": None}}
    out = best_crops(pieces, {1: 0, 2: 0, 3: 1})
    assert out["obj_1"].sum() == 48 and "obj_2" not in out


def test_label_from_vlm_alone_is_not_a_review_reason(cfg):
    frames = frames_for({"obj_1": [(0, 50)]})
    contacts = [contact("obj_1", 1000, 2000, review_needed=True, reasons=["label_from_vlm", "low_contact_confidence"])]
    tools = summary.build_tools(frames, contacts, GENERIC, MS, cfg)
    queue = summary.build_review_queue(tools, contacts, frames, [], {}, MS, cfg)
    assert [i["reason"] for i in queue] == ["low_contact_confidence"]
