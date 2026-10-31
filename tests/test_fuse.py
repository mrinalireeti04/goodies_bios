import json

import numpy as np
import pytest

from ps2 import fuse, masks, safety
from ps2.hands import Hand, assign_tracks, filter_bystanders
from test_hands import make_kp

LEDGER = {
    "tools": [
        {"tool_key": "hammer_1", "name": "hammer", "synonyms": [], "appearance": "red handle", "type": "portable", "powered": False,
         "workpiece": False, "uses": [{"start_s": 10, "end_s": 20, "hand": "left", "action": "hit", "certainty": "high"}]},
        {"tool_key": "saw_1", "name": "saw", "synonyms": [], "appearance": "long blade", "type": "portable", "powered": False,
         "workpiece": False, "uses": [{"start_s": 10, "end_s": 20, "hand": "unknown", "action": "cut", "certainty": "medium"},
                                      {"start_s": 60, "end_s": 70, "hand": "right", "action": "cut", "certainty": "high"}]},
        {"tool_key": "board_1", "name": "board", "synonyms": [], "appearance": "", "type": "portable", "powered": False,
         "workpiece": True, "uses": [{"start_s": 10, "end_s": 20, "hand": "left", "action": "", "certainty": "high"}]},
    ],
    "running": [], "unsure": [{"description": "small dark thing", "start_s": 30, "end_s": 35}], "duration_s": 100.0,
}
NAMES = {"hammer_1": "hammer", "saw_1": "saw", "board_1": "board"}
VOCAB = {"objects": {"hammer": {"type": "portable", "prompts": ["hammer"]}, "saw": {"type": "portable", "prompts": ["saw"]}},
         "generic": {"tool": {"type": "portable", "prompts": ["tool"]}}, "ignore": []}


def contact(oid, start, end, side="left", hand="h_left", interaction="hold"):
    return {"event_id": "c", "start_ms": start, "end_ms": end, "hand_id": hand, "object_id": oid, "object_label": "tool",
            "state": "portable", "interaction": interaction, "confidence": 0.8, "reasons": [], "_side": side}


def test_time_candidates_respect_the_tolerance_workpieces_and_the_hand():
    c = [contact("obj_1", 10_500, 12_000)]
    found = dict(fuse.time_candidates(c, LEDGER, 1000))
    assert set(found) == {"hammer_1", "saw_1"}                       # the workpiece is never a candidate
    assert dict(fuse.time_candidates([contact("obj_1", 10_500, 12_000, side="right")], LEDGER, 1000)) == {"saw_1": found["saw_1"]}
    assert fuse.time_candidates([contact("obj_1", 30_000, 31_000)], LEDGER, 1000) == []
    near = fuse.time_candidates([contact("obj_1", 20_500, 21_500)], LEDGER, 1000)   # 0.5 s after the use: inside the tolerance
    assert {k for k, _ in near} == {"hammer_1", "saw_1"}
    assert fuse.time_candidates([contact("obj_1", 22_500, 23_000, side="unknown")], LEDGER, 1000) == []


def test_hand_ok_rules():
    assert fuse.hand_ok("both", "left") and fuse.hand_ok("unknown", "right") and fuse.hand_ok("left", "unknown")
    assert fuse.hand_ok("left", "left") and not fuse.hand_ok("left", "right")


@pytest.mark.parametrize("cands,crop,status,key,suggested", [
    ([("hammer_1", 5.0)], ("tool", "hammer_1"), "identified", "hammer_1", "hammer"),
    ([("hammer_1", 5.0), ("saw_1", 4.0)], ("tool", "saw_1"), "identified", "saw_1", "saw"),             # the crop picks among time candidates
    ([("hammer_1", 5.0)], ("tool", "saw_1"), "conflict", None, "saw"),
    ([("hammer_1", 5.0)], ("other", "chisel"), "conflict", None, "chisel"),
    ([], ("tool", "saw_1"), "single", None, "saw"),
    ([], ("other", "chisel"), "single", None, "chisel"),
    ([("hammer_1", 5.0)], ("unknown", ""), "single", None, "hammer"),
    ([("hammer_1", 5.0)], ("not_a_tool", ""), "single", None, "hammer"),
    ([("hammer_1", 5.0)], None, "single", None, "hammer"),                                              # no crop answer (budget used up)
    ([], ("unknown", ""), "none", None, None),
    ([], None, "none", None, None),
])
def test_decision_table(cands, crop, status, key, suggested):
    out = fuse.decide(cands, crop, NAMES)
    assert (out["status"], out["tool_key"], out["suggested"]) == (status, key, suggested)


def test_crop_answers_are_parsed_strictly():
    text = json.dumps({"answers": [{"crop": "C1", "answer": "Hammer_1"}, {"crop": "C2", "answer": "other: Chisel"},
                                   {"crop": "C3", "answer": "not_a_tool"}, {"crop": "C4", "answer": "banana"},
                                   {"crop": "C9", "answer": "hammer_1"}, {"crop": "x", "answer": "hammer_1"}]})
    out = fuse.parse_crop_answers(text, ["a", "b", "c", "d"], {"hammer_1", "saw_1"})
    assert out == {0: ("tool", "hammer_1"), 1: ("other", "chisel"), 2: ("not_a_tool", ""), 3: ("unknown", "")}
    assert fuse.parse_crop_answers("garbage", ["a"], set()) == {} and fuse.parse_crop_answers(None, ["a"], set()) == {}


class FakeGem:
    def __init__(self, replies):
        self.replies, self.calls, self.available, self.requests = list(replies), 0, True, []

    def generate(self, parts, max_tokens, schema=None, timeout=None):
        self.calls += 1
        self.requests.append(parts)
        return self.replies.pop(0)


def det(oid, label="tool", typ="portable"):
    return {"object_id": oid, "label": label, "type": typ, "is_hazard": False, "bbox": [0, 0, 10, 10], "conf": 0.8}


def scene(frames_of):
    """dets_by_frame (30 frames) with each object visible in the given frame ranges."""
    dets = [[] for _ in range(30)]
    for oid, ranges in frames_of.items():
        for a, b in ranges:
            for i in range(a, b):
                dets[i].append(det(oid))
    return dets


def run_fuse(cfg, contacts, dets, replies, crops=None):
    gem = FakeGem(replies)
    decisions = [{"t1": {"object_id": d["object_id"]}} for dets_i in dets for d in dets_i[:1]]
    crops = crops if crops is not None else {c["object_id"]: np.zeros((8, 8, 3), np.uint8) for c in contacts}
    out = fuse.fuse(LEDGER, contacts, dets, [{"t1": {"object_id": c["object_id"]}} for c in contacts], crops, gem, VOCAB,
                    list(range(0, 3000, 100)), 100.0, 1e6, {"left": "h_left"}, cfg)
    return out, gem


def reply(*answers):
    return json.dumps({"answers": [{"crop": f"C{i + 1}", "answer": a} for i, a in enumerate(answers)]})


def test_agreement_identifies_and_relabels_and_marks_the_contact(cfg):
    contacts = [contact("obj_1", 10_000, 15_000)]
    dets = scene({"obj_1": [(0, 30)]})
    out, gem = run_fuse(cfg, contacts, dets, [reply("hammer_1")])
    assert out["force_status"] == {"obj_1": "identified"}
    assert out["tool_info"]["obj_1"] == {"tool_key": "hammer_1", "id_confidence": "high", "evidence": ["ledger_time", "crop_match"]}
    assert dets[0][0]["label"] == "hammer" and contacts[0]["object_label"] == "hammer" and "label_from_vlm" in contacts[0]["reasons"]
    assert [i["reason"] for i in out["review_items"] if i["object_id"] == "obj_1"] == []
    text = json.dumps(gem.requests[0])
    assert "red handle" in text and "candidates: hammer_1, saw_1" in text


def test_single_evidence_and_conflict_stay_unidentified_with_a_review_reason(cfg):
    contacts = [contact("obj_1", 10_000, 15_000), contact("obj_2", 60_000, 65_000, side="right", hand="h_right")]
    dets = scene({"obj_1": [(0, 30)], "obj_2": [(0, 30)]})
    out, _ = run_fuse(cfg, contacts, dets, [reply("saw_1", "unknown")])      # obj_1: time says hammer/saw, crop saw -> agrees on saw
    assert out["force_status"]["obj_1"] == "identified"
    contacts = [contact("obj_1", 10_000, 15_000, side="left")]
    out, _ = run_fuse(cfg, contacts, scene({"obj_1": [(0, 30)]}), [reply("other: chisel")])
    item = next(i for i in out["review_items"] if i["object_id"] == "obj_1")
    assert (item["reason"], item["suggested_label"]) == ("tool_evidence_conflict", "chisel") and out["force_status"]["obj_1"] == "unidentified"
    out, _ = run_fuse(cfg, [contact("obj_9", 80_000, 85_000)], scene({"obj_9": [(0, 30)]}), [reply("saw_1")])
    assert next(i for i in out["review_items"] if i["object_id"] == "obj_9")["reason"] == "tool_single_evidence"


def test_fixed_objects_are_never_relabelled(cfg):
    dets = scene({"obj_1": [(0, 30)]})
    for d in dets[0]:
        d["type"] = "fixed"
    out, _ = run_fuse(cfg, [contact("obj_1", 10_000, 15_000)], dets, [reply("hammer_1")])
    assert out["force_status"] == {} and out["suggestions"] == {"obj_1": "hammer"} and dets[0][0]["label"] == "tool"


def test_same_tool_never_visible_together_is_merged_but_visible_together_is_not(cfg):
    contacts = [contact("obj_1", 10_000, 12_000), contact("obj_2", 15_000, 18_000)]
    dets = scene({"obj_1": [(0, 10)], "obj_2": [(12, 20)]})
    out, _ = run_fuse(cfg, contacts, dets, [reply("hammer_1", "hammer_1")])
    assert out["merged"] == {"obj_2": "obj_1"} and {c["object_id"] for c in contacts} == {"obj_1"}
    assert {d["object_id"] for ds in dets for d in ds} == {"obj_1"}
    contacts = [contact("obj_1", 10_000, 12_000), contact("obj_2", 15_000, 18_000)]
    dets = scene({"obj_1": [(0, 15)], "obj_2": [(10, 20)]})
    out, _ = run_fuse(cfg, contacts, dets, [reply("hammer_1", "hammer_1")])
    assert out["merged"] == {} and {c["object_id"] for c in contacts} == {"obj_1", "obj_2"}
    assert [i["reason"] for i in out["review_items"] if i["object_id"] == "obj_2"] == ["tool_evidence_conflict"]


def test_ledger_uses_without_a_local_contact_and_unsure_entries_become_review_items(cfg):
    out, _ = run_fuse(cfg, [contact("obj_1", 10_000, 15_000)], scene({"obj_1": [(0, 30)]}), [reply("hammer_1")])
    ai_only = [i for i in out["review_items"] if i["reason"] == "tool_use_ai_only"]
    assert [(i["suggested_label"], i["start_ms"], i["end_ms"], i["hand_id"], i["object_id"]) for i in ai_only] == \
           [("saw", 60_000, 70_000, None, None)]            # the saw use at 10-20 s is explained by the contact; the board is a workpiece
    unsure = [i for i in out["review_items"] if i["reason"] == "vlm_unsure_object"]
    assert unsure == [{"start_ms": 30_000, "end_ms": 35_000, "hand_id": None, "object_id": None, "reason": "vlm_unsure_object",
                       "suggested_label": "small dark thing"}]


def test_the_crop_check_uses_batches_and_respects_the_call_limit(cfg):
    contacts = [contact(f"obj_{i}", 10_000 + i * 100, 12_000 + i * 100) for i in range(1, 11)]
    dets = scene({f"obj_{i}": [(0, 30)] for i in range(1, 11)})
    out, gem = run_fuse(cfg, contacts, dets, [reply(*["hammer_1"] * 8), reply(*["hammer_1"] * 2)])
    assert gem.calls == 2 and len(gem.requests[0]) == 1 + 8 * 2     # prompt + 8 x (label, image)
    assert sum(1 for v in out["tool_info"].values() if v["tool_key"] or v["evidence"]) >= 1


# ---------------------------------------------------------------- masks, tracks, bystanders, safety

def test_polygons_are_simplified_to_at_most_40_points():
    circle = np.array([[100 + 50 * np.cos(a), 100 + 50 * np.sin(a)] for a in np.linspace(0, 6.28, 400, endpoint=False)])
    poly = masks.simplify(circle)
    assert 8 <= len(poly) <= 40 and all(len(p) == 2 for p in poly)
    square = masks.simplify(np.array([[0, 0], [10, 0], [10, 10], [0, 10]]))
    assert len(square) == 4


def test_mask_targets_are_hold_contacts_of_portable_objects_longest_first():
    dets = [[] for _ in range(100)]
    for i in range(10, 90):
        dets[i] = [{"object_id": "obj_1", "type": "portable", "bbox": [1, 2, 3, 4]}, {"object_id": "obj_2", "type": "fixed", "bbox": [0, 0, 1, 1]}]
    cs = [dict(contact("obj_1", 1000, 3000), _first=10, _last=40), dict(contact("obj_1", 5000, 9000), _first=50, _last=85),
          dict(contact("obj_2", 1000, 9000), _first=10, _last=80), dict(contact("obj_1", 100, 200, interaction="touch"), _first=60, _last=62)]
    targets = masks.hold_targets(cs, dets, 100, 15)
    assert [(t["first"], t["last"], t["object_id"]) for t in targets] == [(50, 99, "obj_1"), (10, 55, "obj_1")]


def H(hand_id, side, wrist, scale=100.0, box=None):
    kp = make_kp()
    kp = kp - kp[0] + np.array(wrist, float)
    kp[9] = kp[0] + (0, -scale)
    return Hand(hand_id, kp, np.array(box if box is not None else [wrist[0] - 60, wrist[1] - 150, wrist[0] + 60, wrist[1] + 10], float), "detected", side)


def test_tracks_survive_hand_id_changes_and_short_gaps_and_split_on_long_ones(cfg):
    frames = {i: [H(f"p{i}", "left", (300 + 3 * i, 800))] for i in range(0, 5)}      # PS1 gives a new id every frame
    frames[5] = []
    frames[9] = [H("z", "left", (320, 800))]                                          # 4 frames later: still the same track
    frames[40] = [H("q", "left", (330, 800))]                                         # 31 frames later: a new track
    assign_tracks(frames, cfg)
    assert {h.track for i in range(5) for h in frames[i]} == {"t1"} and frames[9][0].track == "t1" and frames[40][0].track == "t2"


def test_two_hands_keep_two_tracks_and_a_side_flip_on_the_same_hand_does_not_split_it(cfg):
    frames = {i: [H("a", "left", (300, 800)), H("b", "right", (900, 800))] for i in range(5)}
    frames[5] = [H("a", "right", (301, 800)), H("b", "right", (901, 800))]          # PS1 flips the left hand's side label
    assign_tracks(frames, cfg)
    assert frames[0][0].track != frames[0][1].track
    assert frames[5][0].track == frames[0][0].track and frames[5][1].track == frames[0][1].track


def test_bystander_filter_drops_small_far_and_top_of_frame_hands_and_keeps_at_most_two(cfg):
    wearer_l, wearer_r = H("l", "left", (300, 900)), H("r", "right", (900, 950))
    tiny = H("tiny", "left", (1500, 900), scale=40.0)                  # less than half the size of the largest
    high = H("high", "right", (700, 200))                              # wrist in the top 35% of a 1080 px frame
    third = H("third", "left", (1200, 700), scale=90.0)               # a third large hand: only two are kept
    out, dropped = filter_bystanders({0: [wearer_l, wearer_r, tiny, high, third], 1: [], 2: [tiny]}, 1080.0, cfg)
    assert sorted(h.hand_id for h in out[0]) == ["l", "r"] and out[1] == [] and dropped == 3
    assert [h.hand_id for h in out[2]] == ["tiny"]                     # alone it is the largest, so it stays


def test_safety_uses_ledger_running_intervals_and_names_the_source(cfg):
    kp = make_kp()
    hand = Hand("h_right", kp, np.array([30, -60, 150, 110.0]), "detected", "right", "t1")
    box = [60, -80, 70, -70]
    dets = [[{"object_id": "obj_1", "label": "saw", "type": "portable", "is_hazard": True, "bbox": box, "conf": 0.8, "diff": 0.0}] for _ in range(30)]
    hands = {i: [hand] for i in range(30)}
    times = [round(i * 1000 / 30) for i in range(30)]
    assert safety.find_events(hands, dets, times, 30.0, cfg) == []                       # still object: nothing without the ledger
    events = safety.find_events(hands, dets, times, 30.0, cfg, {"obj_1": [(0.0, 5000.0)]})
    assert len(events) == 1 and events[0]["reasons"] == ["2d_distance_only", "hazard_running_from_ledger"]
    for d in dets:
        d[0]["diff"] = 40.0
    both = safety.find_events(hands, dets, times, 30.0, cfg, {"obj_1": [(0.0, 5000.0)]})
    assert both[0]["reasons"] == ["2d_distance_only", "hazard_motion_estimated", "hazard_running_from_ledger"]
    assert safety.find_events(hands, dets, times, 30.0, cfg)[0]["reasons"] == ["2d_distance_only", "hazard_motion_estimated"]
