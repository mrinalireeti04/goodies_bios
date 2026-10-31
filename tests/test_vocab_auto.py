import json

import cv2
import numpy as np

from ps2 import vlm, vocab_auto
from ps2.hands import Hand, check_ps1
from test_hands import make_kp, ps1_frame

BASE = {"objects": {"hammer": {"type": "portable", "prompts": ["hammer"]},
                    "workbench": {"type": "fixed", "prompts": ["workbench", "table"]},
                    "table_saw": {"type": "fixed", "hazard": True, "prompts": ["table saw"]},
                    "floor": {"type": "fixed", "prompts": ["floor"]}},
        "generic": {"tool": {"type": "portable", "prompts": ["tool"]}}, "ignore": ["hand", "glove"]}


def obj(name, frames, synonyms=(), kind="portable", powered=False):
    return {"name": name, "synonyms": list(synonyms), "type": kind, "powered": powered, "frames": list(frames)}


def batch(*objects, unsure=()):
    return {"objects": list(objects), "unsure": list(unsure)}


def test_normalise_lowercases_and_singularises():
    assert vocab_auto.normalise("  Power DRILLS ") == "power drill"
    assert vocab_auto.normalise("Pliers") == "pliers" and vocab_auto.normalise("scissors") == "scissors"
    assert vocab_auto.normalise("boxes") == "box" and vocab_auto.normalise("batteries") == "battery"
    assert vocab_auto.normalise("glass") == "glass" and vocab_auto.normalise("!!") == ""


def test_parse_batch_drops_invalid_frames_types_and_empty_entries():
    text = json.dumps({"objects": [
        {"name": "Hammer", "synonyms": ["claw hammer", "Hammer"], "type": "portable", "powered_cutting_or_rotating": False, "frames": [3, 99]},
        {"name": "ghost", "synonyms": [], "type": "portable", "powered_cutting_or_rotating": False, "frames": [99]},
        {"name": "thing", "synonyms": [], "type": "weird", "powered_cutting_or_rotating": False, "frames": [3]},
        {"name": "saw", "synonyms": [], "type": "portable", "powered_cutting_or_rotating": "yes", "frames": [4]}],
        "unsure": [{"description": "  shiny   thing ", "frames": [4, 100]}, {"description": "x", "frames": [100]}]})
    out = vocab_auto.parse_batch(text, {3, 4})
    assert [o["name"] for o in out["objects"]] == ["hammer", "saw"]
    assert out["objects"][0]["frames"] == [3] and out["objects"][0]["synonyms"] == ["claw hammer"]
    assert out["objects"][1]["powered"] is False                       # only a real true counts
    assert out["unsure"] == [{"description": "shiny thing", "frames": [4]}]
    assert vocab_auto.parse_batch("not json", {1}) is None and vocab_auto.parse_batch(None, {1}) is None
    assert vocab_auto.parse_batch('{"objects": 5}', {1}) is None


def test_name_must_appear_in_two_of_three_batches():
    entries, _ = vocab_auto.merge_batches([batch(obj("drill", [1]), obj("clamp", [1])),
                                           batch(obj("drill", [2])), batch(obj("sander", [3]))], 2)
    assert [e["name"] for e in entries] == ["drill"] and entries[0]["batches"] == 2 and entries[0]["frames"] == [1, 2]


def test_synonyms_merge_only_when_both_sides_agree():
    mutual = [batch(obj("drill", [1], ["power drill"])), batch(obj("power drill", [2], ["drill"]))]
    assert len(vocab_auto.merge_batches(mutual, 2)[0]) == 1
    one_sided = [batch(obj("drill", [1])), batch(obj("drill press", [2], ["drill"], kind="fixed"))]
    assert vocab_auto.merge_batches(one_sided, 2)[0] == []            # "drill" and "drill press" stay apart


def test_hazard_needs_powered_in_half_the_mentions_and_two_frames():
    def hazard(*b):
        return vocab_auto.merge_batches(list(b), 2)[0][0]["hazard"]
    assert hazard(batch(obj("grinder", [1], powered=True)), batch(obj("grinder", [2], powered=True))) is True
    assert hazard(batch(obj("grinder", [1], powered=True)), batch(obj("grinder", [1], powered=True))) is False   # one frame
    assert hazard(batch(obj("saw", [1])), batch(obj("saw", [2]))) is False
    assert hazard(batch(obj("saw", [1], powered=True)), batch(obj("saw", [2]), obj("saw", [3]))) is False       # 1 of 3 mentions


def test_type_is_the_majority_and_unsure_items_are_merged_by_description():
    entries, unsure = vocab_auto.merge_batches(
        [batch(obj("bench", [1], kind="fixed"), unsure=[{"description": "Red thing", "frames": [1]}]),
         batch(obj("bench", [2], kind="fixed"), unsure=[{"description": "red   thing", "frames": [2]}]),
         batch(obj("bench", [3], kind="portable"))], 2)
    assert entries[0]["type"] == "fixed" and unsure == [{"description": "Red thing", "frames": [1, 2]}]


def test_final_vocabulary_is_gemini_objects_plus_fixed_surfaces_only():
    entry = {"name": "power drill", "synonyms": ["drill", "cordless drill"], "type": "portable", "hazard": True}
    vocab = vocab_auto.build_vocab([entry], BASE)
    assert set(vocab["objects"]) == {"power_drill", "workbench", "floor"}       # not hammer, not the fixed hazard table_saw
    assert vocab["objects"]["power_drill"] == {"type": "portable", "prompts": ["power drill", "drill", "cordless drill"], "hazard": True}
    assert vocab["generic"] == BASE["generic"] and vocab["ignore"] == BASE["ignore"]
    clash = vocab_auto.build_vocab([{"name": "workbench", "synonyms": ["bench"], "type": "fixed", "hazard": False}], BASE)
    assert clash["objects"]["workbench"]["prompts"] == ["workbench", "bench"]    # the Gemini entry wins


class FakeDetector:
    prompts = ["drill", "bench"]
    prompt_to_key = {"drill": "drill", "bench": "bench"}

    def __init__(self, found):
        self.found = found

    def detect(self, frame, hands):
        names = self.found.get(int(frame[0, 0, 0]), [])
        return np.zeros((len(names), 4)), np.ones(len(names)), np.array([self.prompts.index(n) for n in names], float)


def test_confirm_marks_vlm_only_when_the_detector_never_finds_it():
    entries = [{"name": "drill", "frames": [1, 2]}, {"name": "bench", "frames": [2]}]
    frames = {1: np.full((4, 4, 3), 1, np.uint8), 2: np.full((4, 4, 3), 2, np.uint8)}
    vocab_auto.confirm(entries, frames, FakeDetector({2: ["drill"]}))
    assert [e["status"] for e in entries] == ["confirmed", "vlm_only"]


def test_review_items_for_unverified_labels_and_unsure_objects():
    dets = [[], [{"object_id": "obj_3", "label": "tool"}, {"object_id": "obj_4", "label": "bench"}], [{"object_id": "obj_3", "label": "tool"}]]
    times = [0, 100, 200]
    info = {"entries": [{"name": "bench", "frames": [1, 2], "status": "vlm_only"},
                        {"name": "drill", "frames": [1], "status": "confirmed"}],
            "unsure": [{"description": "orange handle", "frames": [1, 2]}, {"description": "far away", "frames": [50]}]}
    items = vocab_auto.review_items(info, dets, times, 100.0, {"tool"})
    assert items == [
        {"start_ms": 100, "end_ms": 300, "hand_id": None, "object_id": "obj_4", "reason": "vocab_unverified", "suggested_label": "bench"},
        {"start_ms": 100, "end_ms": 300, "hand_id": None, "object_id": "obj_3", "reason": "vlm_unsure_object", "suggested_label": "orange handle"}]


def test_meta_vocab_lists_label_status_hazard_and_times():
    info = {"entries": [{"name": "power drill", "status": "confirmed", "hazard": False, "frames": [0, 2]}]}
    assert vocab_auto.meta_vocab(info, [0, 100, 200]) == [{"label": "power_drill", "status": "confirmed", "hazard": False, "seen_ms": [0, 200]}]


def test_sampling_spreads_frames_and_prefers_hands(tmp_path):
    path = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(100):
        w.write(np.full((48, 64, 3), i, np.uint8))
    w.release()
    hand = Hand("h", make_kp(), np.zeros(4), "detected")
    out = vocab_auto.sample_frames(path, 100, {10: [hand], 37: [hand]}, 4, 32)
    idx = [i for i, _ in out]
    assert len(idx) == 4 and idx == sorted(idx)
    assert 10 in idx and 37 in idx and max(idx) > 60                      # hands preferred; whole video covered
    assert max(out[0][1].shape[:2]) == 32                                  # shrunk to the limit
    assert vocab_auto.split_batches(list(range(24)))[1] == list(range(1, 24, 3))


def test_without_gemini_the_base_vocabulary_is_used(cfg, monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    gem = vlm.Gemini(cfg)
    info = vocab_auto.build(tmp_path / "x.mp4", {}, 100, 10.0, "clip", cfg, BASE, None, gem, tmp_path / "cache")
    assert info["source"] == "base" and info["vocab"] is BASE and info["entries"] == [] and gem.calls == 0


# ---------------------------------------------------------------- PS1 assumption checks

def checks(frames, n=3, fps=30.0, w=400, h=400):
    return check_ps1(frames, w, h, n, fps)


def good(i, **kw):
    """A record whose keypoints lie inside a 400x400 frame."""
    kw.setdefault("kp", make_kp() + np.array([150.0, 150.0]))
    return ps1_frame(i, w=400, h=400, **kw)


def test_clean_ps1_data_passes():
    out = checks([good(i) for i in range(3)])
    assert out["ok"] and out["warnings"] == [] and out["distinct_hand_ids"] == 1


def test_each_violation_is_counted_and_never_raises():
    frames = [good(0), dict(good(1), t_ms=500), good(7), dict(good(2), frame_width=500)]
    out = checks(frames, n=4)
    assert out["missing_frames"] == 1                # frame 3 has no record
    assert out["t_ms_mismatch"] == 1 and out["frame_idx_beyond_video"] == 1 and out["frame_size_mismatch"] == 1
    assert not out["ok"] and len(out["warnings"]) == 4


def test_out_of_bounds_many_ids_and_suspect_order():
    far = good(0, kp=make_kp() + 900)                                  # keypoints far outside the 400x400 frame
    many = [good(i, hand_id=f"h{i}") for i in range(1, 7)]
    out = checks([far] + many, n=7)
    assert out["coords_out_of_bounds"] == 1 and out["distinct_hand_ids"] == 7 and out["too_many_hand_ids"] == 7
    scrambled = make_kp() + np.array([150.0, 150.0])
    scrambled[[9, 12]] = scrambled[[12, 9]]                            # middle fingertip nearer the wrist than its knuckle
    out = checks([good(0, kp=scrambled)], n=1)
    assert out["keypoint_order_suspect"] == 1 and any("MediaPipe order" in w for w in out["warnings"])
    mixed = checks([good(0, kp=scrambled)] + [good(i) for i in range(1, 6)], n=6)   # 1 of 6 hands: counted, no warning
    assert mixed["keypoint_order_suspect"] == 1 and mixed["ok"]


def test_garbage_records_do_not_crash_the_checks():
    out = checks([{"frame_idx": "x"}, {"nothing": 1}, {"frame_idx": 0, "hands": [{"kp2d_px": "bad"}]}], n=1)
    assert out["missing_frames"] == 0
