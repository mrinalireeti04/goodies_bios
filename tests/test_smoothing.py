from ps2 import contact

FPS = 30.0
MS = 1000 / FPS
HAND = "h_right"
NONE = {"object_id": None, "label": None, "kind": "none", "state": "none", "interaction": "unknown",
        "confidence": None, "reasons": []}


def contact_frame(oid="a", state="portable", reasons=()):
    return {"object_id": oid, "label": oid, "kind": "contact", "state": state, "interaction": "hold",
            "confidence": 0.8, "reasons": list(reasons)}


def build(pattern, missing=()):
    """pattern: a decision dict per frame (None = no contact). missing: frame indices with no hand."""
    decisions = [{HAND: dict(p or NONE)} for p in pattern]
    present = [set() if i in missing else {HAND} for i in range(len(pattern))]
    times = [round(i * MS) for i in range(len(pattern))]
    return decisions, present, times


def run(cfg, pattern, missing=()):
    decisions, present, times = build(pattern, missing)
    return contact.find_contacts(decisions, present, times, FPS, cfg)


def test_one_frame_blip_and_short_run_are_dropped(cfg):
    assert run(cfg, [None] * 5 + [contact_frame()] + [None] * 10) == []
    assert run(cfg, [None] * 5 + [contact_frame()] * 5 + [None] * 10) == []  # 5 frames = 167 ms < 200 ms


def test_boundaries_in_ms(cfg):
    out = run(cfg, [None] * 5 + [contact_frame()] * 10 + [None] * 10)
    assert len(out) == 1
    assert out[0]["start_ms"] == round(5 * MS) and out[0]["end_ms"] == round(round(14 * MS) + MS)


def test_short_gap_is_merged_long_gap_splits(cfg):
    on = [contact_frame()] * 8
    assert len(run(cfg, on + [None] * 6 + on + [None] * 5)) == 1   # 200 ms gap < merge_gap_ms
    assert len(run(cfg, on + [None] * 12 + on + [None] * 5)) == 2  # 400 ms gap


def test_other_object_starts_a_new_interval(cfg):
    out = run(cfg, [contact_frame("a")] * 8 + [contact_frame("b")] * 8 + [None] * 5)
    assert [c["object_id"] for c in out] == ["a", "b"]


def test_hand_vanishing_is_bridged_up_to_max_gap(cfg):
    on = [contact_frame()] * 8
    short = run(cfg, on + [None] * 9 + on + [None] * 5, missing=set(range(8, 17)))  # 300 ms
    assert len(short) == 1 and "hand_not_detected" in short[0]["reasons"]
    long_ = run(cfg, on + [None] * 20 + on + [None] * 5, missing=set(range(8, 28)))  # 667 ms
    assert len(long_) == 2


def test_reasons_on_many_frames_trigger_review(cfg):
    weak = dict(contact_frame(), kind="weak", state="unknown", reasons=["low_object_confidence"])
    out = run(cfg, [contact_frame()] * 6 + [weak] * 6 + [None] * 5)
    assert out[0]["review_needed"] and out[0]["reasons"] == ["low_object_confidence"]
    clean = run(cfg, [contact_frame()] * 12 + [None] * 5)
    assert not clean[0]["review_needed"] and clean[0]["reasons"] == []


def test_interaction_and_state_use_majority(cfg):
    touch = dict(contact_frame(), interaction="touch")
    out = run(cfg, [touch] * 3 + [contact_frame()] * 7 + [None] * 5)
    assert out[0]["interaction"] == "hold" and out[0]["state"] == "portable"


def test_apply_to_frames_unknown_keeps_reasons_and_none_elsewhere(cfg):
    on = [contact_frame()] * 8
    grasp = dict(NONE, state="unknown", reasons=["grasp_pose_no_object"])
    pattern = [None] * 2 + on + [None] * 2 + on + [grasp] + [None] * 5
    decisions, present, times = build(pattern, missing={10, 11})  # hand vanishes between the two runs
    contacts = contact.find_contacts(decisions, present, times, FPS, cfg)
    frames = [{"hand_states": []} for _ in pattern]
    contact.apply_to_frames(frames, present, decisions, contacts)

    def state(i):
        return frames[i]["hand_states"][0]

    assert state(0)["state"] == "none" and state(0)["object_id"] is None
    assert state(5)["state"] == "portable" and state(5)["object_id"] == "a"
    assert state(10)["state"] == "unknown" and state(10)["reasons"] == ["hand_not_detected"]
    grasp_frame = 2 + 8 + 2 + 8
    assert state(grasp_frame)["state"] == "unknown" and state(grasp_frame)["reasons"] == ["grasp_pose_no_object"]


def test_low_overall_confidence_adds_its_own_reason_code(cfg):
    weak_conf = dict(contact_frame(), confidence=0.2)
    out = run(cfg, [weak_conf] * 12 + [None] * 5)
    assert out[0]["review_needed"] and out[0]["reasons"] == ["low_contact_confidence"]
    assert not run(cfg, [contact_frame()] * 12 + [None] * 5)[0]["review_needed"]
