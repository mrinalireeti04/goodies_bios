import numpy as np

from ps2 import contact
from ps2.hands import Hand
from test_hands import make_kp

# make_kp(): scale 100, fingertips at (40,60) (80,-60) (100,-60) (120,-60) (140,-60); tip_near_norm 0.35 = 35 px
AREA = 1000 * 1000


def hand(interpolated=False, kp=None):
    kp = make_kp() if kp is None else kp
    return Hand("h_right", kp, np.array([30, -60, 150, 110.0]), "interpolated" if interpolated else "detected")


def obj(object_id, box, typ="portable", conf=0.8, label="hammer"):
    return {"object_id": object_id, "label": label, "type": typ, "bbox": box, "conf": conf}


TWO_TIPS = [60, -70, 70, -50]        # tips at x=80 and x=100 are within 35 px
ONE_TIP = [38, 58, 42, 62]           # only the thumb tip
ALL_TIPS = [30, -70, 150, 70]


def vec(x, y=0.0):
    return np.array([x, y])


def fist_kp():
    fist = make_kp()
    for mcp, pip, dip in ((5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19)):
        fist[pip] = fist[mcp] + (0, -20)
        fist[dip] = fist[pip] + (20, 0)
    return fist


def decide_with(cfg, o, hand_move=0.0, hand_vec=None, obj_vec=None, area=AREA):
    return contact.decide(o, hand_move, np.zeros(2) if hand_vec is None else hand_vec, obj_vec, area, cfg)


def test_two_tips_candidate_one_tip_not(cfg):
    assert contact.observe(hand(), [obj("a", TWO_TIPS)], cfg)["obj"]["object_id"] == "a"
    assert contact.observe(hand(), [obj("a", ONE_TIP)], cfg)["obj"] is None


def test_most_tips_wins_then_portable_then_smaller(cfg):
    two, five = obj("two", TWO_TIPS), obj("five", ALL_TIPS)
    assert contact.observe(hand(), [two, five], cfg)["obj"]["object_id"] == "five"
    fixed, portable = obj("f", ALL_TIPS, "fixed"), obj("p", [20, -75, 160, 75])
    o = contact.observe(hand(), [fixed, portable], cfg)
    assert o["obj"]["object_id"] == "p" and not o["ambiguous"]


def test_equal_candidates_are_ambiguous(cfg):
    o = contact.observe(hand(), [obj("a", ALL_TIPS), obj("b", [31, -71, 149, 69])], cfg)
    assert o["ambiguous"]
    d = decide_with(cfg, o)
    assert d["state"] == "unknown" and "ambiguous_candidates" in d["reasons"] and d["object_id"] is None


def test_hold_when_object_moves_with_hand(cfg):
    assert contact.interaction_from_motion(0.5, vec(50), vec(48), 0.1, cfg)[:2] == ("hold", True)


def test_touch_when_hand_moves_object_does_not(cfg):
    assert contact.interaction_from_motion(0.5, vec(50), vec(2), 0.9, cfg)[:2] == ("touch", False)


def test_in_between_motion_is_unknown(cfg):
    assert contact.interaction_from_motion(0.5, vec(50), vec(0, 50), 0.5, cfg)[0] == "unknown"
    assert contact.interaction_from_motion(0.5, vec(50), None, 0.5, cfg)[0] == "unknown"


def test_still_hand_decides_from_pose(cfg):
    assert contact.interaction_from_motion(0.0, vec(0), None, 0.9, cfg)[0] == "hold"
    assert contact.interaction_from_motion(0.0, vec(0), None, 0.05, cfg)[0] == "touch"
    assert contact.interaction_from_motion(0.0, vec(0), None, 0.4, cfg)[0] == "unknown"


def test_camera_only_motion_is_not_a_hold(cfg):
    # after camera compensation both hand and object show ~0 residual motion; a flat hand is a touch
    o = contact.observe(hand(), [obj("a", ALL_TIPS)], cfg)
    d = decide_with(cfg, o, 0.0, np.zeros(2), np.zeros(2))
    assert d["interaction"] == "touch" and d["state"] == "portable"


def test_grasp_pose_without_object_is_unknown_with_reason(cfg):
    d = decide_with(cfg, contact.observe(hand(kp=fist_kp()), [], cfg))
    assert d["state"] == "unknown" and d["reasons"] == ["grasp_pose_no_object"]
    assert decide_with(cfg, contact.observe(hand(), [], cfg))["state"] == "none"


def test_low_detector_confidence_is_unknown_but_keeps_the_object(cfg):
    d = decide_with(cfg, contact.observe(hand(), [obj("a", ALL_TIPS, conf=0.16)], cfg))
    assert d["kind"] == "weak" and d["state"] == "unknown" and "low_object_confidence" in d["reasons"]
    assert d["object_id"] == "a"


def test_big_fixed_box_with_moving_hand_is_unverified(cfg):
    o = contact.observe(hand(), [obj("floor", ALL_TIPS, "fixed", label="floor")], cfg)
    moving = decide_with(cfg, o, 0.5, vec(50), vec(1), area=1000.0)  # box covers >15% of a tiny frame
    assert "fixed_contact_unverified" in moving["reasons"] and moving["state"] == "unknown"
    assert decide_with(cfg, o, 0.0, area=1000.0)["state"] == "fixed"  # hand resting on it: verified


def test_fixed_object_moving_with_hand_becomes_portable(cfg):
    o = contact.observe(hand(), [obj("v", ALL_TIPS, "fixed", label="vise")], cfg)
    d = decide_with(cfg, o, 0.5, vec(50), vec(49))
    assert d["state"] == "portable" and "type_overridden_by_motion" in d["reasons"]


def test_interpolated_hand_lowers_confidence_and_adds_reason(cfg):
    base = decide_with(cfg, contact.observe(hand(), [obj("a", ALL_TIPS)], cfg))
    interp = decide_with(cfg, contact.observe(hand(interpolated=True), [obj("a", ALL_TIPS)], cfg))
    assert interp["confidence"] < base["confidence"]
    assert "hand_interpolated" in interp["reasons"]


def test_window_motion_sums_only_overlapping_frames():
    aux = [{"wrist_res": {"h": vec(1.0)}, "obj_res": {"o": vec(1.0)}} for _ in range(5)]
    aux[2]["obj_res"] = {}
    hand_all, pair = contact.window_motion(aux, 2, "h", "o", 2)
    assert hand_all[0] == 5 and pair[0][0] == 4 and pair[1][0] == 4
