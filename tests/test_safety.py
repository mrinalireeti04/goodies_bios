import numpy as np

from ps2 import safety, schema
from ps2.hands import Hand
from test_hands import make_kp

FPS = 30.0
MS = 1000 / FPS
# make_kp(): scale 100, fingertips at (40,60) (80,-60) (100,-60) (120,-60) (140,-60); safety_near_norm 0.5 = 50 px
NEAR_BOX = [60, -80, 70, -70]      # the closest fingertip (80,-60) is about 22 px away
FAR_BOX = [500, 500, 600, 600]


def hand():
    return Hand("h_right", make_kp(), np.array([30, -60, 150, 110.0]), "detected")


def hazard(box, diff, oid="obj_1", hazard_flag=True, conf=0.8):
    return {"object_id": oid, "label": "angle_grinder", "type": "portable", "is_hazard": hazard_flag,
            "bbox": box, "conf": conf, "diff": diff}


def scene(n, box, diff, hand_frames=None, **kw):
    dets = [[hazard(box, diff, **kw)] for _ in range(n)]
    hands = {i: [hand()] for i in (range(n) if hand_frames is None else hand_frames)}
    return hands, dets, [round(i * MS) for i in range(n)]


def events(cfg, hands, dets, times):
    return safety.find_events(hands, dets, times, FPS, cfg)


def test_fingertip_near_moving_hazard_creates_an_event(cfg):
    out = events(cfg, *scene(30, NEAR_BOX, diff=40.0))
    assert len(out) == 1
    e = out[0]
    assert e["hand_id"] == "h_right" and e["hazard_object_id"] == "obj_1" and e["hazard_moving"] is True
    assert e["start_ms"] == 0 and e["end_ms"] == round(round(29 * MS) + MS)
    assert e["min_distance_norm"] < cfg["safety_near_norm"]
    assert "2d_distance_only" in e["reasons"] and "hazard_motion_estimated" in e["reasons"]


def test_near_a_still_hazard_makes_no_event(cfg):
    assert events(cfg, *scene(30, NEAR_BOX, diff=2.0)) == []


def test_far_from_a_moving_hazard_makes_no_event(cfg):
    assert events(cfg, *scene(30, FAR_BOX, diff=40.0)) == []


def test_non_hazard_objects_are_ignored(cfg):
    assert events(cfg, *scene(30, NEAR_BOX, diff=40.0, hazard_flag=False)) == []


def test_brief_closeness_is_not_a_warning(cfg):
    hands, dets, times = scene(30, NEAR_BOX, diff=40.0, hand_frames=range(5, 10))  # 5 frames = 167 ms < 300 ms
    assert events(cfg, hands, dets, times) == []


def test_short_dropouts_are_merged_into_one_event(cfg):
    hands, dets, times = scene(60, NEAR_BOX, diff=40.0, hand_frames=list(range(0, 20)) + list(range(25, 45)))
    assert len(events(cfg, hands, dets, times)) == 1


def test_motion_must_hold_for_most_of_the_window(cfg):
    hands, dets, times = scene(30, NEAR_BOX, diff=40.0)
    for i, d in enumerate(dets):  # changes in only 1 of every 4 frames: not "most of the window"
        d[0]["diff"] = 40.0 if i % 4 == 0 else 1.0
    assert events(cfg, hands, dets, times) == []


def test_events_pass_the_schema(cfg):
    hands, dets, times = scene(30, NEAR_BOX, diff=40.0)
    out = events(cfg, hands, dets, times)
    frames = [{"frame_idx": i, "t_ms": times[i], "hand_states": [],
               "objects": [{"object_id": "obj_1", "label": "angle_grinder", "type": "portable",
                            "bbox_px": NEAR_BOX, "confidence": 0.8}]} for i in range(30)]
    schema.validate(schema.build_output("c", frames, [], safety_events=out))
