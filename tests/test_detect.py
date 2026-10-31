from collections import Counter

import numpy as np

from ps2.objects import (apply_identities, hand_crop_box, merge_detections, set_object_label)
from ps2.summary import build_tools

VOCAB = {
    "objects": {
        "hammer": {"type": "portable", "prompts": ["hammer"]},
        "angle_grinder": {"type": "portable", "hazard": True, "prompts": ["angle grinder"]},
    },
    "generic": {"tool": {"type": "portable", "prompts": ["tool"]}},
    "ignore": [],
}
FRAME_AREA = 1000 * 1000


def test_hand_crop_is_a_scaled_square_clipped_to_the_frame():
    # hand 100 x 200 px centred at (500, 500): side = 200 * 2.5 = 500
    assert hand_crop_box([450, 400, 550, 600], 2.5, 1920, 1080) == (250, 250, 750, 750)
    assert hand_crop_box([0, 0, 100, 100], 2.5, 1920, 1080) == (0, 0, 175, 175)      # clipped at the corner
    assert hand_crop_box([1900, 1060, 1920, 1080], 1.0, 1920, 1080) is None            # too small to use


def test_merge_keeps_the_more_confident_of_two_same_class_boxes_only():
    xyxy = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [0, 0, 100, 100], [500, 500, 600, 600]], float)
    conf = np.array([0.4, 0.6, 0.5, 0.3])
    cls = np.array([1, 1, 2, 1])
    keep = merge_detections(xyxy, conf, cls, 0.5)
    assert list(keep) == [1, 2, 3]            # box 0 loses to box 1; the other class and the far box stay
    assert len(merge_detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0), 0.5)) == 0


def detection(box, label="angle_grinder", tid=1):
    return {"track_id": tid, "label": label, "type": "portable", "is_hazard": True, "bbox": box, "conf": 0.8}


def identities(boxes, cfg, label="angle_grinder"):
    dets = [[detection(b, label)] for b in boxes]
    pieces = {1: {"labels": Counter({label: 1.0})}}
    apply_identities(dets, [{"obj_res": {}} for _ in dets], pieces, {1: 0}, VOCAB, FRAME_AREA, cfg)
    return [d[0] for d in dets]


def test_hazard_needs_the_vocab_flag_and_a_small_box(cfg):
    small, huge = [0, 0, 200, 200], [0, 0, 800, 800]          # 4% vs 64% of the frame
    out = identities([small, huge], cfg)
    assert [d["is_hazard"] for d in out] == [True, False]
    assert [d["is_hazard"] for d in identities([small], cfg, label="hammer")] == [False]


def test_relabelling_an_object_updates_label_type_and_hazard(cfg):
    dets = identities([[0, 0, 200, 200]] * 3, cfg, label="hammer")
    assert dets[0]["is_hazard"] is False
    set_object_label([[d] for d in dets], "obj_1", "angle_grinder", VOCAB, FRAME_AREA, cfg)
    assert all(d["label"] == "angle_grinder" and d["is_hazard"] for d in dets)


def test_tool_is_a_hazard_only_if_flagged_in_most_of_its_frames(cfg):
    def frame(i, flag):
        return {"frame_idx": i, "t_ms": i * 100, "hand_states": [],
                "objects": [{"object_id": "obj_1", "label": "angle_grinder", "type": "portable",
                             "bbox_px": [0, 0, 10, 10], "confidence": 0.8, "is_hazard": flag}]}
    frames = [frame(i, i < 12) for i in range(20)] + [frame(i, False) for i in range(20, 60)]
    assert build_tools(frames, [], {"tool"}, 100.0, cfg)[0]["is_hazard"] is False
    frames = [frame(i, i < 40) for i in range(60)]
    assert build_tools(frames, [], {"tool"}, 100.0, cfg)[0]["is_hazard"] is True
