import numpy as np

from ps2.hands import Hand
from ps2.objects import drop_hand_like
from test_hands import make_kp


def test_glove_box_over_the_hand_is_removed_and_other_objects_stay(cfg):
    hand = Hand("h_right", make_kp(), np.array([30, -60, 150, 110.0]), "detected")
    glove = {"bbox": [28, -62, 152, 112.0], "label": "tool"}  # a "gloved hand" detected as some tool
    far_tool = {"bbox": [400, 400, 500, 500.0], "label": "hammer"}
    assert drop_hand_like([glove, far_tool], [hand], cfg["hand_overlap_iou"]) == [far_tool]


def test_no_hands_keeps_everything(cfg):
    det = {"bbox": [0, 0, 10, 10.0]}
    assert drop_hand_like([det], [], cfg["hand_overlap_iou"]) == [det]
