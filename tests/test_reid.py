from collections import Counter

import numpy as np

from ps2.objects import HIST_BINS, hand_overlap, identity_review_items, merge_pieces
from ps2.hands import Hand

N_BINS = HIST_BINS[0] * HIST_BINS[1]
GENERIC = {"tool", "object"}
FRAME_MS = 100.0


def hist(bin_):
    h = np.zeros(N_BINS)
    h[bin_] = 1.0
    return h


def piece(frames, label="hammer", bin_=3, with_hist=True):
    return {"frames": list(frames), "labels": Counter({label: 0.5}), "hist": hist(bin_) if with_hist else None}


def merge(pieces, cfg):
    return merge_pieces(pieces, GENERIC, cfg, FRAME_MS)


def test_same_tool_after_re_entry_gets_one_id(cfg):
    mapping, uncertain = merge({1: piece(range(0, 50)), 2: piece(range(300, 350))}, cfg)
    assert mapping[1] == mapping[2] and uncertain == []


def test_pieces_visible_together_never_merge(cfg):
    mapping, _ = merge({1: piece(range(0, 50)), 2: piece(range(40, 90))}, cfg)  # overlap at frames 40-49
    assert mapping[1] != mapping[2]


def test_different_colour_or_label_does_not_merge(cfg):
    mapping, _ = merge({1: piece(range(0, 50), bin_=3), 2: piece(range(300, 350), bin_=90)}, cfg)
    assert mapping[1] != mapping[2]
    mapping, _ = merge({1: piece(range(0, 50)), 2: piece(range(300, 350), label="pliers")}, cfg)
    assert mapping[1] != mapping[2]


def test_generic_label_can_merge_with_specific(cfg):
    mapping, _ = merge({1: piece(range(0, 50)), 2: piece(range(300, 350), label="tool")}, cfg)
    assert mapping[1] == mapping[2]


def test_near_tie_merges_with_the_best_and_asks_for_review(cfg):
    a = piece(range(0, 50))
    near = piece(range(300, 350))
    near["hist"] = 0.97 * hist(3) + 0.03 * hist(4)          # slightly less similar to a
    nearer = piece(range(500, 550))
    nearer["hist"] = 0.99 * hist(3) + 0.01 * hist(4)
    mapping, uncertain = merge({1: a, 2: near, 3: nearer}, cfg)
    assert mapping[1] == mapping[2] == mapping[3]
    assert uncertain, "two similar candidates must create an identity_uncertain item"


def test_no_fingerprint_merges_on_label_and_time_but_is_uncertain(cfg):
    pieces = {1: piece(range(0, 50)), 2: piece(range(60, 100), with_hist=False)}  # 1 s apart
    mapping, uncertain = merge(pieces, cfg)
    assert mapping[1] == mapping[2] and len(uncertain) == 1
    far = {1: piece(range(0, 50)), 2: piece(range(2000, 2050), with_hist=False)}  # 195 s apart
    mapping, uncertain = merge(far, cfg)
    assert mapping[1] != mapping[2] and uncertain == []


def test_final_ids_follow_first_appearance(cfg):
    mapping, _ = merge({7: piece(range(100, 120), bin_=1), 3: piece(range(0, 20), label="pliers", bin_=50)}, cfg)
    assert mapping[3] == 0 and mapping[7] == 1


def test_review_item_spans_the_two_appearances(cfg):
    mapping = {1: 0, 2: 0}
    times = [i * 100 for i in range(400)]
    items = identity_review_items([{"tracker_id": 1, "frame_a": 49, "frame_b": 300}], mapping, times, FRAME_MS)
    assert items == [{"item_id": "r_1", "start_ms": 4900, "end_ms": 30100, "hand_id": None, "object_id": "obj_1",
                      "reason": "identity_uncertain", "suggested_label": None}]


def test_hand_overlap_fraction():
    hand = Hand("h", np.zeros((21, 2)), np.array([0, 0, 50, 100.0]), "detected")
    assert hand_overlap([0, 0, 100, 100], [hand]) == 0.5
    assert hand_overlap([200, 200, 300, 300], [hand]) == 0


def test_ties_between_brief_noise_pieces_are_not_flagged(cfg):
    a = piece(range(0, 4))                                   # 0.4 s: shorter than min_tool_ms
    near, nearer = piece(range(300, 304)), piece(range(500, 504))
    near["hist"] = 0.97 * hist(3) + 0.03 * hist(4)
    nearer["hist"] = 0.99 * hist(3) + 0.01 * hist(4)
    mapping, uncertain = merge({1: a, 2: near, 3: nearer}, cfg)
    assert mapping[1] == mapping[2] == mapping[3] and uncertain == []
