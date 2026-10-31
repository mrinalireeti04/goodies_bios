import json

import numpy as np
import pytest

from ps2.hands import finger_curl, hand_scale, load_hands, tip_distances


def make_kp(wrist=(100, 100), knuckle=(100, 0)):
    """Synthetic hand: wrist-to-knuckle 100 px, fingers straight up from the knuckles."""
    kp = np.zeros((21, 2))
    kp[0], kp[9] = wrist, knuckle
    for finger, x in zip((range(5, 9), range(9, 13), range(13, 17), range(17, 21)), (80, 100, 120, 140)):
        for step, j in enumerate(finger):
            kp[j] = (x, 0 - 20 * step)
    kp[1:5] = [(70, 90), (60, 80), (50, 70), (40, 60)]
    return kp


def ps1_frame(idx=0, w=100, h=100, kp=None, hand_id="h_right"):
    kp = make_kp() if kp is None else kp
    return {"schema_version": "1.0", "clip_id": "c", "frame_idx": idx, "t_ms": idx * 33,
            "frame_width": w, "frame_height": h,
            "hands": [{"hand_id": hand_id, "side": "right", "handedness_score": 0.9,
                       "bbox_px": [0, 0, 10, 20], "kp2d_px": kp.tolist(), "kp3d_hand_m": None,
                       "source": "detected"}]}


def test_loads_list_json_and_jsonl(tmp_path):
    frames = [ps1_frame(0), ps1_frame(1)]
    (tmp_path / "a.json").write_text(json.dumps(frames))
    (tmp_path / "a.jsonl").write_text("\n".join(json.dumps(f) for f in frames))
    for src in (frames, tmp_path / "a.json", str(tmp_path / "a.jsonl")):
        out = load_hands(src, 100, 100)
        assert sorted(out) == [0, 1] and out[0][0].hand_id == "h_right"


def test_rescales_to_video_size():
    out = load_hands([ps1_frame(0, w=100, h=100)], 200, 400)
    hand = out[0][0]
    assert np.allclose(hand.kp[9], (200, 0)) and np.allclose(hand.kp[0], (200, 400))
    assert np.allclose(hand.bbox, (0, 0, 20, 80))


def test_rejects_wrong_keypoint_count_and_missing_file():
    bad = ps1_frame(0)
    bad["hands"][0]["kp2d_px"] = bad["hands"][0]["kp2d_px"][:20]
    with pytest.raises(ValueError):
        load_hands([bad], 100, 100)
    with pytest.raises(ValueError):
        load_hands("does_not_exist.jsonl", 100, 100)


def test_empty_hands_frame_means_no_hand():
    f = ps1_frame(0)
    f["hands"] = []
    assert load_hands([f], 100, 100) == {0: []}


def test_tip_inside_box_is_zero_and_distance_is_in_hand_sizes():
    kp = make_kp()
    assert hand_scale(kp) == 100
    tips = kp[[4, 8, 12, 16, 20]]
    box = [tips[:, 0].min() - 1, tips[:, 1].min() - 1, tips[:, 0].max() + 1, tips[:, 1].max() + 1]
    assert tip_distances(kp, box) == [0.0] * 5
    far = [1000, 1000, 1100, 1100]
    assert all(d > 5 for d in tip_distances(kp, far))
    # a box 50 px to the right of the thumb tip (x=40): 0.5 hand sizes at scale 100
    assert tip_distances(kp, [90, 55, 200, 65])[0] == pytest.approx(0.5, abs=0.05)


def test_tip_distance_to_polygon():
    kp = make_kp()
    poly = [(0, -100), (200, -100), (200, 200), (0, 200)]
    assert tip_distances(kp, poly) == [0.0] * 5


def test_curl_flat_vs_fist():
    flat = make_kp()
    fist = make_kp()
    for mcp, pip, dip in ((5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19)):
        fist[pip] = fist[mcp] + (0, -20)
        fist[dip] = fist[pip] + (20, 0)  # bent 90 degrees at the middle joint
    assert finger_curl(flat) < 0.05
    assert finger_curl(fist) > 0.95
