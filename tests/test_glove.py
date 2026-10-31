import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dev"))

import glove_test  # noqa: E402
from ps2.hands import Hand  # noqa: E402
from test_hands import make_kp  # noqa: E402


def test_glove_covers_the_hand_hull_with_flat_colour_plus_noise_and_nothing_else():
    frame = np.full((300, 300, 3), 128, np.uint8)
    before = frame.copy()
    hand = Hand("h", make_kp() + np.array([100.0, 150.0]), np.zeros(4), "detected")
    glove_test.paint_gloves(frame, [hand], np.random.default_rng(0))
    inside = (frame != before).any(axis=2)
    assert inside.any() and inside[150 + 0, 100 + 100]                       # a point between wrist and knuckle is painted
    assert not inside[0, 0] and not inside[299, 299]                            # far corners untouched
    painted = frame[inside].astype(float)
    assert np.allclose(painted.mean(axis=0), glove_test.GLOVE_BGR, atol=3)     # flat glove colour...
    assert (painted.std(axis=0) > 2).all() and (painted.std(axis=0) < 12).all()                                              # ...with slight noise
    glove_test.paint_gloves(frame, [], np.random.default_rng(0))               # no hands: no crash


def test_painted_capture_paints_by_frame_index(tmp_path):
    import cv2
    path = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for _ in range(5):
        w.write(np.full((240, 320, 3), 128, np.uint8))
    w.release()
    hand = Hand("h", make_kp() * 0.5 + np.array([100.0, 150.0]), np.zeros(4), "detected")
    cap = glove_test.PaintedCapture(str(path), {2: [hand]})
    changed = []
    for _ in range(5):
        ok, frame = cap.read()
        changed.append(bool((np.abs(frame.astype(int) - 128) > 30).any()))
    assert changed == [False, False, True, False, False]
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 5                              # other attributes pass through
