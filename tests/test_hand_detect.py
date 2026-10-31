"""
Tests that run without any footage and without the MediaPipe model file.
A synthetic video and a stand-in detector are used.

    python -m unittest discover tests
"""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hand_detect as hd  # noqa: E402

W, H, FPS, N_FRAMES = 320, 180, 30, 20


class Pt:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class Cat:
    def __init__(self, name, score):
        self.category_name, self.score = name, score


class Result:
    pass


class StubLandmarker:
    """Call 1: two hands that MediaPipe both calls 'Left'. Call 2: one 'Right' hand. Call 3: no hands."""

    def __init__(self):
        self.calls = 0

    def detect_for_video(self, image, t_ms):
        self.calls += 1
        r = Result()
        m = self.calls % 3
        if m == 1:
            r.hand_landmarks = [[Pt(0.5, 0.2 + 0.01 * k) for k in range(21)],
                                [Pt(0.2, 0.3 + 0.01 * k) for k in range(21)]]
            r.handedness = [[Cat("Left", 0.9)], [Cat("Left", 0.6)]]
            r.hand_world_landmarks = [[Pt(0.001 * k, 0.0, -0.002 * k) for k in range(21)]] * 2
        elif m == 2:
            r.hand_landmarks = [[Pt(0.5, 0.5) for _ in range(21)]]
            r.handedness = [[Cat("Right", 0.8)]]
            r.hand_world_landmarks = []
        else:
            r.hand_landmarks, r.handedness, r.hand_world_landmarks = [], [], []
        return r

    def close(self):
        pass


def make_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(N_FRAMES):
        writer.write(np.full((H, W, 3), 40 + i, np.uint8))
    writer.release()


class RunClipTests(unittest.TestCase):
    def run_clip(self, swap_lr):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        video = Path(tmp.name) / "synthetic_clip.mp4"
        make_video(video)
        with mock.patch.object(hd, "make_landmarker", lambda *a, **k: StubLandmarker()):
            summary = hd.run_clip(video, out_root=Path(tmp.name) / "out", step=2,
                                  save_video=False, swap_lr=swap_lr)
        hands_path = Path(tmp.name) / "out" / "synthetic_clip" / "hands.jsonl"
        with open(hands_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        return summary, rows, hands_path

    def test_every_line_matches_the_schema(self):
        _, rows, path = self.run_clip(swap_lr=False)
        self.assertEqual(hd.validate_file(path), (len(rows), None))

    def test_one_record_per_processed_frame(self):
        summary, rows, _ = self.run_clip(swap_lr=False)
        self.assertEqual(len(rows), N_FRAMES // 2)
        self.assertEqual([r["frame_idx"] for r in rows], list(range(0, N_FRAMES, 2)))
        self.assertEqual(summary["frames_processed"], N_FRAMES // 2)

    def test_frames_without_hands_are_kept(self):
        _, rows, _ = self.run_clip(swap_lr=False)
        self.assertTrue(any(r["hands"] == [] for r in rows))

    def test_pixels_are_in_original_frame_space(self):
        _, rows, _ = self.run_clip(swap_lr=False)
        one_hand = next(r for r in rows if len(r["hands"]) == 1)
        self.assertEqual(one_hand["frame_width"], W)
        self.assertAlmostEqual(one_hand["hands"][0]["kp2d_px"][0][0], 0.5 * W, places=1)

    def test_second_hand_with_same_side_becomes_unknown(self):
        _, rows, _ = self.run_clip(swap_lr=False)
        two = next(r for r in rows if len(r["hands"]) == 2)
        self.assertEqual(sorted(h["side"] for h in two["hands"]), ["left", "unknown"])
        self.assertEqual(two["hands"][0]["handedness_score"], 0.9)

    def test_swap_changes_only_the_side_label(self):
        _, plain, _ = self.run_clip(swap_lr=False)
        _, swapped, _ = self.run_clip(swap_lr=True)
        a = next(r for r in plain if len(r["hands"]) == 1)["hands"][0]
        b = next(r for r in swapped if len(r["hands"]) == 1)["hands"][0]
        self.assertEqual((a["side"], b["side"]), ("right", "left"))
        self.assertEqual(a["kp2d_px"], b["kp2d_px"])


class ValidatorTests(unittest.TestCase):
    def good_record(self):
        hand = {"hand_id": "L", "side": "left", "handedness_score": 0.9,
                "bbox_px": [1.0, 2.0, 3.0, 4.0],
                "kp2d_px": [[float(i), float(i)] for i in range(21)],
                "kp3d_hand_m": None, "source": "detected"}
        return {"schema_version": "1.0", "clip_id": "c", "frame_idx": 0, "t_ms": 0,
                "frame_width": 10, "frame_height": 10, "hands": [hand]}

    def assert_rejected(self, mutate):
        rec = copy.deepcopy(self.good_record())
        mutate(rec)
        with self.assertRaises(ValueError):
            hd.validate_record(rec)

    def test_accepts_a_good_record(self):
        hd.validate_record(self.good_record())

    def test_rejects_extra_key(self):
        self.assert_rejected(lambda r: r.__setitem__("confidence", 1))

    def test_rejects_wrong_number_of_points(self):
        self.assert_rejected(lambda r: r["hands"][0]["kp2d_px"].pop())

    def test_rejects_bad_side(self):
        self.assert_rejected(lambda r: r["hands"][0].__setitem__("side", "both"))

    def test_rejects_nan(self):
        self.assert_rejected(lambda r: r["hands"][0]["kp2d_px"][0].__setitem__(0, float("nan")))

    def test_rejects_score_above_one(self):
        self.assert_rejected(lambda r: r["hands"][0].__setitem__("handedness_score", 1.5))

    def test_rejects_duplicate_hand_id(self):
        def dup(r):
            r["hands"].append(copy.deepcopy(r["hands"][0]))
        self.assert_rejected(dup)

    def test_rejects_short_3d_list(self):
        self.assert_rejected(lambda r: r["hands"][0].__setitem__("kp3d_hand_m", [[0, 0, 0]] * 5))


if __name__ == "__main__":
    unittest.main()
