"""Tests for the idle-frame compute saver in hand_detect.py (synthetic video, stub detector)."""
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

W, H, FPS, N = 160, 90, 30, 60


class Res:
    hand_landmarks, handedness, hand_world_landmarks = [], [], []


class Pt:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class Cat:
    category_name, score = "Right", 0.9


class CountingLandmarker:
    """Sees a hand only in frames whose index is in `hand_at` (frame index = t_ms * FPS / 1000)."""

    def __init__(self, hand_at):
        self.hand_at, self.calls = set(hand_at), 0

    def detect_for_video(self, image, t_ms):
        self.calls += 1
        r = Res()
        if round(t_ms * FPS / 1000) in self.hand_at:
            r = Res()
            r.hand_landmarks = [[Pt(0.5, 0.5 + 0.01 * k) for k in range(21)]]
            r.handedness = [[Cat()]]
            r.hand_world_landmarks = []
        return r

    def close(self):
        pass


def make_video(path):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(N):
        wr.write(np.full((H, W, 3), 30 + i, np.uint8))
    wr.release()


def run(hand_at, **kw):
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "c.mp4"
        make_video(video)
        lm = CountingLandmarker(hand_at)
        with mock.patch.object(hd, "make_landmarker", lambda *a, **k: lm):
            summary = hd.run_clip(video, out_root=Path(tmp) / "out", step=2, save_video=False, **kw)
        path = Path(tmp) / "out" / "c" / "hands.jsonl"
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        n, err = hd.validate_file(path)
    return summary, rows, lm, err


class GateUnitTests(unittest.TestCase):
    def test_off_by_default(self):
        g = hd.IdleGate()
        for _ in range(50):
            self.assertFalse(g.skip_next())
            g.update(False)

    def test_skips_every_other_frame_once_idle(self):
        g = hd.IdleGate(after=3, every=2)
        seq = []
        for _ in range(12):
            s = g.skip_next()
            seq.append(s)
            if not s:
                g.update(False)
        self.assertEqual(seq, [False] * 3 + [True, False, True, False, True, False, True, False, True])

    def test_hand_found_goes_back_to_full_rate(self):
        g = hd.IdleGate(after=2, every=3)
        for _ in range(2):
            g.update(False)
        self.assertTrue(g.skip_next())
        self.assertTrue(g.skip_next())
        self.assertFalse(g.skip_next())     # probe frame
        g.update(True)
        for _ in range(10):
            self.assertFalse(g.skip_next())
            g.update(True)


class RunClipGateTests(unittest.TestCase):
    def test_default_run_checks_every_frame(self):
        summary, rows, lm, err = run(hand_at=[])
        self.assertEqual(lm.calls, N // 2)
        self.assertIsNone(err)
        self.assertNotIn("idle_skipped_frames", summary)

    def test_gate_saves_calls_and_keeps_every_record(self):
        summary, rows, lm, err = run(hand_at=[], idle_after=4, idle_every=2)
        self.assertEqual(len(rows), N // 2)
        self.assertLess(lm.calls, N // 2)
        self.assertEqual(lm.calls + summary["idle_skipped_frames"], N // 2)
        self.assertIsNone(err)

    def test_hand_that_stays_is_found_at_most_one_frame_late(self):
        # hand present from frame 40 on; the full run would see it at processed frame 20
        full, rows_full, _, _ = run(hand_at=range(40, N))
        gated, rows_g, _, err = run(hand_at=range(40, N), idle_after=4, idle_every=2)
        first = lambda rows: next(i for i, r in enumerate(rows) if r["hands"])
        self.assertLessEqual(first(rows_g) - first(rows_full), 1)
        self.assertIsNone(err)

    def test_busy_clip_is_not_skipped(self):
        summary, rows, lm, _ = run(hand_at=range(N), idle_after=4, idle_every=2)
        self.assertEqual(summary["idle_skipped_frames"], 0)
        self.assertEqual(lm.calls, N // 2)

    def test_cli_flags_reach_run_clip(self):
        with mock.patch.object(hd, "run_clip", return_value={}) as rc:
            hd.main(["clip.mp4", "--idle-after", "8", "--idle-every", "3", "--no-video"])
        self.assertEqual(rc.call_args.kwargs["idle_after"], 8)
        self.assertEqual(rc.call_args.kwargs["idle_every"], 3)


if __name__ == "__main__":
    unittest.main()
