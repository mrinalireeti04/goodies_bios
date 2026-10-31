"""Tests for tools/fill_gaps.py with synthetic hands (no footage)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fill_gaps as fg  # noqa: E402
import hand_detect as hd  # noqa: E402

W, H = 1920, 1080


def hand(cx, side="left", size=100.0, cy=500.0):
    kp = [[cx + size * (k % 5) / 5, cy + size * (k // 5) / 5] for k in range(21)]
    return {"hand_id": "L" if side == "left" else "R", "side": side, "handedness_score": 0.9,
            "bbox_px": [round(v, 2) for v in fg.bbox_of(kp)], "kp2d_px": kp,
            "kp3d_hand_m": None, "source": "detected"}


def rec(i, hands):
    return {"schema_version": "1.0", "clip_id": "c", "frame_idx": 2 * i, "t_ms": round(2 * i * 1000 / 30),
            "frame_width": W, "frame_height": H, "hands": hands}


class FillTests(unittest.TestCase):
    def test_fills_short_gap_linearly(self):
        recs = [rec(0, [hand(400)]), rec(1, []), rec(2, []), rec(3, [hand(430)])]
        out, st = fg.fill(recs)
        self.assertEqual(st["frames_gained"], 2)
        h = out[1]["hands"][0]
        self.assertEqual((h["source"], h["handedness_score"], h["side"]), ("interpolated", None, "left"))
        self.assertAlmostEqual(h["kp2d_px"][0][0], 410.0, places=1)
        self.assertAlmostEqual(out[2]["hands"][0]["kp2d_px"][0][0], 420.0, places=1)

    def test_input_not_modified_and_output_valid(self):
        recs = [rec(0, [hand(400)]), rec(1, []), rec(2, [hand(420)])]
        before = [len(r["hands"]) for r in recs]
        out, _ = fg.fill(recs)
        self.assertEqual([len(r["hands"]) for r in recs], before)
        for r in out:
            hd.validate_record(r)

    def test_long_gap_not_filled(self):
        recs = [rec(0, [hand(400)])] + [rec(i, []) for i in range(1, 6)] + [rec(6, [hand(420)])]
        out, st = fg.fill(recs, max_gap=3)
        self.assertEqual(st["frames_filled"], 0)

    def test_hand_at_frame_edge_not_filled(self):
        recs = [rec(0, [hand(5)]), rec(1, []), rec(2, [hand(20)])]
        _, st = fg.fill(recs)
        self.assertEqual(st["frames_filled"], 0)
        self.assertEqual(st["skipped_implausible"], 1)

    def test_big_jump_not_filled(self):
        recs = [rec(0, [hand(200)]), rec(1, []), rec(2, [hand(1500)])]
        _, st = fg.fill(recs)
        self.assertEqual(st["frames_filled"], 0)

    def test_no_duplicate_when_another_hand_is_there(self):
        unknown = hand(410, "unknown")
        unknown["hand_id"] = "U1"
        recs = [rec(0, [hand(400)]), rec(1, [unknown]), rec(2, [hand(420)])]
        out, st = fg.fill(recs)
        self.assertEqual(st["skipped_duplicate"], 1)
        self.assertEqual(len(out[1]["hands"]), 1)

    def test_right_and_left_filled_independently_and_ordered(self):
        recs = [rec(0, [hand(300, "left"), hand(900, "right")]), rec(1, [hand(900, "right")]),
                rec(2, [hand(320, "left"), hand(920, "right")])]
        out, _ = fg.fill(recs)
        self.assertEqual([h["side"] for h in out[1]["hands"]], ["left", "right"])

    def test_3d_interpolated_only_when_both_ends_have_it(self):
        a, b = hand(400), hand(420)
        a["kp3d_hand_m"] = [[0.0, 0.0, 0.0]] * 21
        b["kp3d_hand_m"] = [[0.02, 0.0, 0.0]] * 21
        out, _ = fg.fill([rec(0, [a]), rec(1, []), rec(2, [b])])
        self.assertAlmostEqual(out[1]["hands"][0]["kp3d_hand_m"][0][0], 0.01, places=4)
        b["kp3d_hand_m"] = None
        out, _ = fg.fill([rec(0, [a]), rec(1, []), rec(2, [b])])
        self.assertIsNone(out[1]["hands"][0]["kp3d_hand_m"])

    def test_evaluate_linear_beats_hold_on_moving_hand(self):
        recs = [rec(i, [hand(300 + 15 * i)]) for i in range(12)]
        res = fg.evaluate(recs, gaps=(2,))
        self.assertLess(res[2]["linear"]["median"], 0.001)
        self.assertGreater(res[2]["hold_last"]["median"], 0.05)


if __name__ == "__main__":
    unittest.main()
