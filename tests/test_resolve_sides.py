"""Tests for tools/resolve_sides.py with synthetic hands (no footage)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import resolve_sides as rs  # noqa: E402
import hand_detect as hd  # noqa: E402

W, H = 1920, 1080


def hand(cx, side, cy=500.0, size=100.0):
    kp = [[cx + size * (k % 5) / 5, cy + size * (k // 5) / 5] for k in range(21)]
    xs, ys = [p[0] for p in kp], [p[1] for p in kp]
    hid = {"left": "L", "right": "R"}.get(side, "U1")
    return {"hand_id": hid, "side": side, "handedness_score": 0.9, "bbox_px": [min(xs), min(ys), max(xs), max(ys)],
            "kp2d_px": kp, "kp3d_hand_m": None, "source": "detected"}


def rec(i, hands):
    return {"schema_version": "1.0", "clip_id": "c", "frame_idx": 2 * i, "t_ms": round(2 * i * 1000 / 30),
            "frame_width": W, "frame_height": H, "hands": hands}


class ResolveTests(unittest.TestCase):
    def test_two_hands_ordered_by_x(self):
        recs = [rec(0, [hand(300, "left"), hand(900, "unknown")])]
        out, st = rs.resolve(recs)
        self.assertEqual([h["side"] for h in out[0]["hands"]], ["left", "right"])
        self.assertEqual(st["fixed_by_x_order"], 1)
        hd.validate_record(out[0])

    def test_wrong_both_left_gets_fixed(self):
        out, _ = rs.resolve([rec(0, [hand(300, "left"), hand(900, "left")])])
        self.assertEqual(sorted(h["side"] for h in out[0]["hands"]), ["left", "right"])

    def test_lone_unknown_follows_neighbours(self):
        recs = [rec(0, [hand(400, "right")]), rec(1, [hand(420, "unknown")]), rec(2, [hand(440, "right")])]
        out, st = rs.resolve(recs)
        self.assertEqual(out[1]["hands"][0]["side"], "right")
        self.assertEqual(st["fixed_by_neighbours"], 1)

    def test_conflicting_neighbours_stay_unknown(self):
        recs = [rec(0, [hand(400, "right")]), rec(1, [hand(420, "unknown")]), rec(2, [hand(440, "left")])]
        out, _ = rs.resolve(recs)
        self.assertEqual(out[1]["hands"][0]["side"], "unknown")

    def test_far_neighbour_not_used(self):
        recs = [rec(0, [hand(100, "right")]), rec(1, [hand(1500, "unknown")])]
        out, _ = rs.resolve(recs)
        self.assertEqual(out[1]["hands"][0]["side"], "unknown")

    def test_input_unchanged_and_other_fields_kept(self):
        recs = [rec(0, [hand(300, "left"), hand(900, "unknown")])]
        out, _ = rs.resolve(recs)
        self.assertEqual(recs[0]["hands"][1]["side"], "unknown")
        self.assertEqual(out[0]["hands"][1]["kp2d_px"], recs[0]["hands"][1]["kp2d_px"])
        self.assertEqual(out[0]["hands"][1]["source"], "detected")


if __name__ == "__main__":
    unittest.main()
