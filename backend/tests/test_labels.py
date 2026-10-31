import unittest

from backend.inference.labels import correct_hand_sides


def record(index, x, side, score):
    points = [[x + i * 0.1, 10 + i * 0.1] for i in range(21)]
    return {"frame_idx": index, "frame_width": 1000, "frame_height": 700,
            "hands": [{"hand_id": "U1" if side == "unknown" else side[0].upper(),
                       "side": side, "handedness_score": score,
                       "bbox_px": [x - 10, 0, x + 10, 20], "kp2d_px": points}]}


class ClipLabelCorrectionTests(unittest.TestCase):
    def test_corrects_unknown_and_single_frame_label_flip(self):
        raw = [record(0, 100, "left", .9), record(1, 104, "unknown", None),
               record(2, 110, "right", .55), record(3, 114, "left", .9)]
        result = correct_hand_sides(raw)
        self.assertEqual([r["hands"][0]["side"] for r in result], ["left"] * 4)
        self.assertEqual(result[1]["hands"][0]["side_source"], "clip_track_consensus")
        self.assertEqual(raw[2]["hands"][0]["side"], "right")  # source data preserved

    def test_does_not_fill_unknown_without_consensus(self):
        raw = [record(0, 100, "unknown", None), record(1, 103, "unknown", None)]
        self.assertEqual([r["hands"][0]["side"] for r in correct_hand_sides(raw)], ["unknown", "unknown"])


if __name__ == "__main__":
    unittest.main()
