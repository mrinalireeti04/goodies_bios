"""Tests for tools/quality_gate.py with synthetic hands (no footage)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import quality_gate as qg  # noqa: E402

W, H = 1920, 1080


def hand(cx, side="right", score=0.9, cy=500.0, size=120.0, source="detected"):
    kp = [[cx + size * (k % 5) / 5, cy + size * (k // 5) / 5] for k in range(21)]
    xs, ys = [p[0] for p in kp], [p[1] for p in kp]
    return {"hand_id": "R", "side": side, "handedness_score": score, "bbox_px": [min(xs), min(ys), max(xs), max(ys)],
            "kp2d_px": kp, "kp3d_hand_m": None, "source": source}


def rec(i, hands):
    return {"schema_version": "1.0", "clip_id": "c", "frame_idx": 2 * i, "t_ms": round(2 * i * 1000 / 30),
            "frame_width": W, "frame_height": H, "hands": hands}


def clip(builder, n=30):
    return [rec(i, builder(i)) for i in range(n)]


class FrameTests(unittest.TestCase):
    def test_clean_hand_is_good(self):
        self.assertEqual(qg.classify_frames(clip(lambda i: [hand(800 + i)], 3))[0][0], "good")

    def test_empty_frame(self):
        self.assertEqual(qg.classify_frames(clip(lambda i: [], 2))[0], ("empty", []))

    def test_each_weakness_is_reported(self):
        cases = {
            "interpolated": hand(800, source="interpolated"),
            "unknown_side": hand(800, side="unknown"),
            "low_score": hand(800, score=0.3),
            "cut_off": hand(1900),
            "tiny": hand(800, size=20),
        }
        for reason, h in cases.items():
            state, reasons = qg.classify_frames([rec(0, [h])])[0]
            self.assertEqual(state, "check", reason)
            self.assertIn(reason, reasons)

    def test_teleporting_hand_is_flagged(self):
        recs = [rec(0, [hand(200)]), rec(1, [hand(1500)])]
        self.assertIn("jump", qg.classify_frames(recs)[1][1])

    def test_missing_score_is_not_penalised(self):
        self.assertEqual(qg.classify_frames([rec(0, [hand(800, score=None)])])[0][0], "good")


class ClipTests(unittest.TestCase):
    def test_all_good_clip_is_grade_a(self):
        r = qg.analyse(clip(lambda i: [hand(800 + i)], 90))
        self.assertEqual(r["grade"], "A")
        self.assertEqual(r["usable_time_share"], 1.0)
        self.assertEqual(r["review_queue_total"], 0)

    def test_empty_clip_is_grade_d(self):
        r = qg.analyse(clip(lambda i: [], 90))
        self.assertEqual((r["grade"], r["usable_time_share"], r["hand_visible_share"]), ("D", 0.0, 0.0))

    def test_weak_second_goes_to_review_queue_with_reason(self):
        r = qg.analyse(clip(lambda i: [hand(800, side="unknown")], 45))
        self.assertGreaterEqual(r["review_queue_total"], 1)
        self.assertIn("unknown_side", r["review_queue"][0]["reasons"])

    def test_queue_puts_busiest_unreliable_seconds_first(self):
        def build(i):
            if i < 15:
                return [hand(800, side="unknown")]                      # second 0: hand in every frame
            if i < 30:
                return [hand(800, side="unknown")] if i % 3 == 0 else []   # second 1: hand in a third of frames
            return []
        q = qg.analyse(clip(build, 45))["review_queue"]
        self.assertEqual(q[0]["start_ms"], 0)

    def test_report_is_json_serialisable(self):
        json.dumps(qg.analyse(clip(lambda i: [hand(800)], 30)), allow_nan=False)


class FilesAndFacesTests(unittest.TestCase):
    def test_prefers_final_file_and_writes_quality_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "out" / "c"
            d.mkdir(parents=True)
            for name in ("hands.jsonl", "hands_final.jsonl"):
                with open(d / name, "w", encoding="utf-8") as f:
                    for r in clip(lambda i: [hand(800)], 30):
                        f.write(json.dumps(r) + "\n")
            self.assertEqual(qg.find_inputs([Path(tmp) / "out"]), [d / "hands_final.jsonl"])
            self.assertEqual(qg.main([str(Path(tmp) / "out")]), 0)
            self.assertTrue((d / "quality.json").exists())

    def test_face_check_reports_unavailable_when_no_model(self):
        rep = qg.analyse(clip(lambda i: [hand(800)], 30))
        with mock.patch.object(qg, "face_seconds", return_value=None):
            qg.add_privacy(rep, "x.mp4")
        self.assertFalse(rep["privacy"]["checked"])

    def test_face_seconds_flag_review(self):
        rep = qg.analyse(clip(lambda i: [hand(800)], 30))
        with mock.patch.object(qg, "face_seconds", return_value=[3, 4]):
            qg.add_privacy(rep, "x.mp4")
        self.assertTrue(rep["privacy"]["review_needed"])
        self.assertEqual(rep["privacy"]["flagged_seconds"], [3, 4])

    def test_face_seconds_on_missing_video_is_none(self):
        self.assertIsNone(qg.face_seconds("does_not_exist.mp4", cascade=object()))


if __name__ == "__main__":
    unittest.main()
