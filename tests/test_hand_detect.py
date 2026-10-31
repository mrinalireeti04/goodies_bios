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
import rtm_hand  # noqa: E402

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


class StubDet:
    """Hand detector: always proposes one box on the right of the frame."""

    def __call__(self, image):
        h, w = image.shape[:2]
        return np.array([[0.7 * w, 0.4 * h, 0.9 * w, 0.8 * h]])


class StubPose:
    """RTMPose: 21 points spread inside each box; score is configurable."""

    def __init__(self, score=0.9):
        self.score = score
        self.calls = 0

    def __call__(self, image, bboxes=()):
        self.calls += 1
        kps, scs = [], []
        for x1, y1, x2, y2 in bboxes:
            kps.append([[x1 + (x2 - x1) * k / 20, y1 + (y2 - y1) * k / 20] for k in range(21)])
            scs.append([self.score] * 21)
        return np.array(kps), np.array(scs)


class NoHandsLandmarker(StubLandmarker):
    """MediaPipe finds nothing, so only RTMPose can produce hands."""

    def detect_for_video(self, image, t_ms):
        r = Result()
        r.hand_landmarks, r.handedness, r.hand_world_landmarks = [], [], []
        return r


class RTMPoseTests(unittest.TestCase):
    def run_rtm(self, score=0.9, landmarker=StubLandmarker):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        video = Path(tmp.name) / "synthetic_clip.mp4"
        make_video(video)
        engine = rtm_hand.RTMHandEngine(det=StubDet(), pose=StubPose(score))
        with mock.patch.object(hd, "make_landmarker", lambda *a, **k: landmarker()):
            summary = hd.run_clip(video, out_root=Path(tmp.name) / "out", step=2,
                                  save_video=False, engine=engine)
        path = Path(tmp.name) / "out" / "synthetic_clip" / "hands.jsonl"
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        return summary, rows, path

    def test_output_still_matches_the_schema(self):
        _, rows, path = self.run_rtm()
        self.assertEqual(hd.validate_file(path), (len(rows), None))

    def test_summary_records_the_engine(self):
        summary, _, _ = self.run_rtm()
        self.assertEqual(summary["run_config"]["engine"], "rtmpose")
        self.assertIn("rtmpose_stats", summary)

    def test_low_score_hands_are_dropped(self):
        summary, rows, _ = self.run_rtm(score=0.05, landmarker=NoHandsLandmarker)
        self.assertTrue(all(r["hands"] == [] for r in rows))
        self.assertGreater(summary["rtmpose_stats"]["dropped_low_score"], 0)

    def test_kp3d_is_null_and_points_are_rtmpose_points(self):
        _, rows, _ = self.run_rtm(landmarker=NoHandsLandmarker)
        hand = next(h for r in rows for h in r["hands"])
        self.assertIsNone(hand["kp3d_hand_m"])
        self.assertEqual(len(hand["kp2d_px"]), 21)


class FixedDet:
    """Detector that returns a given list of boxes."""

    def __init__(self, boxes=()):
        self.boxes = list(boxes)

    def __call__(self, image):
        return np.array(self.boxes, dtype=float).reshape(-1, 4)


class SquashPose:
    """RTMPose that squeezes every skeleton into a 10 px box."""

    def __call__(self, image, bboxes=()):
        kps = [[[x1 + 10 * k / 20, y1 + 10 * k / 20] for k in range(21)]
               for x1, y1, x2, y2 in bboxes]
        return np.array(kps), np.array([[0.9] * 21] * len(bboxes))


FRAME = np.zeros((300, 600, 3), np.uint8)


def mp_hand(side, score, bbox, with_3d=True):
    """A MediaPipe hand as a schema dict (points spread along the box diagonal)."""
    x1, y1, x2, y2 = bbox
    kp = [[round(x1 + (x2 - x1) * k / 20, 2), round(y1 + (y2 - y1) * k / 20, 2)]
          for k in range(21)]
    return {"hand_id": "x", "side": side, "handedness_score": score, "bbox_px": list(bbox),
            "kp2d_px": kp, "source": "detected",
            "kp3d_hand_m": [[0.001 * k, 0.0, -0.002 * k] for k in range(21)] if with_3d else None}


class MediaPipeFallbackTests(unittest.TestCase):
    def test_unmatched_mediapipe_hand_is_kept_as_is(self):
        eng = rtm_hand.RTMHandEngine(det=FixedDet(), pose=SquashPose())
        m = mp_hand("right", 0.93, [100, 100, 220, 220])
        hands = rtm_hand.to_schema_hands(eng.estimate(FRAME, [m]))
        self.assertEqual(len(hands), 1)
        h = hands[0]
        self.assertEqual((h["side"], h["handedness_score"], h["source"]), ("right", 0.93, "detected"))
        self.assertEqual(h["kp2d_px"], m["kp2d_px"])
        self.assertEqual(h["kp3d_hand_m"], m["kp3d_hand_m"])
        self.assertEqual(eng.stats["mediapipe_fallback"], 1)

    def test_matched_mediapipe_hand_is_not_duplicated(self):
        eng = rtm_hand.RTMHandEngine(det=FixedDet(), pose=StubPose(0.9))
        m = mp_hand("left", 0.9, [100, 100, 220, 220])
        hands = rtm_hand.to_schema_hands(eng.estimate(FRAME, [m]))
        self.assertEqual(len(hands), 1)
        self.assertIsNone(hands[0]["kp3d_hand_m"])   # the RTMPose hand, not the fallback
        self.assertEqual(eng.stats["mediapipe_fallback"], 0)

    def test_fallback_is_marked_nowhere_in_the_schema(self):
        eng = rtm_hand.RTMHandEngine(det=FixedDet(), pose=SquashPose())
        hands = rtm_hand.to_schema_hands(
            eng.estimate(FRAME, [mp_hand("left", 0.9, [100, 100, 220, 220])]))
        self.assertEqual(list(hands[0].keys()), hd.HAND_KEYS)


class SquashedSkeletonTests(unittest.TestCase):
    def test_tiny_skeleton_is_dropped_and_counted(self):
        eng = rtm_hand.RTMHandEngine(det=FixedDet([[100, 100, 220, 220]]), pose=SquashPose())
        self.assertEqual(eng.estimate(FRAME, []), [])
        self.assertEqual(eng.stats["dropped_low_score"], 1)
        self.assertEqual(eng.stats["kept"], 0)

    def test_skeleton_with_a_long_enough_side_is_kept(self):
        eng = rtm_hand.RTMHandEngine(det=FixedDet([[100, 100, 150, 120]]), pose=StubPose(0.9))
        self.assertEqual(len(eng.estimate(FRAME, [])), 1)   # 50 x 20: longest side >= 40


class MinScoreTests(unittest.TestCase):
    def test_default_cutoff_is_0_2(self):
        self.assertEqual(rtm_hand.MIN_KP_SCORE, 0.2)
        for score, kept in ((0.25, 1), (0.15, 0)):
            eng = rtm_hand.RTMHandEngine(det=FixedDet([[100, 100, 220, 220]]), pose=StubPose(score))
            self.assertEqual(len(eng.estimate(FRAME, [])), kept)

    def test_cutoff_can_be_changed(self):
        eng = rtm_hand.RTMHandEngine(min_kp_score=0.5, det=FixedDet([[100, 100, 220, 220]]),
                                     pose=StubPose(0.4))
        self.assertEqual(eng.estimate(FRAME, []), [])

    def test_run_clip_passes_the_cutoff_and_records_it(self):
        real = rtm_hand.RTMHandEngine
        seen = {}

        def factory(**kw):
            seen.update(kw)
            return real(det=StubDet(), pose=StubPose(0.9), **kw)

        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "synthetic_clip.mp4"
            make_video(video)
            with mock.patch.object(hd, "make_landmarker", lambda *a, **k: StubLandmarker()), \
                    mock.patch.object(rtm_hand, "RTMHandEngine", factory):
                summary = hd.run_clip(video, out_root=Path(tmp) / "out", step=2,
                                      save_video=False, engine_name="rtmpose",
                                      rtm_min_score=0.35)
        self.assertEqual(seen, {"min_kp_score": 0.35})
        self.assertEqual(summary["run_config"]["rtm_min_score"], 0.35)
        self.assertIn("mediapipe_fallback", summary["rtmpose_stats"])

    def test_cli_flag_reaches_run_clip(self):
        with mock.patch.object(hd, "run_clip", return_value={}) as rc:
            hd.main(["clip.mp4", "--engine", "rtmpose", "--rtm-min-score", "0.35", "--no-video"])
        self.assertEqual(rc.call_args.kwargs["rtm_min_score"], 0.35)


class PreviousFrameSideTests(unittest.TestCase):
    A = [100, 100, 220, 220]

    def engine(self, box):
        return rtm_hand.RTMHandEngine(det=FixedDet([box]), pose=StubPose(0.9))

    @staticmethod
    def next_frame(eng, det_boxes):
        """Next frame: MediaPipe finds nothing, only the detector proposes boxes
        (previous-frame proposals are cleared so only the side rule is tested)."""
        eng.prev_boxes = []
        eng.det = FixedDet(det_boxes)
        return eng.estimate(FRAME, [])

    def test_lone_hand_takes_side_from_previous_frame(self):
        eng = self.engine(self.A)
        first = eng.estimate(FRAME, [mp_hand("right", 0.9, self.A)])
        self.assertEqual(first[0]["side"], "right")
        second = self.next_frame(eng, [self.A])     # MediaPipe lost it this frame
        self.assertEqual([h["side"] for h in second], ["right"])

    def test_no_overlap_with_previous_hand_stays_unknown(self):
        eng = self.engine(self.A)
        eng.estimate(FRAME, [mp_hand("right", 0.9, self.A)])
        second = self.next_frame(eng, [[400, 100, 520, 220]])
        self.assertEqual([h["side"] for h in second], ["unknown"])

    def test_previous_side_must_be_left_or_right(self):
        eng = self.engine(self.A)
        eng.estimate(FRAME, [mp_hand("unknown", None, self.A)])
        second = self.next_frame(eng, [self.A])
        self.assertEqual([h["side"] for h in second], ["unknown"])

    def test_side_already_used_in_this_frame_is_not_given_twice(self):
        eng = rtm_hand.RTMHandEngine(det=FixedDet([[300, 100, 420, 220]]), pose=StubPose(0.9))
        eng.prev_hands = [([300, 100, 420, 220], "right")]
        # a labelled 'right' hand sits left of the lone hand, so x-order would not label it
        found = eng.estimate(FRAME, [mp_hand("right", 0.9, [100, 100, 220, 220])])
        sides = sorted(f["side"] for f in found)
        self.assertEqual(sides, ["right", "unknown"])


class RTMHelperTests(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(rtm_hand.iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(rtm_hand.iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_merge_keeps_first_of_overlapping_boxes(self):
        kept = rtm_hand.merge_boxes([[0, 0, 100, 100], [5, 5, 100, 100], [200, 200, 300, 300]])
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0], [0.0, 0.0, 100.0, 100.0])

    def test_two_unlabelled_hands_are_ordered_by_x(self):
        a = {"bbox": [500, 0, 600, 100], "kp_score": .9}
        b = {"bbox": [100, 0, 200, 100], "kp_score": .8}
        found = [a, b]
        rtm_hand.RTMHandEngine._assign_sides(found, [], 1000)
        self.assertEqual((b["side"], a["side"]), ("left", "right"))

    def test_side_comes_from_overlapping_mediapipe_hand(self):
        f = {"bbox": [10, 10, 100, 100], "kp_score": .9}
        mp = [{"bbox_px": [12, 12, 98, 98], "side": "right", "handedness_score": .93}]
        rtm_hand.RTMHandEngine._assign_sides([f], mp, 1000)
        self.assertEqual((f["side"], f["score"]), ("right", .93))

    def test_same_side_twice_becomes_unknown_for_the_weaker_hand(self):
        kp = np.array([[float(i), float(i)] for i in range(21)])
        found = [{"side": "left", "score": .9, "kp": kp, "kp_score": .9},
                 {"side": "left", "score": .6, "kp": kp + 50, "kp_score": .9}]
        hands = rtm_hand.to_schema_hands(found)
        self.assertEqual([h["side"] for h in hands], ["left", "unknown"])
        self.assertEqual([h["hand_id"] for h in hands], ["L", "U1"])


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
