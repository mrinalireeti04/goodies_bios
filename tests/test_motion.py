import cv2
import numpy as np

from ps2.motion import Motion

W, H, SHIFT = 1280, 720, 6


def scene():
    rng = np.random.default_rng(0)
    return cv2.GaussianBlur(rng.integers(0, 255, (H, W + 200, 3), dtype=np.uint8), (0, 0), 2)


def frame(big, t):
    """The camera pans right by SHIFT px per frame: the world appears to move left."""
    return np.ascontiguousarray(big[:, SHIFT * t:SHIFT * t + W])


def test_camera_pan_is_removed_from_residual(cfg):
    big, motion = scene(), Motion(cfg)
    motion.step(frame(big, 0), [])
    motion.step(frame(big, 1), [])
    assert motion.reliable
    world_point = np.array([600.0, 300.0])
    seen_now = world_point - (SHIFT, 0)                       # where a static world point shows up now
    assert np.linalg.norm(motion.residual(world_point, seen_now)) < 1.5
    moved = seen_now + (10, 0)                                # something that really moved 10 px
    assert abs(np.linalg.norm(motion.residual(world_point, moved)) - 10) < 1.5


def test_aligned_diff_low_for_static_scene_high_where_something_changed(cfg):
    big, motion = scene(), Motion(cfg)
    f0, f1 = frame(big, 0), frame(big, 1).copy()
    f1[300:500, 500:700] = 0                                  # something dark appears here
    motion.step(f0, [])
    motion.step(f1, [])
    assert motion.aligned_diff([800, 100, 1000, 250]) < 8
    assert motion.aligned_diff([500, 300, 700, 500]) > 40


def test_too_little_texture_is_flagged_unreliable(cfg):
    motion = Motion(cfg)
    blank = np.full((H, W, 3), 128, np.uint8)
    motion.step(blank, [])
    motion.step(blank, [])
    assert not motion.reliable
    assert np.allclose(motion.residual([10, 10], [10, 10]), 0)


def test_hand_region_is_ignored_when_judging_machine_motion(cfg):
    big, motion = scene(), Motion(cfg)
    f0, f1 = frame(big, 0), frame(big, 1).copy()
    f1[300:500, 500:700] = 0                                  # a "hand" moves in front of the machine
    motion.step(f0, [])
    motion.step(f1, [])
    machine = [400, 250, 900, 550]
    assert motion.aligned_diff(machine) > 5                    # the hand alone makes the machine look like it moves
    assert motion.aligned_diff(machine, exclude=[[490, 290, 710, 510]]) < 5
