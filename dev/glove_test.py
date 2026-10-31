"""Synthetic glove stress test (dev only).

Paints the convex hull of each hand's 21 keypoints with a flat glove colour plus slight noise, frame by frame in
memory, runs run_ps2 on that with the SAME hand data, and compares with the normal run. Nothing is written to disk
except outputs/<clip>/glove_test.json.

Usage: python dev/glove_test.py <video_path> [hands_path] [--offline]
  --offline  ignore GEMINI_API_KEY for both runs (no API variance in the comparison)
"""
import json
import logging
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ps2.pipeline as pipeline  # noqa: E402
from ps2.hands import load_hands  # noqa: E402

GLOVE_BGR = (200, 90, 30)   # flat blue glove (not a skin colour)
NOISE_SIGMA = 6.0           # grey levels of texture noise inside the glove

RealCapture = cv2.VideoCapture


def paint_gloves(frame: np.ndarray, hands, rng: np.random.Generator) -> None:
    """Fill the convex hull of every hand's keypoints with the glove colour + noise (in place)."""
    mask = np.zeros(frame.shape[:2], np.uint8)
    for hand in hands:
        cv2.fillConvexPoly(mask, cv2.convexHull(np.round(hand.kp).astype(np.int32)), 255)
    inside = mask > 0
    if inside.any():
        noise = rng.normal(0, NOISE_SIGMA, (int(inside.sum()), 3))
        frame[inside] = np.clip(np.array(GLOVE_BGR) + noise, 0, 255).astype(np.uint8)


class PaintedCapture:
    """cv2.VideoCapture that paints gloves onto every frame it returns (for run_ps2 and the vocabulary sampling)."""

    def __init__(self, path, hands):
        self.cap, self.hands, self.rng = RealCapture(path), hands, np.random.default_rng(0)

    def read(self):
        idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        ok, frame = self.cap.read()
        if ok:
            paint_gloves(frame, self.hands.get(idx, []), self.rng)
        return ok, frame

    def __getattr__(self, name):
        return getattr(self.cap, name)


class LogGrabber(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())

    def number(self, pattern: str) -> int | None:
        for m in reversed(self.messages):
            found = re.search(pattern, m)
            if found:
                return int(found.group(1))
        return None


def run(video: Path, hands_path: Path, painted: bool) -> dict:
    """One run_ps2 call; returns the numbers we compare."""
    w, h = (int(RealCapture(str(video)).get(c)) for c in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    hands = load_hands(hands_path, w, h)
    removed = [0]
    original_filter = pipeline.drop_hand_like

    def counting_filter(dets, hs, max_iou):
        kept = original_filter(dets, hs, max_iou)
        removed[0] += len(dets) - len(kept)
        return kept

    grabber = LogGrabber()
    logging.getLogger("ps2").addHandler(grabber)
    logging.getLogger("ps2").setLevel(logging.INFO)
    pipeline.drop_hand_like = counting_filter
    if painted:
        cv2.VideoCapture = lambda path: PaintedCapture(path, hands)
    try:
        result = pipeline.run_ps2(video, hands_path)
    finally:
        cv2.VideoCapture = RealCapture
        pipeline.drop_hand_like = original_filter
        logging.getLogger("ps2").removeHandler(grabber)
    contacts = result["contacts"]
    return {
        "contacts": len(contacts),
        "holds": sum(c["interaction"] == "hold" for c in contacts),
        "touches": sum(c["interaction"] == "touch" for c in contacts),
        "objects_with_contact": len({c["object_id"] for c in contacts}),
        "tools_identified": sum(t["label_status"] == "identified" for t in result["tools"]),
        "held_objects_named_by_vlm": grabber.number(r"VLM naming: (\d+) objects named"),
        "labels_changed_by_vlm": grabber.number(r"objects named, (\d+) labels changed"),
        "detections_removed_by_hand_filter": removed[0],
        "vocab_source": result["meta"].get("vocab_source"),
        "gemini_requests": grabber.number(r"Gemini requests this clip: (\d+)"),
    }


def main(video_path: str, hands_path: str | None = None, offline: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    video = Path(video_path)
    out_dir = ROOT / "outputs" / video.stem
    hands_path = Path(hands_path) if hands_path else out_dir / "ps1_stub.jsonl"
    if offline:
        os.environ.pop("GEMINI_API_KEY", None)
    normal = run(video, hands_path, painted=False)
    gloved = run(video, hands_path, painted=True)
    (out_dir / "glove_test.json").write_text(json.dumps({"normal": normal, "gloved": gloved}, indent=1), encoding="utf-8")
    print(f"\n{'':36}{'normal':>10}{'gloved':>10}")
    for key in normal:
        print(f"{key:36}{str(normal[key]):>10}{str(gloved[key]):>10}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not 1 <= len(args) <= 2:
        sys.exit(__doc__)
    main(args[0], args[1] if len(args) == 2 else None, "--offline" in sys.argv)
