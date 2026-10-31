"""MediaPipe -> PS1Frame list, until the real PS1 output exists.

Usage: python dev/hands_stub.py <video_path> [out_jsonl]
"""
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "outputs" / "models" / "hand_landmarker.task"
BOX_PAD = 0.1  # grow the keypoint box by 10% so it covers the whole hand


def run_stub(video_path, out_path=None) -> list[dict]:
    video = Path(video_path)
    out_path = Path(out_path) if out_path else ROOT / "outputs" / video.stem / "ps1_stub.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    frames = []
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t_ms = round(idx * 1000 / fps)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(image, t_ms)
            hands, used = [], set()
            for i, lms in enumerate(result.hand_landmarks):
                handed = result.handedness[i][0]
                side = handed.category_name.lower()
                hand_id = f"h_{side}"
                if hand_id in used:  # two hands got the same side label
                    side, hand_id = "unknown", f"h_{i}"
                used.add(hand_id)
                kp = [[p.x * w, p.y * h] for p in lms]
                xs, ys = [p[0] for p in kp], [p[1] for p in kp]
                px, py = BOX_PAD * (max(xs) - min(xs)), BOX_PAD * (max(ys) - min(ys))
                world = result.hand_world_landmarks[i] if result.hand_world_landmarks else None
                hands.append({
                    "hand_id": hand_id,
                    "side": side,
                    "handedness_score": handed.score,
                    "bbox_px": [min(xs) - px, min(ys) - py, max(xs) + px, max(ys) + py],
                    "kp2d_px": kp,
                    "kp3d_hand_m": [[p.x, p.y, p.z] for p in world] if world else None,
                    "source": "detected",
                })
            frames.append({
                "schema_version": "1.0", "clip_id": video.stem, "frame_idx": idx, "t_ms": t_ms,
                "frame_width": w, "frame_height": h, "hands": hands,
            })
            idx += 1
    with out_path.open("w", encoding="utf-8") as f:
        for fr in frames:
            f.write(json.dumps(fr) + "\n")
    return frames


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: python dev/hands_stub.py <video_path> [out_jsonl]")
    out = run_stub(*sys.argv[1:])
    seen = sum(1 for fr in out if fr["hands"])
    print(f"{len(out)} frames, hands seen in {seen}")
