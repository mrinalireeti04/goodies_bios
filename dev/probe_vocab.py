"""Does YOLOE find our tools?

Usage: python dev/probe_vocab.py <video_path> [tag] [vocab_path]
Samples 50 frames, prints detections per label, saves crops and outputs/probe/<clip>/summary_<tag>.json.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLOE

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "ps2" / "config" / "ps2.yaml").read_text(encoding="utf-8"))
N_FRAMES = 50
CROPS_PER_LABEL = 3


def build_prompts(vocab):
    """Return (prompt list, prompt -> vocab key)."""
    prompt_to_key = {}
    for group in ("objects", "generic"):
        for key, spec in vocab[group].items():
            for p in spec["prompts"]:
                prompt_to_key.setdefault(p, key)
    for name in vocab["ignore"]:
        prompt_to_key.setdefault(name, "ignore:" + name)
    return list(prompt_to_key), prompt_to_key


def main(video_path, tag="now", vocab_path=None):
    VOCAB = yaml.safe_load(Path(vocab_path or ROOT / "ps2" / "config" / "vocab.yaml").read_text(encoding="utf-8"))
    video = Path(video_path)
    out_dir = ROOT / "outputs" / "probe" / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in np.linspace(0, total - 1, N_FRAMES).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append((int(idx), frame))

    prompts, prompt_to_key = build_prompts(VOCAB)
    model = YOLOE(ROOT / CFG["det_model"])
    model.set_classes(prompts, model.get_text_pe(prompts))

    confs = defaultdict(list)
    best = defaultdict(list)  # key -> [(conf, frame_idx, crop)]
    for idx, frame in frames:
        res = model.predict(frame, imgsz=CFG["imgsz"], conf=CFG["det_conf"], half=CFG["half"], device=0, verbose=False)[0]
        for box in res.boxes:
            key = prompt_to_key[prompts[int(box.cls)]]
            conf = float(box.conf)
            confs[key].append(conf)
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            best[key].append((conf, idx, frame[max(y1, 0):y2, max(x1, 0):x2].copy()))

    print(f"\n== {video.name}: {len(frames)} frames sampled ==")
    print(f"{'label':<20}{'detections':>11}{'mean conf':>11}")
    for key in sorted(confs, key=lambda k: -len(confs[k])):
        print(f"{key:<20}{len(confs[key]):>11}{np.mean(confs[key]):>11.2f}")
        for conf, idx, crop in sorted(best[key], key=lambda t: -t[0])[:CROPS_PER_LABEL]:
            if crop.size:
                name = key.replace(":", "_")
                cv2.imwrite(str(out_dir / f"{name}_f{idx}_{conf:.2f}.jpg"), crop)
    silent = [k for k in list(VOCAB["objects"]) + list(VOCAB["generic"]) if k not in confs]
    print("never detected:", ", ".join(silent))

    summary = {k: [len(v), round(float(np.mean(v)), 3)] for k, v in confs.items()}
    total = sum(len(v) for k, v in confs.items() if not k.startswith("ignore:"))
    print(f"total detections (excluding body/PPE names): {total}")
    (out_dir / f"summary_{tag}.json").write_text(json.dumps({"total": total, "labels": summary}), encoding="utf-8")


if __name__ == "__main__":
    if not 2 <= len(sys.argv) <= 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
