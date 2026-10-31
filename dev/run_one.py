"""Run run_ps2 on one video and save the JSON to outputs/<clip>/ps2.json.

Usage: python dev/run_one.py <video_path> [hands_path] [--set key=value ...] [--out name]
--set overrides a config value for this run only (experiments); --out NAME writes outputs/<clip>/NAME.json instead of ps2.json.
Without hands_path, uses outputs/<clip>/ps1_stub.jsonl (made by hands_stub.py if missing).
"""
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from hands_stub import run_stub  # noqa: E402
import ps2.pipeline  # noqa: E402
from ps2 import run_ps2  # noqa: E402


def main(video_path, hands_path=None, overrides=None, out_name="ps2"):
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    video = Path(video_path)
    out_dir = ROOT / "outputs" / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    if hands_path is None:
        hands_path = out_dir / "ps1_stub.jsonl"
        if not hands_path.exists():
            run_stub(video, hands_path)
    if overrides:
        base = ps2.pipeline.load_config
        ps2.pipeline.load_config = lambda: {**base(), **overrides}
    result = run_ps2(video, hands_path)
    out = out_dir / f"{out_name}.json"
    out.write_text(json.dumps(result), encoding="utf-8")
    m = result["meta"]
    print(f"{m['frames_processed']} frames at {m['processing_fps']} fps on {m['device']}; "
          f"{len(result['contacts'])} contacts -> {out}")


if __name__ == "__main__":
    import yaml
    args, overrides, out_name, rest = [], {}, "ps2", sys.argv[1:]
    while rest:
        a = rest.pop(0)
        if a == "--set":
            k, v = rest.pop(0).split("=", 1)
            overrides[k] = yaml.safe_load(v)
        elif a == "--out":
            out_name = rest.pop(0)
        else:
            args.append(a)
    if len(args) not in (1, 2):
        sys.exit(__doc__)
    main(args[0], args[1] if len(args) == 2 else None, overrides, out_name)
