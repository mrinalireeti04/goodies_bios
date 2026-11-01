"""Bridge the standalone PS2 package into private Studio job outputs."""
from __future__ import annotations

import json
from pathlib import Path


def run_ps2(video_path, hands_path, output_path, clip_id, work_dir, *, config_overrides=None):
    from ps2 import run_ps2 as infer_ps2

    result = infer_ps2(video_path, hands_path, config_overrides=config_overrides, work_dir=work_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
    temporary.replace(output)
    return output
