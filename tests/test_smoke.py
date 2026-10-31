from pathlib import Path

import yaml

import ps2

CONFIG = Path(ps2.__file__).parent / "config" / "ps2.yaml"


def test_run_ps2_is_exported():
    assert callable(ps2.run_ps2)


def test_config_loads_with_core_keys():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["det_model"].startswith("yoloe")
    assert cfg["min_tips"] == 2
