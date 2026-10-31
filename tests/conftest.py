import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))  # tests import helpers from each other

import ps2  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return yaml.safe_load((Path(ps2.__file__).parent / "config" / "ps2.yaml").read_text(encoding="utf-8"))
