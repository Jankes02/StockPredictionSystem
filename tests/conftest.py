import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def paper_cfg():
    from config import load_config
    return load_config(ROOT / "config.yaml")


@pytest.fixture(scope="session")
def paper_data(paper_cfg):
    from config import get_data_dir
    from main import load_price_data
    return load_price_data(get_data_dir(paper_cfg), paper_cfg["symbols"])
