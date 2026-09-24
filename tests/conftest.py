import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic import write_pack  # noqa: E402

REAL_DATA = ROOT / "data"


def has_real_data():
    return any(REAL_DATA.glob("*tickets.csv"))


@pytest.fixture
def pack(tmp_path):
    """Folder with a small synthetic data pack."""
    return write_pack(tmp_path / "data")


@pytest.fixture(scope="session")
def real_tickets():
    if not has_real_data():
        pytest.skip("real data pack not in data/")
    from vireo.labels import add_reference_labels
    from vireo.load import load
    return add_reference_labels(load(REAL_DATA))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs on the real data pack (about a minute each)")


def pytest_collection_modifyitems(config, items):
    """Skip slow real-data tests unless asked for with -m. Lives here, not in pytest.ini, because
    pytest.ini is only read when pytest starts inside the repo; this file is loaded from anywhere."""
    if config.option.markexpr:
        return
    slow = [i for i in items if i.get_closest_marker("slow")]
    if slow:
        config.hook.pytest_deselected(items=slow)
        items[:] = [i for i in items if not i.get_closest_marker("slow")]
