"""The code and the test setup behave the same wherever they are run from."""
import subprocess
import sys
import warnings

from conftest import ROOT
from synthetic import make_tickets, write_pack


def test_pipeline_code_raises_no_deprecation_warnings(tmp_path):
    from vireo.evaluate import out_of_time
    from vireo.labels import add_reference_labels
    from vireo.load import load
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        t = add_reference_labels(load(write_pack(tmp_path, make_tickets())))
        out_of_time(t)


def test_slow_tests_are_skipped_by_default_even_from_another_folder():
    # exactly what was typed: 'python3 -m pytest' with no path, from the folder above the repo
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                       cwd=ROOT.parent, capture_output=True, text=True, timeout=120)
    assert "test_real_data.py" not in r.stdout
    assert "7 deselected" in r.stdout
