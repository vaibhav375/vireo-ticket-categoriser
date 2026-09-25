"""Regression + resource tests on the real Set E pack (skipped if data/ is empty). Run: python -m pytest -m slow

The expected numbers were checked by hand earlier in the project and appear in docs/MEMO.md and
docs/submission-form.md. If one moves, either a bug crept in or the docs are now wrong.
"""
import re
import resource
import subprocess
import sys
import time

import pandas as pd
import pytest

from conftest import REAL_DATA, ROOT, has_real_data

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not has_real_data(), reason="real data pack not in data/")]


@pytest.fixture(scope="module")
def run_output(tmp_path_factory):
    out = tmp_path_factory.mktemp("real") / "out"
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    start = time.perf_counter()
    r = subprocess.run([sys.executable, str(ROOT / "run.py"), "--data", str(REAL_DATA), "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True, timeout=900)
    seconds = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024  # macOS reports bytes, Linux KB
    assert r.returncode == 0, r.stdout + r.stderr
    return {"out": out, "stdout": r.stdout, "seconds": seconds, "peak_bytes": peak_bytes, "before": before}


@pytest.fixture(scope="module")
def tickets(run_output):
    return pd.read_csv(run_output["out"] / "tickets_categorised.csv", parse_dates=["created_at"])


def test_data_fixes_hold_on_the_real_export(real_tickets):
    assert len(real_tickets) == 11_641               # 11,780 rows minus 139 from 2024
    assert real_tickets.ticket_id.is_unique
    assert (real_tickets.resolution_h.dropna() >= 0).all()   # UTC fix: nothing resolves before it opens
    assert real_tickets.transfers.isna().sum() == 3_913      # legacy rows in window stay unknown


def test_headline_shares_are_unchanged(tickets):
    real = tickets.true_owner.value_counts(normalize=True)
    bot = tickets.bot_owner.value_counts(normalize=True)
    assert round(bot["Billing"], 3) == 0.208 and round(real["Billing"], 3) == 0.140
    assert round(bot["Logistics"], 3) == 0.164 and round(real["Logistics"], 3) == 0.260


def test_recent_misroute_rate_is_unchanged(tickets, run_output):
    recent = tickets[tickets.created_at >= "2026-01-01"]
    assert round(recent.misrouted.mean(), 3) == 0.168
    assert "Misrouted (Jan 2026 – Jun 2026): 16.8%" in run_output["stdout"]


def test_evaluation_numbers_are_unchanged(run_output):
    ev = (run_output["out"] / "evaluation.md").read_text()
    assert "Out-of-time test (train Jan 2025 – Mar 2026, test Apr 2026 – Jun 2026)" in ev
    assert "Category accuracy: AI **100.0%** vs bot tag 70.5%" in ev
    assert "Unseen phrasings: **89.0%** (range 86.7%–90.4%)" in ev
    assert "live-chat noise (typos, dropped words, cut-off messages): **78.1%** (range 76.5%–79.1%)" in ev
    assert re.search(r"AI category: 150/150", ev)


def test_refund_and_replacement_list_is_unchanged(run_output):
    rr = pd.read_csv(run_output["out"] / "refund_and_replacement.csv")
    assert len(rr) == 139 and rr.likely_double_payout.sum() == 73


def test_run_fits_an_8gb_laptop(run_output):
    assert run_output["peak_bytes"] < 1.0e9, f"peak {run_output['peak_bytes'] / 1e9:.2f} GB"
    assert run_output["seconds"] < 180, f"took {run_output['seconds']:.0f} s"


def test_per_ticket_prediction_is_fast_enough_for_intake(real_tickets):
    from vireo.classify import fit_model, predict
    model = fit_model(real_tickets)
    msgs = real_tickets.customer_message.sample(200, random_state=0)
    times = []
    for m in msgs:
        start = time.perf_counter()
        predict(model, pd.Series([m]))
        times.append(time.perf_counter() - start)
    assert pd.Series(times).median() < 0.005   # 5 ms budget; measured ~1 ms
