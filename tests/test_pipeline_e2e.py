"""run.py end to end on a synthetic pack, as a user would run it."""
import re
import subprocess
import sys

import pandas as pd

from conftest import ROOT
from synthetic import make_tickets, write_pack
from vireo.taxonomy import CATEGORIES


def run(*args, cwd=ROOT):
    return subprocess.run([sys.executable, str(ROOT / "run.py"), *map(str, args)], cwd=cwd,
                          capture_output=True, text=True, timeout=600)


def test_full_run_writes_every_output_with_consistent_numbers(pack, tmp_path):
    out = tmp_path / "new" / "out"  # does not exist yet
    r = run("--data", pack, "--out", out)
    assert r.returncode == 0, r.stderr
    for f in ["report.html", "evaluation.md", "tickets_categorised.csv", "monthly_by_category.csv",
              "monthly_by_team.csv", "refund_and_replacement.csv"]:
        assert (out / f).stat().st_size > 0, f
    t = pd.read_csv(out / "tickets_categorised.csv")
    assert len(t) == 660 and t.ticket_id.is_unique
    assert set(t.ai_category) <= set(CATEGORIES)
    assert pd.read_csv(out / "monthly_by_team.csv", index_col=0).to_numpy().sum() == 660
    assert pd.read_csv(out / "monthly_by_category.csv", index_col=0).to_numpy().sum() == 660
    html = (out / "report.html").read_text()
    assert "Monthly tickets by owning team" in html and not re.search(r"\bnan\b", html.lower())


def test_audit_file_without_matching_tickets_is_reported_not_a_crash(pack, tmp_path):
    r = run("--data", pack, "--out", tmp_path / "out")
    assert r.returncode == 0, r.stderr
    assert "none of the audited tickets are in this export" in (tmp_path / "out" / "evaluation.md").read_text()


def test_run_from_another_folder_still_finds_the_audit_file(pack, tmp_path):
    r = run("--data", pack, "--out", tmp_path / "out", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "audit_labels.csv missing" not in (tmp_path / "out" / "evaluation.md").read_text()


def test_audit_mode_prints_evidence_and_writes_nothing(pack, tmp_path):
    r = run("--data", pack, "--audit", "--out", tmp_path / "out")
    assert r.returncode == 0 and "Resolved before created" in r.stdout
    assert not (tmp_path / "out").exists()


def test_data_problems_exit_cleanly_without_a_traceback(tmp_path):
    folder = write_pack(tmp_path / "d", make_tickets().drop(columns=["channel"]))
    r = run("--data", folder, "--out", tmp_path / "out")
    assert r.returncode == 2
    assert "missing columns: channel" in r.stdout + r.stderr and "Traceback" not in r.stderr


def test_missing_data_folder_exits_cleanly(tmp_path):
    r = run("--data", tmp_path / "nothing-here", "--out", tmp_path / "out")
    assert r.returncode == 2 and "Traceback" not in r.stderr
    assert "tickets.csv not found" in r.stdout + r.stderr


def test_other_period_runs_end_to_end_with_start_and_end(tmp_path):
    folder = write_pack(tmp_path / "d", make_tickets(start="2026-07-01", end="2027-06-30", out_of_window=0))
    r = run("--data", folder, "--out", tmp_path / "out", "--start", "2026-07-01", "--end", "2027-07-01")
    assert r.returncode == 0, r.stderr
    ev = (tmp_path / "out" / "evaluation.md").read_text()
    assert "Out-of-time test (train Jul 2026 – Mar 2027, test Apr 2027 – Jun 2027)" in ev
