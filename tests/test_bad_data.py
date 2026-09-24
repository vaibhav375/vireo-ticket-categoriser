"""Malformed or unexpected exports: the pipeline either copes or stops with a clear message."""
import numpy as np
import pandas as pd
import pytest

from synthetic import make_tickets, write_pack
from vireo import business_case as bc
from vireo.classify import categorise
from vireo.labels import add_reference_labels
from vireo.load import load, load_raw


def run_core(folder, **kw):
    t, _ = categorise(add_reference_labels(load(folder, **kw)))
    return bc.summary(t, load_raw(folder)[1])[0]


def test_missing_required_column_is_named(tmp_path):
    t = make_tickets().drop(columns=["agent_notes", "channel"])
    with pytest.raises(ValueError, match="missing columns: agent_notes, channel"):
        load(write_pack(tmp_path, t))


def test_unreadable_created_at_is_reported_with_an_example(tmp_path):
    t = make_tickets().astype({"created_at": str})
    t.loc[5, "created_at"] = "not a date"
    with pytest.raises(ValueError, match=r"created_at.*TK-00005.*not a date"):
        load(write_pack(tmp_path, t))


def test_unreadable_resolution_time_is_treated_as_unknown(tmp_path, capsys):
    t = make_tickets().astype({"resolved_at": str})
    t.loc[5, "resolved_at"] = "31/31/2025"
    out = load(write_pack(tmp_path, t)).set_index("ticket_id")
    assert pd.isna(out.at["TK-00005", "resolved_at"])
    assert "1 unreadable resolved_at" in capsys.readouterr().out


def test_duplicate_ticket_ids_keep_the_first_copy(tmp_path, capsys):
    t = make_tickets()
    dup = t.iloc[[3, 4]].assign(customer_message="second copy")
    out = load(write_pack(tmp_path, pd.concat([t, dup])))
    assert len(out) == 660 and not out.ticket_id.duplicated().any()
    assert "second copy" not in set(out.customer_message)
    assert "2 duplicate ticket_id rows dropped" in capsys.readouterr().out


def test_blank_messages_and_notes_do_not_break_the_pipeline(tmp_path):
    t = make_tickets()
    t.loc[::7, "customer_message"] = np.nan
    t.loc[::5, "agent_notes"] = np.nan
    s = run_core(write_pack(tmp_path, t))
    assert s["tickets"] == 660


def test_unknown_agent_and_channel_do_not_break_the_pipeline(tmp_path):
    t = make_tickets()
    t.loc[::9, "agent_id"] = "A999"       # not in the roster
    t.loc[::11, "channel"] = "whatsapp"   # no SLA target defined
    folder = write_pack(tmp_path, t)
    out = load(folder)
    assert out.loc[out.channel == "whatsapp", "sla_breach"].eq(False).all()
    assert run_core(folder)["tickets"] == 660


def test_export_from_another_period_explains_how_to_set_the_window(tmp_path):
    t = make_tickets(start="2026-07-01", end="2027-06-30", out_of_window=0)
    with pytest.raises(ValueError, match="--start"):
        load(write_pack(tmp_path, t))


def test_export_from_another_period_runs_with_its_own_window(tmp_path):
    t = make_tickets(start="2026-07-01", end="2027-06-30", out_of_window=0)
    s = run_core(write_pack(tmp_path, t), start="2026-07-01", end="2027-07-01")
    assert s["tickets"] == 660
    assert 0 < s["misroute_rate_recent"] < 1
