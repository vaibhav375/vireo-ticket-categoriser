"""Data fixes in vireo/load.py, each against a hand-built ticket."""
import math

import pandas as pd
import pytest

from synthetic import make_tickets, write_pack
from vireo.load import load


def one_ticket(**overrides):
    row = make_tickets(n=1, out_of_window=0).iloc[0].to_dict()
    row.update(overrides)
    return pd.DataFrame([row])


def test_legacy_resolved_at_is_shifted_from_utc_to_ist(tmp_path):
    t = one_ticket(created_at="2025-03-01 10:00", first_response_at="2025-03-01 10:05",
                   resolved_at="2025-03-01 06:30", source_system="legacy_fd")
    out = load(write_pack(tmp_path, t))
    assert out.resolved_at.iloc[0] == pd.Timestamp("2025-03-01 12:00")
    assert out.resolution_h.iloc[0] == 2.0


def test_helpdesk_resolved_at_is_not_shifted(tmp_path):
    t = one_ticket(created_at="2025-10-01 10:00", first_response_at="2025-10-01 10:05",
                   resolved_at="2025-10-01 12:00", source_system="helpdesk")
    out = load(write_pack(tmp_path, t))
    assert out.resolved_at.iloc[0] == pd.Timestamp("2025-10-01 12:00")


def test_rows_outside_window_are_dropped(tmp_path):
    t = pd.concat([one_ticket(ticket_id="IN", created_at="2025-01-01 00:00"),
                   one_ticket(ticket_id="BEFORE", created_at="2024-12-31 23:59"),
                   one_ticket(ticket_id="AFTER", created_at="2026-07-01 00:00")])
    out = load(write_pack(tmp_path, t))
    assert out.ticket_id.tolist() == ["IN"]


def test_blank_legacy_transfers_stay_unknown_not_zero(tmp_path):
    t = one_ticket(created_at="2025-03-01 10:00", transfers=None, source_system="legacy_fd")
    out = load(write_pack(tmp_path, t))
    assert math.isnan(out.transfers.iloc[0])
    assert not out.transfers_known.iloc[0]


def test_agents_with_same_name_resolve_to_their_own_team(tmp_path):
    t = pd.concat([one_ticket(ticket_id="X", agent_id="A2"), one_ticket(ticket_id="Y", agent_id="A3")])
    out = load(write_pack(tmp_path, t)).set_index("ticket_id")
    assert out.at["X", "agent_team"] == "Chat Frontline"  # A2 Om Sharma
    assert out.at["Y", "agent_team"] == "Logistics"       # A3 Om Sharma


def test_roster_row_active_on_ticket_date_is_used(tmp_path):
    agents = pd.DataFrame([
        {"agent_id": "A9", "name": "Moved", "site": "Indore", "team": "Billing", "shift": "Day", "tier": 1,
         "from_date": "2020-01-01", "to_date": "2025-06-30"},
        {"agent_id": "A9", "name": "Moved", "site": "Bengaluru", "team": "Logistics", "shift": "Day", "tier": 1,
         "from_date": "2025-07-01", "to_date": None}])
    t = pd.concat([one_ticket(ticket_id="EARLY", agent_id="A9", created_at="2025-03-01 10:00"),
                   one_ticket(ticket_id="LATE", agent_id="A9", created_at="2025-11-01 10:00")])
    out = load(write_pack(tmp_path, t, agents=agents)).set_index("ticket_id")
    assert out.at["EARLY", "agent_team"] == "Billing"
    assert out.at["LATE", "agent_team"] == "Logistics"


def test_sla_breach_uses_channel_target(tmp_path):
    t = pd.concat([  # chat target 15 min, email 8 h
        one_ticket(ticket_id="CHAT_LATE", channel="chat", created_at="2025-10-01 10:00", first_response_at="2025-10-01 10:16"),
        one_ticket(ticket_id="CHAT_OK", channel="chat", created_at="2025-10-01 10:00", first_response_at="2025-10-01 10:15"),
        one_ticket(ticket_id="EMAIL_OK", channel="email", created_at="2025-10-01 10:00", first_response_at="2025-10-01 17:00")])
    out = load(write_pack(tmp_path, t)).set_index("ticket_id").sla_breach
    assert out.to_dict() == {"CHAT_LATE": True, "CHAT_OK": False, "EMAIL_OK": False}


def test_pack_files_with_uuid_prefix_are_found(tmp_path):
    folder = write_pack(tmp_path / "d", uuid_names=True)
    assert len(load(folder)) == 660


def test_missing_pack_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="tickets.csv not found"):
        load(tmp_path)
