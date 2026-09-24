"""Load the Vireo data pack and apply the data fixes, each one documented.

Every fix here comes from something in the README, support-policy.pdf or the email
thread, confirmed against the data. `audit()` prints the evidence for each.
"""
from pathlib import Path

import pandas as pd

WINDOW_START = pd.Timestamp("2025-01-01")
WINDOW_END = pd.Timestamp("2026-07-01")  # exclusive
LEGACY_UTC_OFFSET = pd.Timedelta(minutes=330)  # IST = UTC+5:30

# Policy §3: first-response targets, in hours
SLA_HOURS = {"chat": 0.25, "voice": 2, "social": 4, "email": 8}

# Policy §4 (FY26 planning figures), rupees
COST_PER_CONTACT = {"chat": 210, "email": 260, "voice": 520, "social": 240}
COST_TRANSFER = 305
COST_SLA_CREDIT = 350
COST_AGENT_HOUR = 165

FRONTLINE = {"Chat Frontline", "Email Frontline", "Voice Frontline"}

REQUIRED_COLUMNS = [
    "ticket_id", "created_at", "first_response_at", "resolved_at", "status", "channel", "customer_id", "order_id",
    "product_sku", "category", "agent_id", "transfers", "csat_score", "refund_amount_inr", "refund_reason_code",
    "replacement_issued", "customer_message", "agent_notes", "source_system",
]


class DataError(ValueError):
    """The export can't be used as-is. The message says what to fix."""


def _find(data_dir: Path, name: str) -> Path:
    """Accept both `tickets.csv` and the pack's `<uuid>-tickets.csv` naming."""
    hits = sorted(data_dir.glob(f"*{name}"))
    if not hits:
        raise FileNotFoundError(f"{name} not found in {data_dir}. Copy the data pack there (see README).")
    return hits[0]


def load_raw(data_dir="data"):
    d = Path(data_dir)
    tickets = pd.read_csv(_find(d, "tickets.csv"))
    agents = pd.read_csv(_find(d, "agents.csv"))
    return tickets, agents


def load(data_dir="data", keep_out_of_window=False, start=None, end=None):
    """Return cleaned tickets joined to the resolving agent's team.

    start/end (end exclusive) default to the window the Set E README states; pass the new
    export's window for any other export.
    """
    t, agents = load_raw(data_dir)
    missing = sorted(set(REQUIRED_COLUMNS) - set(t.columns))
    if missing:
        raise DataError(f"tickets.csv is missing columns: {', '.join(missing)}")

    dup = t.ticket_id.duplicated()
    if dup.any():
        print(f"Warning: {int(dup.sum())} duplicate ticket_id rows dropped (kept the first copy)")
        t = t[~dup].copy()

    _parse_times(t)

    # Fix 1 — policy §9: legacy resolution times were rebuilt from a UTC event log,
    # while every other timestamp is IST. Without this, 2,379 legacy tickets resolve
    # before they were created.
    legacy = t.source_system.eq("legacy_fd")
    t.loc[legacy, "resolved_at"] = t.loc[legacy, "resolved_at"] + LEGACY_UTC_OFFSET

    # Fix 2 — README says the export covers Jan 2025 – Jun 2026; 139 tickets from 2024
    # are in the file anyway. They are dropped from all counts.
    start = pd.Timestamp(start) if start is not None else WINDOW_START
    end = pd.Timestamp(end) if end is not None else WINDOW_END
    t["in_window"] = t.created_at.between(start, end, inclusive="left")
    if not t.in_window.any():
        raise DataError(
            f"no tickets between {start:%Y-%m-%d} and {end:%Y-%m-%d}; this export runs "
            f"{t.created_at.min():%Y-%m-%d} to {t.created_at.max():%Y-%m-%d}. Set its window with --start/--end.")
    if not keep_out_of_window:
        t = t[t.in_window].copy()

    # Fix 3 — Sameer: transfers didn't exist in Freshdesk. Blank means unknown, not zero.
    # pandas already reads blanks as NaN; we only make sure nothing fills them.
    t["transfers_known"] = t.transfers.notna()

    # Fix 4 — two agents share a display name ("Om Sharma"). Join on agent_id only.
    # The roster can have several rows per agent (site/shift moves), so pick the row
    # active on the ticket date. Set E has one row per agent, but the join is date-aware anyway.
    t = _attach_agent(t, agents)

    # Derived fields
    t["month"] = t.created_at.dt.to_period("M").astype(str)
    t["first_response_h"] = (t.first_response_at - t.created_at).dt.total_seconds() / 3600
    t["resolution_h"] = (t.resolved_at - t.created_at).dt.total_seconds() / 3600
    t["sla_breach"] = t.first_response_h > t.channel.map(SLA_HOURS)
    t["attended"] = t.status.isin(["resolved", "closed"])  # policy §10
    t["customer_message"] = t.customer_message.fillna("")
    t["agent_notes"] = t.agent_notes.fillna("")
    return t.reset_index(drop=True)


def last_months(t, n):
    """Start of the last n calendar months in the data (the latest month counts as one)."""
    latest = t.created_at.max().to_period("M")
    return (latest - (n - 1)).to_timestamp()


def period_label(start, end_inclusive):
    return f"{start:%b %Y} – {end_inclusive:%b %Y}"


def _parse_times(t):
    """created_at must be readable (every count depends on it). Other times may be unknown."""
    for c in ["created_at", "first_response_at", "resolved_at"]:
        parsed = pd.to_datetime(t[c], errors="coerce", format="mixed")
        bad = t[c].notna() & parsed.isna()
        if bad.any():
            example = t.loc[bad].iloc[0]
            if c == "created_at":
                raise DataError(f"{int(bad.sum())} rows have an unreadable created_at, "
                                f"e.g. {example.ticket_id}: '{example[c]}'")
            print(f"Warning: {int(bad.sum())} unreadable {c} values treated as unknown "
                  f"(e.g. {example.ticket_id}: '{example[c]}')")
        t[c] = parsed


def _attach_agent(t, agents):
    a = agents.copy()
    a["from_date"] = pd.to_datetime(a.from_date)
    a["to_date"] = pd.to_datetime(a.to_date).fillna(pd.Timestamp("2099-01-01"))
    m = t[["ticket_id", "agent_id", "created_at"]].merge(a, on="agent_id", how="left")
    active = m[(m.created_at >= m.from_date) & (m.created_at <= m.to_date)]
    # Fall back to any roster row if no row covers the date (e.g. roster dates start later).
    best = pd.concat([active, m]).drop_duplicates("ticket_id")
    best = best.set_index("ticket_id")
    t = t.copy()
    t["agent_team"] = t.ticket_id.map(best["team"])
    t["agent_site"] = t.ticket_id.map(best["site"])
    t["agent_tier"] = t.ticket_id.map(best["tier"])
    return t


def audit(data_dir="data"):
    """Print the evidence behind each fix, plus the things checked and not found."""
    raw, agents = load_raw(data_dir)
    for c in ["created_at", "resolved_at"]:
        raw[c] = pd.to_datetime(raw[c])
    res_h = (raw.resolved_at - raw.created_at).dt.total_seconds() / 3600
    print("Rows in tickets.csv:", len(raw))
    print("\n[1] Resolved before created, by source system (before fix):")
    print((res_h < 0).groupby(raw.source_system).sum().to_string())
    fixed = res_h + raw.source_system.eq("legacy_fd") * 5.5
    print("    after +5:30 on legacy_fd:", int((fixed < 0).sum()))

    out = ~raw.created_at.between(WINDOW_START, WINDOW_END, inclusive="left")
    print(f"\n[2] Tickets outside Jan 2025 – Jun 2026: {int(out.sum())} "
          f"(earliest {raw.created_at.min():%Y-%m-%d})")

    print("\n[3] transfers blank by source system:")
    print(raw.transfers.isna().groupby(raw.source_system).mean().round(3).to_string())

    dup_names = agents[agents.name.duplicated(keep=False)][["agent_id", "name", "team"]]
    print("\n[4] Agents sharing a display name:\n" + dup_names.to_string(index=False))
    print("    roster rows per agent (max):", agents.agent_id.value_counts().max())

    # Checked, not present in Set E
    key = ["customer_id", "product_sku", "customer_message"]
    print("\n[x] Same customer + SKU + message appearing twice:", int(raw.duplicated(key).sum()))
    print("    ticket_id duplicated:", int(raw.ticket_id.duplicated().sum()))
    ref = raw.groupby("source_system").refund_amount_inr.median()
    print("    median refund by system (paise would show up as ~100x):", ref.to_dict())


if __name__ == "__main__":
    audit()
