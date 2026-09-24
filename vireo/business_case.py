"""The numbers behind the recommendation, in one place.

A ticket is misrouted when the bot sends it to a different owning team from the one its
real category belongs to (policy §6). The three frontline teams count as one owner, because
moving work between chat and email frontline is shift/channel juggling, not a routing error.

Costs (policy §3–4): Rs 305 per transfer, Rs 350 SLA credit per first-response breach.
Transfers only exist in the current helpdesk (from 14 Sep 2025), so cost-per-misroute is
measured on helpdesk tickets only.
"""
import pandas as pd

from .load import COST_SLA_CREDIT, COST_TRANSFER
from .taxonomy import OWNER, team_group

WEEKLY_VOLUME = 650  # Vireo's stated volume (brief); the export holds ~180/week
WEEKS_PER_QUARTER = 13
TARGET_MISROUTE = 0.05  # conservative: the model is ~99% right on this data, real data will be messier
TWO_HIRES_PER_YEAR = 900_000  # Arjun: "about Rs 9 lakh a year"


def add_routing(t):
    t = t.copy()
    t["bot_owner"] = t.category.map(OWNER)
    t["true_owner"] = t.ai_category.map(OWNER)
    t["misrouted"] = t.bot_owner != t.true_owner
    t["resolved_by"] = t.agent_team.map(team_group)
    return t


def volume_shares(t):
    """Share of tickets by bot tag vs by real category, for the teams in the debate."""
    bot = t.bot_owner.value_counts(normalize=True)
    true = t.true_owner.value_counts(normalize=True)
    return pd.DataFrame({"by_bot_tag": bot, "by_real_category": true}).fillna(0).sort_values("by_real_category", ascending=False)


def cost_of_misrouting(t):
    """Excess transfers and SLA breaches on misrouted tickets, like for like.

    Each misrouted ticket is compared with correctly routed tickets of the same real owner
    (a delivery ticket sent to Billing vs a delivery ticket sent to Logistics), so the
    difference is the routing, not the kind of problem. Helpdesk-era tickets only, because
    transfers are unknown before the migration.
    """
    hd = t[t.transfers_known]
    base = hd[~hd.misrouted].groupby("true_owner")[["transfers", "sla_breach"]].mean()
    mis = hd[hd.misrouted]
    extra_transfers = (mis.transfers - mis.true_owner.map(base.transfers)).mean()
    extra_breach = (mis.sla_breach - mis.true_owner.map(base.sla_breach)).mean()
    per_ticket = extra_transfers * COST_TRANSFER + extra_breach * COST_SLA_CREDIT
    by_route = hd.groupby(["true_owner", "bot_owner"]).agg(
        tickets=("ticket_id", "size"), transfers=("transfers", "mean"), sla_breach=("sla_breach", "mean"),
        csat=("csat_score", "mean"), median_resolution_h=("resolution_h", "median")).round(3)
    return by_route, extra_transfers, extra_breach, per_ticket


def recent_misroute_rate(t, since="2026-01-01"):
    r = t[t.created_at >= since]
    return r.misrouted.mean(), len(r)


def workload_per_agent(t, agents):
    """Real workload (tickets resolved, and tickets whose real category a team owns) per agent.
    Tier 2 (Escalations & Warranty) is shown but not compared on volume (policy §6)."""
    heads = agents.groupby("team").agent_id.nunique()
    heads.index = heads.index.map(team_group)
    heads = heads.groupby(level=0).sum()
    recent = t[t.created_at >= "2026-01-01"]
    months = recent.month.nunique()
    df = pd.DataFrame({
        "agents": heads,
        "bot_routed_per_month": recent.bot_owner.value_counts() / months,
        "real_category_per_month": recent.true_owner.value_counts() / months,
        "resolved_per_month": recent.resolved_by.value_counts() / months,
    })
    df["resolved_per_agent_month"] = df.resolved_per_month / df.agents
    return df.round(1)


def summary(t, agents):
    t = add_routing(t)
    shares = volume_shares(t)
    g, x_tr, x_br, per_ticket = cost_of_misrouting(t)
    rate, n_recent = recent_misroute_rate(t)
    q_volume = WEEKLY_VOLUME * WEEKS_PER_QUARTER
    now_cost = rate * q_volume * per_ticket
    saving = (rate - TARGET_MISROUTE) * q_volume * per_ticket
    b2l = t[(t.bot_owner == "Billing") & (t.true_owner == "Logistics")]
    return {
        "tickets": len(t),
        "billing_share_bot": shares.at["Billing", "by_bot_tag"],
        "billing_share_real": shares.at["Billing", "by_real_category"],
        "logistics_share_bot": shares.at["Logistics", "by_bot_tag"],
        "logistics_share_real": shares.at["Logistics", "by_real_category"],
        "billing_queue_really_logistics": len(b2l) / (t.bot_owner == "Billing").sum(),
        "misroute_rate_all": t.misrouted.mean(),
        "misroute_rate_2026h1": rate,
        "recent_tickets": n_recent,
        "extra_transfers_per_misroute": x_tr,
        "extra_breach_rate_per_misroute": x_br,
        "cost_per_misroute_inr": per_ticket,
        "csat_misrouted": t[t.misrouted].csat_score.mean(), "csat_routed_ok": t[~t.misrouted].csat_score.mean(),
        "delivery_via_billing": g.loc[("Logistics", "Billing")].to_dict(),
        "delivery_via_logistics": g.loc[("Logistics", "Logistics")].to_dict(),
        "quarter_volume_at_650_week": q_volume,
        "misroute_cost_per_quarter_inr": now_cost,
        "target_misroute_rate": TARGET_MISROUTE,
        "saving_per_quarter_inr": saving,
        "two_hires_per_quarter_inr": TWO_HIRES_PER_YEAR / 4,
    }, shares, g, workload_per_agent(t, agents), t


# Refund codes that mean the customer already got their money back for the item itself.
# With a replacement on the same order, these look like a double payout (policy §5).
FULL_REFUND_CODES = {"RETURN-QC-OK", "DOA-REPL", "LOST-TRANSIT", "WTY-BUYBACK"}


def refund_and_replacement(t, products):
    """Orders that got both a refund and a replacement across their tickets. Policy §5: never both."""
    x = t[t.order_id.notna()]
    g = x.groupby("order_id").agg(
        tickets=("ticket_id", lambda s: " ".join(s)), sku=("product_sku", "first"),
        refund_inr=("refund_amount_inr", "sum"),
        refund_codes=("refund_reason_code", lambda s: ",".join(sorted(set(s.dropna())))),
        replaced=("replacement_issued", lambda s: (s == "Y").any()))
    g = g[(g.refund_inr > 0) & g.replaced].drop(columns="replaced")
    unit_cost = products.set_index("sku").unit_cost_inr
    g["replacement_cost_inr"] = g.sku.map(unit_cost) + 340  # policy §5 planning cost
    g["likely_double_payout"] = g.refund_codes.str.split(",").map(lambda c: bool(FULL_REFUND_CODES & set(c)))
    return g.sort_values(["likely_double_payout", "refund_inr"], ascending=False)
