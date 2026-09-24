"""HTML report + CSV exports: the monthly chart Priya asked for, drawn by the bot's tag and by
what the ticket is really about."""
import html
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .taxonomy import CATEGORIES

TEAMS = ["Frontline", "Logistics", "Billing", "Returns Desk", "Escalations & Warranty"]
# Reference categorical palette, fixed order (dataviz skill, references/palette.md)
TEAM_COLOURS = dict(zip(TEAMS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
AI_COLOUR, BOT_COLOUR = "#2a78d6", "#eb6834"
WORKLOAD_COLS = {"agents": "Agents", "bot_routed_per_month": "Routed by bot / month",
                 "real_category_per_month": "Really theirs / month", "resolved_per_month": "Resolved / month",
                 "resolved_per_agent_month": "Resolved per agent / month"}
SHARE_COLS = {"by_bot_tag": "By bot tag", "by_real_category": "By real category"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def _layout(fig, height):
    fig.update_layout(
        height=height, margin=dict(l=50, r=20, t=50, b=70), plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12, color=INK),
        legend=dict(orientation="h", yanchor="top", y=-0.08, x=0), hovermode="x unified", bargap=0.25,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    return fig


def team_chart(t):
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=("By the bot's intake tag (today's chart)", "By what the ticket is really about (AI)"))
    for col, field in [(1, "bot_owner"), (2, "true_owner")]:
        m = t.groupby(["month", field]).size().unstack(fill_value=0).reindex(columns=TEAMS, fill_value=0)
        for team in TEAMS:
            fig.add_bar(x=m.index, y=m[team], name=team, marker_color=TEAM_COLOURS[team],
                        marker_line=dict(color="white", width=1), legendgroup=team, showlegend=col == 1,
                        hovertemplate=f"{team}: %{{y}}<extra></extra>", row=1, col=col)
    fig.update_layout(barmode="stack")
    return _layout(fig, 460)


def category_chart(t):
    cats = [c for c in CATEGORIES if c in set(t.ai_category) | set(t.category)]
    rows = (len(cats) + 3) // 4
    fig = make_subplots(rows=rows, cols=4, subplot_titles=cats, shared_xaxes=True, vertical_spacing=0.08)
    ai = t.groupby(["month", "ai_category"]).size().unstack(fill_value=0).reindex(columns=cats, fill_value=0)
    bot = t.groupby(["month", "category"]).size().unstack(fill_value=0).reindex(columns=cats, fill_value=0)
    for i, c in enumerate(cats):
        r, col = i // 4 + 1, i % 4 + 1
        fig.add_scatter(x=ai.index, y=ai[c], name="AI category", line=dict(color=AI_COLOUR, width=2),
                        legendgroup="ai", showlegend=i == 0, row=r, col=col)
        fig.add_scatter(x=bot.index, y=bot[c], name="Bot tag", line=dict(color=BOT_COLOUR, width=2, dash="dot"),
                        legendgroup="bot", showlegend=i == 0, row=r, col=col)
    fig.update_annotations(font=dict(size=12, color=INK))
    fig.update_xaxes(showticklabels=False)
    return _layout(fig, 220 * rows)


def _table(df, fmt=None):
    return df.to_html(classes="tbl", float_format=fmt or (lambda v: f"{v:,.1f}"), border=0)


def write(t, s, shares, by_route, workload, eval_res, audit_res, out="out", llm_usage=None):
    out = Path(out)
    out.mkdir(exist_ok=True)
    cols = ["ticket_id", "created_at", "month", "channel", "category", "ai_category", "ai_confidence",
            "bot_owner", "true_owner", "misrouted", "assigned_team", "agent_team", "status"]
    t[cols].to_csv(out / "tickets_categorised.csv", index=False)
    t.groupby(["month", "ai_category"]).size().unstack(fill_value=0).to_csv(out / "monthly_by_category.csv")
    t.groupby(["month", "true_owner"]).size().unstack(fill_value=0).to_csv(out / "monthly_by_team.csv")

    pct = lambda v: f"{v:.0%}"
    inr = lambda v: f"Rs {v / 1e5:.1f} lakh"
    tiles = [
        ("Billing share of tickets", f"{pct(s['billing_share_bot'])} → {pct(s['billing_share_real'])}", "bot tag → real"),
        ("Logistics share of tickets", f"{pct(s['logistics_share_bot'])} → {pct(s['logistics_share_real'])}", "bot tag → real"),
        ("Billing queue that is really delivery", pct(s["billing_queue_really_logistics"]), "Jan 2025 – Jun 2026"),
        ("Tickets sent to the wrong team", pct(s["misroute_rate_2026h1"]), "Jan – Jun 2026"),
        ("Cost of misrouting", inr(s["misroute_cost_per_quarter_inr"]) + " / qtr", "at 650 tickets/week"),
    ]
    tiles_html = "".join(f'<div class="tile"><div class="k">{html.escape(k)}</div><div class="v">{v}</div>'
                         f'<div class="n">{html.escape(n)}</div></div>' for k, v, n in tiles)

    d_bill, d_log = s["delivery_via_billing"], s["delivery_via_logistics"]
    route_rows = "".join(
        f"<tr><th>{name}</th><td>{int(d['tickets']):,}</td><td>{d['transfers']:.2f}</td><td>{d['sla_breach']:.0%}</td>"
        f"<td>{d['csat']:.2f}</td><td>{d['median_resolution_h']:.0f} h</td></tr>"
        for name, d in [("Bot sent it to Billing", d_bill), ("Bot sent it to Logistics", d_log)])

    a = audit_res[0] if audit_res else None
    audit_line = (f"On a hand-checked random sample of {a['audited_tickets']} tickets the AI category was right "
                  f"{a['ai_correct']}/{a['audited_tickets']}; the bot's tag {a['bot_tag_correct']}/{a['audited_tickets']}."
                  if a else "Hand audit not available.")
    llm_line = (f"<p>Local LLM review: {llm_usage['tickets_reviewed']} low-confidence tickets re-checked with "
                f"{llm_usage['model']} (free, on this machine); it changed {llm_usage['changed']} of them.</p>" if llm_usage else "")

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vireo Ticket Volume</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root {{ color-scheme: light; --ink:{INK}; --muted:{MUTED}; --line:{GRID}; --bg:#fcfcfb; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:26px; margin:0 0 4px; }} h2 {{ font-size:19px; margin:36px 0 6px; }}
.sub {{ color:var(--muted); margin:0 0 20px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; }}
.tile {{ background:white; border:1px solid var(--line); border-radius:8px; padding:12px 14px; }}
.tile .k {{ color:var(--muted); font-size:13px; }} .tile .v {{ font-size:24px; font-weight:650; }}
.tile .n {{ color:var(--muted); font-size:12px; }}
.card {{ background:white; border:1px solid var(--line); border-radius:8px; padding:8px; overflow-x:auto; }}
.tbl, table.plain {{ border-collapse:collapse; font-size:14px; background:white; }}
.tbl th, .tbl td, table.plain th, table.plain td {{ border-bottom:1px solid var(--line); padding:6px 10px; text-align:right; }}
.tbl th:first-child, table.plain th:first-child {{ text-align:left; }}
p, li {{ max-width:75ch; }} .note {{ color:var(--muted); font-size:13px; }}
</style></head><body><main>
<h1>Vireo Audio — where the support work really goes</h1>
<p class="sub">{s['tickets']:,} tickets, Jan 2025 – Jun 2026, after data fixes. Categories are read from the customer's
opening message by a model trained on agents' closing notes.</p>
<div class="tiles">{tiles_html}</div>

<h2>Monthly tickets by owning team</h2>
<p>Left is today's view: the chat bot's intake tag. Right is the same tickets by what they turned out to be about.
Billing shrinks and Logistics grows, mainly from “I paid but nothing arrived” messages the bot reads as a payment problem.</p>
<div class="card">{team_chart(t).to_html(full_html=False, include_plotlyjs=False)}</div>

<h2>Monthly tickets by category — AI vs bot tag</h2>
<p>“Order Changes” (cancel, address, dispatch) is a new category; the bot files these under “Other”.</p>
<div class="card">{category_chart(t).to_html(full_html=False, include_plotlyjs=False)}</div>

<h2>What a misroute costs: delivery tickets, helpdesk era (Sep 2025 – Jun 2026)</h2>
<table class="plain"><tr><th></th><th>Tickets</th><th>Transfers / ticket</th><th>SLA breached</th><th>CSAT</th><th>Median time to resolve</th></tr>{route_rows}</table>
<p class="note">Across all categories a misrouted ticket carries {s['extra_transfers_per_misroute']:.2f} extra transfers and
{s['extra_breach_rate_per_misroute']:.0%} extra SLA breaches versus correctly routed tickets of the same kind,
≈ Rs {s['cost_per_misroute_inr']:.0f} per ticket at policy rates (Rs 305 per transfer, Rs 350 per breach).</p>

<h2>Workload per agent, Jan – Jun 2026 (tickets per month in this export)</h2>
{_table(workload.rename(columns=WORKLOAD_COLS))}
<p class="note">Escalations &amp; Warranty is Tier 2 and measured on days to resolve, not volume (policy §6). The export holds
about 180 tickets a week against the stated 650, so read these as relative, not absolute.</p>

<h2>Share of tickets by owning team</h2>
{_table((shares * 100).rename(columns=SHARE_COLS), lambda v: f"{v:.1f}%")}

<h2>How we know it's right</h2>
<p>Out-of-time test (trained to Mar 2026, tested Apr – Jun 2026): AI category agrees with the agent's closing note on
{eval_res['ai_category_accuracy']:.1%} of {eval_res['test_tickets']:,} tickets; the bot's tag agrees on
{eval_res['bot_category_accuracy']:.1%}. {audit_line} Full detail in <code>out/evaluation.md</code>.</p>
{llm_line}
</main></body></html>"""
    (out / "report.html").write_text(page)
    return out / "report.html"
