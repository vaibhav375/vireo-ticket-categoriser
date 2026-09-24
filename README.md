# Vireo Audio — support ticket categoriser

Reads each support ticket's opening message, works out what the ticket is really about, and
redraws the monthly volume chart by category and by owning team. It then shows how often the
chat bot sends tickets to the wrong team, and what that costs.

**Headline (Set E, Jan 2025 – Jun 2026, 11,641 tickets):** Billing is 21% of tickets by the
bot's tag but 14% by what customers actually need. Logistics goes from 16% to 26%. 39% of the
Billing queue is "I paid but my order hasn't arrived". Those tickets are transferred, breach
first-response SLA 4x as often, and score 0.8 lower on CSAT.

## Run it

Needs Python 3.10+. No API key, no paid calls, about 30 seconds on a laptop.

```bash
git clone <this repo> && cd vireo-support
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Copy the data pack into `data/`. The files can keep their original `<uuid>-tickets.csv` names:

```
data/tickets.csv  data/agents.csv  (orders, customers, products optional)
```

```bash
python run.py            # full pipeline -> out/report.html
python run.py --audit    # print the evidence for each data fix
python run.py --llm      # optional: re-check low-confidence tickets with Claude (needs ANTHROPIC_API_KEY)
```

Open `out/report.html` in a browser (it loads Plotly from a CDN, so it needs internet).

## What comes out

| File | What it is |
|---|---|
| `out/report.html` | The chart Priya asked for, drawn two ways, plus misroute cost and workload per agent |
| `out/evaluation.md` | Out-of-time accuracy, confusion matrix, every disagreement, hand-audit results |
| `out/tickets_categorised.csv` | One row per ticket: bot tag, AI category, confidence, owning team both ways, misrouted flag |
| `out/monthly_by_category.csv`, `out/monthly_by_team.csv` | The chart data |

## How it works

```
tickets.csv ─► load.py        data fixes (UTC legacy timestamps, out-of-window rows, blank ≠ 0 transfers, agent_id joins)
            ─► labels.py      reference label from the agent's closing note (hindsight: what the ticket really was)
            ─► classify.py    model reads ONLY the customer's opening message (what the bot has at intake)
                              TF-IDF words + char n-grams → logistic regression, out-of-fold predictions
                              [--llm] low-confidence tickets (<0.6, about 0.3%) get a Claude second opinion
            ─► evaluate.py    out-of-time test + 150-ticket hand audit (eval/audit_labels.csv)
            ─► business_case.py  misroute rate, like-for-like cost, workload per agent
            ─► report.py      HTML + CSVs
```

Categories are Vireo's own 11, plus **Order Changes** (cancel, address or pincode change,
dispatch status). The bot files these under "Other".

## Docs

- `docs/MEMO.md` — one-page memo to Priya Raman
- `docs/DECISIONS.md` — every judgement call, and why
- `docs/submission-form.md` — the completed submission form
- `docs/PLAN.md` — the plan as written at the start
- `docs/RECORDING.md` — script for the 3-minute screen recording

## Data

The client data pack is **not** in this repo (it contains customer names and messages). `data/`
and `out/` are git-ignored. `eval/audit_labels.csv` holds only ticket IDs and labels.
