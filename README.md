# Vireo Audio — support ticket categoriser

Reads each support ticket's opening message, works out what the ticket is really about, and
redraws the monthly volume chart by category and by owning team. It then shows how often the
chat bot sends tickets to the wrong team, and what that costs.

**Headline (Set E, Jan 2025 – Jun 2026, 11,641 tickets):** Billing is 21% of tickets by the
bot's tag but 14% by what customers actually need. Logistics goes from 16% to 26%. 39% of the
Billing queue is "I paid but my order hasn't arrived". Those tickets are transferred, breach
first-response SLA 4x as often, and score 0.8 lower on CSAT.

## Run it

Needs Python 3.10+. No API key, no paid calls, no LLM. About 80 seconds and 0.7 GB of RAM on an 8 GB laptop.

```bash
git clone <this repo> && cd vireo-support
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Copy the data pack into `data/`. The files can keep their original `<uuid>-tickets.csv` names:

```
data/tickets.csv  data/agents.csv  data/products.csv  (orders, customers not needed)
```

```bash
python run.py            # full pipeline -> out/report.html
python run.py --audit    # print the evidence for each data fix
python run.py --data other/ --start 2026-07-01 --end 2027-01-01   # a different export: give its window
python run.py --llm        # optional: re-check low-confidence tickets with a free local LLM (see below)
python run.py --benchmark  # optional: measure local LLMs against the default model
```

Open `out/report.html` in a browser (it loads Plotly from a CDN, so it needs internet).

## What comes out

| File | What it is |
|---|---|
| `out/report.html` | The chart Priya asked for, drawn two ways, plus misroute cost and workload per agent |
| `out/evaluation.md` | Out-of-time accuracy, confusion matrix, every disagreement, hand-audit results |
| `out/tickets_categorised.csv` | One row per ticket: bot tag, AI category, confidence, owning team both ways, misrouted flag |
| `out/refund_and_replacement.csv` | Orders that got both a refund and a replacement (policy §5 says never both), for Finance |
| `out/monthly_by_category.csv`, `out/monthly_by_team.csv` | The chart data |

## How it works

```
tickets.csv ─► load.py        data fixes (UTC legacy timestamps, out-of-window rows, blank ≠ 0 transfers, agent_id joins)
            ─► labels.py      reference label from the agent's closing note (hindsight: what the ticket really was)
            ─► classify.py    model reads ONLY the customer's opening message (what the bot has at intake)
                              TF-IDF words + char n-grams → linear SVM, trained on messages + training tickets' notes
                              out-of-fold predictions; confidence = margin between the top two categories
                              [--llm] low-confidence tickets get a second opinion from a local LLM (experimental)
            ─► robustness.py  harder test: phrasings the model never saw, clean and with live-chat noise
            ─► evaluate.py    out-of-time test + 150-ticket hand audit (eval/audit_labels.csv)
            ─► business_case.py  misroute rate, like-for-like cost, workload per agent
            ─► report.py      HTML + CSVs
```

Categories are Vireo's own 11, plus **Order Changes** (cancel, address or pincode change,
dispatch status). The bot files these under "Other".

## Tests

```bash
pip install -r requirements-dev.txt
pytest            # 77 fast tests on synthetic data (~45 s)
pytest -m slow    # 7 regression + resource tests on the real pack (~90 s)
```

What's covered and what testing found: `docs/TESTING.md`.

## Optional experiment: free local LLM (not recommended)

Measured on this data it is less accurate than the default model and ~1,000x slower (`docs/LLM_DECISION.md`). It is kept only so the result can be reproduced. It runs on your machine through [Ollama](https://ollama.com), with no API key and no cost. On 8 GB of RAM, use a 3B model: models over 35% of RAM are refused, and the model is unloaded after the run.

```bash
brew install ollama          # or the installer from ollama.com
ollama serve &               # starts the local server on port 11434
ollama pull qwen2.5:3b       # ~1.9 GB, once
python run.py --llm
```

Use `VIREO_LLM_MODEL=<name>` to try another model. See `docs/LLM_DECISION.md` for the measured results.

## Docs

- `docs/MEMO.md` — one-page memo to Priya Raman
- `docs/DECISIONS.md` — every judgement call, and why
- `docs/submission-form.md` — the completed submission form
- `docs/PLAN.md` — the plan as written at the start
- `docs/RECORDING.md` — script for the 3-minute screen recording
- `docs/LLM_DECISION.md` — local LLM vs the default model: measured accuracy, latency, memory
- `docs/MODEL_IMPROVEMENT.md` — the harder test, the 26 candidates tried, and why the current model won
- `docs/TESTING.md` — the test suite, and the 12 problems it found

## Data

The client data pack is **not** in this repo (it contains customer names and messages). `data/`
and `out/` are git-ignored. `eval/audit_labels.csv` holds only ticket IDs and labels.
