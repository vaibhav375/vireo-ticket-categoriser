# Screen recording script (≤ 3 minutes)

The brief asks for: the prompts used, what changed between versions, what was thrown away. No slides.
Screens to have open: the Claude Code session, the terminal, `out/report.html`, and `git log --oneline`.

**0:00–0:25 — The ask and the answer** (show `out/report.html`, top)
"Priya asked for a monthly chart by category and team, with two hires going to the biggest team.
Here's her chart on the left, by the bot's tag. Billing looks biggest. On the right, the same tickets by what the
customer actually needed: Billing drops from 21% to 14%, and Logistics rises from 16% to 26%."

**0:25–1:05 — Prompts used** (show the Claude Code session, then `vireo/classify.py` → `SYSTEM_PROMPT`)
- "My opening prompt to Claude Code was the full brief plus the submission questions, asking it to plan first
  and commit each step." (scroll the first message)
- "The first thing it did was profile the data. That's where the pattern showed: 31% of Billing tickets were
  closed by Logistics agents."
- "The only prompt inside the tool is this one: the optional local-LLM step for low-confidence tickets. It lists
  the categories and has one explicit rule: paid-but-not-arrived is Delivery, not Billing. The output is
  constrained to the category list with a JSON schema." (show `docs/LLM_DECISION.md` table)

**1:05–1:55 — What changed between versions** (show `git log --oneline`, then `out/evaluation.md`)
- "Version 1 of the labelling took the first keyword in the agent's note. The out-of-time test showed
  'wrong product delivered' labelled as Order Changes. The notes said 'Replacement unit dispatched. Issue: …',
  and 'pair' was matching inside 'repair'. Version 2 reads from the 'Issue:' marker. Agreement went from 98.6% to 99.9%.
  I tuned that after seeing the test errors, so I lean on the 150-ticket audit, not the 99.9%."
- "Cost per misroute: v1 compared misrouted tickets with all tickets. v2 compares like for like: a
  delivery ticket via Billing vs one sent straight to Logistics. 36% SLA breach vs 9%."
- "The memo: v1 said the audit was 'by hand'. It was AI-labelled, so I corrected that. It also claimed leads in
  defects and shifts before I'd checked. I checked, found nothing, and cut them."

**1:55–2:35 — What I threw away** (show `docs/submission-form.md` → "Thrown away")
- "Misroute = resolving team ≠ assigned team. Too noisy: Billing agents resolve some delivery tickets themselves."
- "The LLM. First a Claude API fallback, then a free local model through Ollama. I benchmarked three local models:
  84–87% vs 100% for the plain model, and 1,000x slower. A 7B model even froze my laptop. So it's off by default."
- "A 12-colour stacked chart. Nobody can tell 12 colours apart, so it became small multiples."

**2:35–3:00 — Close** (terminal: `python run.py`, show it finish in ~40s)
"One command, about 40 seconds, no paid calls, no LLM needed. The goal: cut wrong-team tickets from 17% to 5%, about Rs 1.5 lakh a quarter.
Hold the hires eight weeks; if they're still needed, Logistics, not Billing."
