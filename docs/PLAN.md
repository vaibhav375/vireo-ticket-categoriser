# Plan — Vireo Audio support tickets (Set E)

Time cap: 5 hours. Anything that doesn't fit is listed under "Left out" with a reason.

## What the client asked vs what the data says

Priya asked for a monthly chart of volume by category and by team, auto-categorised,
so the two hires go to "the biggest team". She expects Billing (22% of tickets).

A first look at the data shows the 22% is a count of the chat bot's intake tag, not of work:

- `assigned_team` is a straight mapping of the bot's `category` (Billing & Payments -> Billing, etc.).
- 31% of tickets tagged Billing are resolved by a Logistics agent. They read like
  "I paid, nothing arrived", and the bot hears "paid" and tags Billing.
- Misrouted tickets breach first-response SLA about twice as often (19.6% vs 9.4%), score about
  0.55 lower on CSAT and carry about 0.9 transfers each. Each transfer costs Rs 305 and each breach Rs 350 (policy §3–4).

So the deliverable is the chart Priya asked for, drawn two ways: by the bot's tag and by
what the ticket is actually about. We then state the headcount answer and the process fix
that Arjun (Finance) asked about.

## Steps (one commit each)

1. Scaffold: README, requirements, .gitignore (client data is not committed).
2. Load and clean the data (`vireo/load.py`). Every fix is documented and applied once:
   - legacy_fd `resolved_at` is UTC -> +5:30 (2,379 legacy tickets resolve "before" they open)
   - keep 1 Jan 2025 – 30 Jun 2026 only (139 tickets from 2024 are outside the stated window)
   - legacy `transfers` stays blank, never zero
   - join agents on `agent_id`, never the name (two agents are both called "Om Sharma")
   - checked and not found in Set E: cross-system duplicates, legacy money units
3. Reference labels from agent closing notes (`vireo/labels.py`): rules that read what the agent
   actually did. These are hindsight labels, used for training and evaluation only.
4. Classifier (`vireo/classify.py`): reads only the customer's opening message (what is
   known at intake) and predicts the real category and owning team. TF-IDF + logistic regression,
   runs offline for free. An optional Claude Haiku pass re-checks low-confidence tickets when an API key is set.
5. Evaluation (`vireo/evaluate.py`): held-out accuracy against the note labels, a confusion
   matrix, plus a hand-audited sample of 150 tickets to measure how wrong the reference labels are.
6. Report (`vireo/report.py`): an HTML page with monthly charts by category and by team,
   bot tag vs AI category, workload per agent, and misroute cost.
7. Business case (`vireo/business_case.py`): the numbers behind the goal, one place.
8. Memo to Priya (one page, non-technical).
9. Submission form, decisions log, recording script.

## Left out (on purpose)

- Real-time integration with the helpdesk. Out of scope for an evaluation, and Sameer would need to be involved.
- Lot-code / product-defect analysis and refund-policy audits. Interesting, but they don't answer
  the headcount question. Flagged as leads only.
- Staffing model by shift and hour (Erlang etc.). The export is roughly a 28% sample of the
  stated 650 tickets/week, so absolute staffing from it would be false precision.
