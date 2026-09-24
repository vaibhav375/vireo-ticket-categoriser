# Submission form — Vireo Audio, Set E

### What did you build, and what business outcome does it move? *State the number and the money.*

A ticket categoriser that reads the customer's opening message (the only thing the chat bot has
at intake) and assigns the category and owning team it really belongs to. It then redraws Priya's
monthly chart by category and team, both by the bot's tag and by real category, and measures misrouting.

**Business goal:** cut tickets sent to the wrong team first from **17% to 5%**. That is worth about
**Rs 1.5 lakh a quarter** in avoided transfers (Rs 305 each) and first-response SLA credits
(Rs 350 each) at Vireo's 650 tickets/week, out of about Rs 2.1 lakh a quarter misrouting costs today.
The bigger outcome is the headcount decision. Billing is 21% of tickets by tag but **14%** by real
need; Logistics is 16% → **26%**. The two hires (Rs 9 lakh/year) should not go to Billing. Fix
routing first, then hire into Logistics if still needed.

Arithmetic: 650 × 13 = 8,450 tickets/quarter. Misroute cost per ticket, like for like, is
0.358 extra transfers × Rs 305 + 11.0% extra breaches × Rs 350 = Rs 148.
Today: 16.8% × 8,450 × Rs 148 = Rs 2.1 lakh/quarter. At 5%: (16.8% − 5%) × 8,450 × Rs 148 = **Rs 1.47 lakh/quarter saved**.

### What does one run cost, and what would a month cost at Vireo's volume (roughly 650 tickets a week)? *Show the arithmetic. If you used no paid calls, say so.*

**Rs 0 per run and Rs 0 per month. No paid calls anywhere.** Measured on an M1 laptop with 8 GB of RAM:
- **Full run** (18 months, 11,641 tickets, including training, 5-fold cross-validation and the hard test): about 80 s, 0.70 GB peak RAM.
- **Per ticket at intake:** 0.9 ms median (p95 1.2 ms). At 650/week (650 × 52 / 12 ≈ 2,817 tickets/month) that is under 3 seconds of CPU a month.
- **Optional local LLM** (`--llm`, qwen2.5:3b via Ollama, free): about 1.5 s per ticket, only for low-confidence tickets. The cost is still
  Rs 0 (electricity only), but it is *less* accurate on those tickets (69% vs 88%), so it's off by default. See `docs/LLM_DECISION.md`.

### How do you know it works? *Sample size, how you checked, error rate, and the kind of case it gets wrong.*

Three checks (full detail in `out/evaluation.md` after a run):

1. **Out-of-time test.** Trained on Jan 2025 – Mar 2026, tested on Apr – Jun 2026: **2,163 tickets**
   the model never saw. The truth is the category written in the agent's closing note.
   AI agrees on **100.0%** (1 disagreement); the bot's tag agrees on **70.5%**. Owning team: AI 100.0% vs bot 82.7%.
2. **Unseen phrasings: the honest number.** Test 1 is saturated because the same phrasings appear in train and test.
   So whole phrasings (167 of them, 7,348 tickets) are held out, and the model is scored only on wordings it never saw:
   **89.0%** (range 86.7–90.4% across 3 different splits), or **78.1%** (76.5–79.1%) with live-chat noise
   (typos, dropped words, cut-off messages). If the least certain ~12% go to a person first, it is **93.7%** right
   on the rest. This is my best estimate for live messages.
3. **Random audit of 150 tickets.** Each judged from both the message and the note. AI **150/150**
   (95% CI 97.5–100%). The note-reading rules 142/150: the 8 misses are notes like "sorted" that
   say nothing. Bot tag **101/150 (67%)**.

**What it gets wrong:** on unseen phrasings, words that mean different things in different queues.
"Card charged two times" pulls toward Charging & Battery, "refund not received yet" toward Delivery, and
"no update on my repair" toward App & Firmware. Also heavily garbled or cut-off messages, and cases that
sit between two owners (a cancellation after dispatch; a no-power fault that becomes a warranty claim).

**Honest caveats.** The data is very templated, so the 100% will not survive contact with live
messages. The unseen-phrasing test (about 89%, or 78% with noise) is the better guide. I tuned the note-reading rules after looking at
test-period disagreements (they had two bugs), which flatters the out-of-time number. The audit is
the cleaner check, but **the 150 audit labels were made by Claude, not by a human**, and need a
human spot-check.

### Did you change, narrow, or push back on the client's ask? *What, when, and why.*

Yes, from the first hour of looking at the data:
- **Pushed back on "biggest team gets the hires".** The bot's tag makes Billing look biggest, and
  31–39% of Billing's queue is delivery work (31% resolved by a Logistics agent; 39% by message
  content). Recommended fixing routing, holding the hires eight weeks, then Logistics if needed.
  This answers Arjun's email as well as Priya's.
- **Delivered the chart twice** (bot tag vs real category) rather than just "auto-categorising".
  The old tag chart alone would confirm a wrong decision.
- **Narrowed "by team"** to owning team, with the three frontline teams as one. Chat vs email is
  channel juggling, not a routing error. Tier 2 is excluded from volume comparisons (policy §6).
- **Added one category (Order Changes)** because "Other" (14%) is mostly cancellations and address changes.
- **Flagged that Priya's 22% includes 139 tickets from 2024**, outside the stated window, all tagged Billing.

### What is wrong with what you are handing us? *Be specific: bugs, shortcuts, things you know are off.*

- **The audit labels are AI-made** (Claude reading message + note), not human-verified.
- **Evaluation leakage.** The note-label rules were fixed after seeing test-period errors, so the out-of-time number is optimistic.
  The model itself was chosen on the unseen-phrasing test, and I looked at that test's errors, so 89.0% is slightly optimistic too.
  The audit was never used for any choice.
- **Real accuracy on live messages is unknown.** The best estimate is about 89% (78% with noise), but that comes from rewording this export,
  not from real tickets. A few hundred hand-labelled live tickets are needed.
- **The `--llm` local-LLM path works but makes things worse** on the tickets it handles (69% vs 88%).
  It's kept as a guarded experiment, not a feature. A 7B model froze an 8 GB laptop before memory guards were added.
- **Misrouting is measured against the AI's own category.** Any AI error shows up as a false misroute or a missed one.
- **Money scaled from about 180 tickets/week in the export to the stated 650/week.** If the export
  isn't a representative sample, the rupee figures move.
- **Only transfers and SLA credits are costed.** CSAT damage (2.3 vs 3.1) and slower resolution are
  reported but not priced, so the saving is understated. Agent time inside the Rs 305 transfer cost may overlap.
- **Workload per agent uses today's roster for every month.** The roster has no end dates, so leavers and joiners aren't modelled.
- **Tests run on macOS only** (M1, 8 GB, Python 3.12). 86 tests: 79 on synthetic data and 7 on the real pack (`docs/TESTING.md`).
  Not tested: Linux or Windows, browser rendering of the report beyond its content, and real live tickets.
- **The report loads Plotly from a CDN**, so it needs internet to render. Light mode only.
- **Resolution-time medians exclude open and pending tickets** (no resolved_at).
- Legacy money units and cross-system duplicates were checked and not found in Set E. The checks stay in `--audit`, but no dedupe step runs.

### What did you deliberately leave out, and why that rather than something else?

- **Live integration with the helpdesk or bot.** Needs Sameer, and it's a build decision for after
  Vireo agrees with the diagnosis. The tool already produces the routing decision per ticket.
- **Shift and hour staffing model (Erlang etc.).** The export is about 28% of real volume, so absolute
  staffing numbers would be false precision. A quick check showed no big shift differences in chat breaches (11–14%).
- **Product-defect / lot-code analysis.** A quick look found no lot with a meaningful excess of hardware tickets.
- **An LLM in the main path.** Measured: free local 3B–7B models were 84–87% accurate on the audit vs 100%
  for the fast model, and 1,000–9,000x slower (`docs/LLM_DECISION.md`).
- **A dashboard or web app.** A static report answers the question; a small thing that runs beat a large thing.

I chose these because none of them changes the headcount answer, which is the decision on the table.

### Anything you built or found that nobody asked for?

- **74 orders received both a refund and a replacement**, which policy §5 forbids. 42 have
  refund codes that suggest a genuine double payout (RETURN-QC-OK, DOA-REPL, LOST-TRANSIT, WTY-BUYBACK
  plus a replacement). That's about Rs 1.4 lakh in replacement cost across all 74. Not verified; flagged for Finance.
- **SLA breaches are charged to the resolving agent** (policy §3). So Logistics is blamed for late
  first replies that happened while its tickets sat in the Billing queue, which is why Neha sees Logistics "drowning".
- **The 139 out-of-window tickets are all Billing-tagged**, which pushes Billing's share up.
- **"Order Changes"**: a missing intake option that accounts for most of "Other".
- A `--audit` mode that prints the evidence for every data fix.

### What did you use AI for? *Which tools and models, where they helped, where they wasted your time, what you threw away. Link your three-minute screen recording here.*

**Tools:** Claude Code (desktop app), model Claude Opus 5.5, for nearly everything: profiling the
data, writing the Python, drafting docs, and labelling the 150-ticket audit sample. Local open models
through Ollama (qwen2.5:3b, llama3.2:3b, qwen2.5:7b), benchmarked as a free LLM option.
No paid API calls from the tool itself.
**Cost:** [FILL IN — your Claude plan / usage for this session].

**Where it helped:** fast data profiling that surfaced the misroute pattern early; spotting the
UTC timestamp issue and the out-of-window Billing tickets; writing and fixing the pipeline quickly.

**Where it wasted time or went wrong:**
- The first note-labelling rule took the first keyword in the note, but some notes lead with the
  action ("Replacement unit dispatched. Issue: …"). "pair" also matched inside "repair". Caught by
  reading the test errors.
- It committed a broken report build once (the pipe hid the exit code); amended.
- The first memo draft said the audit was done "by hand". It was AI-labelled, so the wording was corrected.
- The first memo draft claimed "leads" in defects and shifts before checking. Checked, found nothing, removed.
- The in-app browser wouldn't render the report, so it switched to a headless Chrome screenshot.
- The local-LLM benchmark left a 7B model in memory, and the next run froze the 8 GB laptop.
  Fixed with a model-size guard, one model at a time, and an explicit unload.

**Thrown away:**
- "Misrouted = resolving team ≠ assigned team". Noisy, because Billing agents resolve some delivery tickets
  themselves and frontline teams swap by shift. Replaced with real category vs bot tag.
- A first cost-per-misroute that compared misrouted tickets with *all* tickets. Replaced with a like-for-like comparison.
- An LLM-for-every-ticket design. The local model was already 99%+, so the LLM became an optional low-confidence fallback.
- The Claude API fallback (needed a key and was never run). Replaced with a free local LLM, which was
  then measured and found worse than the fast model, so it's off by default.
- A 12-colour stacked category chart. Replaced with small multiples, since 12 hues can't be told apart.
- 25 of 26 classifier candidates (`docs/MODEL_IMPROVEMENT.md`), including noise augmentation (it hurt),
  text normalisation (no gain) and calibrated probabilities (cost 1 point). Kept: a linear SVM trained on
  messages plus training tickets' notes, which took unseen-phrasing accuracy from 80.5% to about 89%.

**Screen recording:** [FILL IN — link]

### Your Public Google Drive Link

[FILL IN]

### Someone picks this up on Monday and you are unreachable. *The three things they need to know.*

1. **`python run.py` with the pack in `data/` regenerates every number** in the memo and this form
   (`out/report.html`, `out/evaluation.md`). All assumptions live at the top of `vireo/business_case.py`:
   650/week, the 5% target, Rs 9 lakh for two hires. Cost rates are in `vireo/load.py`, taken from policy §3–4.
2. **The ground truth is the agents' closing notes, read by rules in `vireo/labels.py`.** If Vireo
   changes how agents write notes, re-check label coverage and `out/evaluation.md` first. The 150
   audit labels in `eval/audit_labels.csv` are AI-made. Get a Vireo team lead to check 30 of them before anyone quotes the accuracy.
3. **Two open client actions:** (a) Sameer: can the bot add "Has your order arrived?" or call the
   classifier at intake? That is the change the Rs 1.5 lakh depends on. (b) Arjun: the refund-plus-replacement
   orders. Every run writes the list to `out/refund_and_replacement.csv`, with a `likely_double_payout` flag.

### Honest hours spent. *One number.*

[FILL IN]

### Github Repo Link

[FILL IN]
