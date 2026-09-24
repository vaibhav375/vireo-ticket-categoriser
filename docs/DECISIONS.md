# Decisions log

The brief says: when something is unclear, decide, write it down, explain why. Here they are.

## About the ask

**1. Answered the question Priya asked, and the one underneath it.**
Priya asked for a monthly chart by category and team, with two hires going to the biggest team.
The chart is delivered as asked, but drawn twice: once by the bot's tag, once by what the ticket is really about.
The bot's tag is what makes Billing look biggest. By real work, Billing is 14% and Logistics 26%.
*Why:* the brief itself says "our tags are probably rubbish", and Arjun (Finance) asked whether
a process fix beats a hire. Handing over the tag chart alone would confirm a decision the data doesn't support.

**2. Pushed back on "biggest team gets the hires".**
Raw volume is the wrong test. Fixing routing moves no work *into* Logistics: Logistics already
resolves those tickets after a transfer. It removes the wasted touch in Billing. So the memo says:
fix routing first, re-measure, and if hires are still needed, put them in Logistics, not Billing.
Tier 2 (Escalations & Warranty) is not compared on volume, per policy §6.

**3. Business goal = misroute rate, not "visibility".**
Cut tickets sent to the wrong team from 17% (Jan – Jun 2026) to 5%. That is worth about Rs 1.5 lakh
a quarter in transfers and SLA credits at 650 tickets/week (policy §4 rates), against Rs 2.25 lakh
a quarter for the two hires. The 5% target is deliberately conservative: the model is about 99%
right on this data, and real data will be messier.

## About the data

**4. Legacy `resolved_at` is UTC. Shifted +5:30.**
Policy §9 says so, and 2,379 legacy tickets resolve before they were created without the fix (0 after).

**5. Dropped 139 tickets dated before 1 Jan 2025.**
The README says the export covers Jan 2025 – Jun 2026. All 139 are tagged Billing, so they inflate Billing:
21.8% of all rows (Priya's "22%") becomes 20.8% inside the stated window.

**6. `transfers` blank on legacy rows stays blank.**
Sameer: "blank, not zero". So transfer costs are measured on helpdesk-era tickets only
(14 Sep 2025 onwards), then applied as a rate.

**7. Agents joined by `agent_id`, date-aware.**
Two agents are both called "Om Sharma" (A3006 Chat Frontline, A3029 Logistics). The roster join
picks the row active on the ticket date, even though Set E has one row per agent.

**8. Checked and not found in Set E:** cross-system duplicate tickets (policy §9 warns of re-imports),
and legacy money in a different unit (refund medians are Rs 2,249 vs Rs 2,499, and refund never exceeds order value).
The checks stay in `python run.py --audit` in case another export has them.

**9. The export holds about 180 tickets a week; the brief says 650.**
Rates (misroute %, cost per misroute) come from the data; money is scaled to 650/week because that is
Vireo's real volume. Per-agent workload tables are relative, not absolute.

## About the method

**10. The model reads only the customer's opening message.**
That is all the bot has at intake, so it is the only fair test of "could routing have been right?".
Agent notes are hindsight and are used only to create training and evaluation labels.

**11. Labels come from agent closing notes by rule, not from the resolving team.**
The resolving team is noisy: Billing agents resolve some delivery tickets themselves, and frontline
teams swap work by shift. Rules read the issue the agent wrote down. 91.6% of notes yield a label.
The rest ("sorted", "cx ok") carry no information and are left unlabelled, not guessed.

**12. Added one category: Order Changes.**
14% of tickets are tagged "Other". The notes show most are cancellations and address or dispatch changes,
all frontline-owned. The other 11 categories keep Vireo's names so the charts compare like for like.

**13. The three frontline teams count as one owner.**
Chat vs Email Frontline is decided by channel and shift, not topic. Counting chat→email hand-offs
as misroutes would inflate the problem.

**14. No LLM in the normal run. A free local LLM is an off-by-default experiment.**
TF-IDF + logistic regression is 99.9% on the out-of-time test, runs in about 40 s using 0.4 GB, costs nothing, and
runs on a clean machine without a key. The first version had a Claude API fallback. It was replaced with
a free local model (Ollama) so there's no key and no cost, and then measured. On the tickets it would handle,
qwen2.5:3b is 69% accurate against the fast model's 88%, and about 1,000x slower. On an 8 GB laptop a 7B
model froze the machine. So `--llm` stays as a guarded experiment (size limit, one model at a time,
unload when done) and is not recommended. Full numbers: `docs/LLM_DECISION.md`.

**15. Cost per misroute is like for like.**
A delivery ticket sent to Billing is compared with a delivery ticket sent straight to Logistics, not
with the average ticket. Otherwise delivery tickets' naturally longer resolution would be blamed on routing.
Only transfers (Rs 305) and SLA credits (Rs 350) are costed; CSAT and resolution-time damage are reported, not priced.

**16. Repeat contacts not costed.**
Policy §10 prices repeat contacts, but the repeat rate barely differs between misrouted and
correctly routed tickets (4.0% vs 3.3%). Leaving it out keeps the number conservative.
