"""Run the whole pipeline: load -> label -> categorise -> evaluate -> business case -> report.

    python run.py                 # offline, free
    python run.py --llm           # also re-check low-confidence tickets with Claude (needs ANTHROPIC_API_KEY)
    python run.py --audit         # print the data-fix evidence and exit
"""
import argparse
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from vireo import business_case, evaluate, report  # noqa: E402
from vireo.classify import categorise, llm_review  # noqa: E402
from vireo.labels import add_reference_labels  # noqa: E402
from vireo.load import audit, load, load_raw  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data", help="folder with the Vireo data pack")
    p.add_argument("--out", default="out")
    p.add_argument("--llm", action="store_true", help="re-check low-confidence tickets with Claude")
    p.add_argument("--audit", action="store_true", help="print data-fix evidence and exit")
    a = p.parse_args()

    if a.audit:
        audit(a.data)
        return

    t0 = time.time()
    t = add_reference_labels(load(a.data))
    print(f"Loaded {len(t):,} tickets (Jan 2025 – Jun 2026, data fixes applied)")
    t, _ = categorise(t)
    print(f"Categorised; {(t.ai_confidence < 0.6).sum()} low-confidence tickets")

    usage = None
    if a.llm:
        t, usage = llm_review(t)
        print(f"LLM review: {usage}")

    eval_res, audit_res = evaluate.write_report(t, f"{a.out}/evaluation.md")
    s, shares, by_route, workload, t = business_case.summary(t, load_raw(a.data)[1])
    path = report.write(t, s, shares, by_route, workload, eval_res, audit_res, a.out, usage)

    print(f"\nBilling share:   {s['billing_share_bot']:.1%} by bot tag -> {s['billing_share_real']:.1%} real")
    print(f"Logistics share: {s['logistics_share_bot']:.1%} by bot tag -> {s['logistics_share_real']:.1%} real")
    print(f"Misrouted (Jan-Jun 2026): {s['misroute_rate_2026h1']:.1%}; "
          f"cost ~Rs {s['misroute_cost_per_quarter_inr']:,.0f}/quarter at 650 tickets/week")
    print(f"Out-of-time accuracy: AI {eval_res['ai_category_accuracy']:.1%} vs bot {eval_res['bot_category_accuracy']:.1%}")
    print(f"\nReport: {path}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
