"""Run the whole pipeline: load -> label -> categorise -> evaluate -> business case -> report.

    python run.py                 # offline, free
    python run.py --llm           # experimental: re-check low-confidence tickets with a local LLM (Ollama)
    python run.py --benchmark     # measure local LLMs vs the default model: accuracy + latency
    python run.py --audit         # print the data-fix evidence and exit
"""
import argparse
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from vireo import business_case, evaluate, report  # noqa: E402
from vireo.classify import LOW_CONFIDENCE, categorise, llm_review  # noqa: E402
from vireo.labels import add_reference_labels  # noqa: E402
from vireo.load import DataError, _find, audit, load, load_raw  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
import pandas as pd  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data", help="folder with the Vireo data pack")
    p.add_argument("--out", default="out")
    p.add_argument("--llm", action="store_true", help="re-check low-confidence tickets with a local Ollama model")
    p.add_argument("--benchmark", nargs="*", metavar="MODEL",
                   help="benchmark Ollama models (default: qwen2.5:3b llama3.2:3b) and exit")
    p.add_argument("--audit", action="store_true", help="print data-fix evidence and exit")
    p.add_argument("--start", help="first day of the export's window (default: Set E, 2025-01-01)")
    p.add_argument("--end", help="day after the window ends (default: Set E, 2026-07-01)")
    a = p.parse_args()
    try:
        _run(a)
    except (DataError, FileNotFoundError) as e:
        print(f"Data problem: {e}")
        sys.exit(2)


def _run(a):
    if a.audit:
        audit(a.data)
        return

    Path(a.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    t = add_reference_labels(load(a.data, start=a.start, end=a.end))
    print(f"Loaded {len(t):,} tickets ({t.created_at.min():%b %Y} – {t.created_at.max():%b %Y}, data fixes applied)")
    t, _ = categorise(t)
    print(f"Categorised; {(t.ai_confidence < LOW_CONFIDENCE).sum()} low-confidence tickets")

    if a.benchmark is not None:
        from vireo import benchmark
        res = benchmark.run(t, a.benchmark or ["qwen2.5:3b", "llama3.2:3b"], f"{a.out}/llm_benchmark.md")
        print(res.to_string(index=False))
        return

    usage = None
    if a.llm:
        t, usage = llm_review(t)
        print(f"LLM review: {usage}")

    eval_res, audit_res = evaluate.write_report(t, f"{a.out}/evaluation.md")
    s, shares, by_route, workload, t = business_case.summary(t, load_raw(a.data)[1])
    path = report.write(t, s, shares, by_route, workload, eval_res, audit_res, a.out, usage)
    try:
        orders = pd.read_csv(_find(Path(a.data), "orders.csv"))
    except FileNotFoundError:  # optional: without it, only order IDs quoted on tickets are used
        orders = None
    rr = business_case.refund_and_replacement(t, pd.read_csv(_find(Path(a.data), "products.csv")), orders)
    rr.to_csv(f"{a.out}/refund_and_replacement.csv")

    print(f"\nBilling share:   {s['billing_share_bot']:.1%} by bot tag -> {s['billing_share_real']:.1%} real")
    print(f"Logistics share: {s['logistics_share_bot']:.1%} by bot tag -> {s['logistics_share_real']:.1%} real")
    print(f"Misrouted ({s['recent_period']}): {s['misroute_rate_recent']:.1%}; "
          f"cost ~Rs {s['misroute_cost_per_quarter_inr']:,.0f}/quarter at 650 tickets/week")
    print(f"Out-of-time accuracy: AI {eval_res['ai_category_accuracy']:.1%} vs bot {eval_res['bot_category_accuracy']:.1%}")
    rb = eval_res["robustness"]
    print(f"Unseen phrasings: {rb['unseen']:.1%} [{rb['unseen_min']:.1%}–{rb['unseen_max']:.1%}] "
          f"(with noise {rb['noisy']:.1%}); "
          f"{rb['confident_accuracy']:.1%} on the {1 - rb['triage_share']:.0%} it is confident about")
    print(f"Orders with both refund and replacement: {len(rr)} "
          f"({int(rr.likely_double_payout.sum())} look like double payouts) -> {a.out}/refund_and_replacement.csv")
    print(f"\nReport: {path}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
