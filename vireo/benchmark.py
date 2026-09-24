"""Is a local LLM worth adding? Measure it rather than assume.

For each Ollama model: accuracy and latency on the 150 hand-audited tickets (known answers),
and on the tickets the fast model is unsure about (the only ones --llm would send).
The fast TF-IDF model is timed the same way for comparison.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .classify import LOW_CONFIDENCE, ask_llm, check_llm, fit_model, predict, unload_llm
from .evaluate import AUDIT_FILE


def _llm_run(msgs, model):
    ask_llm("warm-up", model)  # first call loads the model into memory; timed separately
    out = [ask_llm(m, model) for m in msgs]
    return [c for c, _ in out], np.array([s for _, s in out])


def run(t, models, out_path="out/llm_benchmark.md"):
    audit = pd.read_csv(AUDIT_FILE).merge(t[["ticket_id", "customer_message", "ai_category"]], on="ticket_id")
    msgs = audit.customer_message
    low = t[(t.ai_confidence < LOW_CONFIDENCE) & t.ref_category.notna()]

    # Fast model: per-ticket latency, one message at a time (how it would run at intake)
    lab = t[t.ref_category.notna()]
    m = fit_model(lab)
    start = time.perf_counter()
    for x in audit.customer_message:
        predict(m, pd.Series([x]))
    fast_ms = (time.perf_counter() - start) / len(msgs) * 1000

    rows = [{
        "model": "TF-IDF + linear SVM (default)", "size": "~5 MB",
        "audit_accuracy": (audit.ai_category == audit.true_category).mean(),
        "low_conf_accuracy": (low.ai_category == low.ref_category).mean(),
        "median_latency_s": fast_ms / 1000, "p95_latency_s": np.nan, "cold_start_s": np.nan,
    }]
    for model in models:
        problem = check_llm(model)  # also unloads any other model first
        if problem:
            print(f"skip {model}: {problem}")
            continue
        try:
            start = time.perf_counter()
            ask_llm("warm-up", model)
            cold = time.perf_counter() - start
            print(f"{model}: audit sample ({len(audit)})...", flush=True)
            a_pred, a_sec = _llm_run(audit.customer_message, model)
            print(f"{model}: low-confidence tickets ({len(low)})...", flush=True)
            l_pred, _ = _llm_run(low.customer_message, model)
        finally:
            unload_llm(model)
        rows.append({
            "model": model, "size": "",
            "audit_accuracy": np.mean(np.array(a_pred, dtype=object) == audit.true_category.values),
            "low_conf_accuracy": np.mean(np.array(l_pred, dtype=object) == low.ref_category.values),
            "median_latency_s": float(np.median(a_sec)), "p95_latency_s": float(np.percentile(a_sec, 95)),
            "cold_start_s": cold,
        })
        audit[f"pred_{model}"] = a_pred

    res = pd.DataFrame(rows)
    lines = ["# Local LLM benchmark", "",
             f"Audit sample: {len(audit)} tickets with known answers. Low-confidence set: {len(low)} tickets where the fast "
             f"model's confidence < {LOW_CONFIDENCE} and the agent note gives a label (the only tickets `--llm` sends).",
             "Latency is per ticket, one at a time, on this machine. Cold start = first call, loading the model into memory.", "",
             res.to_markdown(index=False, floatfmt=".3f"), ""]
    for model in models:
        col = f"pred_{model}"
        if col in audit:
            wrong = audit[audit[col] != audit.true_category]
            lines += [f"## {model}: audit errors ({len(wrong)})", "",
                      wrong.assign(msg=wrong.customer_message.str.replace("\n", " ").str[:110])[["ticket_id", "true_category", col, "msg"]].to_markdown(index=False), ""]
    Path(out_path).write_text("\n".join(lines) + "\n")
    return res
