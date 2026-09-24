"""How do we know the categories are right, and how often are they wrong?

1. Out-of-time test: train on everything before the export's last 3 months, score those 3 months
   against the reference labels (agent notes). For Set E: train Jan 2025 – Mar 2026, test Apr – Jun 2026. Compared with the bot's intake tag on the same tickets.
2. Unseen phrasings (robustness.py): whole phrasings held out, clean and with live-chat noise.
   Step 1 is saturated on this templated data; this is the better guide to live messages.
3. Hand audit: a random sample of tickets judged by reading both the message and the
   note (eval/audit_labels.csv). This checks the reference labels themselves, which
   steps 1 and 2 take on trust.
"""
import datetime as dt
from pathlib import Path

import pandas as pd

from .classify import LOW_CONFIDENCE, fit_model, predict
from .load import last_months, period_label
from .taxonomy import OWNER

AUDIT_FILE = Path(__file__).resolve().parent.parent / "eval" / "audit_labels.csv"  # works from any folder
TEST_MONTHS = 3


def out_of_time(t):
    lab = t[t.ref_category.notna()]
    split = last_months(t, TEST_MONTHS)
    tr, te = lab[lab.created_at < split], lab[lab.created_at >= split].copy()
    te["pred"], _ = predict(fit_model(tr), te.customer_message)
    # The bot never outputs "Order Changes"; its nearest equivalent is "Other"
    bot = te.category
    res = {
        "train_tickets": len(tr), "test_tickets": len(te),
        "train_period": period_label(t.created_at.min(), split - dt.timedelta(days=1)),
        "test_period": period_label(split, t.created_at.max()),
        "ai_category_accuracy": (te.pred == te.ref_category).mean(),
        "bot_category_accuracy": (bot == te.ref_category).mean(),
        "ai_owner_accuracy": (te.pred.map(OWNER) == te.ref_category.map(OWNER)).mean(),
        "bot_owner_accuracy": (bot.map(OWNER) == te.ref_category.map(OWNER)).mean(),
    }
    confusion = pd.crosstab(te.ref_category, te.pred, rownames=["reference"], colnames=["predicted"])
    errors = te[te.pred != te.ref_category][["ticket_id", "ref_category", "pred", "customer_message"]]
    return res, confusion, errors


def audit_sample(t, n=150, seed=7, path="eval/audit_sample.csv"):
    """Draw the random sample to hand-label. Run once; the labels live in AUDIT_FILE."""
    s = t.sample(n, random_state=seed)[["ticket_id", "customer_message", "agent_notes"]]
    s.to_csv(path, index=False)
    return s


def audit(t):
    if not AUDIT_FILE.exists():
        return None
    a = pd.read_csv(AUDIT_FILE).merge(
        t[["ticket_id", "category", "ref_category", "ai_category", "customer_message"]], on="ticket_id")
    n = len(a)
    res = {
        "audited_tickets": n,
        "ai_correct": int((a.ai_category == a.true_category).sum()),
        "reference_label_correct": int((a.ref_category == a.true_category).sum()),
        "reference_label_missing": int(a.ref_category.isna().sum()),
        "bot_tag_correct": int((a.category == a.true_category).sum()),
        "ai_owner_correct": int((a.ai_category.map(OWNER) == a.true_category.map(OWNER)).sum()),
        "bot_owner_correct": int((a.category.map(OWNER) == a.true_category.map(OWNER)).sum()),
    }
    wrong = a[a.ai_category != a.true_category][["ticket_id", "true_category", "ai_category", "category", "customer_message", "comment"]]
    return res, wrong


def wilson(k, n, z=1.96):
    """95% interval for a proportion; honest about small samples."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - r) / d, (c + r) / d)


def write_report(t, path="out/evaluation.md"):
    res, conf, errors = out_of_time(t)
    lines = ["# Evaluation", "", f"## 1. Out-of-time test (train {res['train_period']}, test {res['test_period']})", ""]
    lines += [f"- Test tickets with a reference label: {res['test_tickets']:,} (trained on {res['train_tickets']:,})"]
    lines += [f"- Category accuracy: AI **{res['ai_category_accuracy']:.1%}** vs bot tag {res['bot_category_accuracy']:.1%}"]
    lines += [f"- Owning-team accuracy: AI **{res['ai_owner_accuracy']:.1%}** vs bot tag {res['bot_owner_accuracy']:.1%}"]
    lines += ["", "Confusion matrix (rows = reference label from agent note, columns = AI):", "", conf.to_markdown(), ""]
    lines += [f"### All {len(errors)} disagreements in the test period", "",
              errors.assign(customer_message=errors.customer_message.str.replace("\n", " ").str[:140]).to_markdown(index=False), ""]

    from .robustness import current_model
    r = current_model(t)
    res["robustness"] = r
    lines += ["## 2. Harder test: phrasings the model has never seen", "",
              "The standard test above is near 100% because the same phrasings appear in training and test. Here whole",
              f"phrasings are held out ({r['phrasings']} canonical phrasings, {r['tickets']:,} tickets, 5 folds), so every test",
              "ticket words its problem in a way the model never saw. This is the better guide to live messages.",
              f"Repeated over {r['repeats']} different random splits of phrasings into folds; the range shows how much the",
              "score depends on which phrasings happen to be held out.", "",
              f"- Unseen phrasings: **{r['unseen']:.1%}** (range {r['unseen_min']:.1%}–{r['unseen_max']:.1%})",
              f"- Unseen phrasings with live-chat noise (typos, dropped words, cut-off messages): **{r['noisy']:.1%}** "
              f"(range {r['noisy_min']:.1%}–{r['noisy_max']:.1%})",
              f"- If the least certain tickets (margin < {LOW_CONFIDENCE}, {r['triage_share']:.0%} of them) go to a person first: "
              f"**{r['confident_accuracy']:.1%}** on the rest", "",
              "Model choice was made on this test; see experiments/model_search.py and docs/MODEL_IMPROVEMENT.md.", ""]
    a = audit(t)
    lines += ["## 3. Hand audit of a random sample", ""]
    if a is None:
        lines += ["Not run: eval/audit_labels.csv missing."]
    elif a[0]["audited_tickets"] == 0:
        lines += ["Not run: none of the audited tickets are in this export (the audit belongs to Set E)."]
    else:
        r, wrong = a
        n = r["audited_tickets"]
        for key, label in [("ai_correct", "AI category"), ("reference_label_correct", "Reference label (agent note rules)"),
                           ("bot_tag_correct", "Bot intake tag"), ("ai_owner_correct", "AI owning team"),
                           ("bot_owner_correct", "Bot owning team")]:
            lo, hi = wilson(r[key], n)
            lines += [f"- {label}: {r[key]}/{n} = **{r[key] / n:.1%}** (95% CI {lo:.0%}–{hi:.0%})"]
        lines += [f"- Reference label missing (note said nothing usable): {r['reference_label_missing']}/{n}", ""]
        lines += ["### Where the AI was wrong in the audit", "",
                  wrong.assign(customer_message=wrong.customer_message.str.replace("\n", " ").str[:140]).to_markdown(index=False)]
    Path(path).write_text("\n".join(lines) + "\n")
    return res, a
