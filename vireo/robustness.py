"""A harder test than the standard evaluation, which is saturated (the same phrasings appear in
train and test, so ~100% says little about live messages).

1. Unseen phrasings: the export is built from a limited set of issue phrasings wrapped in
   templates. Tickets are grouped by canonical phrasing (taken from the structured "Issue:" lines).
   Whole phrasings are held out, and the model is trained ONLY on other phrasings, so every test
   ticket says its problem in words the model has never seen.
2. Noise: the same held-out tickets with extra typos, dropped words, and messages cut short, as live chat would have.

Model choices are made on this test. The 150-ticket audit stays untouched as the final check.
"""
import random
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

_norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", s.lower())).strip()


def phrase_groups(t):
    """Canonical issue phrasing per ticket (None when no known phrasing is found)."""
    issue = t.customer_message.str.lower().str.extract(r"issue:\s*([^\n]+)")[0].dropna().map(_norm)
    counts = issue.value_counts()
    canon = sorted((p for p, c in counts.items() if c >= 2 and len(p) >= 10), key=len, reverse=True)
    rx = re.compile("|".join(re.escape(p) for p in canon))
    return t.customer_message.map(lambda s: (m := rx.search(_norm(s))) and m.group(0))


def corrupt(text, rng, typo=0.06, drop=0.15, cut=0.3):
    """Live-chat style noise: character typos, dropped words, sometimes only the first part of the message."""
    words = text.split()
    if len(words) > 6 and rng.random() < cut:
        words = words[: max(4, int(len(words) * rng.uniform(0.4, 0.7)))]
    words = [w for w in words if rng.random() > drop] or words[:1]
    out = []
    for w in words:
        chars = list(w)
        if len(chars) > 3 and rng.random() < typo * len(chars) / 4:
            i = rng.randrange(len(chars) - 1)
            op = rng.random()
            if op < 0.4:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]  # swap
            elif op < 0.7:
                del chars[i]  # drop
            else:
                chars.insert(i, chars[i])  # double
        out.append("".join(chars))
    return " ".join(out)


def score(t, fit_predict, folds=5, seed=0):
    """fit_predict(train_frame, test_texts) -> predictions. train_frame has customer_message,
    agent_notes and ref_category for training tickets only.
    Returns accuracy on unseen phrasings, clean and noisy."""
    t = t[t.ref_category.notna()].copy()
    t["phrase"] = phrase_groups(t)
    grouped = t[t.phrase.notna()]
    rng = random.Random(seed)
    clean_hits = noisy_hits = n = 0
    for tr, te in GroupKFold(n_splits=folds).split(grouped, groups=grouped.phrase):
        train, test = grouped.iloc[tr], grouped.iloc[te]
        noisy = test.customer_message.map(lambda s: corrupt(s, rng))
        both = pd.concat([test.customer_message, noisy], ignore_index=True)
        pred = np.asarray(fit_predict(train, both))
        k = len(test)
        clean_hits += (pred[:k] == test.ref_category.values).sum()
        noisy_hits += (pred[k:] == test.ref_category.values).sum()
        n += k
    return {"unseen_phrasing": clean_hits / n, "unseen_phrasing_noisy": noisy_hits / n, "tickets": n}


def current_model(t, folds=5, seed=0):
    """The hard test for the model the pipeline actually uses (classify.fit_model), plus how
    accurate the confident tickets are when the least certain ones go to a person."""
    from .classify import LOW_CONFIDENCE, fit_model, predict

    t = t[t.ref_category.notna()].copy()
    t["phrase"] = phrase_groups(t)
    g = t[t.phrase.notna()]
    rng = random.Random(seed)
    ok, ok_noisy, margins = [], [], []
    for tr, te in GroupKFold(n_splits=folds).split(g, groups=g.phrase):
        train, test = g.iloc[tr], g.iloc[te]
        model = fit_model(train)
        pred, margin = predict(model, test.customer_message)
        pred_noisy, _ = predict(model, test.customer_message.map(lambda s: corrupt(s, rng)))
        ok += list(pred == test.ref_category.values)
        ok_noisy += list(pred_noisy == test.ref_category.values)
        margins += list(margin)
    ok, margins = np.array(ok), np.array(margins)
    confident = margins >= LOW_CONFIDENCE
    return {"tickets": len(g), "phrasings": g.phrase.nunique(), "unseen": ok.mean(), "noisy": np.mean(ok_noisy),
            "triage_share": 1 - confident.mean(), "confident_accuracy": ok[confident].mean()}
