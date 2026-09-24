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
    if not canon:  # no structured messages: an empty pattern would match everything
        return pd.Series([None] * len(t), index=t.index, dtype=object)
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


def current_model(t, folds=5, repeats=3):
    """The hard test for the model the pipeline actually uses (classify.fit_model).

    Which phrasings land in which fold moves the score by a couple of points (seen when a
    single extra labelled ticket shifted a one-split score from 88.6% to 89.8%). So the test
    is repeated over `repeats` shuffled group splits (fixed seeds) and the mean and range are
    reported. Also: accuracy on confident tickets when the least certain go to a person.
    """
    from .classify import LOW_CONFIDENCE, fit_model, predict

    t = t[t.ref_category.notna()].copy()
    t["phrase"] = phrase_groups(t)
    g = t[t.phrase.notna()]
    if g.phrase.nunique() < folds:  # not enough distinct phrasings to hold any out
        nan = float("nan")
        return {"tickets": 0, "phrasings": int(g.phrase.nunique()), "repeats": 0, "unseen": nan, "noisy": nan,
                "unseen_min": nan, "unseen_max": nan, "noisy_min": nan, "noisy_max": nan,
                "triage_share": nan, "confident_accuracy": nan}
    clean, noisy, all_ok, all_margin = [], [], [], []
    for seed in range(repeats):
        rng = random.Random(seed)
        ok, ok_noisy = [], []
        for tr, te in GroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(g, groups=g.phrase):
            train, test = g.iloc[tr], g.iloc[te]
            model = fit_model(train)
            pred, margin = predict(model, test.customer_message)
            pred_noisy, _ = predict(model, test.customer_message.map(lambda s: corrupt(s, rng)))
            ok += list(pred == test.ref_category.values)
            ok_noisy += list(pred_noisy == test.ref_category.values)
            all_margin += list(margin)
        clean.append(np.mean(ok))
        noisy.append(np.mean(ok_noisy))
        all_ok += ok
    all_ok, all_margin = np.array(all_ok), np.array(all_margin)
    confident = all_margin >= LOW_CONFIDENCE
    return {"tickets": len(g), "phrasings": g.phrase.nunique(), "repeats": repeats,
            "unseen": float(np.mean(clean)), "unseen_min": float(min(clean)), "unseen_max": float(max(clean)),
            "noisy": float(np.mean(noisy)), "noisy_min": float(min(noisy)), "noisy_max": float(max(noisy)),
            "triage_share": float(1 - confident.mean()), "confident_accuracy": float(all_ok[confident].mean())}
