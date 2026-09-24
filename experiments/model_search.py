"""Model search, scored on the hard test (vireo/robustness.py): unseen phrasings, clean and noisy.
The 150-ticket audit is NOT used here; it stays the final, untouched check.

    python experiments/model_search.py [name ...]
"""
import random
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression, SGDClassifier  # noqa: E402
from sklearn.naive_bayes import ComplementNB  # noqa: E402
from sklearn.pipeline import make_pipeline, make_union  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402

from vireo.classify import _text  # noqa: E402
from vireo.labels import add_reference_labels  # noqa: E402
from vireo.load import load  # noqa: E402
from vireo.robustness import corrupt, score  # noqa: E402

PRODUCTS = r"\b(pulse ?2?|strata ?[23]?|orbit( mini| smart speaker)?|nexa ?(2|fit)?( band| watch| smartwatch)?|fit band|airlite|arc( neckband)?|gan charger|mini speaker|spare case)\b"
SCAFFOLD = [
    r"\[ivr transcript\]", r"\bvr\s?\d+\b", r"\d+", PRODUCTS,
    r"^(dear|hi|hello|hey|helo|hii|namaste|to)\b[^\n,.]*[,.\n]",  # greeting line
    r"i am writ\w* (with|w/|wth) ref\w* to my order of", r"this is regarding",
    r"(regards|thanks|thank you|sincerely|rgds|yours faithfully)[\s\S]*$",  # sign-off to end
    r"\b(tried|expected|purchased|order|product):",
]


def normalise(s):
    s = s.lower()
    for p in SCAFFOLD:
        s = re.sub(p, " ", s, flags=re.M)
    return re.sub(r"\s+", " ", s).strip()


def feats(word=(1, 2), char=(3, 5), analyzer="char_wb", min_df=(2, 3)):
    parts = []
    if word:
        parts.append(TfidfVectorizer(ngram_range=word, min_df=min_df[0], sublinear_tf=True))
    if char:
        parts.append(TfidfVectorizer(analyzer=analyzer, ngram_range=char, min_df=min_df[1], sublinear_tf=True))
    return make_union(*parts)


def note_text(notes):
    """Agent notes as extra training text: drop signatures, SOP refs and status tags."""
    n = notes.str.lower()
    n = n.str.replace(r"\(sop [\d.]+\)|\[closed\]|//\w+|~\w+|-[a-z]{2}\b", " ", regex=True)
    return n.str.replace(r"\s+", " ", regex=True).str.strip()


def pipe(clf, prep=_text, augment=0, notes=False, **fk):
    def fit_predict(train, Xte):
        Xtr = prep(train.customer_message).reset_index(drop=True)
        ytr = train.ref_category.reset_index(drop=True)
        if notes:  # the training tickets' own notes: same issues in agents' vocabulary
            Xtr = pd.concat([Xtr, note_text(train.agent_notes).reset_index(drop=True)], ignore_index=True)
            ytr = pd.concat([ytr, ytr], ignore_index=True)
        if augment:
            rng = random.Random(1)
            extra = [Xtr.map(lambda s: corrupt(s, rng)) for _ in range(augment)]
            Xtr = pd.concat([Xtr, *extra], ignore_index=True)
            ytr = pd.concat([ytr] * (augment + 1), ignore_index=True)
        m = make_pipeline(feats(**fk), clf).fit(Xtr, ytr)
        return m.predict(prep(pd.Series(Xte)))
    return fit_predict


norm_prep = lambda s: s.map(normalise)
LR = lambda C=5, cw=None: LogisticRegression(C=C, max_iter=3000, class_weight=cw)

CANDIDATES = {
    "baseline":            pipe(LR()),
    "C=1":                 pipe(LR(1)),
    "C=20":                pipe(LR(20)),
    "balanced":            pipe(LR(5, "balanced")),
    "char_only":           pipe(LR(), word=None),
    "word_only":           pipe(LR(), char=None),
    "char_2_6":            pipe(LR(), char=(2, 6)),
    "word_1_3":            pipe(LR(), word=(1, 3)),
    "linear_svc":          pipe(LinearSVC(C=0.5)),
    "sgd_huber":           pipe(SGDClassifier(loss="modified_huber", alpha=1e-5, random_state=0)),
    "complement_nb":       pipe(ComplementNB(alpha=0.3)),
    "normalise":           pipe(LR(), prep=norm_prep),
    "augment_noise_x1":    pipe(LR(), augment=1),
    "normalise+augment":   pipe(LR(), prep=norm_prep, augment=1),
    # round 2: combine the round-1 winners
    "svc_C0.1":            pipe(LinearSVC(C=0.1)),
    "svc_C0.25":           pipe(LinearSVC(C=0.25)),
    "svc_C1":              pipe(LinearSVC(C=1)),
    "svc_balanced":        pipe(LinearSVC(C=0.5, class_weight="balanced")),
    "svc_bal+norm":        pipe(LinearSVC(C=0.5, class_weight="balanced"), prep=norm_prep),
    "svc+norm":            pipe(LinearSVC(C=0.5), prep=norm_prep),
    "lr_C20_bal":          pipe(LR(20, "balanced")),
    "lr_C20_bal+norm":     pipe(LR(20, "balanced"), prep=norm_prep),
    "svc_bal_calibrated":  pipe(CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3)),
    # round 3: learn vocabulary from agents' notes of training tickets
    "svc_bal+notes":       pipe(LinearSVC(C=0.5, class_weight="balanced"), notes=True),
    "lr_bal+notes":        pipe(LR(20, "balanced"), notes=True),
    "svc_bal+notes+norm":  pipe(LinearSVC(C=0.5, class_weight="balanced"), notes=True, prep=norm_prep),
    "svc_bal+notes_C0.25": pipe(LinearSVC(C=0.25, class_weight="balanced"), notes=True),
    "svc_bal+notes_C1":    pipe(LinearSVC(C=1, class_weight="balanced"), notes=True),
    "svc_bal+notes_char2_6": pipe(LinearSVC(C=0.5, class_weight="balanced"), notes=True, char=(2, 6)),
}

if __name__ == "__main__":
    t = add_reference_labels(load())
    names = sys.argv[1:] or list(CANDIDATES)
    rows = []
    for name in names:
        start = time.time()
        r = score(t, CANDIDATES[name])
        rows.append({"model": name, "unseen": r["unseen_phrasing"], "unseen_noisy": r["unseen_phrasing_noisy"],
                     "seconds": time.time() - start})
        print(f"{name:22s} unseen {r['unseen_phrasing']:.3f}  noisy {r['unseen_phrasing_noisy']:.3f}  ({time.time() - start:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv("out/model_search.csv", index=False)
