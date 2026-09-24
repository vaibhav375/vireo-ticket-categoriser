"""Categorise tickets from the customer's opening message only.

The opening message is all the intake bot has when it routes a ticket, so it's the only fair
input at prediction time. The classifier learns from the reference labels (agent closing notes, see labels.py).

Two stages:
1. Local model (default, free, offline): TF-IDF on words + character n-grams (the messages are
   full of typos) -> linear SVM with balanced class weights. Training text = the customer
   messages PLUS the same training tickets' agent notes, which describe the same issues in
   support vocabulary ("refund not credited", "double charge", "RMA status"). That teaches the model
   words it would otherwise only learn from phrasings it happens to have seen. Chosen on the
   unseen-phrasing test in robustness.py: 80.5% -> 88.6% vs the first version (experiments/model_search.py).
2. Optional LLM second opinion (--llm): only low-confidence tickets go to a small open model
   running locally in Ollama. Free, offline, no key, but measured worse (docs/LLM_DECISION.md).

Confidence is the margin between the top two category scores. On unseen phrasings, tickets with
margin < LOW_CONFIDENCE are about the least certain 10%; the rest are ~93% right.
"""
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC

from .taxonomy import CATEGORIES

LOW_CONFIDENCE = 0.2  # score margin; see module docstring


def _text(s: pd.Series) -> pd.Series:
    # "[IVR transcript]" marks the channel, not the topic
    return s.str.replace(r"^\[IVR transcript\]\s*", "", regex=True)


def note_text(notes: pd.Series) -> pd.Series:
    """Agent notes as extra training text: drop SOP refs, status tags and agent signatures."""
    n = notes.fillna("").str.lower()
    n = n.str.replace(r"\(sop [\d.]+\)|\[closed\]|//\w+|~\w+|-[a-z]{2}\b", " ", regex=True)
    return n.str.replace(r"\s+", " ", regex=True).str.strip()


def build_model():
    features = make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True),
    )
    return make_pipeline(features, LinearSVC(C=0.5, class_weight="balanced"))


def fit_model(train: pd.DataFrame):
    """Train on labelled tickets' messages and their notes. Only ever pass training tickets."""
    train = train[train.ref_category.notna()]
    X = pd.concat([_text(train.customer_message), note_text(train.agent_notes)], ignore_index=True)
    y = pd.concat([train.ref_category, train.ref_category], ignore_index=True)
    return build_model().fit(X, y)


def predict(model, messages: pd.Series):
    """Return (category, confidence margin) for customer messages."""
    scores = model.decision_function(_text(messages))
    top2 = np.sort(scores, axis=1)[:, -2:]
    return model.classes_[scores.argmax(1)], top2[:, 1] - top2[:, 0]


def categorise(t: pd.DataFrame, folds=5, seed=0) -> pd.DataFrame:
    """Add ai_category and ai_confidence to every ticket.

    Labelled tickets get out-of-fold predictions: the model that scores a ticket never saw that
    ticket's label or its note. Tickets without a reference label are scored by a model trained
    on all labelled tickets.
    """
    t = t.copy()
    t["ai_category"] = None
    t["ai_confidence"] = np.nan
    lab = t.index[t.ref_category.notna()]
    unl = t.index[t.ref_category.isna()]

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(lab, t.ref_category[lab]):
        _assign(t, lab[te], fit_model(t.loc[lab[tr]]))

    final = fit_model(t.loc[lab])
    if len(unl):
        _assign(t, unl, final)
    return t, final


def _assign(t, idx, model):
    category, margin = predict(model, t.loc[idx, "customer_message"])
    t.loc[idx, "ai_category"] = category
    t.loc[idx, "ai_confidence"] = margin


# ---------------------------------------------------------------- optional local LLM stage
# Experimental, off by default. A small open model served by Ollama on the same machine (no API
# key, no per-call cost). Measured on this data it is LESS accurate than the fast model and
# ~1,000x slower (docs/LLM_DECISION.md), so the normal run never uses it.
#
# Memory safety: Ollama keeps a model in RAM for 5 minutes by default. On an 8 GB laptop two
# loaded models plus the pipeline froze the machine. So: refuse models too large for this
# machine, keep a model loaded only briefly, and always unload it when done.

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.environ.get("VIREO_LLM_MODEL", "qwen2.5:3b")
MAX_MODEL_SHARE_OF_RAM = 0.35  # leave room for macOS, the browser and the pipeline (~0.4 GB)
KEEP_ALIVE = "30s"  # Ollama default is 5 minutes

SYSTEM_PROMPT = """You route customer-support tickets for Vireo Audio, an Indian consumer-audio brand
(earbuds, headphones, speakers, smartwatches). Read the customer's opening message and pick the ONE
category that describes what the customer needs resolved.

Categories:
- Billing & Payments: payment taken but no order created, charged twice, failed payment, invoice / GST invoice, coupon or discount not applied.
- Delivery & Shipping: order paid for but not delivered, delayed, stuck in transit, wrong item or damaged on arrival. If the customer paid AND the parcel hasn't arrived, this is Delivery, not Billing.
- Returns & Refunds: refund for a return not received yet, reverse pickup missed or pending.
- Warranty & Repair: warranty claim, RMA or repair status, physical damage on a watch (strap, display, touch).
- Connectivity: Bluetooth pairing, dropouts, device not discoverable, Wi-Fi setup.
- Charging & Battery: battery drain, bud or case not charging, device not powering on.
- App & Firmware: app crashing or not opening, firmware update stuck or failed.
- Audio Quality: no sound on one side, crackling, distortion, static, microphone problems.
- Account & Login: cannot log in, OTP not received.
- Product Enquiry: pre-sales, compatibility or spec questions.
- Order Changes: cancel an order, change address or pincode, dispatch status before shipping.
- Other: none of the above.

Answer with the category name exactly as written above."""

SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string", "enum": CATEGORIES}},
    "required": ["category"],
}


def ask_llm(message: str, model: str = LLM_MODEL, timeout=120):
    """Classify one message with a local Ollama model. Returns (category or None, seconds)."""
    body = json.dumps({
        "model": model, "stream": False, "format": SCHEMA, "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0},
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": message}],
    }).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=body, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        reply = json.loads(r.read())
    elapsed = time.perf_counter() - start
    try:
        category = json.loads(reply["message"]["content"])["category"]
    except (KeyError, ValueError):
        return None, elapsed
    return (category if category in CATEGORIES else None), elapsed


def _ollama(path, payload=None, timeout=5):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"{OLLAMA_URL}{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _ram_bytes():
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def check_llm(model: str = LLM_MODEL):
    """Return None if the model is safe to use here, else the reason it isn't."""
    try:
        models = {m["name"]: m["size"] for m in _ollama("/api/tags")["models"]}
    except OSError:
        return "Ollama isn't running. Start it with: ollama serve"
    if model not in models:
        return f"{model} isn't pulled. Run: ollama pull {model}"
    limit = MAX_MODEL_SHARE_OF_RAM * _ram_bytes()
    if models[model] > limit:
        return (f"{model} is {models[model] / 1e9:.1f} GB; this machine allows up to {limit / 1e9:.1f} GB "
                f"({MAX_MODEL_SHARE_OF_RAM:.0%} of RAM). Use a smaller model, e.g. qwen2.5:3b.")
    try:
        loaded = [m["name"] for m in _ollama("/api/ps")["models"] if m["name"] != model]
    except OSError:
        loaded = []
    for other in loaded:  # only one model in memory at a time
        unload_llm(other)
    return None


def unload_llm(model: str = LLM_MODEL):
    """Free the model's memory now instead of after Ollama's keep-alive timeout."""
    try:
        _ollama("/api/generate", {"model": model, "keep_alive": 0}, timeout=30)
    except OSError:
        pass


def llm_review(t: pd.DataFrame, threshold=LOW_CONFIDENCE, model: str = LLM_MODEL):
    """Send low-confidence tickets to the local LLM; its answer replaces the model's. Returns (tickets, usage)."""
    problem = check_llm(model)
    if problem:
        raise SystemExit(problem)
    t = t.copy()
    t["llm_category"] = None
    idx = t.index[t.ai_confidence < threshold]
    seconds = []
    try:
        for i in idx:
            category, secs = ask_llm(t.at[i, "customer_message"], model)
            seconds.append(secs)
            t.at[i, "llm_category"] = category
    finally:
        unload_llm(model)
    reviewed = t.llm_category.notna()
    changed = int((t.loc[reviewed, "llm_category"] != t.loc[reviewed, "ai_category"]).sum())
    t.loc[reviewed, "ai_category"] = t.loc[reviewed, "llm_category"]
    usage = {"model": model, "tickets_sent": len(idx), "tickets_reviewed": int(reviewed.sum()), "changed": changed,
             "median_seconds": round(float(np.median(seconds)), 2) if seconds else 0.0, "cost": "free (local)"}
    return t, usage
