"""Categorise tickets from the customer's opening message only.

The opening message is all the intake bot has when it routes a ticket, so it's the only fair
input. The classifier learns from the reference labels (agent closing notes, see labels.py).

Two stages:
1. Local model (default, free, offline): TF-IDF on words + character n-grams (the messages
   are full of typos) -> logistic regression.
2. Optional LLM second opinion (--llm): only tickets where the local model's confidence is
   below a threshold go to Claude with the category definitions. Needs ANTHROPIC_API_KEY.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline, make_union

from .taxonomy import CATEGORIES

LOW_CONFIDENCE = 0.6


def _text(s: pd.Series) -> pd.Series:
    # "[IVR transcript]" marks the channel, not the topic
    return s.str.replace(r"^\[IVR transcript\]\s*", "", regex=True)


def build_model():
    features = make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True),
    )
    return make_pipeline(features, LogisticRegression(C=5, max_iter=2000))


def categorise(t: pd.DataFrame, folds=5, seed=0) -> pd.DataFrame:
    """Add ai_category and ai_confidence to every ticket.

    Labelled tickets get out-of-fold predictions: the model that scores a ticket never
    saw that ticket's label. Tickets without a reference label are scored by a model trained
    on all labelled tickets.
    """
    t = t.copy()
    t["ai_category"] = None
    t["ai_confidence"] = np.nan
    lab = t.index[t.ref_category.notna()]
    unl = t.index[t.ref_category.isna()]
    X, y = _text(t.customer_message), t.ref_category

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(lab, y[lab]):
        m = build_model().fit(X[lab[tr]], y[lab[tr]])
        _assign(t, lab[te], m, X)

    final = build_model().fit(X[lab], y[lab])
    if len(unl):
        _assign(t, unl, final, X)
    return t, final


def _assign(t, idx, model, X):
    proba = model.predict_proba(X[idx])
    t.loc[idx, "ai_category"] = model.classes_[proba.argmax(1)]
    t.loc[idx, "ai_confidence"] = proba.max(1)


# ---------------------------------------------------------------- optional LLM stage

LLM_MODEL = os.environ.get("VIREO_LLM_MODEL", "claude-opus-5")
# USD per million tokens (input, output), Anthropic list prices
PRICES = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0), "claude-haiku-4-5": (1.0, 5.0)}

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
    "additionalProperties": False,
}


def llm_review(t: pd.DataFrame, threshold=LOW_CONFIDENCE, limit=None):
    """Send low-confidence tickets to Claude. Returns (tickets, usage summary)."""
    import anthropic  # optional dependency

    client = anthropic.Anthropic()
    idx = t.index[t.ai_confidence < threshold]
    if limit:
        idx = idx[:limit]
    t = t.copy()
    t["llm_category"] = None
    tokens_in = tokens_out = 0
    for i in idx:
        try:
            r = client.messages.create(
                model=LLM_MODEL,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": t.at[i, "customer_message"]}],
            )
        except anthropic.RateLimitError:
            print("Rate limited; stopping LLM review early.")
            break
        except anthropic.APIStatusError as e:
            print(f"{t.at[i, 'ticket_id']}: API error {e.status_code}, skipped")
            continue
        tokens_in += r.usage.input_tokens
        tokens_out += r.usage.output_tokens
        if r.stop_reason == "refusal":
            continue
        text = next((b.text for b in r.content if b.type == "text"), None)
        if text:
            t.at[i, "llm_category"] = json.loads(text)["category"]

    reviewed = t.llm_category.notna()
    t.loc[reviewed, "ai_category"] = t.loc[reviewed, "llm_category"]
    p_in, p_out = PRICES.get(LLM_MODEL, (np.nan, np.nan))
    usage = {
        "model": LLM_MODEL, "tickets_sent": len(idx), "tickets_reviewed": int(reviewed.sum()),
        "input_tokens": tokens_in, "output_tokens": tokens_out,
        "cost_usd": round(tokens_in / 1e6 * p_in + tokens_out / 1e6 * p_out, 4),
    }
    return t, usage
