"""Classifier (vireo/classify.py): valid output for any input, uses notes, deterministic, copes with small data."""
import random
import string

import numpy as np
import pandas as pd
import pytest

from synthetic import make_tickets
from vireo.classify import categorise, fit_model, note_text, predict
from vireo.labels import add_reference_labels
from vireo.taxonomy import CATEGORIES


@pytest.fixture(scope="module")
def train():
    t = make_tickets(out_of_window=0)
    return add_reference_labels(t.assign(agent_notes=t.agent_notes.fillna("")))


@pytest.fixture(scope="module")
def model(train):
    return fit_model(train)


def weird_inputs():
    rng = random.Random(3)
    return [
        "", "   ", "\n\n", "[IVR transcript]", "[IVR transcript] ", "12345 67890", "VR908311", "?!?!?!",
        "😡😡😡 where is my order 📦", "मेरा ऑर्डर नहीं आया", "mera order abhi tak nahi aaya bhai",
        "<script>alert(1)</script>", "a" * 20000, "refund " * 3000,
        "".join(rng.choice(string.printable) for _ in range(500)),
        "\x00\x01\x02 control chars",
    ] + ["".join(rng.choice(string.ascii_lowercase + " ") for _ in range(rng.randint(1, 200))) for _ in range(30)]


def test_any_message_gets_one_valid_category_and_a_nonnegative_confidence(model):
    msgs = pd.Series(weird_inputs())
    cats, conf = predict(model, msgs)
    assert len(cats) == len(msgs)
    assert set(cats) <= set(CATEGORIES)
    assert np.all(np.isfinite(conf)) and np.all(conf >= 0)


def test_missing_messages_are_treated_as_empty_not_an_error(model):
    cats, conf = predict(model, pd.Series([None, np.nan, "my order has not been delivered yet"]))
    assert set(cats) <= set(CATEGORIES)
    assert cats[2] == "Delivery & Shipping"


def test_note_text_strips_sop_refs_status_tags_and_signatures():
    got = note_text(pd.Series(["Cx reported charged twice. Refund done. (SOP 2.7) [closed] //HAR ~Megha -SM", None]))
    assert got.tolist() == ["cx reported charged twice. refund done.", ""]


def test_vocabulary_seen_only_in_notes_is_learned(train):
    # 'zorblat' never appears in any customer message, only in Warranty notes
    t = train.copy()
    w = t.ref_category == "Warranty & Repair"
    t.loc[w, "agent_notes"] = t.loc[w, "agent_notes"] + " zorblat zorblat"
    cats, _ = predict(fit_model(t), pd.Series(["zorblat zorblat zorblat"]))
    assert cats[0] == "Warranty & Repair"


def test_training_twice_gives_identical_predictions(train):
    msgs = pd.Series(weird_inputs())
    a, ca = predict(fit_model(train), msgs)
    b, cb = predict(fit_model(train), msgs)
    assert list(a) == list(b) and np.allclose(ca, cb)


def test_categorise_labels_every_ticket(train):
    out, _ = categorise(train)
    assert out.ai_category.notna().all()
    assert set(out.ai_category) <= set(CATEGORIES)


def test_categorise_works_when_a_category_has_very_few_tickets(train):
    # A new or rare category (3 labelled tickets) must not crash 5-fold cross-validation
    rare = train[train.ref_category == "Product Enquiry"].index[3:]
    t = train.drop(rare)
    with pytest.warns(UserWarning, match="least populated class"):  # sklearn's notice, not a failure
        out, _ = categorise(t)
    assert out.ai_category.notna().all()


def test_categorise_with_no_labelled_tickets_fails_clearly(train):
    t = train.assign(ref_category=None)
    with pytest.raises(ValueError, match="no labelled tickets"):
        categorise(t)
