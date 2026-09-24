"""Evaluation helpers and the unseen-phrasing test (vireo/evaluate.py, vireo/robustness.py)."""
import math
import random

import pandas as pd
import pytest

from synthetic import make_tickets
from vireo.evaluate import wilson
from vireo.labels import add_reference_labels
from vireo.robustness import corrupt, current_model, phrase_groups


def test_wilson_interval_for_a_perfect_150_is_not_100_percent():
    lo, hi = wilson(150, 150)
    assert round(lo, 3) == 0.975 and hi == pytest.approx(1.0)


def test_wilson_interval_with_no_data_is_undefined():
    assert all(math.isnan(x) for x in wilson(0, 0))


def test_corrupt_is_reproducible_with_a_seed_and_never_empties_a_message():
    msgs = ["my order has not been delivered yet", "x", "refund please", ""]
    a = [corrupt(m, random.Random(5)) for m in msgs]
    b = [corrupt(m, random.Random(5)) for m in msgs]
    assert a == b
    assert all(out for m, out in zip(msgs, a) if m)


def test_phrase_groups_find_the_structured_issue_phrasing():
    t = pd.DataFrame({"customer_message": [
        "Product: X\nIssue: my order has not been delivered yet\nExpected: fix",
        "Product: Y\nIssue: my order has not been delivered yet\nExpected: refund",
        "hello my order has not been delivered yet please help",
        "something else entirely"]})
    assert phrase_groups(t).tolist() == ["my order has not been delivered yet"] * 3 + [None]


def test_phrase_groups_return_none_when_there_are_no_structured_messages():
    t = pd.DataFrame({"customer_message": ["hello", "my parcel is late", "refund please"]})
    assert phrase_groups(t).isna().all()


def test_unseen_phrasing_test_reports_instead_of_crashing_without_enough_phrasings():
    t = add_reference_labels(make_tickets(out_of_window=0))
    t = t.assign(customer_message=t.customer_message.str.replace("Issue:", "Problem:"))  # no structured lines
    r = current_model(t)
    assert r["tickets"] == 0 and math.isnan(r["unseen"])


def test_unseen_phrasing_score_is_averaged_over_several_splits_and_reports_its_range():
    t = add_reference_labels(make_tickets(out_of_window=0))
    r = current_model(t, repeats=3)
    assert r["repeats"] == 3
    assert r["unseen_min"] <= r["unseen"] <= r["unseen_max"]
    assert r["noisy_min"] <= r["noisy"] <= r["noisy_max"]
