"""Business-case arithmetic (vireo/business_case.py) on hand-built tickets with hand-computed answers."""
import pandas as pd
import pytest

from vireo import business_case as bc


def tickets(rows):
    base = {"ticket_id": "T", "created_at": pd.Timestamp("2026-02-01"), "month": "2026-02", "category": "Connectivity",
            "ai_category": "Connectivity", "agent_team": "Chat Frontline", "transfers": 0.0, "transfers_known": True,
            "sla_breach": False, "csat_score": 4.0, "resolution_h": 1.0, "order_id": None, "product_sku": "VA-EB-PL2",
            "refund_amount_inr": None, "refund_reason_code": None, "replacement_issued": "N"}
    return pd.DataFrame([{**base, "ticket_id": f"T{i}", **r} for i, r in enumerate(rows)])


def test_misrouted_means_bot_owner_differs_from_real_owner():
    t = bc.add_routing(tickets([
        {"category": "Billing & Payments", "ai_category": "Delivery & Shipping"},  # Billing vs Logistics
        {"category": "Delivery & Shipping", "ai_category": "Delivery & Shipping"},
        {"category": "Connectivity", "ai_category": "Audio Quality"},              # both frontline
        {"category": "Other", "ai_category": "Order Changes"},                     # both frontline
    ]))
    assert t.misrouted.tolist() == [True, False, False, False]


def test_cost_per_misroute_is_like_for_like_within_the_real_owner():
    t = bc.add_routing(tickets([
        # delivery tickets routed right: 0 transfers, no breach
        {"category": "Delivery & Shipping", "ai_category": "Delivery & Shipping"},
        {"category": "Delivery & Shipping", "ai_category": "Delivery & Shipping"},
        # delivery tickets the bot sent to Billing: 1 transfer each, one breach
        {"category": "Billing & Payments", "ai_category": "Delivery & Shipping", "transfers": 1.0, "sla_breach": True},
        {"category": "Billing & Payments", "ai_category": "Delivery & Shipping", "transfers": 1.0},
        # a slow but correctly routed Returns ticket must not inflate the delivery baseline
        {"category": "Returns & Refunds", "ai_category": "Returns & Refunds", "transfers": 2.0, "sla_breach": True},
    ]))
    _, extra_transfers, extra_breach, per_ticket = bc.cost_of_misrouting(t)
    assert extra_transfers == 1.0
    assert extra_breach == 0.5
    assert per_ticket == 1.0 * 305 + 0.5 * 350  # Rs 480


def test_unknown_transfers_are_left_out_of_cost_not_counted_as_zero():
    t = bc.add_routing(tickets([
        {"category": "Delivery & Shipping", "ai_category": "Delivery & Shipping"},
        {"category": "Billing & Payments", "ai_category": "Delivery & Shipping", "transfers": 1.0},
        {"category": "Billing & Payments", "ai_category": "Delivery & Shipping", "transfers": None, "transfers_known": False},
    ]))
    _, extra_transfers, _, _ = bc.cost_of_misrouting(t)
    assert extra_transfers == 1.0


def test_orders_with_refund_and_replacement_are_flagged():
    products = pd.DataFrame([{"sku": "VA-EB-PL2", "unit_cost_inr": 1480}])
    t = tickets([
        {"order_id": "O1", "refund_amount_inr": 3499.0, "refund_reason_code": "RETURN-QC-OK"},
        {"order_id": "O1", "replacement_issued": "Y"},
        {"order_id": "O2", "refund_amount_inr": 200.0, "refund_reason_code": "DUP-PAYMENT", "replacement_issued": "Y"},
        {"order_id": "O3", "refund_amount_inr": 3499.0, "refund_reason_code": "DOA-REPL"},  # refund only
        {"order_id": None, "refund_amount_inr": 100.0, "replacement_issued": "Y"},        # no order id
    ])
    got = bc.refund_and_replacement(t, products)
    assert sorted(got.index) == ["O1", "O2"]
    assert got.at["O1", "likely_double_payout"] and not got.at["O2", "likely_double_payout"]
    assert got.at["O1", "replacement_cost_inr"] == 1480 + 340


def test_summary_works_when_the_export_has_no_billing_tickets(pack):
    from vireo.classify import categorise
    from vireo.labels import add_reference_labels
    from vireo.load import load, load_raw
    t = load(pack)
    t = t[~t.category.eq("Billing & Payments") & ~t.ai_category.eq("Billing & Payments")] if "ai_category" in t else \
        t[t.category != "Billing & Payments"]
    t = t[~t.customer_message.str.contains("charged|payment|gst", regex=True)]
    t, _ = categorise(add_reference_labels(t))
    s, shares, *_ = bc.summary(t, load_raw(pack)[1])
    assert s["billing_share_real"] == pytest.approx(0.0, abs=0.02)
    assert shares.sum().round(6).tolist() == [1.0, 1.0]
