"""Small synthetic data pack with the same shape as the Vireo export, for fast tests.

Built so the answers are known in advance:
- every category has messages + agent notes, including structured "Issue:" messages
- delivery messages that mention paying are tagged Billing by the "bot" (the real misroute pattern)
- legacy rows (before 14 Sep 2025) have resolved_at in UTC, as in the real export
- a few rows fall before the stated window
"""
import random
from pathlib import Path

import pandas as pd

CASES = {  # category: (message phrasings, note)
    "Billing & Payments": (["i was charged twice for one order", "payment deducted but no order created",
                            "please share gst invoice for my order"], "cx: charged twice | refund processed"),
    "Delivery & Shipping": (["my order has not been delivered yet", "parcel stuck in transit for a week",
                             "wrong item delivered in the box"], "issue: order not delivered | raised with courier"),
    "Returns & Refunds": (["refund for my return not received yet", "nobody came for the pickup"],
                          "re: refund pending | checked qc status"),
    "Warranty & Repair": (["what is the status of my warranty claim", "no news on my repair at service centre"],
                          "issue: rma status query | followed up"),
    "Connectivity": (["bluetooth pairing fails every time", "keeps disconnecting from my phone"],
                     "issue: pairing failure | reset done"),
    "Charging & Battery": (["battery drains very fast now", "charging case not charging at all"],
                           "issue: battery draining fast | fw pushed"),
    "App & Firmware": (["the app crashes when i open it", "firmware update stuck at half"],
                       "issue: app crashing | cleared cache"),
    "Audio Quality": (["no sound from the left side", "crackling noise in the audio"],
                      "issue: audio distortion | replacement raised"),
    "Account & Login": (["cannot login to my account", "otp never arrives on my phone"],
                        "issue: login issue | otp resent"),
    "Product Enquiry": (["is this compatible with iphone", "question about specs before i buy"],
                        "issue: compatibility query | shared spec sheet"),
    "Order Changes": (["please cancel my order it was a mistake", "need to change my delivery address"],
                      "issue: cancellation request | cancelled before dispatch"),
}
BOT_CATEGORY = {c: c for c in CASES} | {"Order Changes": "Other"}
TEAM = {"Billing & Payments": "Billing", "Delivery & Shipping": "Logistics", "Returns & Refunds": "Returns Desk",
        "Warranty & Repair": "Escalations & Warranty"}
AGENTS = [("A1", "Asha Rao", "Chat Frontline"), ("A2", "Om Sharma", "Chat Frontline"), ("A3", "Om Sharma", "Logistics"),
          ("A4", "Ravi Das", "Billing"), ("A5", "Meena Iyer", "Returns Desk"), ("A6", "Kiran Pal", "Escalations & Warranty"),
          ("A7", "Divya Sen", "Email Frontline")]
TEAM_AGENT = {"Chat Frontline": "A1", "Logistics": "A3", "Billing": "A4", "Returns Desk": "A5",
              "Escalations & Warranty": "A6", "Email Frontline": "A7"}
GO_LIVE = pd.Timestamp("2025-09-14")


def make_tickets(n=660, seed=0, start="2025-01-01", end="2026-06-30", out_of_window=6):
    rng = random.Random(seed)
    cats = list(CASES)
    rows = []
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    span = (end - start).total_seconds()
    for i in range(n + out_of_window):
        cat = cats[i % len(cats)]
        msgs, note = CASES[cat]
        msg = rng.choice(msgs)
        if rng.random() < 0.25:
            msg = f"Product: Pulse 2\nIssue: {msg}\nExpected: fix"
        bot = BOT_CATEGORY[cat]
        if cat == "Delivery & Shipping" and rng.random() < 0.4:
            msg, bot = f"paid already. {msg}", "Billing & Payments"  # the bot hears "paid"
        created = start + pd.Timedelta(seconds=rng.random() * span)
        if i >= n:
            created = start - pd.Timedelta(days=30 + i)  # before the stated window
        assigned = TEAM.get(bot, "Chat Frontline")
        resolver = TEAM.get(cat, "Chat Frontline")
        legacy = created < GO_LIVE
        resolved = created + pd.Timedelta(hours=rng.uniform(1, 48))
        rows.append({
            "ticket_id": f"TK-{i:05d}", "created_at": created.floor("min"),
            "first_response_at": (created + pd.Timedelta(minutes=rng.uniform(1, 600))).floor("min"),
            "resolved_at": (resolved - pd.Timedelta(minutes=330) if legacy else resolved).floor("min"),
            "status": "resolved", "channel": rng.choice(["chat", "email", "voice", "social"]),
            "customer_id": f"C{i % 97:03d}", "order_id": f"VR{i % 150:04d}", "product_sku": "VA-EB-PL2",
            "category": bot, "priority": "Normal", "assigned_team": assigned, "agent_id": TEAM_AGENT[resolver],
            "transfers": None if legacy else float(assigned != resolver),
            "csat_score": rng.choice([None, 2, 3, 4, 5]), "refund_amount_inr": None, "refund_reason_code": None,
            "replacement_issued": "N", "customer_message": msg, "agent_notes": note,
            "source_system": "legacy_fd" if legacy else "helpdesk",
        })
    return pd.DataFrame(rows)


def write_pack(folder: Path, tickets=None, agents=None, uuid_names=False):
    folder.mkdir(parents=True, exist_ok=True)
    tickets = make_tickets() if tickets is None else tickets
    agents = agents if agents is not None else pd.DataFrame(
        [{"agent_id": a, "name": n, "site": "Bengaluru", "team": t, "shift": "Day", "tier": 2 if "Escal" in t else 1,
          "from_date": "2020-01-01", "to_date": None} for a, n, t in AGENTS])
    products = pd.DataFrame([{"sku": "VA-EB-PL2", "product_name": "Pulse 2", "family": "earbuds", "launch_date": "2025-07-15",
                              "unit_cost_inr": 1480, "retail_price_inr": 3499, "warranty_months": 12}])
    prefix = "0f0f0f0f-1111-2222-3333-444444444444-" if uuid_names else ""
    tickets.to_csv(folder / f"{prefix}tickets.csv", index=False)
    agents.to_csv(folder / f"{prefix}agents.csv", index=False)
    products.to_csv(folder / f"{prefix}products.csv", index=False)
    return folder
