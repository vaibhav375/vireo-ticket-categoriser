"""Categories and who owns them (support-policy.pdf §6).

Vireo's own 11 bot categories are kept so charts compare like for like, plus one:
"Order Changes" (cancel before dispatch, address / pincode change, dispatch status).
The bot files these under "Other", which is 14% of tickets, but agents' notes show they are a
distinct, frontline-owned request type.
"""

FRONTLINE = "Frontline"  # Chat / Email / Voice Frontline; which one depends on channel, not topic

OWNER = {
    "Billing & Payments": "Billing",
    "Delivery & Shipping": "Logistics",
    "Returns & Refunds": "Returns Desk",
    "Warranty & Repair": "Escalations & Warranty",
    "Connectivity": FRONTLINE,
    "Charging & Battery": FRONTLINE,
    "App & Firmware": FRONTLINE,
    "Audio Quality": FRONTLINE,
    "Account & Login": FRONTLINE,
    "Product Enquiry": FRONTLINE,
    "Order Changes": FRONTLINE,
    "Other": FRONTLINE,
}
CATEGORIES = list(OWNER)


def team_group(team):
    """Collapse the three frontline teams: moving work between them is channel juggling, not misrouting."""
    if isinstance(team, str) and team.endswith("Frontline"):
        return FRONTLINE
    return team
