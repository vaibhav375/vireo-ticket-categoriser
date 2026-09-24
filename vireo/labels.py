"""Reference labels from the agent's closing note.

The closing note is written after the agent has dealt with the ticket, so it says what the
ticket was really about. That makes it the best label source in the pack, but only in
hindsight: it doesn't exist at intake. So we use it to train and check the classifier,
never as the classifier's input.

Rule: the issue mentioned first in the note wins. Notes describe the issue first and the
action second ("bt dropouts | ... -> replacement approved, reverse pickup arranged"), and
action words must not override the issue.

Notes with no recognisable issue ("sorted", "cx ok", "see prev") get no label.
"""
import re

KEYWORDS = {
    "Delivery & Shipping": [
        r"not deliver", r"shipment not r", r"deliv\w* delay", r"dlvry (?:delay|query)", r"delivery query",
        r"transit", r"wrong (?:item|variant)", r"incorrect product", r"(?:received|rcvd) damaged",
        r"out for delivery", r"parcel stuck", r"order not recei", r"shipment issue", r"dl\w?v\w*ry (?:delay|query)",
        r"\brto\b", r"reship",
    ],
    "Returns & Refunds": [
        r"re?fu?nd (?:pending|not cr\w*dited|delay)", r"rfnd (?:pending|not cr\w*dited|delay)",
        r"reverse (?:pickup|pkp) pending", r"(?:pickup|pkp) (?:mi\w*s\w*d|not done)", r"\barn\b",
    ],
    "Billing & Payments": [
        r"payment debited", r"deducted without", r"failed ord\w* after payment", r"charged twice",
        r"double (?:charge|payment)", r"duplicate (?:payment|txn)", r"invoice", r"\bgst", r"c[ou]{1,2}pon",
        r"promo", r"discount", r"credit issued", r"payment (?:failed|pending)",
    ],
    "Order Changes": [
        r"cance?l", r"canel", r"address (?:update|change)", r"wrong pincode", r"\bdispatch",
    ],
    "Warranty & Repair": [
        r"\brma\b", r"repair status", r"warranty claim", r"wty claim", r"strap", r"unresponsive display",
        r"touch (?:issue|not respond)",
    ],
    "Connectivity": [
        r"pair", r"discoverable", r"dropouts", r"disconnect", r"connection drop", r"wi-?fi", r"network setup",
    ],
    "Charging & Battery": [
        r"battery", r"not charging", r"not taking charge", r"no ch\w*ge\b", r"case dead", r"powering on",
        r"no power",
    ],
    "App & Firmware": [
        r"app crash", r"app not opening", r"update (?:hang|stuck|failed)", r"firmware",
    ],
    "Audio Quality": [
        r"audio", r"crackling", r"distortion", r"static", r"side silent", r"\bmic\b",
    ],
    "Account & Login": [r"log ?in", r"\botp\b"],
    "Product Enquiry": [r"compatib", r"prod\w* enquiry", r"pre-sales"],
}
_COMPILED = {cat: re.compile("|".join(pats), re.I) for cat, pats in KEYWORDS.items()}


def label_note(note: str):
    """Return the category whose keyword appears earliest in the note, or None."""
    best, best_pos = None, None
    for cat, rx in _COMPILED.items():
        m = rx.search(note or "")
        if m and (best_pos is None or m.start() < best_pos):
            best, best_pos = cat, m.start()
    return best


def add_reference_labels(t):
    t = t.copy()
    t["ref_category"] = t.agent_notes.map(label_note)
    return t
