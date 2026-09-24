"""Reference labels read from agent closing notes (vireo/labels.py). Real note styles from the export."""
import pytest

from vireo.labels import label_note


@pytest.mark.parametrize("note, want", [
    # plain issue statements
    ("issue: order not delivered | raised ticket with courier partner", "Delivery & Shipping"),
    ("Cx reported charged twice. Confirmed UTRs. Refund of Rs 1234 initiated.", "Billing & Payments"),
    ("re: rfnd not credited | checked reverse pickup qc status -> refund approved", "Returns & Refunds"),
    ("ticket raised for rma status query. followed up with service centre.", "Warranty & Repair"),
    ("cx states pairing failure. Asked cx to reset the device.", "Connectivity"),
    ("Contact re battery draining fast. chk FW 3.4.12.", "Charging & Battery"),
    ("issue: app not opening | checked app version -> cleared cache", "App & Firmware"),
    ("cx says no audio one side. asked cx to check bud in case.", "Audio Quality"),
    ("issue: login issue | checked otp logs -> account unlocked", "Account & Login"),
    ("issue: product enquiry | shared spec sheet", "Product Enquiry"),
    ("Cx reported cancellation request. Cancelled before dispatch.", "Order Changes"),
    # action written before the issue: the issue marker wins
    ("Replacement unit dispatched. Issue: wrong item delivered. Verified photos.", "Delivery & Shipping"),
    ("rplc raised, RMA shared with cx. Issue: damaged in transit.", "Delivery & Shipping"),
    # 'pair' inside 'repair' must not mean Connectivity
    ("repair completed, shipped back. issue: rma status query.", "Warranty & Repair"),
    # typos seen in the export
    ("RTO confirmed, reshipped. Issue: dlvrry delayed. Checked AWB.", "Delivery & Shipping"),
    ("cx: pkp msised | re-raised pkp with courier", "Returns & Refunds"),
    ("cx: canel ord | chk shipment status -> refund approved", "Order Changes"),
])
def test_note_is_labelled_by_the_issue_it_describes(note, want):
    assert label_note(note) == want


@pytest.mark.parametrize("note", ["sorted", "cx ok ~Megha", "see prev (SOP 3.1) //SIM", "-", "", None])
def test_notes_without_an_issue_get_no_label(note):
    assert label_note(note) is None
