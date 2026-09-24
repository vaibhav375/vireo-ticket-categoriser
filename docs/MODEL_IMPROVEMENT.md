# Making the offline model as good as it can be

## The problem with the first numbers

The first model scored 99.9% on the out-of-time test and 150/150 on the audit. That looks perfect,
but it says little about live tickets. The export is built from a limited set of issue phrasings
(about 170 canonical ones, plus typos) wrapped in templates, so the test tickets reuse phrasings the model
had already seen in training. Tuning against a saturated test would only overfit it.

## A harder test

`vireo/robustness.py` groups tickets by canonical issue phrasing (167 phrasings, 7,348 tickets) and
holds out **whole phrasings**. The model trains only on *other* phrasings, so every test ticket words
its problem in a way the model has never seen. It also re-runs the held-out tickets with live-chat
noise: extra typos, about 15% of words dropped, and some messages cut off partway.

**The first model on this test: 80.5% on unseen phrasings, 69.5% with noise.** That is the honest
starting point for live messages, not 99.9%.

All model choices were made on this test. The 150-ticket audit was not used for any choice and stays the final check.

## What was tried (`python experiments/model_search.py`)

| Candidate | Unseen phrasings | + noise | Kept? |
|---|---|---|---|
| First model: TF-IDF words + chars → logistic regression | 80.5% | 69.5% | baseline |
| Weaker / stronger regularisation (C=1 / C=20) | 78.2% / 81.3% | 67.6% / 70.4% | no |
| Class-balanced weights | 83.0% | 71.9% | yes, combined |
| Char n-grams only / word n-grams only | 78.2% / 75.7% | 68.5% / 63.4% | no, keep both |
| Wider char range (2–6) / word trigrams | 80.6% / 79.9% | 69.4% / 69.0% | no |
| Linear SVM | 84.8% | 73.0% | yes |
| SGD (modified Huber) / Complement Naive Bayes | 75.8% / 78.3% | 66.0% / 67.7% | no |
| Strip greetings, sign-offs, order IDs, product names | 81.3% | 69.2% | no gain with SVM |
| Train on noisy copies (augmentation) | 79.6% | 68.5% | no, hurt |
| Linear SVM + balanced | 85.0% | 73.1% | yes |
| Calibrated SVM (for probabilities) | 83.9% | 71.9% | no, costs 1 point |
| **Linear SVM + balanced + training tickets' agent notes as extra text** | **88.6%** | **78.1%** | **adopted** |
| … same, C=0.25 / C=1 / char 2–6 | 88.3% / 88.3% / 88.5% | 78.2% / 77.9% / 78.0% | no difference |

## Why the notes help

The remaining errors were vocabulary gaps. "Card charged two times" was read as *Charging & Battery*
(the characters "charg"). "Refund not received yet" was read as *Delivery* ("not received").
"No update on my repair" was read as *App & Firmware* ("update"). The model only knew what "refund"
or "charged" meant from phrasings it had happened to see.

Agents' closing notes describe the same issues in consistent support vocabulary ("refund not credited",
"double charge", "RMA status query"). Adding each **training** ticket's note as an extra training example
teaches that vocabulary. Nothing leaks:
- the model still reads only the customer's message when it predicts;
- held-out tickets' notes never enter training (out-of-fold everywhere, including the out-of-time test).

I did *not* write a hand-made keyword list to fix the errors above. I had just looked at those test
errors, so that would have been tuning on the test.

## Knowing when it's unsure

The SVM gives no probabilities, so confidence is the **margin between its top two category scores**.
On unseen phrasings:

| Least-certain share sent to a person first | Accuracy on the rest |
|---|---|
| 0% | 88.6% |
| 5% | 90.7% |
| 10% | 92.7% |
| 20% | 95.5% |
| 30% | 96.8% |

The pipeline flags margin < 0.2, which is about 11% of unseen-phrasing tickets, with **93.0%** accuracy on
the rest. On the export itself only 4 tickets fall below it, because their phrasings were seen in training.
For live use this is the lever: auto-route the confident ones, and send the rest to frontline triage as today.

## Result

| | First model | Current model |
|---|---|---|
| Unseen phrasings | 80.5% | **88.6%** |
| Unseen phrasings + noise | 69.5% | **78.1%** |
| Confident tickets (flag the least certain ~11%) | — | **93.0%** |
| Out-of-time test (Apr–Jun 2026) | 99.9% | 100.0% (1 disagreement) |
| 150-ticket audit (never used for choices) | 150/150 | 150/150 |
| Per ticket at intake | 1.4 ms | 0.9 ms |
| Full run, 18 months | ~40 s, 0.4 GB | ~50 s, 0.63 GB |

The business numbers didn't move: Billing is 14.0% by real need and Logistics 26.0%, and 16.8% of tickets are misrouted.
The first model already categorised this export correctly. The gain is in how it would cope with new wordings.

## What would help next (not done)

- **Real live tickets, hand-labelled** (a few hundred). This is the only true test of live accuracy, and the best extra training data.
- **Hindi / Hinglish.** The export is English only; live chat in India won't be.
- **Small sentence-embedding models** (e.g. a 30–130 MB ONNX model) could help unseen phrasings further.
  Not tried: it adds a model download and dependencies, and the gain on this templated data would be hard to
  measure honestly. Worth testing once real tickets exist.
