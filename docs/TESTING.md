# Testing

```bash
pip install -r requirements-dev.txt
python -m pytest              # 84 fast tests, ~50 s, synthetic data only (no client data needed)
python -m pytest -m slow      # 7 regression + resource tests on the real pack in data/ (~90 s)
```

Use `python -m pytest`, not bare `pytest`: a `pytest` launcher left over from another Python install
(this happened on the development machine) runs the wrong interpreter and reports `No module named 'pytest'`.

Every bug fix started with a test that failed first. Tests describing behaviour that was already right passed
straight away, so for the core data fixes the code was deliberately broken (UTC shift removed, issue-marker rule removed, roster dates
ignored) to confirm a test catches it.

## What is covered

| Layer | File | What it proves |
|---|---|---|
| Data fixes | `test_load.py` | UTC→IST shift on legacy rows only; window drops out-of-range rows; blank transfers stay unknown; same-name agents join by ID; dated roster rows pick the one active on the ticket date; SLA breach per channel target; uuid-prefixed file names |
| Note labelling | `test_labels.py` | 17 real note styles from the export, including action-first notes, "pair" inside "repair", and typos; notes with no issue get no label |
| Classifier | `test_classify.py` | Any input gives exactly one valid category and a finite, non-negative confidence (fuzzed: empty, 20,000 chars, Hindi, Hinglish, emoji, HTML, control characters, random text); missing messages; vocabulary seen only in notes is learned; training is deterministic; rare categories; no labels at all |
| Business case | `test_business_case.py` | Misroute definition (frontline teams are one owner); like-for-like cost against a hand-computed Rs 480; unknown transfers left out, not counted as zero; refund+replacement flags; an export with no Billing tickets |
| Evaluation | `test_evaluate_and_robustness.py` | Wilson interval; reproducible noise; phrase grouping; the hard test copes with too few phrasings and reports mean and range over repeated splits |
| Bad data | `test_bad_data.py` | Missing columns named; unreadable created_at names the row; unreadable other times become unknown; duplicate IDs; blank messages and notes; unknown agent or channel; an export from another period |
| End to end | `test_pipeline_e2e.py` | `run.py` as a user runs it: every output written, counts consistent (monthly tables sum to the ticket count), no "nan" in the report, new output folder, run from another folder, `--audit`, clean exit code 2 with a one-line message for data problems, `--start/--end` for another export |
| Local LLM | `test_llm_safety.py` | Against a fake Ollama HTTP server: oversized model refused before loading, Ollama down, model not pulled, other models unloaded first, only uncertain tickets sent, unload afterwards, a failed call or garbage reply keeps the fast model's answer |
| Setup | `test_setup.py` | The pipeline raises no deprecation warnings; slow tests are skipped by default even when pytest is started outside the repo |
| Real data | `test_real_data.py` (slow) | Headline numbers unchanged (Billing 20.8%→14.0%, Logistics 16.4%→26.0%, misroute 16.8%, out-of-time 100%, unseen phrasings 89.0% with range, audit 150/150, refund list 139/73); peak memory < 1 GB; run time < 3 min; prediction < 5 ms per ticket |

## What testing found (all fixed)

1. **Any export other than Set E crashed the whole run.** The audit file's ticket IDs didn't match, so evaluation divided by zero.
2. **Another export's dates dropped every row.** The window, out-of-time split and "recent" period were hardcoded to Set E.
   There are now `--start/--end` flags, the split is the data's last 3 months, and "recent" is its last 6 months.
3. **The audit file was only found when run from the repo folder.**
4. **One unreadable date killed the run** with a pandas format error.
5. **Duplicate ticket IDs were double-counted.**
6. **Missing columns gave `AttributeError`** instead of saying which columns.
7. **A missing customer message crashed prediction.** That's the realistic case at intake.
8. **An export with no Billing tickets raised `KeyError`** in the business case.
9. **No structured "Issue:" lines** made phrase grouping match an empty pattern, and the hard test then crashed.
10. **One HTTP error from Ollama aborted the whole `--llm` run.**
11. **The note typo "pkp msised" got no label.** The pickup pattern only allowed typos starting "mi".
12. **The unseen-phrasing score moved ±1–2 points** depending on how phrasings fell into folds. It is now repeated over
    3 shuffled splits and reported with its range: 89.0% (86.7–90.4%).
13. **Running pytest from the parent folder ignored `pytest.ini`.** Slow tests ran by default and 42,846 warnings
    appeared. The warnings came from `pd.Timedelta(...)` on numpy 2.5, and were hidden rather than fixed. Now Python's
    `datetime.timedelta` is used instead (0 warnings), and the slow-test rule lives in `conftest.py`, which loads from any folder.

Data problems now print one line (`Data problem: ...`) and exit with code 2, never a stack trace.

## Not covered

- **Live helpdesk integration.** It doesn't exist yet.
- **Real live-ticket accuracy.** This needs hand-labelled live tickets; the synthetic and real-pack tests can't stand in for it.
- **Browser rendering of `report.html`** beyond checking its content. It was checked by eye with a headless Chrome screenshot.
- **Linux/Windows.** Tested on macOS (M1, 8 GB) with Python 3.12. The memory check reads `ru_maxrss` in both macOS and Linux units.
