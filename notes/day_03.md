# Day 3 Log - July 17, 2026

**Today's goal:** implement Lightning Module and tests

**Start Time:** 10.30 am

**End time:** 3.30 pm

---

## 1. Module Architecture

* [x] TransformerPredictor Lightning Module Implementation
* [x] Set Anomaly Predictor Lightning Module Implementation
* [x] Test TransformerPredictor Implementation
* [x] Test Set Anomaly Predictor Implementation

---


## 2. Unit Test Results
* **Smoke Test Status (smoke_test.py):** [Pass / Fail] not yet write
* **Pytest Run (`pytest -q`):** [10 failed, 78 passed, 2 skipped, 4 warnings] in 0.94s 

predictor‑specific tests (18) all pass, but other modules (attention, positional encoding, encoder blocks) have regressions that need investigation.

---

## 3. Today's Retrospective
* **What went well:**
* Successfully implemented both Lightning modules with clear separation of concerns.

* Overrode forward in the subclass to handle per‑element outputs without duplicating code.

* Wrote focused, non‑redundant tests; all predictor‑specific assertions pass.

* **What I struggled with:**
* Jupyter kernel / VS Code integration – wasted time troubleshooting kernel visibility;

* Unexpected test failures in other modules – likely due to changes made during predictor implementation (e.g., modified MultiheadAttention masking or PositionalEncoding odd‑dim handling). These need debugging tomorrow.

* **First action for tomorrow:** 

* - Debug and fix the 10 failing tests (run pytest -v to see the specific errors).

* - Resolve profiling notebook path issues so I can generate the FLOPs plot.

* - Once tests are green, start implementing the LightningDataModule and a training script.


