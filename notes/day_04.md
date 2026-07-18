# Day 4 Log - July 18, 2026

**Today's goal:** Stabilise the test suite, complete profiling, and build the training pipeline.

**Start Time:** 11.45 am

**End time:** 3.30 pm

---

## 1. Tasks Completed

- [x] Fix the 10 failing tests (device mismatches)
- [x] Smoke test (6 tests passed in 0.2s)
- [x] Complete model profiling notebook – saved plot to `results/`
- [ ] Implement LightningDataModule → *carried over to Day 5*
- [ ] Write minimal training script → *carried over to Day 5*

---

## 2. Unit Test Results

- **Initial status:** 10 failed, 78 passed, 2 skipped
- **Final status:** 86 passed, 2 skipped, 4 warnings in 0.58s
- **Smoke test:** 6 passed in 0.2s

---



## 3. Today's Retrospective

- **What went well:** 
    - Fixed test regressions quickly by moving `pe` and `mask` to the correct device.
    - Profiling notebook now runs and saves outputs reliably – `Agg` backend solved blank image issue.

- **What I struggled with:** 
    - VS Code kernel selection took time; resolved by selecting `.venv/bin/python` directly.
    - `thop` wasn't pre‑installed; added it manually.
    - Needed to adjust `sys.path` in notebooks for imports.

- **First action for tomorrow:** 

    1. Implement LightningDataModule    

        - Define train_dataloader, val_dataloader, test_dataloader.

    2. Write a minimal training script (scripts/train.py)

        - Instantiate model, datamodule, and pl.Trainer.

        - Run a few epochs to verify the loss decreases.


## 4. Profiling Summary

- **Model:** `SetAnomalyPredictor` (embed_dim=128, heads=8, layers=2)
- **Parameters:** 298.754K  
- **FLOPs for set size 16:** 4.772M  
- **Scaling:** FLOPs grow quadratically with set size (expected for self‑attention). See `results/flops_vs_set_size.png`.



---

## 5. Git Status

- **Branch merged:** `feature/training-pipeline` → `main`
- **Next branch:** `feature/training-loop` (Day 5)

