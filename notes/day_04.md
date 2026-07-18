# Day 4 Log - July 18, 2026

**Today's goal:** Stabilise the test suite, complete profiling, and build the training pipeline.

**Start Time:** 11.45 am

**End time:** 

---

## 1. Tasks Completed

- [ ] Fix the 10 failing tests
- [ ] Complete model profiling notebook
- [ ] Implement LightningDataModule
- [ ] Write minimal training script

---

## 2. Unit Test Results

- **Initial status:** 10 failed, 78 passed, 2 skipped
- **Final status:** [fill at end]

---

## 3. Today's Retrospective

- **What went well:** 
- **What I struggled with:** 
- **First action for tomorrow:** 




---
## Priority Tasks

1. Fix the 10 failing tests

    - Run pytest -v --tb=short to list failures.

    - Likely culprits:

        - test_modules.py – mask handling, dropout tests, or positional encoding odd‑dim.

        - test_attention.py – shape or gradient tests affected by mask changes.

    - Fix them one by one (I can help if you share the error logs).

2. Complete model profiling notebook

    - Adjust sys.path or run from project root to import src.

    - Save FLOPs vs. set‑size plot to results/.

3. Implement LightningDataModule

    - Use the UvA tutorial’s synthetic data generation.

    - Define train_dataloader, val_dataloader, test_dataloader.

4. Write a minimal training script (scripts/train.py)

    - Instantiate model, datamodule, and pl.Trainer.

    - Run a few epochs to verify the loss decreases.

