# Day 4 Log - July 18, 2026

**Today's goal:** Stabilise the test suite, complete profiling, and build the training pipeline.

**Start Time:** 11.45 am

**End time:** 

---

## 1. Tasks Completed

- [x] Fix the 10 failing tests
- [ ] Complete model profiling notebook
- [ ] Implement LightningDataModule
- [ ] Write minimal training script

---

## 2. Unit Test Results

- **Initial status:** 10 failed, 78 passed, 2 skipped
- **Final status:** 86 passed, 2 skipped, 4 warnings in 0.58s

---



## 3. Today's Retrospective

- **What went well:** 
- **What I struggled with:** 
- **First action for tomorrow:** 

Parameters: 298.754K
FLOPs per forward pass (set size 16): 4.772M

## 4. Profiling Summary

- **Model:** `SetAnomalyPredictor` (embed_dim=128, heads=8, layers=2)
- **Parameters:** 298.754K  
- **FLOPs for set size 16:** 4.772M  
- **Scaling:** FLOPs grow quadratically with set size (expected for self‑attention). See `results/flops_vs_set_size.png`.

---
## Priority Tasks

1. Fix the 10 failing tests

    - Run pytest -v --tb=short to list failures.

    - Likely culprits:

        - test_modules.py – mask handling, dropout tests, or positional encoding odd‑dim.

        - test_attention.py – shape or gradient tests affected by mask changes.

    - Fix them one by one

2. Complete model profiling notebook

    - Adjust sys.path or run from project root to import src.

    - Save FLOPs vs. set‑size plot to results/.

3. Implement LightningDataModule    

    - Define train_dataloader, val_dataloader, test_dataloader.

4. Write a minimal training script (scripts/train.py)

    - Instantiate model, datamodule, and pl.Trainer.

    - Run a few epochs to verify the loss decreases.

### notes 
In VS Code (just before commit):
Right-click the notebook file => "Clear All Outputs" (if using the Jupyter extension)