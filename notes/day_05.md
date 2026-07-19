# Day 5 Log - July 19, 2026

**Today's goal:** build the training pipeline.

**Start Time:** 10.30  am

**End time:** 3.00 pm

---

## 1. Tasks Completed

- [x] Implement `SetAnomalyDataset` with proper class means and tensor conversion
- [x] Add `SetAnomalyDataModule` with train/val/test splits
- [x] Create YAML config (`config/trainer/default.yaml`) with dataset, model, and training params
- [x] Write `train.py` with argparse, config loading, and seed reproducibility
- [x] Write `run_sweep.py` to run multiple seeds sequentially
- [x] Fix TensorBoard dependency and config path resolution
- [x] Resolve hyperparameter conflict between model and datamodule (`ignore=['num_classes', 'anomaly_labels']`)
- [x] Run sweep on 2 seeds (42, 43) – training completed successfully

---

## 2. Unit Test Results

- **Initial status:** 92 passed, 2 skipped, 4 warnings in 0.71s
- **Final status:** 92 passed, 2 skipped, 4 warnings in 0.71s
- **Smoke test:** 6 passed in 0.02s

---


## 3. Training Results

| Seed | Test Accuracy | Test Loss |
|------|---------------|-----------|
| 42   | 98.90%        | 0.0458    |
| 43   | 98.45%        | 0.0693    |
| **Average** | **98.68%** | **0.0576** |

The model reliably identifies the anomaly in synthetic sets, confirming the pipeline works.


## 4. Profiling Summary

- **Model:** `SetAnomalyPredictor` (embed_dim=128, heads=8, layers=2)
- **Trainable params:** 298 K (~1.2 MB)
- **FLOPs (set size 16):** 4.77 M (see `results/flops_vs_set_size.png`)



## 5. Today's Retrospective

- **What went well:** 
    - Training script is fully configurable and reproducible.
    - Multi‑seed sweep ran without errors and produced solid results.
    - Logging with TensorBoard works per seed.

- **What I struggled with:** 
    - Dataset generation – needed to pre‑compute class means per set and convert tensors outside the loop.
    - Config values were sometimes strings; added explicit casting to int/float.
    - Hyperparameter merge conflict between model and datamodule; fixed with `save_hyperparameters(ignore=...)`.

- **First action for tomorrow:** 
    - Analyse the training curves across seeds (TensorBoard logs).
    - Create `notebooks/03_attention_viz.ipynb` to visualise attention maps on sample sets.
    - Compute additional metrics (precision, recall, F1) and confusion matrix.
    - Prepare final project README and documentation.

    

---

## 6. Git Status

- **Branch merged:** `feature/training-loop`  -> `main`
- **Next branch:** `feature/training-analysis`

