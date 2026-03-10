#!/usr/bin/env python3
"""
prepare_dataset.py
==================
Computes normalization stats, normalizes per-month raw data,
and saves everything to /kaggle/temp/data/ for train.py and infer.py.
"""

import os
import sys
import numpy as np

from src.utils.config import load_config
from src.utils.data import (
    compute_normalization_stats,
    save_norm_stats,
    load_and_save_latlon,
    load_raw_month,
    save_month_data,
)

# ─────────────────────────────────────────────────────────────────────────────
# Load config
# ─────────────────────────────────────────────────────────────────────────────

cfg = load_config("configs/prepare_dataset.yaml")

RAW_PATH  = cfg.paths.raw_path
SAVE_DIR  = cfg.paths.save_dir
MONTHS    = cfg.data.months
H, W      = cfg.data.H, cfg.data.W
TARGET    = cfg.features.target
MET_FEATS = cfg.features.met_features

os.makedirs(SAVE_DIR, exist_ok=True)

print("=" * 60)
print("PREPARE DATASET")
print(f"  Raw path : {RAW_PATH}")
print(f"  Save dir : {SAVE_DIR}")
print(f"  Months   : {MONTHS}")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Compute & save normalization stats
# ─────────────────────────────────────────────────────────────────────────────

print("\nComputing normalization stats...")
norm_stats = compute_normalization_stats(RAW_PATH, MONTHS, TARGET, MET_FEATS, H, W)
save_norm_stats(norm_stats, SAVE_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Load, normalize & save lat/lon grids
# ─────────────────────────────────────────────────────────────────────────────

print("\nProcessing lat/lon grids...")
load_and_save_latlon(RAW_PATH, SAVE_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: For each month, normalize all features and save arrays
# ─────────────────────────────────────────────────────────────────────────────

all_features = [TARGET] + MET_FEATS

print("\nProcessing monthly data...")
for month in MONTHS:
    print(f"\n  Month: {month}")
    data = load_raw_month(RAW_PATH, month, all_features, norm_stats, H, W)
    save_month_data(data, SAVE_DIR, month)
    T = data[TARGET].shape[0]
    print(f"    Timesteps: {T}")
    del data

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)
