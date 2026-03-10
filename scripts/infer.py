#!/usr/bin/env python3
"""
infer.py
========
Mega-ensemble inference: Phase 1 best + Phase 2 SWA + best snapshots.
4-fold TTA (horizontal/vertical flips). Outputs preds.npy.
"""

import os
import sys
import gc
import json
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

from src.utils.config import load_config
from src.utils.data import (
    load_norm_stats,
    load_latlon,
    load_test_data,
)
from models.baseline_model import build_model

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

cfg = load_config("configs/infer.yaml")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR  = cfg.paths.model_dir
DATA_DIR   = cfg.paths.data_dir
OUTPUT_FILE = cfg.paths.output_file
BS = cfg.inference.batch_size

print(f"Device: {DEVICE}")
print(f"Model dir: {MODEL_DIR}")
print(f"Output: {OUTPUT_FILE}")

# ─────────────────────────────────────────────────────────────────────────────
# Load pre-computed stats and test data
# ─────────────────────────────────────────────────────────────────────────────

print("\nLoading norm stats & lat/lon...")
norm_stats = load_norm_stats(DATA_DIR)
lat_grid, lon_grid = load_latlon(DATA_DIR)

print("\nLoading test data...")
test_input = load_test_data(cfg, norm_stats, lat_grid, lon_grid)

# ─────────────────────────────────────────────────────────────────────────────
# TTA Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_tta(model, test_input, norm_stats, device, bs=16):
    """4-fold flip TTA: original + h-flip + v-flip + hv-flip."""
    model.eval()
    target_key = cfg.features.target
    std_v = norm_stats[target_key]['std']
    mean_v = norm_stats[target_key]['mean']
    N = test_input.shape[0]
    preds = []

    for i in range(0, N, bs):
        b = torch.from_numpy(test_input[i:i + bs]).float().to(device)
        p0 = model(b)
        p1 = torch.flip(model(torch.flip(b, [-1])), [-1])
        p2 = torch.flip(model(torch.flip(b, [-2])), [-2])
        p3 = torch.flip(model(torch.flip(b, [-1, -2])), [-1, -2])
        p = ((p0 + p1 + p2 + p3) / 4.0).cpu().numpy()
        p = p * std_v + mean_v
        p = p.transpose(0, 2, 3, 1)  # (B, H, W, 16)
        preds.append(p)

    return np.concatenate(preds, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Load result manifests from train.py
# ─────────────────────────────────────────────────────────────────────────────

with open(os.path.join(MODEL_DIR, 'ensemble_results.json'), 'r') as f:
    ensemble_results = json.load(f)

with open(os.path.join(MODEL_DIR, 'snapshot_results.json'), 'r') as f:
    snapshot_results = json.load(f)

with open(os.path.join(MODEL_DIR, 'retrain_results.json'), 'r') as f:
    retrain_results = json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# Build mega-ensemble model list
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- Mega-Ensemble Inference ---")
all_model_paths = []

# Phase 1 best per seed
for seed, res in ensemble_results.items():
    all_model_paths.append(('P1', seed, res['path']))

# Phase 2 all-data retrained
for seed, res in retrain_results.items():
    all_model_paths.append(('P2', seed, res['path']))

# Best snapshot per seed (if different from Phase 1 best)
for seed, snaps in snapshot_results.items():
    if snaps:
        best_snap = min(snaps, key=lambda s: s['rmse'])
        if best_snap['path'] != ensemble_results[seed]['path']:
            all_model_paths.append(('SN', seed, best_snap['path']))

print(f"Total models in mega-ensemble: {len(all_model_paths)}")

# ─────────────────────────────────────────────────────────────────────────────
# Run inference
# ─────────────────────────────────────────────────────────────────────────────

all_preds = []
for tag, seed, path in all_model_paths:
    print(f"  Inferring {tag} seed={seed}...")
    m = build_model(cfg, DEVICE)
    sd = torch.load(path, map_location=DEVICE, weights_only=False)
    m.load_state_dict(sd)
    p = infer_tta(m, test_input, norm_stats, DEVICE, bs=BS)
    all_preds.append(p)
    del m, sd
    gc.collect()
    sys.stdout.flush()

# ─────────────────────────────────────────────────────────────────────────────
# Average & save
# ─────────────────────────────────────────────────────────────────────────────

preds = np.clip(np.mean(all_preds, axis=0), 0, None)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
np.save(OUTPUT_FILE, preds)
print(f"\nFinal preds → {OUTPUT_FILE}  shape={preds.shape}")
print(f"  range=[{preds.min():.2f}, {preds.max():.2f}], mean={preds.mean():.2f}")

# Also save Phase1-only and Phase2-only for comparison
n_ens = len(ensemble_results)
n_ret = len(retrain_results)

p1_preds = np.clip(np.mean(all_preds[:n_ens], axis=0), 0, None)
np.save(os.path.join(os.path.dirname(OUTPUT_FILE), 'preds_v33_phase1.npy'), p1_preds)

p2_preds = np.clip(np.mean(all_preds[n_ens:n_ens + n_ret], axis=0), 0, None)
np.save(os.path.join(os.path.dirname(OUTPUT_FILE), 'preds_v33_phase2.npy'), p2_preds)

print("\nINFERENCE COMPLETE")
