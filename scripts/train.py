#!/usr/bin/env python3
"""
train.py
========
Phase 1: 5-seed ensemble with CosineAnnealingWarmRestarts, snapshots, early stopping.
Phase 2: All-data retrain with SWA for each seed.
Saves all checkpoints + result manifests to /kaggle/working/.
"""

import os
import sys
import time
import gc
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel
import warnings
warnings.filterwarnings('ignore')

from src.utils.config import load_config
from src.utils.data import (
    load_norm_stats,
    load_latlon,
    build_datasets,
    build_all_data,
)
from src.utils.metrics import (
    make_horizon_weights,
    weighted_huber_loss,
    domain_rmse,
    mixup_batch,
)
from models.baseline_model import build_model

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

cfg = load_config("configs/train.yaml")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = cfg.paths.data_dir
OUT_DIR  = cfg.paths.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Data dir: {DATA_DIR}")
print(f"Output dir: {OUT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# Seed utility
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ─────────────────────────────────────────────────────────────────────────────
# Load pre-computed data
# ─────────────────────────────────────────────────────────────────────────────

print("\nLoading norm stats & lat/lon...")
norm_stats = load_norm_stats(DATA_DIR)
lat_grid, lon_grid = load_latlon(DATA_DIR)

print("\nBuilding datasets (stride=1)...")
train_ds, val_ds = build_datasets(cfg, DATA_DIR, lat_grid, lon_grid)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

# Horizon weights for loss
HORIZON_WEIGHTS = make_horizon_weights(cfg.data.forecast_hours, DEVICE)

# Print model info
_tmp = build_model(cfg, DEVICE)
n_params = sum(p.numel() for p in _tmp.parameters() if p.requires_grad)
print(f"\nModel: TemporalFNO (modes_hi={cfg.model.fno_modes_hi}, modes_lo={cfg.model.fno_modes_lo}, "
      f"width={cfg.model.fno_width}, layers={cfg.model.fno_layers}, gru={cfg.model.gru_hidden})")
print(f"Params: {n_params:,}  (~{n_params*4/1024**2:.1f} MB)")
del _tmp

# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, norm_stats, device):
    model.eval()
    total_loss = 0; total_rmse = 0; n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += F.mse_loss(pred, y).item()
        total_rmse += domain_rmse(pred.float(), y.float(), norm_stats, cfg.features.target)
        n += 1
    return total_loss / n, total_rmse / n

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Seed Ensemble Training
# ═══════════════════════════════════════════════════════════════════════════════

ensemble_results = {}
snapshot_results = {}

SEEDS = cfg.ensemble.seeds
EPOCHS = cfg.training.epochs
BATCH_SIZE = cfg.training.batch_size
LR = cfg.training.lr
WEIGHT_DECAY = cfg.training.weight_decay
GRAD_CLIP = cfg.training.grad_clip
EARLY_STOP = cfg.training.early_stop_patience
SNAP_INTERVAL = cfg.training.snapshot_interval
MIXUP_ALPHA = cfg.augmentation.mixup_alpha

for seed in SEEDS:
    print(f"\n{'=' * 65}")
    print(f"  SEED {seed}  ({SEEDS.index(seed) + 1}/{len(SEEDS)})")
    print(f"{'=' * 65}")

    seed_everything(seed)
    model = build_model(cfg, DEVICE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=SNAP_INTERVAL, T_mult=1, eta_min=1e-6)

    best_rmse = float('inf'); best_epoch = 0; no_improve = 0
    save_path = os.path.join(OUT_DIR, f'best_model_v33_seed{seed}.pt')
    seed_snapshots = []

    print(f"  {'Ep':>4} | {'TrLoss':>9} | {'VaLoss':>9} | {'VaRMSE':>9} | {'LR':>9}")
    print(f"  {'-' * 52}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        total_loss = 0; n_batch = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x, y = mixup_batch(x, y, alpha=MIXUP_ALPHA)
            optimizer.zero_grad()
            pred = model(x)
            loss = weighted_huber_loss(pred, y, HORIZON_WEIGHTS)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item(); n_batch += 1

        scheduler.step()
        tr_loss = total_loss / n_batch

        va_loss, va_rmse = validate(model, val_loader, norm_stats, DEVICE)
        elapsed = time.time() - t0

        if va_rmse < best_rmse:
            best_rmse = va_rmse; best_epoch = epoch; no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1

        # Snapshot at LR valleys
        if epoch % SNAP_INTERVAL == 0:
            snap_path = os.path.join(OUT_DIR, f'snap_v33_seed{seed}_ep{epoch}.pt')
            torch.save(model.state_dict(), snap_path)
            seed_snapshots.append({'path': snap_path, 'rmse': va_rmse, 'epoch': epoch})
            print(f"  >>> Snapshot saved @ ep {epoch}, val RMSE: {va_rmse:.4f}")

        lr = optimizer.param_groups[0]['lr']
        if epoch % 10 == 0 or epoch <= 3 or no_improve == 0:
            best_mark = ' *' if no_improve == 0 else ''
            print(f"  {epoch:4d} | {tr_loss:9.5f} | {va_loss:9.5f} | "
                  f"{va_rmse:9.4f} | {lr:9.2e}{best_mark}  [{elapsed:.1f}s]")
            sys.stdout.flush()

        if no_improve >= EARLY_STOP:
            print(f"  Early stop ep {epoch}")
            break

    print(f"\n  Seed {seed} best RMSE: {best_rmse:.4f} @ ep {best_epoch}")
    ensemble_results[seed] = {
        'best_rmse': best_rmse,
        'best_epoch': best_epoch,
        'path': save_path,
    }
    snapshot_results[seed] = seed_snapshots

    del model, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\n{'=' * 65}")
print("ENSEMBLE SUMMARY (Phase 1)")
for s, r in ensemble_results.items():
    print(f"  seed={s:5d}  RMSE={r['best_rmse']:.4f}  ep={r['best_epoch']}")
mean_rmse = np.mean([r['best_rmse'] for r in ensemble_results.values()])
print(f"  Mean val RMSE: {mean_rmse:.4f}")
sys.stdout.flush()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Retrain on ALL data with SWA
# ═══════════════════════════════════════════════════════════════════════════════

avg_ep = int(np.mean([r['best_epoch'] for r in ensemble_results.values()]))
retrain_epochs = max(avg_ep + cfg.phase2.extra_epochs, cfg.phase2.min_retrain_epochs)
print(f"\n{'=' * 65}")
print(f"PHASE 2: Retraining on ALL data for {retrain_epochs} epochs")
print(f"{'=' * 65}")

all_data_ds = build_all_data(cfg, DATA_DIR, lat_grid, lon_grid)
print(f"All-data: {len(all_data_ds)} samples")

retrain_results = {}
for seed in SEEDS:
    print(f"\n  Retrain seed {seed}...")
    seed_everything(seed)
    model = build_model(cfg, DEVICE)

    all_loader = DataLoader(all_data_ds, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, pin_memory=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=retrain_epochs, T_mult=1, eta_min=1e-6)
    swa_model = AveragedModel(model)
    swa_start = int(retrain_epochs * cfg.phase2.swa_start_frac)

    for epoch in range(1, retrain_epochs + 1):
        model.train()
        for x, y in all_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x, y = mixup_batch(x, y, alpha=MIXUP_ALPHA)
            optimizer.zero_grad()
            pred = model(x)
            loss = weighted_huber_loss(pred, y, HORIZON_WEIGHTS)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        scheduler.step()
        if epoch >= swa_start:
            swa_model.update_parameters(model)
        if epoch % 20 == 0:
            print(f"    Ep {epoch}/{retrain_epochs}")
            sys.stdout.flush()

    torch.optim.swa_utils.update_bn(all_loader, swa_model, device=DEVICE)
    save_p = os.path.join(OUT_DIR, f'best_model_v33_all_seed{seed}.pt')
    torch.save(swa_model.module.state_dict(), save_p)
    retrain_results[seed] = {'path': save_p}

    del model, swa_model, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("Phase 2 complete!")
sys.stdout.flush()

# ═══════════════════════════════════════════════════════════════════════════════
# Save result manifests for infer.py
# ═══════════════════════════════════════════════════════════════════════════════

# Convert paths to strings for JSON serialization
ens_json = {str(k): {**v, 'path': str(v['path'])} for k, v in ensemble_results.items()}
snap_json = {str(k): [{'path': str(s['path']), 'rmse': s['rmse'], 'epoch': s['epoch']}
                       for s in v] for k, v in snapshot_results.items()}
retrain_json = {str(k): {'path': str(v['path'])} for k, v in retrain_results.items()}

with open(os.path.join(OUT_DIR, 'ensemble_results.json'), 'w') as f:
    json.dump(ens_json, f, indent=2)
with open(os.path.join(OUT_DIR, 'snapshot_results.json'), 'w') as f:
    json.dump(snap_json, f, indent=2)
with open(os.path.join(OUT_DIR, 'retrain_results.json'), 'w') as f:
    json.dump(retrain_json, f, indent=2)

print("\nSaved result manifests to", OUT_DIR)
print("\nTRAINING COMPLETE")
