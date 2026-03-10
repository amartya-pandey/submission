"""
Loss functions & training metrics for PM2.5 forecasting.
"""

import numpy as np
import torch


def make_horizon_weights(forecast_hours, device):
    """Create per-horizon weights: heavier on short-range, lighter on long-range."""
    w = torch.linspace(1.5, 0.5, forecast_hours)
    w = w / w.mean()
    return w.view(1, forecast_hours, 1, 1).to(device)


def weighted_huber_loss(pred, target, horizon_weights, delta=0.5):
    """Huber loss with per-horizon weighting. Robust to DEC outliers."""
    diff = pred - target
    abs_diff = diff.abs()
    quadratic = torch.clamp(abs_diff, max=delta)
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return (loss * horizon_weights).mean()


def domain_rmse(pred, target, norm_stats, target_key='cpm25'):
    """Compute RMSE in original (denormalized) domain."""
    std_v = norm_stats[target_key]['std']
    mean_v = norm_stats[target_key]['mean']
    pred_o = pred * std_v + mean_v
    tgt_o = target * std_v + mean_v
    rmse_per = torch.sqrt(((pred_o - tgt_o)**2).mean(dim=(-2, -1)))
    return rmse_per.mean().item()


def mixup_batch(x, y, alpha=0.3):
    """Mixup augmentation on a batch."""
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# Numpy-based metrics (for evaluation)
# ═══════════════════════════════════════════════════════════════════════════════

def rmse(actual, pred, dim=(1, 2)):
    error = actual - pred
    return np.sqrt(np.nanmean(error**2, axis=dim))


def mfb(actual, pred, dim=(1, 2)):
    error = (pred - actual) / (pred + actual)
    error = np.where(np.isfinite(error), error, np.nan)
    return 2 * np.nanmean(error, axis=dim)


def smape(actual, pred, dim=(1, 2)):
    error = np.abs((actual - pred) / (actual + pred))
    error = np.where(np.isfinite(error), error, np.nan)
    return 200 * np.nanmean(error, axis=dim)
