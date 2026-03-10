"""
PM2.5 Dataset & data-loading utilities.
Shared by prepare_dataset.py, train.py, and infer.py.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════════════════════════════

def compute_normalization_stats(raw_dir, months, target, met_features, H, W):
    """Compute z-score stats over all months for each feature + wind_speed."""
    raw_dir = Path(raw_dir)
    all_features = [target] + met_features
    stats = {}

    for feat in all_features:
        rsum = 0.0; rsq = 0.0; cnt = 0
        for month in months:
            fpath = raw_dir / month / f'{feat}.npy'
            if not fpath.exists():
                continue
            arr = np.load(fpath).astype(np.float64)
            rsum += arr.sum(); rsq += (arr**2).sum(); cnt += arr.size
        if cnt > 0:
            mean = rsum / cnt
            std = max(np.sqrt(rsq / cnt - mean**2), 1e-8)
            stats[feat] = {'mean': float(mean), 'std': float(std)}

    # Wind speed stats
    ws_sum = 0.0; ws_sq = 0.0; ws_cnt = 0
    for month in months:
        u10 = np.load(raw_dir / month / 'u10.npy').astype(np.float64)
        v10 = np.load(raw_dir / month / 'v10.npy').astype(np.float64)
        if u10.shape[0] == H:
            u10 = u10.transpose(2, 0, 1)
            v10 = v10.transpose(2, 0, 1)
        ws = np.sqrt(u10**2 + v10**2)
        ws_sum += ws.sum(); ws_sq += (ws**2).sum(); ws_cnt += ws.size

    ws_mean = ws_sum / ws_cnt
    ws_std = max(np.sqrt(ws_sq / ws_cnt - ws_mean**2), 1e-8)
    stats['wind_speed'] = {'mean': float(ws_mean), 'std': float(ws_std)}

    for k, v in stats.items():
        print(f"  {k:15s}  mean={v['mean']:10.4f}  std={v['std']:10.4f}")

    return stats


def save_norm_stats(stats, save_dir):
    """Save normalization stats as JSON."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'norm_stats.json')
    with open(path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved norm_stats → {path}")


def load_norm_stats(data_dir):
    """Load normalization stats from JSON."""
    path = os.path.join(data_dir, 'norm_stats.json')
    with open(path, 'r') as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# Lat/Lon
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_save_latlon(raw_dir, save_dir):
    """Load lat/lon from raw, normalize, and save to save_dir."""
    raw_dir = Path(raw_dir)
    ll = np.load(raw_dir / 'lat_long.npy').astype(np.float32)
    lat = ll[:, :, 0]; lon = ll[:, :, 1]
    lat = (lat - lat.mean()) / (lat.std() + 1e-8)
    lon = (lon - lon.mean()) / (lon.std() + 1e-8)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'lat.npy'), lat)
    np.save(os.path.join(save_dir, 'lon.npy'), lon)
    print(f"  Saved lat/lon grids → {save_dir}")
    return lat, lon


def load_latlon(data_dir):
    """Load pre-saved normalized lat/lon grids."""
    lat = np.load(os.path.join(data_dir, 'lat.npy'))
    lon = np.load(os.path.join(data_dir, 'lon.npy'))
    return lat, lon


# ═══════════════════════════════════════════════════════════════════════════════
# Raw month loading & normalization
# ═══════════════════════════════════════════════════════════════════════════════

def load_raw_month(raw_dir, month_name, features, norm_stats, H, W):
    """Load a single month's data, normalize, and compute wind features."""
    month_dir = Path(raw_dir) / month_name
    data = {}

    for feat in features:
        arr = np.load(month_dir / f'{feat}.npy').astype(np.float32)
        if arr.ndim == 3 and arr.shape[0] == H and arr.shape[1] == W:
            arr = arr.transpose(2, 0, 1)
        if feat in norm_stats:
            arr = (arr - norm_stats[feat]['mean']) / norm_stats[feat]['std']
        data[feat] = arr

    # Wind derived features
    u10_raw = np.load(month_dir / 'u10.npy').astype(np.float32)
    v10_raw = np.load(month_dir / 'v10.npy').astype(np.float32)
    if u10_raw.shape[0] == H:
        u10_raw = u10_raw.transpose(2, 0, 1)
        v10_raw = v10_raw.transpose(2, 0, 1)
    ws = np.sqrt(u10_raw**2 + v10_raw**2)
    wd = np.arctan2(v10_raw, u10_raw)
    data['wind_speed'] = (ws - norm_stats['wind_speed']['mean']) / norm_stats['wind_speed']['std']
    data['wind_sin'] = np.sin(wd)
    data['wind_cos'] = np.cos(wd)

    return data


def save_month_data(data, save_dir, month_name):
    """Save all feature arrays for a month to disk."""
    month_dir = os.path.join(save_dir, month_name)
    os.makedirs(month_dir, exist_ok=True)
    for feat, arr in data.items():
        np.save(os.path.join(month_dir, f'{feat}.npy'), arr)
    print(f"    Saved {len(data)} features → {month_dir}")


def load_month_data(data_dir, month_name, features):
    """Load pre-saved normalized month arrays from disk."""
    month_dir = os.path.join(data_dir, month_name)
    data = {}
    for feat in features:
        data[feat] = np.load(os.path.join(month_dir, f'{feat}.npy'))
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class PM25Dataset(Dataset):
    """Sliding-window dataset for PM2.5 forecasting with augmentation."""

    def __init__(self, month_data_list, lat_grid, lon_grid, cfg, stride=1, augment=False):
        """
        Args:
            month_data_list: list of dicts, each dict has feature_name → (T, H, W) array
            lat_grid: (H, W) normalized latitude grid
            lon_grid: (H, W) normalized longitude grid
            cfg: config namespace with data.*, augmentation.* fields
            stride: sliding window stride
            augment: whether to apply augmentations
        """
        self.samples = []
        self.augment = augment
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.cfg = cfg

        target = cfg.features.target
        total_window = cfg.data.total_window

        for data in month_data_list:
            T = data[target].shape[0]
            for start in range(0, T - total_window + 1, stride):
                self.samples.append({'data': data, 'start': start})
        print(f"  Dataset: {len(self.samples)} samples (augment={augment})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        data, t0 = info['data'], info['start']
        cfg = self.cfg

        target_key = cfg.features.target
        met_features = cfg.features.met_features
        input_hours = cfg.data.input_hours
        total_window = cfg.data.total_window
        H = cfg.data.H
        W = cfg.data.W

        target = data[target_key][t0 + input_hours : t0 + total_window]   # (16, H, W)
        cpm25_hist = data[target_key][t0 : t0 + input_hours]              # (10, H, W)

        # Temporal difference
        cpm25_diff = cpm25_hist[1:] - cpm25_hist[:-1]  # (9, H, W)

        # Met + wind features for full 26h window
        met_feats = []
        for feat in met_features:
            met_feats.append(data[feat][t0 : t0 + total_window])
        met_feats.append(data['wind_speed'][t0 : t0 + total_window])
        met_feats.append(data['wind_sin'][t0 : t0 + total_window])
        met_feats.append(data['wind_cos'][t0 : t0 + total_window])
        met_wind = np.concatenate(met_feats, axis=0)  # (286, H, W)

        lat_lon = np.stack([self.lat_grid, self.lon_grid], axis=0)  # (2, H, W)

        # Convert to tensors
        inp_pm25 = torch.from_numpy(cpm25_hist.copy()).float()
        inp_diff = torch.from_numpy(cpm25_diff.copy()).float()
        inp_met  = torch.from_numpy(met_wind.copy()).float()
        inp_ll   = torch.from_numpy(lat_lon.copy()).float()
        target   = torch.from_numpy(target.copy()).float()

        if self.augment:
            aug = cfg.augmentation

            # Flip augmentation
            if aug.augment_flips:
                if torch.rand(1).item() < 0.5:
                    inp_pm25 = torch.flip(inp_pm25, [-1])
                    inp_diff = torch.flip(inp_diff, [-1])
                    inp_met  = torch.flip(inp_met, [-1])
                    inp_ll   = torch.flip(inp_ll, [-1])
                    target   = torch.flip(target, [-1])
                if torch.rand(1).item() < 0.5:
                    inp_pm25 = torch.flip(inp_pm25, [-2])
                    inp_diff = torch.flip(inp_diff, [-2])
                    inp_met  = torch.flip(inp_met, [-2])
                    inp_ll   = torch.flip(inp_ll, [-2])
                    target   = torch.flip(target, [-2])

            # Gaussian noise on inputs
            if aug.noise_std > 0:
                inp_pm25 = inp_pm25 + torch.randn_like(inp_pm25) * aug.noise_std
                inp_diff = inp_diff + torch.randn_like(inp_diff) * aug.noise_std

            # Spatial cutout
            if aug.cutout_prob > 0 and torch.rand(1).item() < aug.cutout_prob:
                cs = aug.cutout_size
                y0 = torch.randint(0, H - cs, (1,)).item()
                x0 = torch.randint(0, W - cs, (1,)).item()
                inp_pm25[:, y0:y0+cs, x0:x0+cs] = 0
                inp_met[:, y0:y0+cs, x0:x0+cs] = 0

        # Pack into single tensor: (307, H, W)
        inp = torch.cat([inp_pm25, inp_diff, inp_met, inp_ll], dim=0)
        return inp, target


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset builders (used by train.py)
# ═══════════════════════════════════════════════════════════════════════════════

ALL_MONTH_FEATURES = None  # will be set dynamically


def _get_all_features(cfg):
    """Return list of all feature keys stored per month."""
    target = cfg.features.target
    met = cfg.features.met_features
    return [target] + met + ['wind_speed', 'wind_sin', 'wind_cos']


def build_datasets(cfg, data_dir, lat_grid, lon_grid):
    """Build train/val PM25Datasets from pre-saved month arrays."""
    months = cfg.data.months
    val_ratio = cfg.data.val_ratio
    stride = cfg.data.stride
    all_feats = _get_all_features(cfg)
    target = cfg.features.target

    train_list, val_list = [], []
    for month in months:
        print(f"  Loading {month}...")
        data = load_month_data(data_dir, month, all_feats)
        T = data[target].shape[0]
        split_t = int(T * (1 - val_ratio))
        train_list.append({k: v[:split_t] for k, v in data.items()})
        val_list.append({k: v[split_t:] for k, v in data.items()})
        print(f"    T={T}, train={split_t}, val={T - split_t}")

    train_ds = PM25Dataset(train_list, lat_grid, lon_grid, cfg, stride=stride, augment=True)
    val_ds   = PM25Dataset(val_list,   lat_grid, lon_grid, cfg, stride=stride, augment=False)
    return train_ds, val_ds


def build_all_data(cfg, data_dir, lat_grid, lon_grid):
    """Build a PM25Dataset from ALL month data (no split) for Phase 2 retrain."""
    months = cfg.data.months
    stride = cfg.data.stride
    all_feats = _get_all_features(cfg)

    data_list = []
    for month in months:
        data = load_month_data(data_dir, month, all_feats)
        data_list.append(data)

    return PM25Dataset(data_list, lat_grid, lon_grid, cfg, stride=stride, augment=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Test data loading (used by infer.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_test_data(cfg, norm_stats, lat_grid, lon_grid):
    """Load and assemble test data into packed tensor format."""
    test_dir = Path(cfg.paths.test_dir)
    target = cfg.features.target
    met_features = cfg.features.met_features
    total_window = cfg.data.total_window
    H, W = cfg.data.H, cfg.data.W
    input_hours = cfg.data.input_hours

    # Compute expected channel count
    n_pm25 = input_hours                                # 10
    n_diff = input_hours - 1                            # 9
    n_met  = len(met_features) * total_window           # 208
    n_wind = 3 * total_window                           # 78
    n_ll   = 2                                          # 2
    n_raw_in = n_pm25 + n_diff + n_met + n_wind + n_ll  # 307

    # PM2.5 history
    cpm25 = np.load(test_dir / 'cpm25.npy').astype(np.float32)
    print(f"  cpm25 test: {cpm25.shape}")
    cpm25 = (cpm25 - norm_stats[target]['mean']) / norm_stats[target]['std']
    n = cpm25.shape[0]

    cpm25_diff = cpm25[:, 1:, :, :] - cpm25[:, :-1, :, :]

    # Met features
    met_feats = []
    for feat in met_features:
        fp = test_dir / f'{feat}.npy'
        if not fp.exists():
            fp = test_dir / f'{feat.lower()}.npy'
        if fp.exists():
            arr = np.load(fp).astype(np.float32)
            if feat in norm_stats:
                arr = (arr - norm_stats[feat]['mean']) / norm_stats[feat]['std']
            met_feats.append(arr)
        else:
            met_feats.append(np.zeros((n, total_window, H, W), dtype=np.float32))

    # Wind
    u10 = np.load(test_dir / 'u10.npy').astype(np.float32)
    v10 = np.load(test_dir / 'v10.npy').astype(np.float32)
    ws = np.sqrt(u10**2 + v10**2)
    wd = np.arctan2(v10, u10)
    ws = (ws - norm_stats['wind_speed']['mean']) / norm_stats['wind_speed']['std']
    met_feats.append(ws)
    met_feats.append(np.sin(wd))
    met_feats.append(np.cos(wd))
    met_wind = np.concatenate(met_feats, axis=1)

    lat_ch = np.broadcast_to(lat_grid[np.newaxis, np.newaxis], (n, 1, H, W)).copy()
    lon_ch = np.broadcast_to(lon_grid[np.newaxis, np.newaxis], (n, 1, H, W)).copy()

    # Pack: [pm25(10), diff(9), met+wind(286), latlon(2)]
    test_input = np.concatenate([cpm25, cpm25_diff, met_wind, lat_ch, lon_ch], axis=1)
    print(f"  Combined: {test_input.shape}")
    assert test_input.shape[1] == n_raw_in, \
        f"Channel mismatch: {test_input.shape[1]} vs {n_raw_in}"

    return test_input
