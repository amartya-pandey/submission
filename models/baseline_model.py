"""
PM2.5 TemporalFNO Model
========================
ConvGRU temporal encoder + Multi-scale Fourier Neural Operator
for PM2.5 forecasting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# Channel count helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_channel_counts(cfg):
    """Compute channel counts from config values."""
    input_hours = cfg.data.input_hours
    total_window = cfg.data.total_window
    met_features = cfg.features.met_features
    gru_hidden = cfg.model.gru_hidden

    n_gru_out = gru_hidden                              # 32
    n_diff_ch = input_hours - 1                         # 9
    n_met_ch = len(met_features) * total_window         # 208
    n_wind_ch = 3 * total_window                        # 78
    n_latlon = 2                                        # 2
    n_fno_in = n_gru_out + n_diff_ch + n_met_ch + n_wind_ch + n_latlon  # 329

    n_pm25_ch = input_hours                             # 10
    n_raw_in = n_pm25_ch + n_diff_ch + n_met_ch + n_wind_ch + n_latlon  # 307

    return {
        'n_gru_out': n_gru_out,
        'n_diff_ch': n_diff_ch,
        'n_met_ch': n_met_ch,
        'n_wind_ch': n_wind_ch,
        'n_latlon': n_latlon,
        'n_fno_in': n_fno_in,
        'n_pm25_ch': n_pm25_ch,
        'n_raw_in': n_raw_in,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ConvGRU
# ═══════════════════════════════════════════════════════════════════════════════

class ConvGRUCell(nn.Module):
    """2D Convolutional GRU cell for spatiotemporal encoding."""
    def __init__(self, input_ch, hidden_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_ch = hidden_ch
        self.gates = nn.Conv2d(input_ch + hidden_ch, 2 * hidden_ch, kernel_size, padding=pad)
        self.cand  = nn.Conv2d(input_ch + hidden_ch, hidden_ch, kernel_size, padding=pad)

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=1)
        gates = torch.sigmoid(self.gates(combined))
        r, z = gates.chunk(2, dim=1)
        combined_r = torch.cat([x, r * h], dim=1)
        n = torch.tanh(self.cand(combined_r))
        return (1 - z) * h + z * n


class TemporalEncoder(nn.Module):
    """Encodes PM2.5 history via stacked ConvGRU → fixed-size spatial feature map."""
    def __init__(self, input_ch=1, hidden_ch=32, n_layers=2):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.n_layers = n_layers
        self.cells = nn.ModuleList()
        for i in range(n_layers):
            in_c = input_ch if i == 0 else hidden_ch
            self.cells.append(ConvGRUCell(in_c, hidden_ch))

    def forward(self, seq):
        # seq: (B, T, H, W)
        B, T, H, W = seq.shape
        h = [torch.zeros(B, self.hidden_ch, H, W, device=seq.device)
             for _ in range(self.n_layers)]
        for t in range(T):
            x_t = seq[:, t:t+1, :, :]  # (B, 1, H, W)
            for i, cell in enumerate(self.cells):
                h[i] = cell(x_t if i == 0 else h[i-1], h[i])
        return h[-1]  # (B, hidden_ch, H, W)


# ═══════════════════════════════════════════════════════════════════════════════
# Spectral / FNO layers
# ═══════════════════════════════════════════════════════════════════════════════

class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.out_ch = out_ch
        scale = 1 / (in_ch * out_ch)
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x.float())
        out_ft = torch.zeros(B, self.out_ch, x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.w1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.w2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNOBlock(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes, modes)
        self.pw       = nn.Conv2d(width, width, 1)
        self.norm     = nn.InstanceNorm2d(width)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral(x) + self.pw(x))) + x


class MultiScaleFNOBlock(nn.Module):
    """Parallel FNO at two resolutions: high modes + low modes → merge."""
    def __init__(self, width, modes_hi, modes_lo):
        super().__init__()
        w2 = width // 2
        self.split = nn.Conv2d(width, width, 1)
        self.hi = FNOBlock(w2, modes_hi)
        self.lo = FNOBlock(w2, modes_lo)
        self.merge = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU())
        self.norm = nn.InstanceNorm2d(width)

    def forward(self, x):
        h = self.split(x)
        h1, h2 = h.chunk(2, dim=1)
        h1 = self.hi(h1)
        h2 = self.lo(h2)
        out = self.merge(torch.cat([h1, h2], dim=1))
        return self.norm(out) + x


class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch // reduction), nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Main model
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalFNO(nn.Module):
    def __init__(self, fno_in_ch, out_ch=16, width=64, modes_hi=20, modes_lo=8,
                 n_layers=4, gru_hidden=32, dropout=0.15):
        super().__init__()
        self.gru_hidden = gru_hidden

        # Temporal encoder for PM2.5 history
        self.temporal_enc = TemporalEncoder(input_ch=1, hidden_ch=gru_hidden, n_layers=2)

        # Channel dropout during training
        self.channel_drop = nn.Dropout2d(p=0.05)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(fno_in_ch, width, 1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU())

        # FNO blocks — alternate multi-scale and regular
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            if i % 2 == 0:
                self.blocks.append(MultiScaleFNOBlock(width, modes_hi, modes_lo))
            else:
                self.blocks.append(FNOBlock(width, modes_hi))

        self.attn = ChannelAttention(width, reduction=8)

        # Decoder
        self.dec = nn.Sequential(
            nn.Conv2d(width, width * 2, 1), nn.GELU(), nn.Dropout2d(dropout),
            nn.Conv2d(width * 2, width, 1), nn.GELU())

        # Per-horizon output heads
        self.head_short = nn.Conv2d(width, 8, 1)   # hours 1-8
        self.head_long  = nn.Conv2d(width, 8, 1)   # hours 9-16

        # Persistence shortcut
        self.persist = nn.Conv2d(1, out_ch, 1)
        nn.init.ones_(self.persist.weight)
        nn.init.zeros_(self.persist.bias)

        # Init heads near zero so model starts from persistence
        for h in [self.head_short, self.head_long]:
            nn.init.zeros_(h.weight)
            nn.init.zeCFGros_(h.bias)

    def forward(self, x):
        # x: (B, N_RAW_IN, H, W) — packed [pm25(10), diff(9), met+wind(286), latlon(2)]
        pm25_hist = x[:, :10, :, :]           # (B, 10, H, W)
        diff      = x[:, 10:19, :, :]         # (B, 9, H, W)
        met_wind  = x[:, 19:305, :, :]        # (B, 286, H, W)
        latlon    = x[:, 305:307, :, :]       # (B, 2, H, W)

        last_pm25 = pm25_hist[:, 9:10, :, :]  # (B, 1, H, W)

        # Encode PM2.5 history via ConvGRU
        gru_out = self.temporal_enc(pm25_hist)  # (B, gru_hidden, H, W)

        # Assemble FNO input
        fno_in = torch.cat([gru_out, diff, met_wind, latlon], dim=1)

        # Channel dropout for regularization
        if self.training:
            fno_in = self.channel_drop(fno_in)

        h = self.encoder(fno_in)
        for blk in self.blocks:
            h = blk(h)
        h = self.attn(h)
        h = self.dec(h) + h

        # Per-horizon heads
        short = self.head_short(h)   # (B, 8, H, W)
        long_ = self.head_long(h)    # (B, 8, H, W)
        pred  = torch.cat([short, long_], dim=1)  # (B, 16, H, W)

        return pred + self.persist(last_pm25)


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(cfg, device):
    """Build TemporalFNO from a config namespace and move to device."""
    ch = compute_channel_counts(cfg)
    model = TemporalFNO(
        fno_in_ch=ch['n_fno_in'],
        out_ch=cfg.data.forecast_hours,
        width=cfg.model.fno_width,
        modes_hi=cfg.model.fno_modes_hi,
        modes_lo=cfg.model.fno_modes_lo,
        n_layers=cfg.model.fno_layers,
        gru_hidden=cfg.model.gru_hidden,
        dropout=cfg.model.dropout,
    ).to(device)
    return model
