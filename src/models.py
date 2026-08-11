# -*- coding: utf-8 -*-
"""
4_12_base/models.py
All 6 model classes + GEV/Gumbel math (numpy & torch).

Models
------
GEVNNSingleTower   (TYPE="gev_st")     – [256,128]→3  (μ,σ,ξ)
GEVNNGumbelST      (TYPE="gumbel_st")  – [256,128]→2  (μ,σ), ξ≡0
GEVNNGumbelDT      (TYPE="gumbel_dt")  – 2×[128,64]→1 (μ, σ separate towers)
GEVNNTripleTower   (TYPE="gev_tt")     – 3×[128,64]→1 (μ, σ, ξ separate)
ANNDirect          (TYPE="ann_direct") – [256,128]→6  log(Q2…Q100)
RF is handled by sklearn; no class here.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

RETURN_PERIODS = [2, 5, 10, 20, 50, 100]

# ============================================================
# GEV / Gumbel math
# ============================================================

def gev_quantile_np(mu, sigma, xi, T):
    p, eps = 1.0 - 1.0 / T, 1e-8
    yp     = -np.log(-np.log(p))
    q      = mu + sigma * yp
    mask   = np.abs(xi) >= eps if np.ndim(xi) > 0 else (abs(xi) >= eps)
    if np.ndim(xi) > 0:
        xi_s  = np.where(mask, xi, eps)
        q_gev = mu + sigma / xi_s * ((-np.log(p)) ** (-xi_s) - 1.0)
        q     = np.where(mask, q_gev, q)
    elif abs(xi) >= eps:
        q = mu + sigma / xi * ((-np.log(p)) ** (-xi) - 1.0)
    return q


def gumbel_quantile_np(mu, sigma, T):
    p  = 1.0 - 1.0 / T
    yp = -np.log(-np.log(p))
    return mu + sigma * yp


def gev_quantile_torch(mu, sigma, xi, T):
    p   = 1.0 - 1.0 / T
    nlp = -np.log(1.0 - 1.0 / T)
    yp  = float(-np.log(nlp))
    eps = 1e-6
    q_g = mu + sigma * yp
    nlp_t  = torch.tensor(nlp, device=mu.device, dtype=mu.dtype)
    xi_abs = xi.abs()
    xi_s   = torch.where(xi_abs >= eps, xi, eps * torch.ones_like(xi))
    q_gev  = mu + sigma / xi_s * (nlp_t.pow(-xi) - 1.0)
    mask   = (xi_abs >= eps).float()
    return mask * q_gev + (1.0 - mask) * q_g


def gumbel_quantile_torch(mu, sigma, T):
    p  = 1.0 - 1.0 / T
    yp = float(-np.log(-np.log(p)))
    return mu + sigma * yp


def gev_nll_torch(mu, sigma, xi, ams, ams_mask):
    eps = 1e-6
    s   = sigma.clamp(min=0.01)
    t   = (ams - mu.unsqueeze(1)) / s.unsqueeze(1)
    xi_b = xi.unsqueeze(1)
    # Gumbel branch
    nll_g = s.log().unsqueeze(1) + t + torch.exp((-t).clamp(max=20))
    # GEV branch
    z      = (1.0 + xi_b * t).clamp(min=eps)
    xi_s   = torch.where(xi_b.abs() >= eps, xi_b, eps * torch.ones_like(xi_b))
    log_z  = torch.log(z)
    nll_v  = s.log().unsqueeze(1) + (1.0 + 1.0 / xi_s) * log_z + torch.exp((-log_z / xi_s).clamp(max=20))
    use_gev = (xi.abs() >= eps).unsqueeze(1).expand_as(nll_v)
    z_ok    = (ams - mu.unsqueeze(1)) * (xi_b.expand_as(ams)) > -s.unsqueeze(1)
    valid   = ams_mask & ((~use_gev) | z_ok)
    nll     = torch.where(use_gev, nll_v, nll_g)
    nll_s   = torch.where(valid, nll, torch.zeros_like(nll))
    return nll_s.sum() / valid.float().sum().clamp(min=1.0)


def gumbel_nll_torch(mu, sigma, ams, ams_mask):
    s   = sigma.clamp(min=0.01)
    t   = (ams - mu.unsqueeze(1)) / s.unsqueeze(1)
    nll = s.log().unsqueeze(1) + t + torch.exp((-t).clamp(max=20))
    nll = torch.where(ams_mask, nll, torch.zeros_like(nll))
    return nll.sum() / ams_mask.float().sum().clamp(min=1.0)


# ============================================================
# MLP factory
# ============================================================

def _make_mlp(in_dim: int, out_dim: int, hidden_dims: list[int],
              dropout: float) -> nn.Sequential:
    layers, prev = [], in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                   nn.LeakyReLU(0.1), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ============================================================
# Model classes
# ============================================================

class GEVNNSingleTower(nn.Module):
    """Single-tower MLP → (μ, σ, ξ).   hidden=[256,128]"""
    TYPE = "gev_st"
    HIDDEN = [256, 128]

    def __init__(self, in_dim: int, dropout: float = 0.1,
                 hidden_dims: list[int] | None = None):
        super().__init__()
        hd = self.HIDDEN if hidden_dims is None else [int(h) for h in hidden_dims]
        self.net = _make_mlp(in_dim, 3, hd, dropout)

    def init_bias(self, mu_med: float, sig_med: float):
        with torch.no_grad():
            self.net[-1].bias[0] = float(np.log(max(mu_med,  1e-6)))
            self.net[-1].bias[1] = float(np.log(max(sig_med, 1e-6)))
            self.net[-1].bias[2] = 0.0

    def forward(self, x):
        r  = self.net(x)
        mu  = torch.exp(r[:, 0].clamp(-6, 6))
        sig = torch.exp(r[:, 1].clamp(-6, 6))
        xi  = torch.tanh(r[:, 2]) * 0.5
        return mu, sig, xi


class GEVNNGumbelST(nn.Module):
    """Single-tower Gumbel-NN → (μ, σ), ξ≡0.  hidden=[256,128]"""
    TYPE = "gumbel_st"
    HIDDEN = [256, 128]

    def __init__(self, in_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = _make_mlp(in_dim, 2, self.HIDDEN, dropout)

    def init_bias(self, mu_med: float, sig_med: float):
        with torch.no_grad():
            self.net[-1].bias[0] = float(np.log(max(mu_med,  1e-6)))
            self.net[-1].bias[1] = float(np.log(max(sig_med, 1e-6)))

    def forward(self, x):
        r   = self.net(x)
        mu  = torch.exp(r[:, 0].clamp(-6, 6))
        sig = torch.exp(r[:, 1].clamp(-6, 6))
        return mu, sig


class GEVNNGumbelDT(nn.Module):
    """Dual-tower Gumbel-NN.  Each tower: [128,64]→1"""
    TYPE = "gumbel_dt"
    HIDDEN = [128, 64]

    def __init__(self, in_dim: int, dropout: float = 0.1):
        super().__init__()
        self.tower_mu  = _make_mlp(in_dim, 1, self.HIDDEN, dropout)
        self.tower_sig = _make_mlp(in_dim, 1, self.HIDDEN, dropout)

    def init_bias(self, mu_med: float, sig_med: float):
        with torch.no_grad():
            self.tower_mu[-1].bias[0]  = float(np.log(max(mu_med,  1e-6)))
            self.tower_sig[-1].bias[0] = float(np.log(max(sig_med, 1e-6)))

    def forward(self, x):
        mu  = torch.exp(self.tower_mu(x)[:, 0].clamp(-6, 6))
        sig = torch.exp(self.tower_sig(x)[:, 0].clamp(-6, 6))
        return mu, sig


class GEVNNTripleTower(nn.Module):
    """Triple-tower GEV-NN.  Each tower: [128,64]→1"""
    TYPE = "gev_tt"
    HIDDEN = [128, 64]

    def __init__(self, in_dim: int, dropout: float = 0.1):
        super().__init__()
        self.tower_mu  = _make_mlp(in_dim, 1, self.HIDDEN, dropout)
        self.tower_sig = _make_mlp(in_dim, 1, self.HIDDEN, dropout)
        self.tower_xi  = _make_mlp(in_dim, 1, self.HIDDEN, dropout)

    def init_bias(self, mu_med: float, sig_med: float):
        with torch.no_grad():
            self.tower_mu[-1].bias[0]  = float(np.log(max(mu_med,  1e-6)))
            self.tower_sig[-1].bias[0] = float(np.log(max(sig_med, 1e-6)))
            self.tower_xi[-1].bias[0]  = 0.0

    def forward(self, x):
        mu  = torch.exp(self.tower_mu(x)[:, 0].clamp(-6, 6))
        sig = torch.exp(self.tower_sig(x)[:, 0].clamp(-6, 6))
        xi  = torch.tanh(self.tower_xi(x)[:, 0]) * 0.5
        return mu, sig, xi


class ANNDirect(nn.Module):
    """Single-tower → 6 outputs = log(Q2,Q5,Q10,Q20,Q50,Q100).  hidden=[256,128]"""
    TYPE = "ann_direct"
    HIDDEN = [256, 128]

    def __init__(self, in_dim: int, dropout: float = 0.1, n_out: int = 6,
                 hidden_dims: list[int] | None = None):
        super().__init__()
        hd = self.HIDDEN if hidden_dims is None else [int(h) for h in hidden_dims]
        self.net = _make_mlp(in_dim, n_out, hd, dropout)

    def init_bias(self, log_q_meds: np.ndarray):
        with torch.no_grad():
            self.net[-1].bias[:] = torch.tensor(log_q_meds.astype(np.float32))

    def forward(self, x):
        return self.net(x)   # (B, 6)  – log-space


class ANNSingle(ANNDirect):
    """Compatibility alias used by legacy experiment scripts.

    The current runner trains multi-output ANN in log-space, so this class
    reuses ANNDirect behavior while preserving old import paths.
    """
    TYPE = "ann_direct"
