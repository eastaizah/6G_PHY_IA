"""
channels.py
===========
Differentiable channel implementations for training and evaluation.

AWGN Channel
------------
  y = x + n,    n ~ N(0, σ²·I)
  σ² = 1 / (2 · Es/N0_linear)   [real baseband, E[||x||²/n] = 1]

Rayleigh Flat-Fading Channel (SISO)
------------------------------------
  y_i = h_i · x_i + n_i,   h_i ~ CN(0,1),  n_i ~ CN(0, σ²)
  In real baseband: split into real/imag parts, apply 2×2 rotation matrix.
  Perfect CSI at receiver: the received signal is equalized before
  being fed to the decoder (coherent detection).

All channels accept and return real (B, n) tensors.
σ² (noise variance) is computed from Es/N0 in linear scale:
  σ² = 1 / (2 · snr_lin)   for real AWGN
  σ² = 1 / (2 · snr_lin)   for the noise component in Rayleigh
  (fading power is normalized: E[|h|²] = 1)
"""

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Helper: dB ↔ linear conversion
# ---------------------------------------------------------------------------

def db_to_lin(snr_db: float) -> float:
    return 10.0 ** (snr_db / 10.0)


def lin_to_db(snr_lin: float) -> float:
    return 10.0 * np.log10(snr_lin + 1e-20)


# ---------------------------------------------------------------------------
# AWGN channel
# ---------------------------------------------------------------------------

def awgn_channel(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add complex-baseband AWGN noise.

    Parameters
    ----------
    x      : (B, n) real transmitted symbols, E[||x||²/n] = 1 per sample
    snr_db : Es/N0 in dB

    Returns
    -------
    y : (B, n) noisy received signal
    """
    snr_lin = db_to_lin(snr_db)
    noise_std = 1.0 / (np.sqrt(2.0 * snr_lin))
    noise = torch.randn_like(x) * noise_std
    return x + noise


def make_awgn_fn(snr_db: float):
    """Return a channel function for a fixed SNR (useful for evaluation)."""
    def fn(x: torch.Tensor) -> torch.Tensor:
        return awgn_channel(x, snr_db)
    return fn


def make_awgn_fn_random(snr_min_db: float, snr_max_db: float):
    """Return a channel function that samples a fresh SNR each call.

    Used during training so the autoencoder generalises across the
    [snr_min_db, snr_max_db] dB curriculum.
    """
    def fn(x: torch.Tensor) -> torch.Tensor:
        snr_db = float(torch.FloatTensor(1).uniform_(snr_min_db, snr_max_db))
        return awgn_channel(x, snr_db)
    return fn


# ---------------------------------------------------------------------------
# Rayleigh flat-fading channel (SISO, real baseband)
# ---------------------------------------------------------------------------

def rayleigh_channel(x: torch.Tensor, snr_db: float,
                     perfect_csi: bool = True) -> torch.Tensor:
    """Rayleigh flat-fading SISO channel in real baseband.

    Each of the n channel uses experiences an independent Rayleigh-fading
    amplitude  h_i ~ Rayleigh(1/√2),  i.e. h_i = |h_c| where
    h_c ~ CN(0,1).  The phase is assumed perfectly known and compensated
    at the receiver (coherent detection), so only amplitude uncertainty
    remains.

    Model
    -----
    y_i = h_i · x_i + n_i,   n_i ~ N(0, σ²),   σ² = 1 / (2·SNR_lin)

    E[h²] = 1, so Es/N0 matches the requested snr_db on average.

    Perfect-CSI equalization: y_eq_i = y_i / h_i = x_i + n_i / h_i.
    Effective instantaneous SNR = h_i² · SNR_lin.

    Parameters
    ----------
    x          : (B, n) real transmitted symbols
    snr_db     : Average Es/N0 in dB
    perfect_csi: If True, amplitude-equalize before returning.

    Returns
    -------
    y_eq : (B, n) equalized received signal (real)
    """
    snr_lin   = db_to_lin(snr_db)
    noise_std = 1.0 / np.sqrt(2.0 * snr_lin)

    B, n = x.shape
    device = x.device

    # Rayleigh fading magnitude: h_i ~ Rayleigh(1/√2), E[h²]=1
    h_re  = torch.randn(B, n, device=device) / np.sqrt(2.0)
    h_im  = torch.randn(B, n, device=device) / np.sqrt(2.0)
    h_abs = torch.sqrt(h_re ** 2 + h_im ** 2)   # (B, n), E[h²] = 1

    # Received signal: y = |h| * x + noise  (amplitude fading, phase compensated)
    noise = torch.randn(B, n, device=device) * noise_std
    y = h_abs * x + noise

    if perfect_csi:
        # ZF/MMSE equalization: divide by |h| to restore signal amplitude
        # After equalization: y_eq = x + noise/h,  effective SNR = h²·SNR_lin
        y = y / (h_abs + 1e-8)

    return y


def make_rayleigh_fn(snr_db: float, perfect_csi: bool = True):
    """Return a fixed-SNR Rayleigh channel function."""
    def fn(x: torch.Tensor) -> torch.Tensor:
        return rayleigh_channel(x, snr_db, perfect_csi=perfect_csi)
    return fn


def make_rayleigh_fn_random(snr_min_db: float, snr_max_db: float,
                             perfect_csi: bool = True):
    """Return a random-SNR Rayleigh channel function (for training)."""
    def fn(x: torch.Tensor) -> torch.Tensor:
        snr_db = float(torch.FloatTensor(1).uniform_(snr_min_db, snr_max_db))
        return rayleigh_channel(x, snr_db, perfect_csi=perfect_csi)
    return fn
