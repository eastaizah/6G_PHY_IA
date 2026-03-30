"""
training.py
===========
Training loop for the communication autoencoder, with integrated
complexity tracking (GPU-hours, wall-clock time, memory).

Usage
-----
    from training import train_autoencoder

    model, history = train_autoencoder(
        n=7, k=4, M=16,
        channel_fn=make_awgn_fn_random(0, 7),
        seed=42,
    )
"""

import time
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from models import Autoencoder, labels_to_bits, bits_to_labels, count_parameters
import config as cfg


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_autoencoder(
    n: int,
    k: int,
    M: int,
    channel_fn,
    seed: int = 42,
    num_epochs: int  = None,
    batch_size: int  = None,
    lr: float        = None,
    device: str      = None,
    verbose: bool    = True,
    log_every: int   = 5000,
) -> tuple:
    """Train an Autoencoder(n, k, M) on the given channel.

    Parameters
    ----------
    n, k, M     : Code / model dimensions.
    channel_fn  : Callable x → y (e.g. make_awgn_fn_random(0, 7)).
    seed        : Random seed for torch, numpy, and Python random.
    num_epochs  : Training epochs (default cfg.NUM_EPOCHS).
    batch_size  : Batch size (default cfg.BATCH_SIZE).
    lr          : Adam learning rate (default cfg.LEARNING_RATE).
    device      : 'cpu' / 'cuda' / None (auto-detect).
    verbose     : Print periodic loss summaries.
    log_every   : Print loss every this many epochs.

    Returns
    -------
    model   : Trained Autoencoder.
    history : Dict with keys 'loss', 'epoch', 'train_time_s',
              'gpu_hours', 'num_params'.
    """
    # -----------------------------------------------------------------------
    # Resolve defaults
    # -----------------------------------------------------------------------
    if num_epochs is None:
        num_epochs = cfg.NUM_EPOCHS
    if batch_size is None:
        batch_size = cfg.BATCH_SIZE
    if lr is None:
        lr = cfg.LEARNING_RATE
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = Autoencoder(
        k=k, n=n, M=M,
        enc_h1=cfg.ENCODER_H1, enc_h2=cfg.ENCODER_H2,
        dec_h1=cfg.DECODER_H1, dec_h2=cfg.DECODER_H2,
    ).to(device)

    num_params = count_parameters(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    history = {"loss": [], "epoch": []}
    t_start = time.perf_counter()

    model.train()
    for epoch in range(1, num_epochs + 1):
        # Sample random message labels → convert to k-bit vectors
        labels = torch.randint(0, M, (batch_size,), device=device)
        bits   = labels_to_bits(labels, k)        # (B, k) float

        # Forward pass through encoder → channel → decoder
        logits = model(bits, channel_fn)           # (B, M)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0:
            history["loss"].append(float(loss.item()))
            history["epoch"].append(epoch)
            if verbose:
                elapsed = time.perf_counter() - t_start
                print(f"  [{epoch:>7d}/{num_epochs}]  loss={loss.item():.4f}"
                      f"  elapsed={elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # Complexity summary
    # -----------------------------------------------------------------------
    train_time_s = time.perf_counter() - t_start
    # GPU-hours: wall-clock × (1 GPU) / 3600
    gpu_hours    = train_time_s / 3600.0 if torch.cuda.is_available() else 0.0

    history["train_time_s"] = train_time_s
    history["gpu_hours"]    = gpu_hours
    history["num_params"]   = num_params
    history["device"]       = device
    history["n"]            = n
    history["k"]            = k
    history["M"]            = M

    if verbose:
        print(f"\n  Training complete: {train_time_s:.1f} s"
              f"  ({gpu_hours*60:.2f} GPU-min)"
              f"  | params={num_params:,}")

    return model, history


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_model(model: Autoencoder, path: str):
    """Save model state dict and config to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "k": model.k,
            "n": model.n,
            "M": model.M,
        },
        path,
    )


def load_model(path: str, device: str = "cpu") -> Autoencoder:
    """Load a saved model from disk."""
    ckpt  = torch.load(path, map_location=device)
    model = Autoencoder(k=ckpt["k"], n=ckpt["n"], M=ckpt["M"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    return model
