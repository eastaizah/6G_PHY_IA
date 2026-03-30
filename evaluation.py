"""
evaluation.py
=============
BER evaluation utilities for the trained autoencoder and conventional
baseline codes.

All eval functions return:
    ber_curve : dict  {'snr_db': list[float], 'ber': list[float]}

The function `snr_at_ber_target` searches for the Es/N0 threshold at a
given BER level (default BER = 10^-3) using linear interpolation on the
log-BER curve.
"""

import numpy as np
import torch

from models import Autoencoder, labels_to_bits, bits_to_labels
import config as cfg


# ---------------------------------------------------------------------------
# Autoencoder BER evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_autoencoder_ber(
    model: Autoencoder,
    channel_fn_factory,          # callable(snr_db) -> channel_fn
    snr_db_range: list,
    batch_size: int  = None,
    num_batches: int = None,
    device: str      = None,
) -> dict:
    """Compute BER curve for the autoencoder over a range of Es/N0 values.

    Parameters
    ----------
    model              : Trained Autoencoder.
    channel_fn_factory : Callable snr_db → channel_fn(x) → y.
    snr_db_range       : Iterable of Es/N0 values in dB.
    batch_size         : Samples per batch (default cfg.BATCH_SIZE).
    num_batches        : Number of batches per SNR point (default cfg.EVAL_BATCHES).
    device             : Torch device string.

    Returns
    -------
    dict with 'snr_db' and 'ber' lists.
    """
    if batch_size is None:
        batch_size = cfg.BATCH_SIZE
    if num_batches is None:
        num_batches = cfg.EVAL_BATCHES
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    k = model.k
    M = model.M

    ber_list = []
    for snr_db in snr_db_range:
        channel_fn = channel_fn_factory(snr_db)
        total_bits = 0
        error_bits = 0

        for _ in range(num_batches):
            labels = torch.randint(0, M, (batch_size,), device=device)
            bits   = labels_to_bits(labels, k)

            logits  = model(bits, channel_fn)
            pred    = torch.argmax(logits, dim=-1)          # (B,)
            pred_bits  = labels_to_bits(pred, k)
            true_bits  = bits

            error_bits += int((pred_bits != true_bits).sum().item())
            total_bits += batch_size * k

        ber = error_bits / total_bits if total_bits > 0 else 1.0
        ber_list.append(max(ber, 1e-7))   # floor for log-scale plots

    return {"snr_db": list(snr_db_range), "ber": ber_list}


# ---------------------------------------------------------------------------
# Threshold search
# ---------------------------------------------------------------------------

def snr_at_ber_target(ber_curve: dict, ber_target: float = 1e-3) -> float:
    """Find Es/N0 (dB) where BER = ber_target via log-linear interpolation.

    Returns NaN if the target BER is never reached in the evaluated range.
    """
    snr_arr = np.asarray(ber_curve["snr_db"])
    ber_arr = np.asarray(ber_curve["ber"])

    log_ber    = np.log10(np.clip(ber_arr, 1e-10, 1.0))
    log_target = np.log10(ber_target)

    # Find crossing: BER decreasing, so log_ber decreasing
    # We want log_ber[i] >= log_target >= log_ber[i+1]
    for i in range(len(log_ber) - 1):
        if log_ber[i] >= log_target >= log_ber[i + 1]:
            # Linear interpolation in (snr_db, log_ber) space
            frac = (log_target - log_ber[i]) / (log_ber[i + 1] - log_ber[i] + 1e-12)
            return float(snr_arr[i] + frac * (snr_arr[i + 1] - snr_arr[i]))

    # If BER never drops below target in range, return the max SNR as upper bound
    if ber_arr[-1] > ber_target:
        return float("nan")
    # BER already below target at minimum SNR
    return float(snr_arr[0])


# ---------------------------------------------------------------------------
# Summarise BER results at BER = 10^-3 for all configs
# ---------------------------------------------------------------------------

def print_results_table(results: dict, title: str = ""):
    """Pretty-print a results table keyed by (n, k) tuples.

    Parameters
    ----------
    results : dict[str, dict[(n,k), float]]
        Keys are method names; values are {(n,k): snr_dB_at_target}.
    title   : Optional header string.
    """
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    # Collect all (n, k) configs
    all_configs = set()
    for method_dict in results.values():
        all_configs |= set(method_dict.keys())
    all_configs = sorted(all_configs)

    # Header
    header = f"{'Method':<30s}" + "".join(f"  (n={n},k={k})" for n, k in all_configs)
    print(header)
    print("-" * len(header))

    for method, method_dict in results.items():
        row = f"{method:<30s}"
        for nk in all_configs:
            val = method_dict.get(nk, float("nan"))
            if np.isnan(val):
                row += f"  {'N/A':>10s}"
            else:
                row += f"  {val:>10.2f}"
        print(row)
    print()
