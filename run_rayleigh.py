"""
run_rayleigh.py
===============
Experiment 2 – Rayleigh flat-fading channel equivalent of Table II.

Trains a fresh autoencoder on a Rayleigh channel and compares with
Polar (SC) and PPV reference at the same block-length configurations.
Results form a "Table III" showing performance on the more realistic
fading channel scenario requested in Section L.2 of the article.

Both perfect-CSI (equalized) and imperfect-CSI (unequalized) scenarios
are evaluated at test time for the AWGN-trained model.

Random seed: SEED_RAYLEIGH = 123.

Usage
-----
    python run_rayleigh.py
    python run_rayleigh.py --quick
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

import config as cfg
from channels import make_rayleigh_fn_random, make_rayleigh_fn
from baselines import ppv_snr_at_ber_target, polar_snr_at_ber_target
from training import train_autoencoder, save_model
from evaluation import eval_autoencoder_ber, snr_at_ber_target, print_results_table


def run_rayleigh(quick: bool = False):
    """Train and evaluate autoencoder on Rayleigh channel."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)

    num_epochs  = 2000 if quick else cfg.NUM_EPOCHS
    num_frames  = 200  if quick else 5000
    num_batches = 20   if quick else cfg.EVAL_BATCHES

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Rayleigh]  device={device}  quick={quick}")
    print(f"  Seed: {cfg.SEED_RAYLEIGH}")
    print(f"  Epochs: {num_epochs}\n")

    # Wider SNR range for fading
    snr_db_range = np.arange(cfg.EVAL_SNR_MIN_DB,
                             cfg.EVAL_SNR_MAX_DB + cfg.EVAL_SNR_STEP,
                             cfg.EVAL_SNR_STEP)
    # Extended range for Rayleigh (needs higher SNR to reach BER=1e-3)
    snr_db_range_ext = np.arange(-4.0, 22.0, 0.5)

    results_rayleigh_pcsi = {}   # perfect CSI
    results_rayleigh_no_csi = {} # no equalization

    for cfg_item in cfg.CONFIGS:
        n, k, M = cfg_item["n"], cfg_item["k"], cfg_item["M"]
        print(f"{'─'*55}")
        print(f"  Config: n={n}, k={k}, M={M}")
        print(f"{'─'*55}")

        # ------------------------------------------------------------------
        # Train on Rayleigh (perfect CSI at receiver)
        # ------------------------------------------------------------------
        print("  Training on Rayleigh (perfect CSI) …")
        ch_train = make_rayleigh_fn_random(
            cfg.SNR_RAYLEIGH_TRAIN_MIN_DB,
            cfg.SNR_RAYLEIGH_TRAIN_MAX_DB,
            perfect_csi=True,
        )
        model_pcsi, hist = train_autoencoder(
            n=n, k=k, M=M,
            channel_fn=ch_train,
            seed=cfg.SEED_RAYLEIGH,
            num_epochs=num_epochs,
            device=device,
            verbose=True,
            log_every=max(num_epochs // 10, 1),
        )
        model_path = os.path.join(cfg.MODELS_DIR,
                                  f"autoencoder_rayleigh_n{n}_k{k}.pt")
        save_model(model_pcsi, model_path)

        # ------------------------------------------------------------------
        # Evaluate: Rayleigh with perfect CSI
        # ------------------------------------------------------------------
        print("  Evaluating BER (perfect CSI) …")
        curve_pcsi = eval_autoencoder_ber(
            model=model_pcsi,
            channel_fn_factory=lambda snr: make_rayleigh_fn(snr, perfect_csi=True),
            snr_db_range=snr_db_range_ext,
            num_batches=num_batches,
            device=device,
        )
        snr_pcsi = snr_at_ber_target(curve_pcsi, cfg.BER_TARGET)
        results_rayleigh_pcsi[(n, k)] = snr_pcsi
        print(f"  Rayleigh pCSI Es/N0 @ BER=1e-3: {snr_pcsi:.2f} dB")

        # ------------------------------------------------------------------
        # Evaluate: Rayleigh WITHOUT CSI (no equalization)
        # ------------------------------------------------------------------
        print("  Evaluating BER (no CSI / no equalisation) …")
        curve_ncsi = eval_autoencoder_ber(
            model=model_pcsi,
            channel_fn_factory=lambda snr: make_rayleigh_fn(snr, perfect_csi=False),
            snr_db_range=snr_db_range_ext,
            num_batches=num_batches,
            device=device,
        )
        snr_ncsi = snr_at_ber_target(curve_ncsi, cfg.BER_TARGET)
        results_rayleigh_no_csi[(n, k)] = snr_ncsi
        print(f"  Rayleigh noCSI Es/N0 @ BER=1e-3:{snr_ncsi:.2f} dB")

        # Save curves
        for label, curve in [("pcsi", curve_pcsi), ("ncsi", curve_ncsi)]:
            cp = os.path.join(cfg.RESULTS_DIR,
                              f"ber_rayleigh_{label}_n{n}_k{k}.json")
            with open(cp, "w") as f:
                json.dump(curve, f, indent=2)

        print()

    # ------------------------------------------------------------------
    # Summary tables
    # ------------------------------------------------------------------
    print_results_table(
        {"Autoencoder pCSI (Rayleigh-trained)":     results_rayleigh_pcsi,
         "Autoencoder no-CSI (Rayleigh-trained)":   results_rayleigh_no_csi},
        title=(f"Table III — Es/N0 (dB) at BER = {cfg.BER_TARGET:.0e}"
               f"  [Rayleigh Flat Fading]"),
    )

    # Save
    serialisable = {
        "rayleigh_perfect_csi": {
            f"n{n}k{k}": v for (n, k), v in results_rayleigh_pcsi.items()
        },
        "rayleigh_no_csi": {
            f"n{n}k{k}": v for (n, k), v in results_rayleigh_no_csi.items()
        },
    }
    out_path = os.path.join(cfg.RESULTS_DIR, "table3_rayleigh.json")
    with open(out_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Results saved to {out_path}")

    return results_rayleigh_pcsi, results_rayleigh_no_csi


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rayleigh channel benchmark")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run_rayleigh(quick=args.quick)
