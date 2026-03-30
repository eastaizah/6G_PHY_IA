"""
run_table2.py
=============
Experiment 1 – Reproduce Table II (Section III.A.7 of the article).

Compares Es/N0 required for BER = 10^-3 on AWGN for:
  • Communication Autoencoder (this work)
  • PPV finite-block-length lower bound
  • Polar code (SC decoder)
  • IEEE 802.11n LDPC  (tabulated reference)
  • 3GPP Turbo LTE     (tabulated reference)

Block lengths: n ∈ {7, 16, 32, 64}.
Random seed: SEED_TABLE2 = 42.

Usage
-----
    python run_table2.py           # run full experiment
    python run_table2.py --quick   # fast smoke-test (fewer epochs)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# Make sibling modules importable when run directly
sys.path.insert(0, os.path.dirname(__file__))

import config as cfg
from channels import make_awgn_fn_random, make_awgn_fn
from baselines import (
    ppv_snr_at_ber_target,
    polar_snr_at_ber_target,
    turbo_snr_reference,
    ldpc_snr_reference,
)
from training import train_autoencoder, save_model
from evaluation import eval_autoencoder_ber, snr_at_ber_target, print_results_table


def run_table2(quick: bool = False):
    """Main entry point for the Table II experiment.

    Parameters
    ----------
    quick : bool
        If True, use reduced epoch count and fewer evaluation frames for
        a fast smoke-test (results will be approximate).
    """
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)

    num_epochs  = 2000   if quick else cfg.NUM_EPOCHS
    num_frames  = 200    if quick else 5000   # for Polar MC simulation
    num_batches = 20     if quick else cfg.EVAL_BATCHES

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Table II]  device={device}  quick={quick}")
    print(f"  Seed: {cfg.SEED_TABLE2}")
    print(f"  Epochs: {num_epochs}  |  BER target: {cfg.BER_TARGET}\n")

    # SNR evaluation grid
    snr_db_range = np.arange(cfg.EVAL_SNR_MIN_DB,
                             cfg.EVAL_SNR_MAX_DB + cfg.EVAL_SNR_STEP,
                             cfg.EVAL_SNR_STEP)

    # Collect results: method → {(n,k): snr_at_ber_target}
    results = {
        "Autoencoder (this work)":   {},
        "PPV Bound (lower bound)":   {},
        "Polar (SC decoder)":        {},
        "IEEE 802.11n LDPC (ref)":   {},
        "3GPP Turbo LTE (ref)":      {},
    }

    complexity_log = []   # training complexity per config

    for cfg_item in cfg.CONFIGS:
        n, k, M = cfg_item["n"], cfg_item["k"], cfg_item["M"]
        print(f"{'─'*55}")
        print(f"  Config: n={n}, k={k}, M={M}")
        print(f"{'─'*55}")

        # ------------------------------------------------------------------
        # 1. Train autoencoder on AWGN
        # ------------------------------------------------------------------
        print("  Training autoencoder …")
        channel_train = make_awgn_fn_random(
            cfg.SNR_TRAIN_MIN_DB, cfg.SNR_TRAIN_MAX_DB
        )
        model, history = train_autoencoder(
            n=n, k=k, M=M,
            channel_fn=channel_train,
            seed=cfg.SEED_TABLE2,
            num_epochs=num_epochs,
            batch_size=cfg.BATCH_SIZE,
            device=device,
            verbose=True,
            log_every=max(num_epochs // 10, 1),
        )

        # Save model
        model_path = os.path.join(cfg.MODELS_DIR,
                                  f"autoencoder_awgn_n{n}_k{k}.pt")
        save_model(model, model_path)
        complexity_log.append({
            "n": n, "k": k, "M": M,
            "train_time_s":   history["train_time_s"],
            "gpu_hours":      history["gpu_hours"],
            "num_params":     history["num_params"],
            "device":         history["device"],
        })
        print(f"  Saved to {model_path}")

        # ------------------------------------------------------------------
        # 2. Evaluate BER curve
        # ------------------------------------------------------------------
        print("  Evaluating BER …")
        ber_curve = eval_autoencoder_ber(
            model=model,
            channel_fn_factory=make_awgn_fn,
            snr_db_range=snr_db_range,
            num_batches=num_batches,
            device=device,
        )
        snr_threshold = snr_at_ber_target(ber_curve, cfg.BER_TARGET)
        results["Autoencoder (this work)"][(n, k)] = snr_threshold
        print(f"  Autoencoder Es/N0 @ BER=1e-3: {snr_threshold:.2f} dB")

        # Save BER curve
        curve_path = os.path.join(cfg.RESULTS_DIR,
                                  f"ber_awgn_autoencoder_n{n}_k{k}.json")
        with open(curve_path, "w") as f:
            json.dump(ber_curve, f, indent=2)

        # ------------------------------------------------------------------
        # 3. PPV bound
        # ------------------------------------------------------------------
        ppv_snr = ppv_snr_at_ber_target(cfg.BER_TARGET, n, k)
        results["PPV Bound (lower bound)"][(n, k)] = ppv_snr
        print(f"  PPV bound:                      {ppv_snr:.2f} dB")

        # ------------------------------------------------------------------
        # 4. Polar code (SC decoder, MC simulation)
        # ------------------------------------------------------------------
        if n in (7, 16, 32, 64):
            print("  Simulating Polar SC …")
            rng_polar = np.random.default_rng(cfg.SEED_TABLE2 + n)
            polar_snr = polar_snr_at_ber_target(
                cfg.BER_TARGET, n, k,
                num_frames=num_frames,
                rng=rng_polar,
            )
            results["Polar (SC decoder)"][(n, k)] = polar_snr
            print(f"  Polar SC Es/N0 @ BER=1e-3:     {polar_snr:.2f} dB")

        # ------------------------------------------------------------------
        # 5. LDPC / Turbo reference values
        # ------------------------------------------------------------------
        ldpc_snr  = ldpc_snr_reference(n, k)
        turbo_snr = turbo_snr_reference(n, k)
        results["IEEE 802.11n LDPC (ref)"][(n, k)] = ldpc_snr
        results["3GPP Turbo LTE (ref)"][(n, k)]    = turbo_snr
        print(f"  LDPC reference:                 {ldpc_snr:.2f} dB"
              f"  (tabulated)")
        print(f"  Turbo reference:                {turbo_snr:.2f} dB"
              f"  (tabulated)")
        print()

    # ------------------------------------------------------------------
    # Print and save summary table
    # ------------------------------------------------------------------
    print_results_table(
        results,
        title=f"Table II — Es/N0 (dB) at BER = {cfg.BER_TARGET:.0e}  [AWGN]",
    )

    # Save numerical results
    # Convert tuple keys to strings for JSON serialisation
    serialisable = {
        method: {f"n{n}k{k}": v for (n, k), v in method_dict.items()}
        for method, method_dict in results.items()
    }
    out_path = os.path.join(cfg.RESULTS_DIR, "table2_awgn.json")
    with open(out_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Results saved to {out_path}")

    # Save complexity log
    cplx_path = os.path.join(cfg.RESULTS_DIR, "training_complexity.json")
    with open(cplx_path, "w") as f:
        json.dump(complexity_log, f, indent=2)
    print(f"Complexity log saved to {cplx_path}")

    # Print training complexity table
    print("\n[Training Complexity]")
    print(f"{'Config':<16} {'Params':>10} {'Time (s)':>12} {'GPU-min':>10} {'Device':>8}")
    print("-" * 60)
    for row in complexity_log:
        print(f"  n={row['n']},k={row['k']:<12}"
              f" {row['num_params']:>10,}"
              f" {row['train_time_s']:>12.1f}"
              f" {row['gpu_hours']*60:>10.2f}"
              f" {row['device']:>8}")

    return results, complexity_log


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table II AWGN benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Fast smoke-test with reduced epochs")
    args = parser.parse_args()
    run_table2(quick=args.quick)
