"""
run_shift.py
============
Experiment 4 – Distributional Shift: AWGN-trained autoencoder on Rayleigh.

Quantifies the BER degradation when an autoencoder trained exclusively on
AWGN is evaluated on a Rayleigh flat-fading channel (with and without
perfect CSI equalization), as described in Section L.2 of the article.

This is a known limitation of end-to-end learned systems: they optimise
for the training distribution and may generalise poorly to different
channel statistics.

Three models are compared:
  1. Autoencoder trained on AWGN  → tested on AWGN          (in-distribution)
  2. Autoencoder trained on AWGN  → tested on Rayleigh pCSI  (distributional shift)
  3. Autoencoder trained on Rayleigh pCSI → tested on Rayleigh pCSI  (matched)

Random seed: SEED_SHIFT = 789.

Usage
-----
    python run_shift.py
    python run_shift.py --quick
    python run_shift.py --no-plot
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

import config as cfg
from channels import make_awgn_fn, make_awgn_fn_random
from channels import make_rayleigh_fn, make_rayleigh_fn_random
from training import train_autoencoder, load_model, save_model
from evaluation import eval_autoencoder_ber, snr_at_ber_target


def run_shift(quick: bool = False, make_plot: bool = True):
    """Run the distributional-shift experiment."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs  = 2000 if quick else cfg.NUM_EPOCHS
    num_batches = 20   if quick else cfg.EVAL_BATCHES

    print(f"\n[Distributional Shift]  device={device}  quick={quick}")
    print(f"  Seed: {cfg.SEED_SHIFT}\n")

    snr_db_range = np.arange(cfg.SHIFT_EVAL_SNR_MIN_DB,
                             cfg.SHIFT_EVAL_SNR_MAX_DB + cfg.SHIFT_EVAL_SNR_STEP,
                             cfg.SHIFT_EVAL_SNR_STEP)

    all_results = {}   # label → BER curve dict

    for cfg_item in cfg.CONFIGS:
        n, k, M = cfg_item["n"], cfg_item["k"], cfg_item["M"]
        label_nk = f"n{n}_k{k}"
        print(f"{'─'*55}")
        print(f"  Config: n={n}, k={k}")
        print(f"{'─'*55}")

        # ──────────────────────────────────────────────────────────────────
        # (A) Load or train AWGN model
        # ──────────────────────────────────────────────────────────────────
        awgn_model_path = os.path.join(cfg.MODELS_DIR,
                                        f"autoencoder_awgn_n{n}_k{k}.pt")
        if os.path.exists(awgn_model_path) and not quick:
            print("  Loading AWGN-trained model …")
            model_awgn = load_model(awgn_model_path, device=device)
        else:
            print("  Training AWGN model …")
            ch_train = make_awgn_fn_random(cfg.SNR_TRAIN_MIN_DB,
                                            cfg.SNR_TRAIN_MAX_DB)
            model_awgn, _ = train_autoencoder(
                n=n, k=k, M=M,
                channel_fn=ch_train,
                seed=cfg.SEED_SHIFT,
                num_epochs=num_epochs,
                device=device,
                verbose=True,
                log_every=max(num_epochs // 10, 1),
            )
            save_model(model_awgn, awgn_model_path)

        # ──────────────────────────────────────────────────────────────────
        # (B) Load or train Rayleigh model
        # ──────────────────────────────────────────────────────────────────
        ray_model_path = os.path.join(cfg.MODELS_DIR,
                                       f"autoencoder_rayleigh_n{n}_k{k}.pt")
        if os.path.exists(ray_model_path) and not quick:
            print("  Loading Rayleigh-trained model …")
            model_ray = load_model(ray_model_path, device=device)
        else:
            print("  Training Rayleigh model …")
            ch_ray = make_rayleigh_fn_random(
                cfg.SNR_RAYLEIGH_TRAIN_MIN_DB,
                cfg.SNR_RAYLEIGH_TRAIN_MAX_DB,
                perfect_csi=True,
            )
            model_ray, _ = train_autoencoder(
                n=n, k=k, M=M,
                channel_fn=ch_ray,
                seed=cfg.SEED_SHIFT + 1,
                num_epochs=num_epochs,
                device=device,
                verbose=True,
                log_every=max(num_epochs // 10, 1),
            )
            save_model(model_ray, ray_model_path)

        # ──────────────────────────────────────────────────────────────────
        # (C) Three evaluation scenarios
        # ──────────────────────────────────────────────────────────────────
        scenarios = [
            ("AWGN-trained → AWGN (in-distribution)",
             model_awgn, make_awgn_fn),
            ("AWGN-trained → Rayleigh pCSI (shift)",
             model_awgn,
             lambda snr: make_rayleigh_fn(snr, perfect_csi=True)),
            ("Rayleigh-trained → Rayleigh pCSI (matched)",
             model_ray,
             lambda snr: make_rayleigh_fn(snr, perfect_csi=True)),
        ]

        for scenario_label, model, ch_factory in scenarios:
            print(f"  Evaluating: {scenario_label} …")
            curve = eval_autoencoder_ber(
                model=model,
                channel_fn_factory=ch_factory,
                snr_db_range=snr_db_range,
                num_batches=num_batches,
                device=device,
            )
            snr_thr = snr_at_ber_target(curve, cfg.BER_TARGET)
            key = f"{scenario_label}  [{label_nk}]"
            all_results[key] = {"curve": curve, "snr_at_ber_target": snr_thr}
            print(f"    → Es/N0 @ BER=1e-3: {snr_thr:.2f} dB")

            # Save curve
            safe_key = key.replace(" ", "_").replace("→", "to").replace("/", "_")
            cp = os.path.join(cfg.RESULTS_DIR,
                              f"shift_{safe_key[:60]}.json")
            with open(cp, "w") as fh:
                json.dump({"key": key, **curve,
                           "snr_at_ber_target": snr_thr}, fh, indent=2)

        print()

    # ------------------------------------------------------------------
    # Summary: degradation at BER=1e-3
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  Distributional Shift Summary — SNR degradation (dB)")
    print(f"{'='*70}")
    print(f"{'Config':<14} {'AWGN→AWGN':>12} {'AWGN→Ray':>12} "
          f"{'Ray→Ray':>12} {'Shift gap':>12}")
    print("-" * 60)

    for cfg_item in cfg.CONFIGS:
        n, k = cfg_item["n"], cfg_item["k"]
        label_nk = f"n{n}_k{k}"
        vals = {}
        for key, data in all_results.items():
            if label_nk in key:
                if "AWGN→AWGN" in key or "in-distribution" in key:
                    vals["awgn"]   = data["snr_at_ber_target"]
                elif "AWGN-trained → Rayleigh" in key or "shift" in key:
                    vals["shift"]  = data["snr_at_ber_target"]
                elif "Rayleigh-trained → Rayleigh" in key or "matched" in key:
                    vals["rayleigh"] = data["snr_at_ber_target"]

        awgn_val   = vals.get("awgn",    float("nan"))
        shift_val  = vals.get("shift",   float("nan"))
        ray_val    = vals.get("rayleigh", float("nan"))
        gap        = shift_val - awgn_val  # additional SNR needed due to shift

        def fmt(v):
            return f"{v:.2f}" if not np.isnan(v) else "N/A"

        print(f"  n={n},k={k:<8} {fmt(awgn_val):>12} {fmt(shift_val):>12}"
              f" {fmt(ray_val):>12} {fmt(gap):>12}")

    # Save summary JSON
    summary = {}
    for key, data in all_results.items():
        summary[key] = {
            "snr_at_ber_target": data["snr_at_ber_target"],
            "snr_db":  data["curve"]["snr_db"],
            "ber":     data["curve"]["ber"],
        }
    out_path = os.path.join(cfg.RESULTS_DIR, "distributional_shift.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results saved to {out_path}")

    if make_plot:
        _plot_shift(all_results, cfg.CONFIGS)

    return all_results


def _plot_shift(all_results: dict, configs: list):
    """Plot Figure: distributional shift comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    line_styles = {
        "AWGN→AWGN":    ("tab:blue",   "-",  "AWGN-trained → AWGN"),
        "AWGN→Ray":     ("tab:red",    "--", "AWGN-trained → Rayleigh pCSI (shift)"),
        "Ray→Ray":      ("tab:green",  "-.", "Rayleigh-trained → Rayleigh pCSI"),
    }

    for idx, cfg_item in enumerate(configs):
        ax = axes.flatten()[idx]
        n, k = cfg_item["n"], cfg_item["k"]
        label_nk = f"n{n}_k{k}"

        for key, data in all_results.items():
            if label_nk not in key:
                continue
            c = data["curve"]

            if "in-distribution" in key or "AWGN→AWGN" in key:
                color, style, lbl_s = line_styles["AWGN→AWGN"]
            elif "shift" in key:
                color, style, lbl_s = line_styles["AWGN→Ray"]
            else:
                color, style, lbl_s = line_styles["Ray→Ray"]

            ax.semilogy(c["snr_db"], c["ber"],
                        color=color, linestyle=style,
                        linewidth=1.8, label=lbl_s)

        ax.axhline(1e-3, color="gray", linestyle=":", linewidth=0.8,
                   alpha=0.6, label="BER = 10⁻³")
        ax.set_xlabel("Es/N0 (dB)", fontsize=10)
        ax.set_ylabel("BER", fontsize=10)
        ax.set_title(f"n={n}, k={k}", fontsize=11)
        ax.set_ylim(1e-5, 1.0)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Distributional Shift: AWGN-trained vs Rayleigh Channel\n"
                 "(Seed=789)", fontsize=12)
    plt.tight_layout()

    fig_path = os.path.join(cfg.FIGURES_DIR, "distributional_shift.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Distributional shift figure saved to {fig_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributional shift experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    run_shift(quick=args.quick, make_plot=not args.no_plot)
