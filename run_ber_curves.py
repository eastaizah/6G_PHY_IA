"""
run_ber_curves.py
=================
Experiment 3 – Complete BER vs Eb/N0 curves (Figure 5 in the article).

Generates the full semi-logarithmic BER curves (BER from 10^-1 to 10^-5)
for all coding schemes over AWGN, corresponding to the pending Figure 5
described in Section III.A.4 of Native_AI_Physical_Layer_6G_IEEE.md.

Plotted on a fine Eb/N0 grid from -2 to 12 dB.
Curves are saved as JSON (for custom plotting) and as a matplotlib PNG.

Random seed: SEED_BER_CURVES = 456.

Usage
-----
    python run_ber_curves.py
    python run_ber_curves.py --quick
    python run_ber_curves.py --no-plot    # headless / CI mode
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

import config as cfg
from channels import make_awgn_fn
from baselines import ppv_ber_approx, polar_ber
from training import train_autoencoder, load_model, save_model
from evaluation import eval_autoencoder_ber
from channels import make_awgn_fn_random


# Es/N0 ↔ Eb/N0 conversion: Eb/N0 = Es/N0 − 10·log10(k/n)
def esn0_to_ebn0(esn0_db, k, n):
    rate = k / n
    return esn0_db - 10.0 * np.log10(rate)


def ebn0_to_esn0(ebn0_db, k, n):
    rate = k / n
    return ebn0_db + 10.0 * np.log10(rate)


def run_ber_curves(quick: bool = False, make_plot: bool = True):
    """Generate and save complete BER vs Eb/N0 curves."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs  = 2000 if quick else cfg.NUM_EPOCHS
    num_batches = 20   if quick else cfg.EVAL_BATCHES * 2   # more for low BER
    num_frames  = 200  if quick else 10000

    print(f"\n[BER Curves]  device={device}  quick={quick}")
    print(f"  Seed: {cfg.SEED_BER_CURVES}\n")

    # Eb/N0 grid: −2 to 12 dB in 0.5 dB steps
    ebn0_range = np.arange(-2.0, 12.5, 0.5)

    all_curves = {}   # method_label → {n_k_label: {ebn0: [], ber: []}}

    for cfg_item in cfg.CONFIGS:
        n, k, M = cfg_item["n"], cfg_item["k"], cfg_item["M"]
        label_nk = f"n={n},k={k}"
        print(f"{'─'*55}")
        print(f"  BER curves for {label_nk}")
        print(f"{'─'*55}")

        # Convert Eb/N0 to Es/N0 for this configuration
        esn0_range = np.array([ebn0_to_esn0(eb, k, n) for eb in ebn0_range])

        # ------------------------------------------------------------------
        # Load or train autoencoder
        # ------------------------------------------------------------------
        model_path = os.path.join(cfg.MODELS_DIR,
                                  f"autoencoder_awgn_n{n}_k{k}.pt")
        if os.path.exists(model_path) and not quick:
            print(f"  Loading saved model from {model_path}")
            model = load_model(model_path, device=device)
        else:
            print("  Training autoencoder …")
            ch_train = make_awgn_fn_random(cfg.SNR_TRAIN_MIN_DB,
                                            cfg.SNR_TRAIN_MAX_DB)
            model, _ = train_autoencoder(
                n=n, k=k, M=M,
                channel_fn=ch_train,
                seed=cfg.SEED_BER_CURVES,
                num_epochs=num_epochs,
                device=device,
                verbose=True,
                log_every=max(num_epochs // 10, 1),
            )
            save_model(model, model_path)

        # ------------------------------------------------------------------
        # Autoencoder BER curve
        # ------------------------------------------------------------------
        print("  Evaluating autoencoder BER curve …")
        ae_curve = eval_autoencoder_ber(
            model=model,
            channel_fn_factory=make_awgn_fn,
            snr_db_range=esn0_range,
            num_batches=num_batches,
            device=device,
        )
        # Convert Es/N0 axis → Eb/N0
        ae_curve_ebn0 = {
            "ebn0_db": list(ebn0_range),
            "ber":     ae_curve["ber"],
        }
        key = f"Autoencoder_{label_nk}"
        all_curves[key] = ae_curve_ebn0

        # ------------------------------------------------------------------
        # Polar SC BER curve
        # ------------------------------------------------------------------
        print("  Simulating Polar SC BER curve …")
        polar_bers = []
        rng_polar  = np.random.default_rng(cfg.SEED_BER_CURVES + n)
        for es in esn0_range:
            ber_val = polar_ber(float(es), n, k,
                                num_frames=num_frames, rng=rng_polar)
            polar_bers.append(max(ber_val, 1e-7))
        key_polar = f"Polar_SC_{label_nk}"
        all_curves[key_polar] = {"ebn0_db": list(ebn0_range), "ber": polar_bers}

        # ------------------------------------------------------------------
        # PPV bound
        # ------------------------------------------------------------------
        ppv_bers = [ppv_ber_approx(float(es), n, k) for es in esn0_range]
        key_ppv  = f"PPV_Bound_{label_nk}"
        all_curves[key_ppv] = {"ebn0_db": list(ebn0_range), "ber": ppv_bers}

        print()

    # ------------------------------------------------------------------
    # Save all curves
    # ------------------------------------------------------------------
    curves_path = os.path.join(cfg.RESULTS_DIR, "ber_curves_awgn.json")
    with open(curves_path, "w") as f:
        json.dump(all_curves, f, indent=2)
    print(f"All BER curves saved to {curves_path}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if make_plot:
        _plot_ber_curves(all_curves, cfg.CONFIGS)

    return all_curves


def _plot_ber_curves(all_curves: dict, configs: list):
    """Create Figure 5: semi-log BER vs Eb/N0 for all configs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.flatten()

    ae_colors     = ["tab:orange", "tab:red", "tab:purple", "tab:brown"]
    polar_colors  = ["tab:blue",   "tab:cyan", "tab:green", "navy"]
    ppv_colors    = ["black",      "gray",     "darkgray",  "lightgray"]

    for idx, cfg_item in enumerate(configs):
        ax = axes_flat[idx]
        n, k = cfg_item["n"], cfg_item["k"]
        label_nk = f"n={n},k={k}"

        key_ae    = f"Autoencoder_{label_nk}"
        key_polar = f"Polar_SC_{label_nk}"
        key_ppv   = f"PPV_Bound_{label_nk}"

        for key, color, style, lbl in [
            (key_ae,    ae_colors[idx],    "-",  f"Autoencoder ({label_nk})"),
            (key_polar, polar_colors[idx], "--", f"Polar SC ({label_nk})"),
            (key_ppv,   ppv_colors[idx],   ":",  f"PPV Bound ({label_nk})"),
        ]:
            if key in all_curves:
                c = all_curves[key]
                ax.semilogy(c["ebn0_db"], c["ber"],
                            color=color, linestyle=style, linewidth=1.8,
                            label=lbl, marker="o" if "Auto" in key else None,
                            markersize=4, markevery=4)

        ax.axhline(1e-3, color="gray", linestyle=":", linewidth=0.8,
                   alpha=0.6, label="BER = 10⁻³")
        ax.set_xlabel("Eb/N0 (dB)", fontsize=10)
        ax.set_ylabel("BER", fontsize=10)
        ax.set_title(f"n={n}, k={k}, rate={k/n:.2f}", fontsize=11)
        ax.set_ylim(1e-5, 1.0)
        ax.set_xlim(-2, 12)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Figure 5 – BER vs Eb/N0: Autoencoder vs Polar SC vs PPV Bound\n"
                 "(AWGN Channel, Seed=456)", fontsize=12)
    plt.tight_layout()

    fig_path = os.path.join(cfg.FIGURES_DIR, "figure5_ber_curves_awgn.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 5 saved to {fig_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BER curves experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    run_ber_curves(quick=args.quick, make_plot=not args.no_plot)
