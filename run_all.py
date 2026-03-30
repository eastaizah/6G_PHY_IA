"""
run_all.py
==========
Master script — runs all four validation experiments in sequence.

Experiments
-----------
  1. Table II  – AWGN benchmark  (seed=42)
  2. Rayleigh  – Fading channel  (seed=123)
  3. BER Curves– Complete curves (seed=456)
  4. Shift     – Distributional  (seed=789)

Usage
-----
    # Full run (may take several hours without GPU):
    python run_all.py

    # Quick smoke-test (reduced epochs, fast approximation):
    python run_all.py --quick

    # Run specific experiments only:
    python run_all.py --experiments table2 rayleigh

    # Headless mode (no matplotlib plots):
    python run_all.py --no-plot
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import config as cfg


def main():
    parser = argparse.ArgumentParser(
        description="Run all 6G AI PHY validation experiments"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Reduced epochs / fast smoke-test mode",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip matplotlib figure generation (headless / CI)",
    )
    parser.add_argument(
        "--experiments", nargs="*",
        choices=["table2", "rayleigh", "ber_curves", "shift"],
        default=["table2", "rayleigh", "ber_curves", "shift"],
        help="Which experiments to run (default: all)",
    )
    args = parser.parse_args()

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)

    print("=" * 65)
    print("  6G AI-Native PHY — Validation Experiments")
    print("  Native_AI_Physical_Layer_6G_IEEE.md  (Section L)")
    print("=" * 65)
    print(f"  Quick mode   : {args.quick}")
    print(f"  Experiments  : {args.experiments}")
    print(f"  Results dir  : {cfg.RESULTS_DIR}")
    print(f"  Figures dir  : {cfg.FIGURES_DIR}")
    print("=" * 65)

    wall_start = time.perf_counter()
    summary = {}

    # ──────────────────────────────────────────────────────────────────────
    # Experiment 1 – Table II (AWGN benchmark)
    # ──────────────────────────────────────────────────────────────────────
    if "table2" in args.experiments:
        print("\n\n" + "#" * 55)
        print("# Experiment 1: Table II — AWGN Benchmark  (seed=42)")
        print("#" * 55)
        from run_table2 import run_table2
        t0 = time.perf_counter()
        results_t2, complexity = run_table2(quick=args.quick)
        summary["table2"] = {
            "duration_s": time.perf_counter() - t0,
            "complexity": complexity,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Experiment 2 – Rayleigh channel
    # ──────────────────────────────────────────────────────────────────────
    if "rayleigh" in args.experiments:
        print("\n\n" + "#" * 55)
        print("# Experiment 2: Rayleigh Channel  (seed=123)")
        print("#" * 55)
        from run_rayleigh import run_rayleigh
        t0 = time.perf_counter()
        run_rayleigh(quick=args.quick)
        summary["rayleigh"] = {"duration_s": time.perf_counter() - t0}

    # ──────────────────────────────────────────────────────────────────────
    # Experiment 3 – BER curves
    # ──────────────────────────────────────────────────────────────────────
    if "ber_curves" in args.experiments:
        print("\n\n" + "#" * 55)
        print("# Experiment 3: BER Curves  (seed=456)")
        print("#" * 55)
        from run_ber_curves import run_ber_curves
        t0 = time.perf_counter()
        run_ber_curves(quick=args.quick, make_plot=not args.no_plot)
        summary["ber_curves"] = {"duration_s": time.perf_counter() - t0}

    # ──────────────────────────────────────────────────────────────────────
    # Experiment 4 – Distributional shift
    # ──────────────────────────────────────────────────────────────────────
    if "shift" in args.experiments:
        print("\n\n" + "#" * 55)
        print("# Experiment 4: Distributional Shift  (seed=789)")
        print("#" * 55)
        from run_shift import run_shift
        t0 = time.perf_counter()
        run_shift(quick=args.quick, make_plot=not args.no_plot)
        summary["shift"] = {"duration_s": time.perf_counter() - t0}

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    total_wall = time.perf_counter() - wall_start
    print("\n\n" + "=" * 55)
    print("  ALL EXPERIMENTS COMPLETE")
    print("=" * 55)
    for exp, data in summary.items():
        print(f"  {exp:<15}: {data['duration_s']:.1f} s")
    print(f"  {'TOTAL':<15}: {total_wall:.1f} s  "
          f"({total_wall / 3600:.2f} h)")
    print(f"\n  Results → {cfg.RESULTS_DIR}")
    print(f"  Figures → {cfg.FIGURES_DIR}")

    # Save overall summary
    summary["total_wall_s"] = total_wall
    with open(os.path.join(cfg.RESULTS_DIR, "run_summary.json"), "w") as f:
        # Remove non-serialisable items
        clean = {k: {kk: vv for kk, vv in v.items() if kk != "complexity"}
                 for k, v in summary.items() if isinstance(v, dict)}
        json.dump(clean, f, indent=2)


if __name__ == "__main__":
    main()
