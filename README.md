# 6G AI-Native Physical Layer — Validation Experiments

Supplementary code for the article
**"Native AI Physical Layer for 6G: A Comprehensive Survey"**
(`Native_AI_Physical_Layer_6G_IEEE.md`).

Implements the complete experimental validation described in **Section L**,
including Table II and the four additional experiments that strengthen the
article's contributions.

---

## Contents

```
experiments/
├── README.md                  ← this file
├── requirements.txt           ← pip dependencies
│
├── config.py                  ← all hyperparameters & random seeds
├── models.py                  ← Encoder / Decoder / Autoencoder (PyTorch)
├── channels.py                ← AWGN and Rayleigh channel layers
├── baselines.py               ← PPV bound, Polar SC decoder, LDPC/Turbo refs
├── training.py                ← training loop + complexity tracking
├── evaluation.py              ← BER curve evaluation + threshold search
│
├── run_table2.py              ← Experiment 1: reproduce Table II (AWGN)
├── run_rayleigh.py            ← Experiment 2: Rayleigh channel Table III
├── run_ber_curves.py          ← Experiment 3: full BER vs Eb/N0 curves
├── run_shift.py               ← Experiment 4: distributional shift
└── run_all.py                 ← master script (runs all experiments)
```

### Output directories (created automatically)

```
experiments/
├── results/    ← JSON files with numerical results
├── figures/    ← PNG plots (Figure 5, distributional shift, etc.)
└── saved_models/ ← checkpoint .pt files per configuration
```

---

## Setup

```bash
pip install -r requirements.txt
```

Requirements: `torch>=2.0`, `numpy>=1.24`, `scipy>=1.10`, `matplotlib>=3.7`,
`tqdm>=4.65`, `pandas>=2.0`.

---

## Quick start — smoke-test (< 5 minutes)

```bash
cd experiments
python run_all.py --quick --no-plot
```

`--quick` uses 2 000 training epochs instead of 100 000 and fewer
Monte-Carlo frames, so results are approximate but the full pipeline
is exercised.

---

## Full experimental run

```bash
cd experiments
python run_all.py
```

Expected wall-clock time on a modern GPU (RTX 3090):
  ≈ 4–8 hours total for all four experiments at full 100 000 epochs.

On CPU only: multiply by ~10×.

---

## Running individual experiments

```bash
# Experiment 1 – Table II (AWGN benchmark)
python run_table2.py

# Experiment 2 – Rayleigh channel equivalent
python run_rayleigh.py

# Experiment 3 – Complete BER vs Eb/N0 curves
python run_ber_curves.py

# Experiment 4 – Distributional shift (AWGN-trained → Rayleigh)
python run_shift.py
```

Each script also accepts `--quick` and (where applicable) `--no-plot`.

---

## Reproducibility

All random seeds are declared in `config.py`:

| Experiment | Seed |
|---|---|
| Table II (AWGN)       | `SEED_TABLE2 = 42`     |
| Rayleigh channel      | `SEED_RAYLEIGH = 123`  |
| BER curves            | `SEED_BER_CURVES = 456`|
| Distributional shift  | `SEED_SHIFT = 789`     |

Seeds are applied via `torch.manual_seed()`, `np.random.seed()`, and
`torch.cuda.manual_seed_all()` at the start of each experiment.

---

## Model architecture

Exactly as specified in Section III.A.7 of the article:

```
Encoder: k → Linear(128,ReLU) → Linear(64,ReLU) → Linear(n)
         → per-sample power normalisation: E[||x||²/n] = 1

Decoder: n → Linear(64,ReLU)  → Linear(128,ReLU) → Linear(M)
         → cross-entropy loss with true message label
```

Training: Adam (`lr=1e-3`), 100 000 epochs, batch=256,
          training Es/N0 ∈ [0, 7] dB (uniform curriculum).

---

## Block-length configurations (Table II)

| n  | k  | M       | Rate   | Notes |
|----|----|---------|--------|-------|
| 7  | 4  | 16      | 4/7 ≈ 0.57 | Standard O'Shea & Hoydis (2017) config |
| 16 | 8  | 256     | 1/2    | Exact rate 1/2 |
| 32 | 8  | 256     | 1/4    | k capped: k=16 → M=65 536 (impractical) |
| 64 | 8  | 256     | 1/8    | k capped: k=32 → M=2^32 (infeasible) |

For n=32 and n=64, k is reduced to 8 (M=256) for computational tractability.
PPV and Polar baselines are recomputed at the same (n,k) to maintain a fair
comparison.  The trade-off in code rate is acknowledged in the article discussion.

---

## Baselines

| Method | Implementation |
|---|---|
| **PPV Bound** | Normal approximation (Polyanskyi–Poor–Verdú 2010) |
| **Polar (SC)** | Custom SC decoder; n=7 uses punctured Polar(8,k) |
| **Polar (SC-List, L=8)** | Article Table II values (reference only) |
| **IEEE 802.11n LDPC** | Tabulated from literature |
| **3GPP Turbo (LTE)** | Tabulated from literature |

The SC-List decoder (L=8) is not implemented; the SC decoder is provided
as an intermediate baseline.  The gap between SC and SC-List is typically
0.2–0.5 dB for these short block lengths.

---

## Experiment descriptions

### Experiment 1 – Table II (AWGN)
Reproduces Table II.  Trains one autoencoder per configuration, evaluates
BER vs Es/N0 over [−4, 12] dB, and reports the threshold at BER = 10⁻³.
Also logs training complexity (wall-clock time, GPU-hours, parameter count).

### Experiment 2 – Rayleigh Channel (Table III)
Trains autoencoders on Rayleigh flat fading (perfect CSI at receiver) and
evaluates under both perfect-CSI and no-CSI conditions.  Shows the
performance gap relative to AWGN (Table II).

### Experiment 3 – Complete BER Curves
Generates semi-logarithmic BER vs Eb/N0 curves from BER ≈ 10⁻¹ down to
≈ 10⁻⁵ for the autoencoder, Polar SC, and PPV bound.  Saved as Figure 5
(`figures/figure5_ber_curves_awgn.png`).

### Experiment 4 – Distributional Shift
Compares three scenarios for each block length:
  1. AWGN-trained model tested on AWGN (in-distribution baseline)
  2. AWGN-trained model tested on Rayleigh pCSI (distributional shift)
  3. Rayleigh-trained model tested on Rayleigh pCSI (matched baseline)

The SNR gap between scenarios 1 and 2 quantifies the performance penalty
due to channel mismatch — a known limitation of end-to-end learned systems.

---

## Citing

If you use this code, please cite the article:

```
Native AI Physical Layer for 6G: A Comprehensive Survey
Native_AI_Physical_Layer_6G_IEEE.md
Repository: eastaizah/Articulos
```
