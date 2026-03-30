"""
config.py
=========
Centralized configuration for all 6G AI-native PHY validation experiments.
All random seeds and hyperparameters are declared here to ensure
full reproducibility of Table II and the four additional experiments
described in Section L of Native_AI_Physical_Layer_6G_IEEE.md.

Random seeds
------------
SEED_TABLE2      = 42   – Table II AWGN benchmark (main result)
SEED_RAYLEIGH    = 123  – Rayleigh channel equivalent of Table II
SEED_BER_CURVES  = 456  – Complete BER vs Eb/N0 curve generation
SEED_SHIFT       = 789  – Distributional-shift evaluation
"""

# ---------------------------------------------------------------------------
# Random seeds (one per experiment for full independence)
# ---------------------------------------------------------------------------
SEED_TABLE2     = 42
SEED_RAYLEIGH   = 123
SEED_BER_CURVES = 456
SEED_SHIFT      = 789

# ---------------------------------------------------------------------------
# Block-length / code configurations
# n  – number of real channel uses (= 2× complex channel uses for SISO)
# k  – number of information bits
# M  – alphabet size = 2^k  (number of distinct messages)
#
# For n=7  k=4, M=16  follows O'Shea & Hoydis (2017) exactly.
# For n=16 k=8, M=256 → rate 1/2; exactly replicates Table II of the article.
# For n=32 k=8, M=256 → rate 1/4. k is capped at 8 for computational
#   tractability: k=16 would require M=65 536 output classes, making the
#   DECODER linear layer 128×65 536 ≈ 8M parameters and cross-entropy
#   impractical to optimise in reasonable time.  PPV and Polar baselines
#   are computed at the same (n=32, k=8) to keep the comparison fair.
# For n=64 k=8, M=256 → rate 1/8, same argument as n=32.
# ---------------------------------------------------------------------------
CONFIGS = [
    {"n":  7, "k": 4, "M":   16},
    {"n": 16, "k": 8, "M":  256},
    {"n": 32, "k": 8, "M":  256},
    {"n": 64, "k": 8, "M":  256},
]

# ---------------------------------------------------------------------------
# Autoencoder training hyper-parameters
# Exactly as specified in Section III.A.7 of the article:
#   Adam, lr=1e-3, 100 000 epochs, batch=256,
#   training SNR drawn uniformly from [SNR_TRAIN_MIN_DB, SNR_TRAIN_MAX_DB].
# ---------------------------------------------------------------------------
LEARNING_RATE   = 1e-3
NUM_EPOCHS      = 100_000
BATCH_SIZE      = 256
SNR_TRAIN_MIN_DB = 0.0    # dB  – lower end of training SNR curriculum
SNR_TRAIN_MAX_DB = 7.0    # dB  – upper end of training SNR curriculum

# Encoder hidden dimensions: k → H1 → H2 → n
ENCODER_H1 = 128
ENCODER_H2 = 64

# Decoder hidden dimensions: n → H1 → H2 → M
DECODER_H1 = 64
DECODER_H2 = 128

# ---------------------------------------------------------------------------
# Evaluation grid
# ---------------------------------------------------------------------------
# Es/N0 range for BER evaluation [dB]
EVAL_SNR_MIN_DB = -4.0
EVAL_SNR_MAX_DB = 12.0
EVAL_SNR_STEP   = 0.5     # dB step

# Number of Monte-Carlo batches per SNR point for reliable BER estimation.
# Total test samples = EVAL_BATCHES × BATCH_SIZE.
EVAL_BATCHES = 200

# Target BER for Table II threshold search
BER_TARGET = 1e-3

# ---------------------------------------------------------------------------
# Rayleigh channel training SNR range
# Wider than AWGN because fading increases required SNR.
# ---------------------------------------------------------------------------
SNR_RAYLEIGH_TRAIN_MIN_DB = 0.0
SNR_RAYLEIGH_TRAIN_MAX_DB = 15.0

# ---------------------------------------------------------------------------
# Distributional-shift experiment
# Evaluate AWGN-trained model on Rayleigh over a fine SNR grid.
# ---------------------------------------------------------------------------
SHIFT_EVAL_SNR_MIN_DB = 0.0
SHIFT_EVAL_SNR_MAX_DB = 20.0
SHIFT_EVAL_SNR_STEP   = 1.0

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
import os
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "saved_models")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
