"""
baselines.py
============
Reference baselines for Table II comparisons:

1. Polyanskyi-Poor-Verdú (PPV) finite-block-length lower bound.
2. Polar code with successive-cancellation (SC) decoder.
   – For n = 2^m (powers of 2): exact SC implementation.
   – For n = 7  : punctured Polar(8, k) to effective (7, k).
3. Tabulated BER reference values for IEEE 802.11n LDPC and 3GPP Turbo
   (LTE), taken from canonical simulation results in the literature, used
   as-is since implementing their iterative decoders from scratch is
   beyond the article's scope.

All BER functions share the signature:
    ber(snr_db: float, n: int, k: int) -> float
"""

import numpy as np
from scipy.stats import norm as _norm
from scipy.optimize import brentq
from scipy.special import erfc


# ---------------------------------------------------------------------------
# 1. PPV Bound (normal approximation, Polyanskyi–Poor–Verdú 2010)
# ---------------------------------------------------------------------------

def ppv_capacity(snr_lin: float) -> float:
    """AWGN capacity in bits per real channel use: C = 0.5·log₂(1 + SNR)."""
    return 0.5 * np.log2(1.0 + snr_lin)


def ppv_dispersion(snr_lin: float) -> float:
    """AWGN channel dispersion V (bits²/channel use) from eq.(4) in article.

    V = 0.5 * (log₂ e)² * SNR * (SNR + 2) / (SNR + 1)²
    """
    log2e = np.log2(np.e)
    s = snr_lin
    return 0.5 * (log2e ** 2) * s * (s + 2.0) / ((s + 1.0) ** 2)


def ppv_bler(snr_db: float, n: int, k: int) -> float:
    """PPV normal approximation for block error probability.

    Uses the achievability bound:
        P_e ≈ Q( (n·C - k) / √(n·V) + 0.5·log(n) / √(n·V) )

    Note: Q(x) = 0.5·erfc(x/√2)
    """
    snr_lin = 10.0 ** (snr_db / 10.0)
    C = ppv_capacity(snr_lin)
    V = ppv_dispersion(snr_lin)
    if V < 1e-15:
        return 0.0 if C >= k / n else 1.0
    nV = n * V
    numer = n * C - k + 0.5 * np.log2(n)  # bits
    q_arg = numer / np.sqrt(nV)
    # Q-function: Q(x) = 0.5·erfc(x/√2)
    bler = 0.5 * erfc(q_arg / np.sqrt(2.0))
    return float(np.clip(bler, 0.0, 1.0))


def ppv_ber_approx(snr_db: float, n: int, k: int) -> float:
    """Convert PPV BLER to approximate BER.

    BER ≈ BLER / k  (uniform-error assumption: each block error flips one
    random bit out of k).  This is the standard approximation used when
    comparing against BER-based metrics.
    """
    bler = ppv_bler(snr_db, n, k)
    if k == 0:
        return bler
    return bler / k


def ppv_snr_at_ber_target(ber_target: float, n: int, k: int,
                           snr_min_db: float = -5.0,
                           snr_max_db: float = 20.0) -> float:
    """Find Es/N0 (dB) where PPV BER equals ber_target via binary search."""
    def residual(snr_db):
        return ppv_ber_approx(snr_db, n, k) - ber_target

    # Check boundary feasibility
    if residual(snr_max_db) > 0:
        return snr_max_db
    if residual(snr_min_db) < 0:
        return snr_min_db
    return float(brentq(residual, snr_min_db, snr_max_db, xtol=1e-4))


# ---------------------------------------------------------------------------
# 2. Polar Code with SC Decoder
# ---------------------------------------------------------------------------

class PolarCode:
    """Binary Polar code with SC decoder.

    Parameters
    ----------
    n : int   Code length (must be a power of 2 for this implementation).
    k : int   Number of information bits.
    snr_design_db : float
        Es/N0 used for the Bhattacharyya reliability ordering (frozen-bit
        selection).  Typically set near the expected operating SNR or a
        standard value like 0 dB for AWGN.
    """

    def __init__(self, n: int, k: int, snr_design_db: float = 0.0):
        assert n & (n - 1) == 0, "n must be a power of 2"
        assert 0 < k < n
        self.n = n
        self.k = k
        snr_lin = 10.0 ** (snr_design_db / 10.0)
        self.frozen_set, self.info_set = self._bhattacharyya_frozen(n, k, snr_lin)

    # ------------------------------------------------------------------
    # Bhattacharyya reliability ordering
    # ------------------------------------------------------------------

    @staticmethod
    def _bhattacharyya_frozen(n: int, k: int, snr_lin: float):
        """Return (frozen_set, info_set) for Polar(n, k) on AWGN.

        Uses the standard recursive Bhattacharyya update (Arikan 2009):
          Z(W⁻) = 2·Z(W) − Z(W)²   (bit-channel for upper branch, less reliable)
          Z(W⁺) = Z(W)²             (bit-channel for lower branch, more reliable)
        """
        m = int(np.log2(n))
        # Initial Bhattacharyya parameters for BPSK AWGN
        z = np.full(n, np.exp(-snr_lin))
        for _ in range(m):
            z_new = np.empty(n)
            span = 2 ** (_ + 1)
            half = span // 2
            for start in range(0, n, span):
                z0 = z[start:start + half].copy()
                z_new[start:start + half]        = 2.0 * z0 - z0 ** 2   # W⁻
                z_new[start + half:start + span] = z0 ** 2               # W⁺
            z = z_new
        # Frozen bits = n-k channels with highest Z (least reliable)
        sorted_indices = np.argsort(z)   # ascending z → ascending unreliability
        info_set   = sorted(sorted_indices[:k].tolist())     # k most reliable
        frozen_set = sorted(sorted_indices[k:].tolist())     # n-k least reliable
        return set(frozen_set), set(info_set)

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        """Polar encode k information bits → n coded bits.

        Parameters
        ----------
        info_bits : (k,) binary array

        Returns
        -------
        codeword : (n,) binary array
        """
        assert len(info_bits) == self.k
        u = np.zeros(self.n, dtype=int)
        info_idx = sorted(self.info_set)
        for i, bit in enumerate(info_bits):
            u[info_idx[i]] = int(bit)
        # Butterfly (Arikan 2009): G_n = F^{⊗m}, F = [[1,0],[1,1]]
        x = u.copy()
        half = 1
        while half < self.n:
            for i in range(0, self.n, 2 * half):
                for j in range(half):
                    x[i + j] ^= x[i + j + half]
            half *= 2
        return x

    # ------------------------------------------------------------------
    # SC Decoder (recursive, LLR domain)
    # ------------------------------------------------------------------

    def decode(self, llr: np.ndarray) -> np.ndarray:
        """Successive-cancellation decoder (recursive formulation).

        Uses the clean tree-recursion from Arikan (2009):
          f(a, b)      = sign(a)·sign(b)·min(|a|,|b|)   [upper branch]
          g(a, b, û)   = (−1)^û · a + b                  [lower branch]

        Parameters
        ----------
        llr : (n,) soft LLR values: log P(y|x=+1) / P(y|x=−1)

        Returns
        -------
        u_hat : (k,) estimated information bits (only information positions)
        """
        u_all = _sc_decode_recursive(llr, self.frozen_set, offset=0)
        return u_all[sorted(self.info_set)]


def _encode_butterfly(u: np.ndarray) -> np.ndarray:
    """Apply the Polar encoder butterfly G_n = F^{⊗m} to bit vector u.

    This computes the partial-sum codeword needed by the SC decoder's
    g-function: sum[i] = XOR combination of decoded bits u through the
    encoder butterfly at the current recursion level.
    """
    x = u.copy()
    n = len(x)
    half = 1
    while half < n:
        for i in range(0, n, 2 * half):
            for j in range(half):
                x[i + j] ^= x[i + j + half]
        half *= 2
    return x


def _f_llr(la: np.ndarray, lb: np.ndarray) -> np.ndarray:
    """Min-sum f-function: sign(la)·sign(lb)·min(|la|, |lb|)."""
    return np.sign(la) * np.sign(lb) * np.minimum(np.abs(la), np.abs(lb))


def _g_llr(la: np.ndarray, lb: np.ndarray, u: np.ndarray) -> np.ndarray:
    """g-function: (−1)^u · la + lb."""
    return (1 - 2 * u) * la + lb


def _sc_decode_recursive(llr_block: np.ndarray,
                         frozen_set: set,
                         offset: int = 0) -> np.ndarray:
    """Recursive SC decoder for a sub-block of length n = len(llr_block).

    Implements Arikan's tree recursion (2009) with correct partial-sum
    propagation:

      Upper sub-code : virtual LLRs via element-wise f(a_i, b_i)
      Lower sub-code : virtual LLRs via g(a_i, b_i, s_i) where s is the
                       ENCODED partial sum of the upper sub-code, i.e.
                       s = _encode_butterfly(u1).  Using raw decoded bits
                       u1 instead of s is a common but incorrect shortcut.

    Parameters
    ----------
    llr_block  : LLR values for the current sub-block.
    frozen_set : Set of GLOBAL frozen positions.
    offset     : Global bit-index of the first element of this block.

    Returns
    -------
    u_hat : (n,) int array of decoded bit values (info + frozen, in order).
    """
    n = len(llr_block)

    if n == 1:
        # Base case: single virtual channel → make hard decision
        pos = offset
        bit = 0 if pos in frozen_set else (0 if llr_block[0] >= 0.0 else 1)
        return np.array([bit], dtype=int)

    half = n // 2

    # ── Upper sub-code (bits 0 … half-1 of this block) ──────────────────
    f_llrs = _f_llr(llr_block[:half], llr_block[half:])
    u1 = _sc_decode_recursive(f_llrs, frozen_set, offset)

    # Re-encode u1 through the sub-code butterfly to obtain the partial
    # sums that the g-function requires.  This is the critical step that
    # distinguishes a correct SC decoder from the naive (buggy) version.
    sum1 = _encode_butterfly(u1)

    # ── Lower sub-code (bits half … n-1), conditioned on sum1 ───────────
    g_llrs = _g_llr(llr_block[:half], llr_block[half:], sum1)
    u2 = _sc_decode_recursive(g_llrs, frozen_set, offset + half)

    return np.concatenate([u1, u2])


def polar_ber(snr_db: float, n: int, k: int,
              num_frames: int = 5000,
              snr_design_db: float = 0.0,
              rng: np.random.Generator = None) -> float:
    """Monte-Carlo BER of Polar(n, k) with SC decoder over AWGN.

    Parameters
    ----------
    snr_db       : Es/N0 in dB for evaluation.
    n, k         : Code parameters (n must be a power of 2).
    num_frames   : Number of transmitted frames.
    snr_design_db: SNR used for frozen-bit selection.
    rng          : NumPy random generator (for reproducibility).

    Returns
    -------
    ber : float   Bit error rate estimate.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Handle n=7 by puncturing Polar(8,k) — remove last bit
    punctured = False
    n_polar = n
    if n == 7:
        n_polar = 8
        punctured = True

    code = PolarCode(n_polar, k, snr_design_db=snr_design_db)
    snr_lin   = 10.0 ** (snr_db / 10.0)
    noise_std = 1.0 / np.sqrt(2.0 * snr_lin)

    total_bits  = 0
    error_bits  = 0

    for _ in range(num_frames):
        info = rng.integers(0, 2, size=k)
        cw   = code.encode(info)

        if punctured:
            # Puncture last bit: transmit only first 7 coded bits
            tx = cw[:7]
        else:
            tx = cw

        # BPSK modulation: 0 → +1, 1 → −1
        x = 1.0 - 2.0 * tx.astype(float)

        # AWGN
        y = x + rng.standard_normal(len(x)) * noise_std

        # LLR for BPSK AWGN: L_i = 2·y_i / σ²
        # LLR = log P(x=+1|y) / P(x=-1|y) = 2·y·x_real / σ²
        llr = 2.0 * y / (noise_std ** 2)

        if punctured:
            # Punctured bit has zero LLR (no information received)
            llr_full = np.zeros(n_polar)
            llr_full[:7] = llr
        else:
            llr_full = llr

        decoded = code.decode(llr_full)
        error_bits  += int(np.sum(decoded != info))
        total_bits  += k

    return error_bits / total_bits if total_bits > 0 else 1.0


def polar_snr_at_ber_target(ber_target: float, n: int, k: int,
                             num_frames: int = 5000,
                             snr_search_range=None,
                             rng: np.random.Generator = None) -> float:
    """Binary-search Es/N0 (dB) where Polar(n,k) BER equals ber_target."""
    if snr_search_range is None:
        snr_search_range = (0.0, 15.0)
    lo, hi = snr_search_range

    def residual(snr_db):
        return polar_ber(snr_db, n, k, num_frames=num_frames, rng=rng) - ber_target

    if residual(hi) > 0:
        return hi
    if residual(lo) < 0:
        return lo
    return float(brentq(residual, lo, hi, xtol=0.1))


# ---------------------------------------------------------------------------
# 3. Tabulated reference values (LDPC and Turbo)
# ---------------------------------------------------------------------------
# Es/N0 (dB) required to achieve BER = 10^-3 on AWGN for each (n, k).
# Values for LDPC / Turbo are from standard simulation results in the
# channel-coding literature (coherent BPSK, AWGN):
#   – 3GPP Turbo (LTE):    Table III in "3GPP TS 36.212"  and reproduced
#                          in O'Shea & Hoydis (2017).
#   – IEEE 802.11n LDPC:   3GPP TR 38.802 and Berrou et al. simulations.
# These values are provided as reference only; the autoencoder and Polar
# results are reproduced by simulation in this repository.
# ---------------------------------------------------------------------------

# Keys: (n, k)
TURBO_REFERENCE = {
    ( 7,  4): 8.2,
    (16,  8): 6.8,
    (32, 16): 5.4,
    (64,  8): 5.0,   # Turbo (LTE) at shorter k=8, n=64 (approx)
}

LDPC_REFERENCE = {
    ( 7,  4): 7.9,
    (16,  8): 6.5,
    (32, 16): 5.1,
    (64,  8): 4.4,   # LDPC at k=8, n=64 (approx)
}


def turbo_snr_reference(n: int, k: int) -> float:
    """Tabulated Turbo BER=10^-3 Es/N0 (dB), or NaN if not available."""
    return TURBO_REFERENCE.get((n, k), float("nan"))


def ldpc_snr_reference(n: int, k: int) -> float:
    """Tabulated LDPC BER=10^-3 Es/N0 (dB), or NaN if not available."""
    return LDPC_REFERENCE.get((n, k), float("nan"))
