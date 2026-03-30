"""
models.py
=========
PyTorch Encoder, Decoder and Autoencoder for the end-to-end communication
system described in Section III.A of Native_AI_Physical_Layer_6G_IEEE.md.

Architecture (Section III.A.7):
  Encoder : k → Linear(128) → ReLU → Linear(64) → ReLU → Linear(n)
            followed by per-sample average-power normalisation to unit
            energy per channel use: E[|x_i|²] = 1.
  Decoder : n → Linear(64) → ReLU → Linear(128) → ReLU → Linear(M)
            (raw logits; apply log-softmax / cross-entropy during training).

Input convention:
  The encoder receives the k information bits as a float vector in {0,1}^k.
  This avoids an M-dimensional one-hot lookup and keeps the architecture
  well-defined for large k (e.g. k=16).

Loss:
  The decoder produces M raw logits.  The message index (integer label
  0 … M-1) is the target for nn.CrossEntropyLoss.  Training labels are
  obtained from the k-bit input via label = int(bits, base=2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Maps k information bits → n normalised real channel symbols.

    Parameters
    ----------
    k : int  Number of information bits.
    n : int  Number of channel uses (real dimensions).
    h1, h2 : int  Hidden-layer widths (default 128, 64 per article spec).
    """

    def __init__(self, k: int, n: int, h1: int = 128, h2: int = 64):
        super().__init__()
        self.k = k
        self.n = n
        self.net = nn.Sequential(
            nn.Linear(k, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, n),
        )

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        bits : (B, k) float tensor with values in {0., 1.}

        Returns
        -------
        x : (B, n) power-normalised real symbols, E[||x||²/n] = 1
        """
        x = self.net(bits.float())
        # Per-sample power normalisation: ||x||² / n = 1
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) * self.n + 1e-8)
        return x / norm


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """Maps n received real values → logits over M messages.

    Parameters
    ----------
    n : int  Number of channel uses.
    M : int  Alphabet size (M = 2^k).
    h1, h2 : int  Hidden-layer widths (default 64, 128 per article spec).
    """

    def __init__(self, n: int, M: int, h1: int = 64, h2: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, M),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        y : (B, n) received signal (real)

        Returns
        -------
        logits : (B, M) unnormalised log-probabilities
        """
        return self.net(y)


# ---------------------------------------------------------------------------
# Full autoencoder (encoder + decoder, no channel layer)
# ---------------------------------------------------------------------------

class Autoencoder(nn.Module):
    """End-to-end communication autoencoder.

    The channel is injected externally during the forward pass so the
    same model can be evaluated on AWGN, Rayleigh, etc.

    Parameters
    ----------
    k : int   Information bits.
    n : int   Channel uses.
    M : int   Alphabet size.
    """

    def __init__(self, k: int, n: int, M: int,
                 enc_h1: int = 128, enc_h2: int = 64,
                 dec_h1: int = 64,  dec_h2: int = 128):
        super().__init__()
        self.k = k
        self.n = n
        self.M = M
        self.encoder = Encoder(k, n, h1=enc_h1, h2=enc_h2)
        self.decoder = Decoder(n, M, h1=dec_h1, h2=dec_h2)

    def encode(self, bits: torch.Tensor) -> torch.Tensor:
        return self.encoder(bits)

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        return self.decoder(y)

    def forward(self, bits: torch.Tensor, channel_fn) -> torch.Tensor:
        """
        Parameters
        ----------
        bits       : (B, k) float tensor, values in {0., 1.}
        channel_fn : callable(x) → y  (e.g. awgn_channel or rayleigh_channel)

        Returns
        -------
        logits : (B, M) raw decoder outputs
        """
        x = self.encode(bits)
        y = channel_fn(x)
        return self.decode(y)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def bits_to_labels(bits: torch.Tensor) -> torch.Tensor:
    """Convert (B, k) binary tensor to (B,) integer labels 0 … 2^k-1."""
    k = bits.shape[-1]
    powers = 2 ** torch.arange(k - 1, -1, -1, device=bits.device, dtype=torch.long)
    return (bits.long() * powers).sum(dim=-1)


def labels_to_bits(labels: torch.Tensor, k: int) -> torch.Tensor:
    """Convert (B,) integer labels to (B, k) binary tensor."""
    device = labels.device
    powers = 2 ** torch.arange(k - 1, -1, -1, device=device, dtype=torch.long)
    bits = ((labels.unsqueeze(-1) & powers) > 0).float()
    return bits


def count_parameters(model: nn.Module) -> int:
    """Return total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
