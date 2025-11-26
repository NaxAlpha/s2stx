import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ResidualVectorQuantizerConfig:
    """Configuration for the residual vector quantizer.

    Attributes:
        dim: Dimensionality of latent vectors to quantize.
        num_codebooks: Number of residual quantization stages.
        codebook_size: Number of entries per codebook.
        commitment_weight: Weight for the commitment loss term.
    """

    dim: int = 128
    num_codebooks: int = 4
    codebook_size: int = 256
    commitment_weight: float = 0.25


class ResidualVectorQuantizer(nn.Module):
    """Simple residual vector quantizer (RVQ) implementation.

    This is intentionally lightweight and ONNX-friendly:
    - Uses only basic tensor ops (no control flow based on data).
    - Exposes indices and quantized latents for downstream models.
    """

    def __init__(self, cfg: ResidualVectorQuantizerConfig):
        super().__init__()
        self.cfg = cfg

        codebooks = []
        for _ in range(cfg.num_codebooks):
            # Codebook is [codebook_size, dim]
            emb = nn.Embedding(cfg.codebook_size, cfg.dim)
            nn.init.uniform_(emb.weight, -1.0 / cfg.codebook_size, 1.0 / cfg.codebook_size)
            codebooks.append(emb)
        self.codebooks = nn.ModuleList(codebooks)

    @property
    def num_codebooks(self) -> int:
        return self.cfg.num_codebooks

    @property
    def codebook_size(self) -> int:
        return self.cfg.codebook_size

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize latents with RVQ.

        Args:
            z: Latent tensor of shape [B, T, D].

        Returns:
            quantized: Quantized latents of shape [B, T, D].
            indices: Long tensor of shape [B, T, num_codebooks].
            vq_loss: Scalar VQ commitment + codebook loss.
        """
        assert z.dim() == 3, "Expected latent tensor of shape [B, T, D]"
        B, T, D = z.shape
        assert D == self.cfg.dim, f"Expected last dim={self.cfg.dim}, got {D}"

        residual = z
        all_indices: List[torch.Tensor] = []
        all_quantized: List[torch.Tensor] = []

        vq_loss = z.new_zeros(())

        for codebook in self.codebooks:
            # Flatten to [B*T, D]
            flat_residual = residual.reshape(-1, D)

            # Compute squared L2 distance to all codes: [B*T, K]
            # dist(x, e) = ||x||^2 + ||e||^2 - 2 x·e
            codebook_weight = codebook.weight  # [K, D]
            x_sq = (flat_residual ** 2).sum(dim=1, keepdim=True)  # [B*T, 1]
            e_sq = (codebook_weight ** 2).sum(dim=1)  # [K]
            # [B*T, K]
            distances = x_sq + e_sq.unsqueeze(0) - 2.0 * flat_residual @ codebook_weight.t()

            # Nearest neighbor indices
            indices = torch.argmin(distances, dim=1)  # [B*T]
            all_indices.append(indices.view(B, T))

            # Straight-through estimator
            z_q = codebook(indices).view(B, T, D)  # [B, T, D]
            all_quantized.append(z_q)

            # VQ loss (EMA-free, simple formulation)
            vq_loss = vq_loss + F.mse_loss(z_q.detach(), residual) + self.cfg.commitment_weight * F.mse_loss(
                z_q, residual.detach()
            )

            residual = residual - z_q.detach()

        # Sum quantized contributions from all codebooks
        quantized = torch.stack(all_quantized, dim=0).sum(dim=0)  # [B, T, D]

        # Combine indices: [B, T, num_codebooks]
        indices_stacked = torch.stack(all_indices, dim=-1)

        # Straight-through estimator for gradient passthrough
        quantized = z + (quantized - z).detach()

        return quantized, indices_stacked, vq_loss



