from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AudioTokenTransformerConfig:
    """Configuration for the audio token Transformer LM.

    Defaults are small enough for Colab, but you can scale them up.
    """

    vocab_size: int = 1024  # e.g. flattened 4x256 RVQ codebooks
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    d_ff: int = 2048
    max_seq_len: int = 2048
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("causal_mask", None, persistent=False)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        # Convert to additive mask: 0 for allowed, -inf for masked
        mask = mask.masked_fill(mask == 1, float("-inf")).masked_fill(mask == 0, 0.0)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, T, C] -> [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, n_heads, T, T]

        if self.causal_mask is None or self.causal_mask.size(0) < T:
            self.causal_mask = self._build_causal_mask(T, x.device)
        mask = self.causal_mask[:T, :T]
        attn_scores = attn_scores + mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        y = attn_weights @ v  # [B, n_heads, T, head_dim]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        return y


class TransformerBlock(nn.Module):
    def __init__(self, cfg: AudioTokenTransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class AudioTokenTransformerLM(nn.Module):
    """Decoder-only Transformer LM over codec token IDs."""

    def __init__(self, cfg: Optional[AudioTokenTransformerConfig] = None):
        super().__init__()
        self.cfg = cfg or AudioTokenTransformerConfig()

        self.token_embed = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.pos_embed = nn.Embedding(self.cfg.max_seq_len, self.cfg.d_model)
        self.drop = nn.Dropout(self.cfg.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(self.cfg) for _ in range(self.cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(self.cfg.d_model)
        self.head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)

        self.register_buffer(
            "position_ids",
            torch.arange(0, self.cfg.max_seq_len, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )

    def forward(
        self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            input_ids: [B, T] codec token IDs.
            labels: optional [B, T] next-token targets.

        Returns:
            logits: [B, T, vocab_size]
            loss: scalar cross-entropy loss if labels are provided.
        """
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, "Sequence length exceeds max_seq_len"

        pos_ids = self.position_ids[:, :T]
        x = self.token_embed(input_ids) + self.pos_embed(pos_ids)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size), labels.view(-1), ignore_index=-100
            )

        return logits, loss



