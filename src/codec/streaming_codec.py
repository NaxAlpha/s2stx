from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rvq import ResidualVectorQuantizer, ResidualVectorQuantizerConfig


@dataclass
class StreamingRVQCodecConfig:
    """Configuration for a lightweight streaming RVQ codec.

    This is intentionally modest so it can be trained on a single Colab GPU.
    You can scale channels or depth up if you have more compute.
    """

    sample_rate: int = 16_000
    # Downsampling by total stride = 8 → 16k / 8 = 2k frames/s (8 ms hop).
    # This is a bit finer than the 20 ms target, but keeps things simple.
    encoder_hidden: int = 128
    encoder_num_layers: int = 3
    rvq_dim: int = 128
    rvq_num_codebooks: int = 4
    rvq_codebook_size: int = 256
    decoder_hidden: int = 128


class ConvEncoder(nn.Module):
    def __init__(self, cfg: StreamingRVQCodecConfig):
        super().__init__()
        ch = cfg.encoder_hidden
        layers = []
        in_ch = 1
        # Three conv layers with stride 2 each → factor 8 downsampling
        for _ in range(cfg.encoder_num_layers):
            layers.append(
                nn.Conv1d(
                    in_ch,
                    ch,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                )
            )
            layers.append(nn.GroupNorm(8, ch))
            layers.append(nn.SiLU())
            in_ch = ch
        # Project to RVQ latent dim
        layers.append(nn.Conv1d(ch, cfg.rvq_dim, kernel_size=3, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to latents.

        Args:
            audio: [B, T] waveform in mono.

        Returns:
            latents: [B, T_enc, D].
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # [B, 1, T]
        x = self.net(audio)  # [B, C, T_enc]
        return x.transpose(1, 2)  # [B, T_enc, C]


class ConvDecoder(nn.Module):
    def __init__(self, cfg: StreamingRVQCodecConfig):
        super().__init__()
        ch = cfg.decoder_hidden
        layers = []

        in_ch = cfg.rvq_dim
        for _ in range(cfg.encoder_num_layers):
            # Mirror of encoder: stride 2 deconvs
            layers.append(
                nn.ConvTranspose1d(
                    in_ch,
                    ch,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    output_padding=1,
                )
            )
            layers.append(nn.GroupNorm(8, ch))
            layers.append(nn.SiLU())
            in_ch = ch
        # Final projection to mono waveform
        layers.append(nn.Conv1d(ch, 1, kernel_size=7, stride=1, padding=3))
        self.net = nn.Sequential(*layers)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode quantized latents back to waveform.

        Args:
            latents: [B, T_enc, D].

        Returns:
            audio: [B, T] waveform.
        """
        x = latents.transpose(1, 2)  # [B, D, T_enc]
        x = self.net(x)  # [B, 1, T]
        return x.squeeze(1)


class StreamingRVQCodec(nn.Module):
    """Lightweight convolutional RVQ codec with simple streaming API.

    The streaming interface is intentionally minimal: we keep a small overlap
    buffer on the encoder and decoder sides and operate on fixed audio chunks.
    """

    def __init__(self, cfg: Optional[StreamingRVQCodecConfig] = None):
        super().__init__()
        self.cfg = cfg or StreamingRVQCodecConfig()

        self.encoder = ConvEncoder(self.cfg)
        rvq_cfg = ResidualVectorQuantizerConfig(
            dim=self.cfg.rvq_dim,
            num_codebooks=self.cfg.rvq_num_codebooks,
            codebook_size=self.cfg.rvq_codebook_size,
        )
        self.rvq = ResidualVectorQuantizer(rvq_cfg)
        self.decoder = ConvDecoder(self.cfg)

    @property
    def num_codebooks(self) -> int:
        return self.rvq.num_codebooks

    @property
    def codebook_size(self) -> int:
        return self.rvq.codebook_size

    def encode(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode and quantize audio.

        Args:
            audio: [B, T] waveform.

        Returns:
            quantized_latents: [B, T_enc, D].
            indices: [B, T_enc, num_codebooks].
            vq_loss: scalar loss.
        """
        latents = self.encoder(audio)
        quantized_latents, indices, vq_loss = self.rvq(latents)
        return quantized_latents, indices, vq_loss

    def decode(self, quantized_latents: torch.Tensor) -> torch.Tensor:
        """Decode quantized latents to waveform.

        Args:
            quantized_latents: [B, T_enc, D].

        Returns:
            audio: [B, T] waveform.
        """
        return self.decoder(quantized_latents)

    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full codec pass: audio → tokens → reconstructed audio.

        Returns:
            recon: [B, T] reconstructed waveform.
            indices: [B, T_enc, num_codebooks].
            vq_loss: scalar loss.
        """
        quantized_latents, indices, vq_loss = self.encode(audio)
        recon = self.decode(quantized_latents)
        return recon, indices, vq_loss



