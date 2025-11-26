"""
Minimal codec training loop using HuggingFace LibriSpeech streaming.

This script is intentionally simple for Google Colab usage, but it also
includes a set of auxiliary losses to improve perceptual quality:

- Time-domain MSE
- Multi-resolution STFT magnitude loss
- Log-mel spectrogram loss
- Loudness (RMS) matching
- Latent smoothness over time
- Latent variance regularization
- Codebook diversity regularization

Example:
    uv run python -m src.train_codec --batch_size 8 --steps 1000
"""

import argparse
from pathlib import Path

import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio

from .codec import StreamingRVQCodec, StreamingRVQCodecConfig
from .data import create_librispeech_streaming_dataloader


def mse_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Simple reconstruction loss in the time domain.
    return ((recon - target) ** 2).mean()


def multi_resolution_stft_loss(
    audio: torch.Tensor,
    recon: torch.Tensor,
    sample_rate: int,
    fft_sizes=(256, 1024, 4096),
) -> torch.Tensor:
    """Multi-resolution STFT magnitude loss."""
    device = audio.device
    loss = audio.new_zeros(())
    for n_fft in fft_sizes:
        hop = n_fft // 4
        win = torch.hann_window(n_fft, device=device)
        x_stft = torch.stft(
            audio, n_fft=n_fft, hop_length=hop, window=win, return_complex=True, center=True
        )
        y_stft = torch.stft(
            recon, n_fft=n_fft, hop_length=hop, window=win, return_complex=True, center=True
        )
        loss = loss + torch.mean(torch.abs(x_stft.abs() - y_stft.abs()))
    return loss / len(fft_sizes)


_mel_spec_transform = None


def log_mel_spectrogram_loss(
    audio: torch.Tensor,
    recon: torch.Tensor,
    sample_rate: int,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> torch.Tensor:
    """Log-mel spectrogram L1 loss."""
    global _mel_spec_transform
    if _mel_spec_transform is None:
        _mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
        )

    _mel_spec_transform.to(audio.device)

    mel_x = _mel_spec_transform(audio)
    mel_y = _mel_spec_transform(recon)

    log_mel_x = torch.log(mel_x + 1e-5)
    log_mel_y = torch.log(mel_y + 1e-5)

    return torch.mean(torch.abs(log_mel_x - log_mel_y))


def loudness_rms_loss(audio: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Match global RMS loudness."""
    rms_x = torch.sqrt(torch.mean(audio ** 2, dim=1) + 1e-8)
    rms_y = torch.sqrt(torch.mean(recon ** 2, dim=1) + 1e-8)
    return torch.mean((rms_x - rms_y) ** 2)


def latent_smoothness_loss(latents: torch.Tensor) -> torch.Tensor:
    """Encourage temporal smoothness in quantized latents."""
    # latents: [B, T_enc, D]
    diff = latents[:, 1:, :] - latents[:, :-1, :]
    return torch.mean(diff ** 2)


def latent_variance_loss(latents: torch.Tensor, target_var: float = 1.0) -> torch.Tensor:
    """Regularize latent variance to be close to a target value."""
    # Compute variance per dimension over batch and time.
    mean = latents.mean(dim=(0, 1), keepdim=True)
    centered = latents - mean
    var = (centered ** 2).mean(dim=(0, 1))  # [D]
    return torch.mean((var - target_var) ** 2)


def codebook_diversity_loss(model: StreamingRVQCodec) -> torch.Tensor:
    """Simple diversity regularizer on codebook embeddings.

    Encourages code vectors within each codebook to be spread out by
    maximizing their variance (implemented as inverse-variance penalty).
    """
    device = next(model.parameters()).device
    loss = torch.tensor(0.0, device=device)
    for emb in model.rvq.codebooks:
        w = emb.weight  # [K, D]
        mean = w.mean(dim=0, keepdim=True)
        centered = w - mean
        var = (centered ** 2).mean()  # scalar
        loss = loss + 1.0 / (var + 1e-5)
    return loss / len(model.rvq.codebooks)


def train_codec(
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 8,
    steps: int = 1000,
    lr: float = 1e-4,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = StreamingRVQCodecConfig()
    model = StreamingRVQCodec(cfg).to(device)

    opt = optim.AdamW(model.parameters(), lr=lr)

    dataloader = create_librispeech_streaming_dataloader(batch_size=batch_size)

    model.train()
    for step in range(1, steps + 1):
        batch = next(dataloader)
        audio = batch.audio.to(device)

        opt.zero_grad()

        # Encode/decode explicitly so we can access quantized latents.
        quantized_latents, indices, vq_loss = model.encode(audio)
        recon = model.decode(quantized_latents)

        rec_loss = mse_loss(recon, audio)
        stft_loss = multi_resolution_stft_loss(audio, recon, cfg.sample_rate)
        mel_loss = log_mel_spectrogram_loss(audio, recon, cfg.sample_rate)
        rms_loss = loudness_rms_loss(audio, recon)
        smooth_loss = latent_smoothness_loss(quantized_latents)
        var_loss = latent_variance_loss(quantized_latents)
        cb_div_loss = codebook_diversity_loss(model)

        # Loss weights (tunable)
        loss = (
            rec_loss
            + vq_loss
            + 0.5 * stft_loss
            + 0.5 * mel_loss
            + 0.1 * rms_loss
            + 0.1 * smooth_loss
            + 0.01 * var_loss
            + 0.001 * cb_div_loss
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            print(
                f"Step {step}/{steps} - "
                f"loss={loss.item():.4f} "
                f"(rec={rec_loss.item():.4f}, vq={vq_loss.item():.4f}, "
                f"stft={stft_loss.item():.4f}, mel={mel_loss.item():.4f}, "
                f"rms={rms_loss.item():.4f}, smooth={smooth_loss.item():.4f}, "
                f"var={var_loss.item():.4f}, cb_div={cb_div_loss.item():.4f})"
            )

        if step % 500 == 0 or step == steps:
            ckpt_path = output_dir / f"codec_step_{step:06d}.pt"
            torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict()}, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

            # Save a few input/reconstruction waveform pairs for quick listening tests.
            sample_dir = output_dir / f"samples_{step:06d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            with torch.no_grad():
                # Use the current batch for samples.
                num_samples = min(5, audio.shape[0])
                audio_cpu = audio[:num_samples].detach().cpu()
                recon_cpu = recon[:num_samples].detach().cpu()

                for i in range(num_samples):
                    in_path = sample_dir / f"input_{i}.wav"
                    out_path = sample_dir / f"recon_{i}.wav"
                    sf.write(str(in_path), audio_cpu[i].numpy(), cfg.sample_rate)
                    sf.write(str(out_path), recon_cpu[i].numpy(), cfg.sample_rate)

            print(f"Saved {num_samples} sample pairs to {sample_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train streaming RVQ codec on LibriSpeech.")
    parser.add_argument("--output_dir", type=str, default="checkpoints/codec")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_codec(
        output_dir=Path(args.output_dir),
        device=args.device,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
    )


