"""
Minimal codec training loop using HuggingFace LibriSpeech streaming.

This script is intentionally simple for Google Colab usage.

Example:
    pip install -r requirements.txt
    python -m src.train_codec --batch_size 8 --steps 1000
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from .codec import StreamingRVQCodec, StreamingRVQCodecConfig
from .data import create_librispeech_streaming_dataloader


def mse_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Simple reconstruction loss in the time domain.
    return ((recon - target) ** 2).mean()


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
        recon, _, vq_loss = model(audio)
        rec_loss = mse_loss(recon, audio)
        loss = rec_loss + vq_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            print(
                f"Step {step}/{steps} - loss={loss.item():.4f} "
                f"(rec={rec_loss.item():.4f}, vq={vq_loss.item():.4f})"
            )

        if step % 200 == 0 or step == steps:
            ckpt_path = output_dir / f"codec_step{step}.pt"
            torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict()}, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")


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


