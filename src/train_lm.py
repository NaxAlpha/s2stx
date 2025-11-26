"""
Minimal LM training loop over codec tokens.

For simplicity and Colab friendliness, this script:
- Loads a *frozen* codec checkpoint.
- On-the-fly encodes LibriSpeech audio into codec token sequences.
- Trains a small decoder-only Transformer LM to predict next tokens.

Example:
    python -m src.train_lm --codec_ckpt checkpoints/codec/codec_step1000.pt
"""

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.optim as optim

from .codec import StreamingRVQCodec, StreamingRVQCodecConfig
from .data import create_librispeech_streaming_dataloader
from .lm import AudioTokenTransformerConfig, AudioTokenTransformerLM


def flatten_rvq_indices(indices: torch.Tensor, codebook_size: int) -> torch.Tensor:
    """Flatten multi-codebook indices to a single token ID.

    Args:
        indices: [B, T_enc, num_codebooks]

    Returns:
        flat_ids: [B, T_enc * num_codebooks]
    """
    B, T_enc, Q = indices.shape
    device = indices.device

    # Compute flat IDs per (time, codebook)
    codebook_ids = torch.arange(Q, device=device).view(1, 1, Q)
    flat_ids = indices + codebook_ids * codebook_size
    # [B, T_enc * Q]
    flat_ids = flat_ids.view(B, T_enc * Q)
    return flat_ids


def make_lm_batch(
    codec: StreamingRVQCodec,
    audio_batch: torch.Tensor,
    codebook_size: int,
    max_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode audio and build input/label sequences for the LM."""
    with torch.no_grad():
        _, indices, _ = codec.encode(audio_batch)
    tokens = flatten_rvq_indices(indices, codebook_size)  # [B, L]

    # Truncate or pad to max_seq_len
    B, L = tokens.shape
    if L > max_seq_len:
        tokens = tokens[:, :max_seq_len]
        L = max_seq_len

    input_ids = tokens.clone()
    labels = tokens.clone()

    # Shift for next-token prediction: labels[t] is the target for input[t]
    # We mask the first position.
    labels[:, 0] = -100
    return input_ids, labels


def train_lm(
    codec_ckpt: Path,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 4,
    steps: int = 1000,
    lr: float = 1e-4,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load codec and freeze
    codec_cfg = StreamingRVQCodecConfig()
    codec = StreamingRVQCodec(codec_cfg).to(device)
    if codec_ckpt.is_file():
        ckpt = torch.load(codec_ckpt, map_location=device)
        codec.load_state_dict(ckpt["state_dict"])
        print(f"Loaded codec checkpoint from {codec_ckpt}")
    codec.eval()
    for p in codec.parameters():
        p.requires_grad = False

    vocab_size = codec.codebook_size * codec.num_codebooks

    lm_cfg = AudioTokenTransformerConfig(
        vocab_size=vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=8,  # slightly smaller than the plan for Colab
        d_ff=2048,
        max_seq_len=1024,
        dropout=0.1,
    )
    model = AudioTokenTransformerLM(lm_cfg).to(device)

    opt = optim.AdamW(model.parameters(), lr=lr)

    dataloader = create_librispeech_streaming_dataloader(batch_size=batch_size)

    model.train()
    for step in range(1, steps + 1):
        batch = next(dataloader)
        audio = batch.audio.to(device)

        input_ids, labels = make_lm_batch(
            codec=codec,
            audio_batch=audio,
            codebook_size=codec.codebook_size,
            max_seq_len=lm_cfg.max_seq_len,
        )
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        opt.zero_grad()
        _, loss = model(input_ids, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            print(f"Step {step}/{steps} - loss={loss.item():.4f}")

        if step % 200 == 0 or step == steps:
            ckpt_path = output_dir / f"lm_step{step}.pt"
            torch.save({"cfg": lm_cfg.__dict__, "state_dict": model.state_dict()}, ckpt_path)
            print(f"Saved LM checkpoint to {ckpt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train audio token Transformer LM.")
    parser.add_argument(
        "--codec_ckpt",
        type=str,
        required=True,
        help="Path to trained codec checkpoint (.pt).",
    )
    parser.add_argument("--output_dir", type=str, default="checkpoints/lm")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_lm(
        codec_ckpt=Path(args.codec_ckpt),
        output_dir=Path(args.output_dir),
        device=args.device,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
    )


