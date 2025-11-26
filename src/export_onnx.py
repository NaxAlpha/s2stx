"""
ONNX export helpers for codec and LM.

This keeps the API very simple so you can run it on Colab after training:

    python -m src.export_onnx --codec_ckpt checkpoints/codec/codec_step1000.pt \\
        --lm_ckpt checkpoints/lm/lm_step1000.pt
"""

import argparse
from pathlib import Path

import torch

from .codec import StreamingRVQCodec, StreamingRVQCodecConfig
from .lm import AudioTokenTransformerConfig, AudioTokenTransformerLM


def export_codec_onnx(codec_ckpt: Path, output_dir: Path, device: str = "cpu") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = StreamingRVQCodecConfig()
    model = StreamingRVQCodec(cfg).to(device)
    if codec_ckpt.is_file():
        ckpt = torch.load(codec_ckpt, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded codec checkpoint from {codec_ckpt}")
    model.eval()

    dummy_audio = torch.zeros(1, cfg.sample_rate, dtype=torch.float32).to(device)

    # Export encoder (audio -> quantized latents + indices)
    encoder_path = output_dir / "encoder.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_audio,
            encoder_path,
            input_names=["audio"],
            output_names=["recon", "indices", "vq_loss"],
            opset_version=17,
            dynamic_axes={"audio": {1: "time"}},
        )
    print(f"Saved codec ONNX to {encoder_path}")


def export_lm_onnx(lm_ckpt: Path, vocab_size: int, output_dir: Path, device: str = "cpu") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = AudioTokenTransformerConfig(vocab_size=vocab_size)
    model = AudioTokenTransformerLM(cfg).to(device)
    if lm_ckpt.is_file():
        ckpt = torch.load(lm_ckpt, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded LM checkpoint from {lm_ckpt}")
    model.eval()

    dummy_tokens = torch.zeros(1, 16, dtype=torch.long).to(device)

    lm_path = output_dir / "lm.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_tokens,
            lm_path,
            input_names=["input_ids"],
            output_names=["logits", "loss"],
            opset_version=17,
            dynamic_axes={"input_ids": {1: "time"}},
        )
    print(f"Saved LM ONNX to {lm_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export codec and LM to ONNX.")
    parser.add_argument("--codec_ckpt", type=str, required=True)
    parser.add_argument("--lm_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="onnx")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    codec_ckpt = Path(args.codec_ckpt)
    lm_ckpt = Path(args.lm_ckpt)
    out_dir = Path(args.output_dir)

    # Derive vocab size from codec config (4 codebooks x 256 codes by default)
    codec_cfg = StreamingRVQCodecConfig()
    vocab = codec_cfg.rvq_codebook_size * codec_cfg.rvq_num_codebooks

    export_codec_onnx(codec_ckpt, out_dir / "codec", device=args.device)
    export_lm_onnx(lm_ckpt, vocab, out_dir / "lm", device=args.device)


