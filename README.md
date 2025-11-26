## Pure-Audio Neural Codec LM (<100M, Streaming, ONNX)

This repository implements a **pure-audio neural codec language model** designed to:

- Use a **streaming neural audio codec** (SoundStream/EnCodec-style) with residual vector quantization.
- Run a **decoder-only Transformer LM** over discrete codec tokens.
- Keep the **total parameter count under 100M** (codec + LM).
- Be **ONNX-exportable** and suitable for **real-time mobile deployment** (e.g. iPhone SE, Android phones).

The code focuses on:

- A small, ONNX-friendly **convolutional RVQ codec** with streaming wrappers.
- A compact **Transformer-based audio token LM** with support for chunked, streaming-style decoding.
- A **training pipeline skeleton** for codec pretraining, LM training, and ONNX export/benchmarking.

> Note: This project provides a solid, extensible starting point. It does **not** include pretrained weights or a full training setup for the very large datasets mentioned in the design plan.

### Quickstart (Google Colab friendly, using `uv`)

- **Install `uv` in the Colab session**:

  ```bash
  pip install uv
  ```

- **Sync dependencies into a virtual env** (created in `.venv`):

  ```bash
  uv sync
  ```

- **Train the codec** (streaming LibriSpeech from HuggingFace):

  ```bash
  uv run python -m src.train_codec --batch_size 8 --steps 1000
  ```

  This saves checkpoints under `checkpoints/codec/`.

- **Train the LM over codec tokens**:

  ```bash
  uv run python -m src.train_lm --codec_ckpt checkpoints/codec/codec_step1000.pt \
      --batch_size 4 --steps 1000
  ```

  This saves checkpoints under `checkpoints/lm/`.

- **Export to ONNX**:

  ```bash
  uv run python -m src.export_onnx \
      --codec_ckpt checkpoints/codec/codec_step1000.pt \
      --lm_ckpt checkpoints/lm/lm_step1000.pt \
      --output_dir onnx
  ```

  You will get basic `encoder.onnx` and `lm.onnx` files you can further optimize for mobile.



