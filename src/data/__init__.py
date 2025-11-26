"""
Data loading and HuggingFace streaming helpers.

The main entry point is `create_librispeech_streaming_dataloader`, which
provides a simple, infinite stream of 16 kHz speech batches suitable for
codec and LM training on Colab.
"""

from .audio_datasets import (
    create_librispeech_streaming_dataloader,
    AudioBatch,
)

__all__ = ["create_librispeech_streaming_dataloader", "AudioBatch"]



