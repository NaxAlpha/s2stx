"""
Streaming neural audio codec components.

The main entry point is `StreamingRVQCodec`, which wraps:
- A convolutional encoder that downsamples audio into latent frames.
- A residual vector quantizer (RVQ) producing discrete codebook indices.
- A convolutional decoder that reconstructs audio from quantized latents.
"""

from .streaming_codec import StreamingRVQCodec, StreamingRVQCodecConfig

__all__ = ["StreamingRVQCodec", "StreamingRVQCodecConfig"]



