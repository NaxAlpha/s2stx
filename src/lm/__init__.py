"""
Language model components for codec tokens.

The main entry point is `AudioTokenTransformerLM`, a decoder-only Transformer
over sequences of codec token IDs.
"""

from .transformer_lm import (
    AudioTokenTransformerConfig,
    AudioTokenTransformerLM,
)

__all__ = ["AudioTokenTransformerConfig", "AudioTokenTransformerLM"]



