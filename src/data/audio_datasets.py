from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset


@dataclass
class AudioBatch:
    audio: torch.Tensor  # [B, T] float32 waveform at target_sr
    lengths: torch.Tensor  # [B] original (pre-padding) lengths


def _normalize_waveform(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    # Simple peak normalization
    max_val = np.max(np.abs(x)) + 1e-9
    return x / max_val


def _pad_batch(batch: list, target_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of 1D numpy arrays to the same length."""
    B = len(batch)
    audio = torch.zeros(B, target_len, dtype=torch.float32)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, arr in enumerate(batch):
        arr = _normalize_waveform(arr)
        arr_t = torch.from_numpy(arr)
        L = min(target_len, arr_t.shape[0])
        audio[i, :L] = arr_t[:L]
        lengths[i] = L
    return audio, lengths


def create_librispeech_streaming_dataloader(
    split: str = "train.100",
    batch_size: int = 4,
    target_sr: int = 16_000,
    max_duration_s: float = 4.0,
) -> Iterator[AudioBatch]:
    """Create an infinite iterator over LibriSpeech audio using HF streaming.

    Args:
        split: Which LibriSpeech split to use (e.g. 'train.100', 'train.clean.100').
        batch_size: Number of examples per batch.
        target_sr: Target sampling rate; LibriSpeech is 16 kHz by default.
        max_duration_s: Truncate/clip audio to this many seconds.

    Yields:
        AudioBatch with padded waveforms and lengths.
    """
    ds = load_dataset("librispeech_asr", "clean", split=split, streaming=True)

    iterator = iter(ds)
    max_len = int(target_sr * max_duration_s)

    while True:
        batch_arrays = []
        for _ in range(batch_size):
            try:
                example = next(iterator)
            except StopIteration:
                iterator = iter(ds)
                example = next(iterator)

            audio_arr = np.asarray(example["audio"]["array"], dtype=np.float32)
            if example["audio"]["sampling_rate"] != target_sr:
                # Simple resampling by linear interpolation; for higher quality
                # you can replace this with torchaudio.resample.
                orig_len = audio_arr.shape[0]
                new_len = int(orig_len * target_sr / example["audio"]["sampling_rate"])
                audio_arr = np.interp(
                    np.linspace(0, orig_len, new_len, endpoint=False),
                    np.arange(orig_len),
                    audio_arr,
                ).astype(np.float32)
            batch_arrays.append(audio_arr[:max_len])

        audio, lengths = _pad_batch(batch_arrays, max_len)
        yield AudioBatch(audio=audio, lengths=lengths)



