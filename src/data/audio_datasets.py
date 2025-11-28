from dataclasses import dataclass
from typing import Iterator, Tuple, List, Dict, Any

import numpy as np
import torch
from datasets import load_dataset
import random


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
    max_duration_s: float = 8.0,
) -> Iterator[AudioBatch]:
    """Create an infinite iterator over LibriSpeech audio using HF streaming."""
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

            # Randomly crop a segment of length max_len if the sample is longer.
            if audio_arr.shape[0] > max_len:
                start = np.random.randint(0, audio_arr.shape[0] - max_len + 1)
                audio_arr = audio_arr[start : start + max_len]
            batch_arrays.append(audio_arr)

        audio, lengths = _pad_batch(batch_arrays, max_len)
        yield AudioBatch(audio=audio, lengths=lengths)


def create_expressive_en_ja_streaming_dataloader(
    batch_size: int = 4,
    target_sr: int = 16_000,
    max_duration_s: float = 8.0,
) -> Iterator[AudioBatch]:
    """Create an infinite iterator over a mixture of expressive EN/JA datasets.

    Uses HuggingFace datasets in streaming mode. The current mixture includes:
    - openslr/librispeech_asr (English audiobooks)
    - MLCommons/peoples_speech (large-scale English speech) [clean subset]
    - japanese-asr/ja_asr.jsut_basic5000 (Japanese JSUT subset)
    - shunyalabs/japanese-speech-dataset (Japanese speech)
    - joujiboi/japanese-anime-speech (Japanese anime / visual-novel style)
    """

    dataset_specs: List[Dict[str, Any]] = [
        {
            "name": "openslr/librispeech_asr",
            "config": "clean",
            "split": "train.100",
        },
        {
            "name": "MLCommons/peoples_speech",
            "config": "clean",
            "split": "train",
        },
        {
            "name": "japanese-asr/ja_asr.jsut_basic5000",
            "config": None,
            "split": "train",
        },
        {
            "name": "shunyalabs/japanese-speech-dataset",
            "config": None,
            "split": "train",
        },
        {
            "name": "joujiboi/japanese-anime-speech",
            "config": None,
            "split": "train",
        },
    ]

    streams: List[Dict[str, Any]] = []
    for spec in dataset_specs:
        # Try preferred split, fall back to 'test' if only a test split exists.
        try:
            ds = load_dataset(
                spec["name"],
                spec["config"],
                split=spec["split"],
                streaming=True,
            )
        except ValueError as e:
            msg = str(e)
            if "Bad split" in msg and "['test']" in msg:
                # Retry using the test split.
                ds = load_dataset(
                    spec["name"],
                    spec["config"],
                    split="test",
                    streaming=True,
                )
            else:
                raise
        streams.append(
            {
                "spec": spec,
                "dataset": ds,
                "iterator": iter(ds),
            }
        )

    max_len = int(target_sr * max_duration_s)

    while True:
        batch_arrays = []
        for _ in range(batch_size):
            # Randomly choose a dataset stream
            stream = random.choice(streams)
            spec = stream["spec"]
            it = stream["iterator"]

            while True:
                try:
                    example = next(it)
                except StopIteration:
                    # Recreate iterator on exhaustion
                    it = iter(stream["dataset"])
                    stream["iterator"] = it
                    example = next(it)

                # For now we don't filter by language; all selected datasets
                # are either English or Japanese speech.
                break

            # Try the common 'audio' field first, then fall back to any
            # value that looks like a decoded Audio feature
            audio_obj = example.get("audio")
            if audio_obj is None:
                for v in example.values():
                    if isinstance(v, dict) and "array" in v and "sampling_rate" in v:
                        audio_obj = v
                        break
            if audio_obj is None:
                raise KeyError(f"No audio-like field found in example from {spec['name']}")
            audio_arr = np.asarray(audio_obj["array"], dtype=np.float32)
            sr = audio_obj["sampling_rate"]

            if sr != target_sr:
                orig_len = audio_arr.shape[0]
                new_len = int(orig_len * target_sr / sr)
                audio_arr = np.interp(
                    np.linspace(0, orig_len, new_len, endpoint=False),
                    np.arange(orig_len),
                    audio_arr,
                ).astype(np.float32)

            # Randomly crop a segment of length max_len if the sample is longer.
            if audio_arr.shape[0] > max_len:
                start = np.random.randint(0, audio_arr.shape[0] - max_len + 1)
                audio_arr = audio_arr[start : start + max_len]

            batch_arrays.append(audio_arr)

        audio, lengths = _pad_batch(batch_arrays, max_len)
        yield AudioBatch(audio=audio, lengths=lengths)

