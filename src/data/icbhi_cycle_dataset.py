"""
ICBHICycleDataset — Per-cycle dataset for ICBHI2017.

Instead of one sample per recording file (10-90s, single label),
each respiratory cycle becomes a separate training sample with its
own label. This 5-10× multiplies training data and provides cleaner
per-cycle supervision.

Each annotation .txt row: start_time  end_time  crackles(0/1)  wheezes(0/1)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from .official_split import parse_split_file


class ICBHICycleDataset(torch.utils.data.Dataset):
    """ICBHI 2017 dataset where each item is one respiratory cycle.

    Parameters
    ----------
    data_dir : Path
        Root dataset directory containing audio/ subdirectory.
    split_file : Path
        Path to official_split.txt.
    split : str
        "train" or "test".
    transform : callable, optional
        Waveform → spectrogram transform (applied per cycle).
    sample_rate : int
        Target sample rate for time→sample conversion.
    """

    LABEL_MAP = {
        (0, 0): "normal",
        (1, 0): "crackles",
        (0, 1): "wheezes",
        (1, 1): "both",
    }

    def __init__(
        self,
        data_dir: Union[str, Path],
        split_file: Union[str, Path],
        split: str,
        transform: Optional[callable] = None,
        sample_rate: int = 4000,
        fixed_length_seconds: float = 5.0,
        return_filename: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.audio_dir = self.data_dir / "audio"
        if not self.audio_dir.is_dir():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")

        self.split = split.lower()
        if self.split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {self.split!r}")

        self.transform = transform
        self.sample_rate = sample_rate
        self.fixed_length_samples = int(fixed_length_seconds * sample_rate)
        self.return_filename = return_filename

        # Parse split and build cycle index — cycles stay in their file's split
        split_map = parse_split_file(Path(split_file))
        filenames = sorted(
            fname for fname, subset in split_map.items() if subset == self.split
        )

        # Build flat list of (filename, start_sample, end_sample, label_key)
        self.cycles: list[tuple[str, int, int, tuple[int, int]]] = []
        self._filename_cache: dict[str, np.ndarray] = {}

        for fname in filenames:
            annot_path = self.audio_dir / f"{fname}.txt"
            if not annot_path.exists():
                continue
            cycles = self._read_cycles(annot_path)
            for start_s, end_s, crackles, wheezes in cycles:
                self.cycles.append((fname, start_s, end_s, (crackles, wheezes)))

        if not self.cycles:
            raise RuntimeError(
                f"No cycles found for split={self.split!r}"
            )

    def _read_cycles(self, annot_path: Path) -> list[tuple[int, int, int, int]]:
        """Parse annotation file into (start_sample, end_sample, crackles, wheezes)."""
        cycles = []
        with open(annot_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                start_t = float(parts[0])
                end_t = float(parts[1])
                crackles = int(parts[2])
                wheezes = int(parts[3])
                # Convert seconds → samples
                start_s = int(start_t * self.sample_rate)
                end_s = int(end_t * self.sample_rate)
                cycles.append((start_s, end_s, crackles, wheezes))
        return cycles

    def __len__(self) -> int:
        return len(self.cycles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, int]]:
        filename, start_s, end_s, (crackles, wheezes) = self.cycles[idx]

        # Load audio (cached per file to avoid repeated disk reads)
        if filename not in self._filename_cache:
            wav_path = self.audio_dir / f"{filename}.wav"
            import soundfile as sf
            audio_np, _sr = sf.read(str(wav_path), dtype="float32")
            if audio_np.ndim == 1:
                audio_np = audio_np[np.newaxis, :]
            else:
                audio_np = audio_np.T
            self._filename_cache[filename] = audio_np

        audio = self._filename_cache[filename]
        # Extract cycle segment
        end_s = min(end_s, audio.shape[1])
        start_s = max(0, min(start_s, end_s - 1))
        segment = audio[:, start_s:end_s]

        # Pad or truncate to fixed length
        seg_len = segment.shape[1]
        target = self.fixed_length_samples
        if seg_len < target:
            # Repeat the cycle waveform (circular padding) — clinically cleaner
            # than zero-padding, preserves respiratory sound characteristics
            repeats = target // seg_len + 1
            segment = np.tile(segment, (1, repeats))[:, :target]
        elif seg_len > target:
            segment = segment[:, :target]

        waveform = torch.from_numpy(segment.copy())

        # Apply transform (e.g., log-mel spectrogram)
        if self.transform is not None:
            waveform = self.transform(waveform)

        # Multi-label: [crackles, wheezes] as binary flags
        labels = torch.tensor([crackles, wheezes], dtype=torch.float32)

        # Periodically clear cache to avoid memory bloat
        if len(self._filename_cache) > 200:
            self._filename_cache.pop(next(iter(self._filename_cache)))

        if self.return_filename:
            return waveform, labels, filename
        return waveform, labels
