# Skill source: external/AI-research-SKILLs/03-fine-tuning/axolotl/references/dataset-formats.md
# - Guidance on building clean, reusable dataset classes.
# Skill source: external/AI-research-SKILLs/03-fine-tuning/unsloth/references/llms-full.md
# - General coding best practices for production pipelines.
# Skill source: external/AI-research-SKILLs/03-fine-tuning/unsloth/references/llms-txt.md
# - Clear, well-documented Python examples.

"""
icbhi_dataset.py — PyTorch Dataset for the ICBHI2017 Respiratory Sound Database
================================================================================

Provides ``ICBHIDataset``, a :class:`torch.utils.data.Dataset` subclass that:

1. Reads the official patient-wise train/test split from *split_file*.
2. Filters recordings belonging to the requested split (``"train"`` or
   ``"test"``).
3. Loads the corresponding ``.wav`` file with :func:`torchaudio.load`.
4. Parses the annotation ``.txt`` file to determine the lung-sound label:
   ``"normal"``, ``"crackles"``, ``"wheezes"``, or ``"both"``.
5. Returns a tuple ``(waveform, labels_dict)`` where *labels_dict* is a
   one-hot-like dictionary with keys ``{"crackles", "wheezes", "both",
   "normal"}``, exactly one flag set to ``1``.

Usage example::

    from pathlib import Path
    from src.data.icbhi_dataset import ICBHIDataset

    ds = ICBHIDataset(
        data_dir=Path("data"),
        split_file=Path("data/official_split.txt"),
        split="train",
    )
    waveform, labels = ds[0]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torchaudio

# Reuse the verified helpers from the official split script
from .official_split import determine_label, parse_split_file


class ICBHIDataset(torch.utils.data.Dataset):
    """ICBHI 2017 lung sound dataset that respects the official 60/40 split.

    Parameters
    ----------
    data_dir : pathlib.Path
        Root directory of the ICBHI dataset.  It must contain an ``audio/``
        sub-directory with the ``.wav`` and ``.txt`` files.
    split_file : pathlib.Path
        Path to ``official_split.txt`` (tab-separated: filename, train|test).
    split : str
        Which subset to load: ``"train"`` or ``"test"``.
    transform : callable, optional
        A function/transform that takes in a waveform tensor and returns a
        transformed version.  Useful for data augmentation (applied before
        returning).
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split_file: Union[str, Path],
        split: str,
        transform: Optional[callable] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split_file = Path(split_file)
        self.split = split.lower()
        if self.split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {self.split!r}")
        self.transform = transform

        self.audio_dir = self.data_dir / "audio"
        if not self.audio_dir.is_dir():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")

        # Parse the official split and keep only recordings for our subset
        split_map = parse_split_file(self.split_file)
        self.filenames = sorted(
            fname for fname, subset in split_map.items() if subset == self.split
        )

        if not self.filenames:
            raise RuntimeError(
                f"No recordings found for split={self.split!r} in {self.split_file}"
            )

    def __len__(self) -> int:
        """Return the total number of recordings in the selected split."""
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Dict[str, int]]:
        """Return the waveform and its label dictionary for index `idx`.

        Returns
        -------
        waveform : torch.Tensor
            Shape ``(1, num_samples)`` — mono waveform.
        labels_dict : dict
            Dictionary with keys ``{"crackles", "wheezes", "both", "normal"}``
            and values ``0`` or ``1``; exactly one key is ``1``.
        """
        filename = self.filenames[idx]

        # --- Load audio ------------------------------------------------------
        wav_path = self.audio_dir / f"{filename}.wav"
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file missing: {wav_path}")
        # torchaudio.load returns (channels, samples)
        waveform, sample_rate = torchaudio.load(wav_path)

        # Convert to mono by averaging channels (lung sounds are usually mono)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Apply optional transform (e.g., spectrogram, augmentation)
        if self.transform is not None:
            waveform = self.transform(waveform)

        # --- Determine label from annotation --------------------------------
        annot_path = self.audio_dir / f"{filename}.txt"
        if not annot_path.exists():
            raise FileNotFoundError(f"Annotation file missing: {annot_path}")
        label_str = determine_label(annot_path)  # "normal", "crackles", ...

        # Build one-hot-like dictionary (exactly one class active)
        labels_dict = {
            "normal": 0,
            "crackles": 0,
            "wheezes": 0,
            "both": 0,
        }
        labels_dict[label_str] = 1

        return waveform, labels_dict


# ---------------------------------------------------------------------------
# Quick sanity test (run this file directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Try to load a small subset to verify the class works.
    # Adjust the paths or set the ICBHI2017_ROOT environment variable as
    # needed.  This block is only for manual testing; it is not used during
    # training.
    try:
        from .official_split import _resolve_data_root

        root = _resolve_data_root()
    except ImportError:
        # Fallback when running as __main__ without full package context
        root = Path(os.environ.get("ICBHI2017_ROOT", "data"))
        if not root.exists():
            print("Set ICBHI2017_ROOT or place the dataset at ./data")
            sys.exit(1)

    split_path = root / "official_split.txt"
    if not split_path.exists():
        print(f"No official_split.txt found in {root}; aborting test.")
        sys.exit(1)

    dataset = ICBHIDataset(data_dir=root, split_file=split_path, split="train")
    print(f"Loaded {len(dataset)} training recordings.")
    waveform, labels = dataset[0]
    print(f"Waveform shape: {waveform.shape}")
    print(f"Labels: {labels}")
