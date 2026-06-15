"""
src/audio/augmentation.py

Augmentation pipeline for lung sound classification.
Implements clinically-safe audio transformations including:
  - SpecAugment (time & frequency masking) on mel spectrograms
  - Time stretching via phase vocoder (STFT + TimeStretch + iSTFT)
  - Pitch shifting using torchaudio
  - Background noise injection (white Gaussian or custom noise file)
  - Composable pipeline with per-class configuration

All transforms are designed to preserve diagnostically relevant features
while improving model robustness. Clinical safety constraints are baked
into the parameter ranges (e.g., maximum pitch shift, max mask lengths).

Author: RespirAI Augmentation Agent
Date: 2026-05-26
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F_audio
import torchaudio.transforms as T


# ==============================================================================
# Clinical safety constraints for each augmentation
# ==============================================================================

# Maximum pitch shift in semitones. Larger shifts can alter crackle/wheeze
# characteristics beyond clinical plausibility.
MAX_PITCH_SHIFT_SEMITONES = 3

# Maximum time stretch ratio (new_rate / orig_rate).
# Ratio > 1.1 or < 0.9 can distort respiratory cycle timing.
MAX_TIME_STRETCH_RATIO = 0.25  # ±25% stretch factor

# Maximum masking fractions for SpecAugment on mel spectrograms.
# Too much masking can remove entire respiratory events.
MAX_TIME_MASK_FRACTION = 0.15  # max 15% of time axis
MAX_FREQ_MASK_FRACTION = 0.1   # max 10% of frequency bins

# Background noise SNR range (in dB). Lower SNR -> stronger noise.
NOISE_SNR_RANGE_DB = (10, 30)  # 10 dB (noisy) to 30 dB (subtle)


# ==============================================================================
# Waveform-level augmentations (applied before spectrogram)
# ==============================================================================

class TimeStretchWaveform(nn.Module):
    """
    Apply time stretching to a waveform using STFT + TimeStretch + iSTFT.

    Clinical safety:
      - stretch_factor clamped to ±MAX_TIME_STRETCH_RATIO to preserve
        respiratory cycle periodicity.
      - Recommended range: 0.9–1.1 for typical training augmentation.
    """

    def __init__(self, stretch_factor: float = None, min_factor: float = 0.9, max_factor: float = 1.1):
        """
        Args:
            stretch_factor: If provided, fixed factor; otherwise random uniform
                            between min_factor and max_factor on each forward pass.
            min_factor: Lower bound for random draw.
            max_factor: Upper bound for random draw.
        """
        super().__init__()
        self.stretch_factor = stretch_factor
        self.min_factor = min_factor
        self.max_factor = max_factor

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (1, T) or (batch, 1, T) single-channel audio.
        Returns:
            Time-stretched waveform with same number of samples.
        """
        if self.stretch_factor is not None:
            factor = self.stretch_factor
        else:
            factor = random.uniform(self.min_factor, self.max_factor)
        factor = max(self.min_factor, min(self.max_factor, factor))

        # Ensure waveform is 2D (channels, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 3:
            # Extract single channel for simplicity
            waveform = waveform.squeeze(1)  # (batch, T)

        # If batch dim present, process each item
        if waveform.dim() == 2:
            orig_shape = waveform.shape
            stretched = []
            for i in range(waveform.shape[0]):
                stretched.append(self._stretch(waveform[i], factor))
            return torch.stack(stretched).squeeze(1) if orig_shape[1] else torch.stack(stretched)
        else:
            return self._stretch(waveform.squeeze(0), factor).unsqueeze(0)

    def _stretch(self, wav: torch.Tensor, factor: float) -> torch.Tensor:
        """
        Perform STFT -> time stretch -> iSTFT on a single waveform (T,).
        """
        # Compute STFT
        spec = torch.stft(
            wav,
            n_fft=512,
            hop_length=256,
            win_length=512,
            window=torch.hann_window(512, device=wav.device),
            return_complex=True,
        )
        # Apply time stretch
        stretched_spec = torchaudio.transforms.TimeStretch(
            hop_length=256, n_freq=spec.size(1)
        ).to(wav.device)(spec, factor)
        # Inverse STFT back to waveform
        stretched_wav = torch.istft(
            stretched_spec,
            n_fft=512,
            hop_length=256,
            win_length=512,
            window=torch.hann_window(512, device=wav.device),
            length=wav.shape[-1],  # keep original length
        )
        return stretched_wav


class PitchShiftWaveform(nn.Module):
    """
    Shift the pitch of the waveform by a random number of semitones.

    Clinical safety:
      - n_steps limited to ±MAX_PITCH_SHIFT_SEMITONES.
      - Using torchaudio's robust phase vocoder algorithm.
    """

    def __init__(self, sample_rate: int, n_steps: float = None, min_steps: int = -MAX_PITCH_SHIFT_SEMITONES,
                 max_steps: int = MAX_PITCH_SHIFT_SEMITONES):
        """
        Args:
            sample_rate: Audio sample rate in Hz.
            n_steps: Fixed semitone shift; if None, uniform random between min_steps and max_steps.
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.n_steps = n_steps
        self.min_steps = min_steps
        self.max_steps = max_steps

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (1, T) or (batch, 1, T).
        Returns:
            Pitch-shifted waveform.
        """
        if self.n_steps is not None:
            steps = self.n_steps
        else:
            steps = random.randint(self.min_steps, self.max_steps)

        # Ensure waveform is 2D (1, T)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 3:
            waveform = waveform.squeeze(1)  # (batch, T)

        # If batch present, process each
        if waveform.dim() == 2:
            shifted = []
            for i in range(waveform.shape[0]):
                shifted.append(
                    F_audio.pitch_shift(waveform[i], self.sample_rate, steps).unsqueeze(0)
                )
            return torch.cat(shifted, dim=0).squeeze(1) if waveform.shape[0] == 1 else torch.cat(shifted, dim=0)
        else:
            return F_audio.pitch_shift(waveform.squeeze(0), self.sample_rate, steps).unsqueeze(0)


class BackgroundNoiseInjector(nn.Module):
    """
    Add background noise to a waveform. Noise can be white Gaussian or loaded from a file.

    Clinical safety:
      - SNR range clamped to [10, 30] dB to prevent overwhelming the lung sounds.
      - Noise file, if provided, should be ambient/hospital noise (not speech/music).
    """

    def __init__(self, sample_rate: int = 16000, noise_file: Optional[str] = None,
                 snr_db: Optional[float] = None, min_snr: float = NOISE_SNR_RANGE_DB[0],
                 max_snr: float = NOISE_SNR_RANGE_DB[1]):
        """
        Args:
            sample_rate: Audio sample rate (Hz).
            noise_file: Path to a .wav noise file. If None, white noise is used.
            snr_db: Fixed SNR; if None, uniform random between min_snr and max_snr.
            min_snr, max_snr: SNR range in dB.
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.noise_file = noise_file
        self.snr_db = snr_db
        self.min_snr = min_snr
        self.max_snr = max_snr
        # Pre-load noise file if provided
        if noise_file is not None:
            noise_wav, noise_sr = torchaudio.load(noise_file)
            if noise_sr != sample_rate:
                resampler = T.Resample(orig_freq=noise_sr, new_freq=sample_rate)
                noise_wav = resampler(noise_wav)
            # Keep a buffer of noise; will be trimmed/looped as needed during forward
            self.noise_buffer = noise_wav.squeeze(0)  # (noise_len,)
        else:
            self.noise_buffer = None

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (1, T) or (batch, 1, T).
        Returns:
            Noisy waveform with same shape.
        """
        if self.snr_db is not None:
            snr = self.snr_db
        else:
            snr = random.uniform(self.min_snr, self.max_snr)
        snr = max(self.min_snr, min(self.max_snr, snr))

        # Ensure waveform is 2D (1, T)
        orig_dim = waveform.dim()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 3:
            waveform = waveform.squeeze(1)  # (batch, T)

        if waveform.dim() == 2:
            noisy = []
            for i in range(waveform.shape[0]):
                noisy.append(self._add_noise(waveform[i], snr).unsqueeze(0))
            result = torch.cat(noisy, dim=0)
        else:
            result = self._add_noise(waveform.squeeze(0), snr).unsqueeze(0)

        # Restore original dimension if needed
        if orig_dim == 1:
            result = result.squeeze(0)
        return result

    def _add_noise(self, clean: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        Combine clean signal with noise at specified SNR.
        """
        if self.noise_buffer is not None:
            # use preloaded noise file
            noise_chunk = self._get_noise_chunk(clean.shape[0])
            noise = noise_chunk.to(clean.device)
        else:
            noise = torch.randn_like(clean)

        # Compute RMS
        rms_signal = clean.pow(2).mean().sqrt()
        rms_noise = noise.pow(2).mean().sqrt()
        scaling = rms_signal / (rms_noise * (10 ** (snr_db / 20)))
        return clean + noise * scaling

    def _get_noise_chunk(self, length: int) -> torch.Tensor:
        """
        Extract a chunk of noise of desired length, looping if necessary.
        """
        buffer = self.noise_buffer
        if len(buffer) >= length:
            start = random.randint(0, len(buffer) - length)
            chunk = buffer[start:start+length]
        else:
            repeats = length // len(buffer) + 1
            chunk = buffer.repeat(repeats)[:length]
        return chunk


# ==============================================================================
# Spectrogram-level augmentations (applied after mel spectrogram computation)
# ==============================================================================

class SpecAugment(nn.Module):
    """
    Applies frequency masking and time masking to mel spectrograms.

    Clinical safety:
      - Mask count and size bounded by MAX_TIME_MASK_FRACTION / MAX_FREQ_MASK_FRACTION
        to ensure that clinically important patterns are not fully erased.
      - Multiple masks can be applied but total masked area is limited.
    """

    def __init__(self,
                 freq_mask_param: int = 10,
                 time_mask_param: int = 20,
                 n_freq_masks: int = 1,
                 n_time_masks: int = 2,
                 max_time_fraction: float = MAX_TIME_MASK_FRACTION,
                 max_freq_fraction: float = MAX_FREQ_MASK_FRACTION):
        """
        Args:
            freq_mask_param: Max number of frequency bins to mask in one stripe.
            time_mask_param: Max number of time steps to mask in one stripe.
            n_freq_masks: Number of frequency mask stripes.
            n_time_masks: Number of time mask stripes.
            max_time_fraction: Fraction of time axis maximum mask width.
            max_freq_fraction: Fraction of frequency bins maximum mask width.
        """
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.max_time_fraction = max_time_fraction
        self.max_freq_fraction = max_freq_fraction

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: Mel spectrogram of shape (batch, freq, time) or (freq, time).
        Returns:
            Masked spectrogram.
        """
        # Ensure spec is 3D: (batch, freq, time)
        orig_dim = spec.dim()
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)
        elif spec.dim() > 3:
            raise ValueError(f"Unexpected spectrogram shape: {spec.shape}")

        batch, n_freq, n_time = spec.shape
        # Adjust mask size based on clinical caps
        max_freq_mask = min(self.freq_mask_param, int(self.max_freq_fraction * n_freq))
        max_time_mask = min(self.time_mask_param, int(self.max_time_fraction * n_time))

        # Frequency masking
        for _ in range(self.n_freq_masks):
            mask_len = random.randint(0, max_freq_mask)
            if mask_len == 0:
                continue
            f0 = random.randint(0, n_freq - mask_len)
            spec[:, f0:f0+mask_len, :] = spec.mean()  # fill with mean value

        # Time masking
        for _ in range(self.n_time_masks):
            mask_len = random.randint(0, max_time_mask)
            if mask_len == 0:
                continue
            t0 = random.randint(0, n_time - mask_len)
            spec[:, :, t0:t0+mask_len] = spec.mean()

        if orig_dim == 2:
            spec = spec.squeeze(0)
        return spec


# ==============================================================================
# Per-class augmentation configuration
# ==============================================================================

@dataclass
class AugmentationConfig:
    """
    Configuration for one class's augmentations.
    Fields:
        enable_time_stretch: bool
        time_stretch_min, _max: range
        enable_pitch_shift: bool
        pitch_shift_steps_range: (min_steps, max_steps)
        enable_noise: bool
        noise_snr_range: (min_snr, max_snr) in dB
        noise_file: path to noise .wav (None = white noise)
        enable_specaug: bool
        specaug_freq_mask_param, time_mask_param, n_freq_masks, n_time_masks
    """
    enable_time_stretch: bool = True
    time_stretch_min: float = 0.9
    time_stretch_max: float = 1.1

    enable_pitch_shift: bool = True
    pitch_shift_steps_range: tuple = (-2, 2)

    enable_noise: bool = True
    noise_snr_range: tuple = (10, 25)
    noise_file: Optional[str] = None  # path to noise file

    enable_specaug: bool = True
    specaug_freq_mask_param: int = 8
    specaug_time_mask_param: int = 15
    specaug_n_freq_masks: int = 1
    specaug_n_time_masks: int = 2


# Default configuration per class index (0=normal, 1=crackles, 2=wheezes, 3=both)
# More augmentation for minority classes to combat imbalance.
DEFAULT_CLASS_AUGMENTATION_CONFIG: Dict[int, AugmentationConfig] = {
    0: AugmentationConfig(  # Normal - minimal augmentation
        enable_time_stretch=False,
        enable_pitch_shift=False,
        enable_noise=True,
        noise_snr_range=(20, 30),
        enable_specaug=True,
        specaug_freq_mask_param=4,
        specaug_time_mask_param=10,
        specaug_n_freq_masks=1,
        specaug_n_time_masks=1,
    ),
    1: AugmentationConfig(  # Crackles - moderate augmentation
        enable_time_stretch=True,
        time_stretch_min=0.95,
        time_stretch_max=1.05,
        enable_pitch_shift=True,
        pitch_shift_steps_range=(-1, 1),
        enable_noise=True,
        noise_snr_range=(15, 25),
        enable_specaug=True,
        specaug_freq_mask_param=6,
        specaug_time_mask_param=12,
        specaug_n_freq_masks=2,
        specaug_n_time_masks=2,
    ),
    2: AugmentationConfig(  # Wheezes - moderate augmentation
        enable_time_stretch=True,
        time_stretch_min=0.95,
        time_stretch_max=1.05,
        enable_pitch_shift=True,
        pitch_shift_steps_range=(-1, 1),
        enable_noise=True,
        noise_snr_range=(15, 25),
        enable_specaug=True,
        specaug_freq_mask_param=6,
        specaug_time_mask_param=12,
        specaug_n_freq_masks=2,
        specaug_n_time_masks=2,
    ),
    3: AugmentationConfig(  # Both - strongest augmentation
        enable_time_stretch=True,
        time_stretch_min=0.9,
        time_stretch_max=1.1,
        enable_pitch_shift=True,
        pitch_shift_steps_range=(-2, 2),
        enable_noise=True,
        noise_snr_range=(10, 20),
        enable_specaug=True,
        specaug_freq_mask_param=8,
        specaug_time_mask_param=15,
        specaug_n_freq_masks=2,
        specaug_n_time_masks=3,
    ),
}


# ==============================================================================
# Pipeline builders
# ==============================================================================

def get_waveform_augmentation(config: AugmentationConfig, sample_rate: int) -> nn.Sequential:
    """
    Build a Sequential module that applies waveform-level augmentations
    (time stretch, pitch shift, background noise) according to config.

    Args:
        config: AugmentationConfig for the target class.
        sample_rate: Audio sample rate in Hz.

    Returns:
        nn.Sequential pipeline that takes waveform (1, T) and returns
        augmented waveform (1, T).
    """
    layers = []
    if config.enable_time_stretch:
        layers.append(
            TimeStretchWaveform(min_factor=config.time_stretch_min, max_factor=config.time_stretch_max)
        )
    if config.enable_pitch_shift:
        layers.append(
            PitchShiftWaveform(
                sample_rate=sample_rate,
                min_steps=config.pitch_shift_steps_range[0],
                max_steps=config.pitch_shift_steps_range[1],
            )
        )
    if config.enable_noise:
        layers.append(
            BackgroundNoiseInjector(
                sample_rate=sample_rate,
                noise_file=config.noise_file,
                min_snr=config.noise_snr_range[0],
                max_snr=config.noise_snr_range[1],
            )
        )
    return nn.Sequential(*layers)


def get_spectrogram_augmentation(config: AugmentationConfig) -> SpecAugment:
    """
    Build a SpecAugment module for mel spectrograms.

    Args:
        config: AugmentationConfig for the target class.

    Returns:
        SpecAugment module.
    """
    return SpecAugment(
        freq_mask_param=config.specaug_freq_mask_param,
        time_mask_param=config.specaug_time_mask_param,
        n_freq_masks=config.specaug_n_freq_masks,
        n_time_masks=config.specaug_n_time_masks,
    )


# ==============================================================================
# Convenience function to obtain pipeline for a given class label
# ==============================================================================

def get_augmentation_pipeline_for_class(
    class_index: int,
    sample_rate: int,
    config_overrides: Optional[AugmentationConfig] = None,
) -> Dict[str, nn.Module]:
    """
    Returns both waveform and spectrogram augmentation modules for a class.

    Args:
        class_index: Integer class label {0: normal, 1: crackles, 2: wheezes, 3: both}.
        sample_rate: Audio sample rate in Hz.
        config_overrides: If provided, use this config instead of the default.

    Returns:
        dict with keys 'waveform' and 'spectrogram' containing the respective modules.
    """
    config = config_overrides if config_overrides is not None else DEFAULT_CLASS_AUGMENTATION_CONFIG.get(class_index)
    if config is None:
        raise ValueError(f"No augmentation config for class index {class_index}")

    waveform_pipeline = get_waveform_augmentation(config, sample_rate)
    spec_pipeline = get_spectrogram_augmentation(config)
    return {"waveform": waveform_pipeline, "spectrogram": spec_pipeline}


# ==============================================================================
# Test / demo
# ==============================================================================

if __name__ == "__main__":
    # Example usage with dummy data
    sample_rate = 16000
    duration = 2  # seconds
    waveform = torch.randn(1, sample_rate * duration) * 0.01  # quiet signal

    print("Original waveform shape:", waveform.shape)

    config = AugmentationConfig()
    wf_pipe = get_waveform_augmentation(config, sample_rate)
    aug_wf = wf_pipe(waveform)
    print("After waveform augmentations shape:", aug_wf.shape)

    # Simulate mel spectrogram (dummy)
    mel_spec = torch.randn(1, 64, 100)  # batch, n_mels, time
    spec_pipe = get_spectrogram_augmentation(config)
    aug_spec = spec_pipe(mel_spec)
    print("After SpecAugment shape:", aug_spec.shape)

    # Per-class pipeline
    for cls_idx in [0, 1, 2, 3]:
        print(f"\n--- Class {cls_idx} pipeline ---")
        pipelines = get_augmentation_pipeline_for_class(cls_idx, sample_rate)
        wf_test = pipelines["waveform"](waveform)
        print(f"Waveform augmented; mean={wf_test.mean().item():.4f}")
        spec_test = pipelines["spectrogram"](mel_spec)
        print(f"Spectrogram augmented; mean={spec_test.mean().item():.4f}")
