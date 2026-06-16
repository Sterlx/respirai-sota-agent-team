# Skill sources:
#   external/scientific-agent-skills/docs/skills.md (Database Lookup for literature on ICBHI preprocessing)
#   external/AI-research-SKILLs/03-fine-tuning/unsloth/references/llms-full.md (general ML best practices)
#   external/scientific-agent-skills/skills/neurokit2/SKILL.md (biosignal processing patterns)

"""
Audio preprocessing module for ICBHI2017 lung sound classification.

This module implements the preprocessing pipeline used by top-performing
papers on the ICBHI2017 dataset. It includes:

  - Resampling to a target sample rate (default 4 kHz, as used by
    Acharya & Basu 2020, Nguyen & Pernkopf 2022)
  - Butterworth bandpass filtering (50-2000 Hz) to focus on lung sound
    frequencies and remove heart sounds, muscle noise, and aliasing
    (Mesquita et al. 2021, Goncharov et al. 2022)
  - Log-mel spectrogram extraction, which is the dominant feature
    representation for deep learning on ICBHI (Fraiwan et al. 2021,
    Mouzali et al. 2022)
  - Variable-length segmentation: recordings are either padded to a
    fixed length or split into overlapping segments for mini-batch
    training.

All components are implemented in PyTorch / torchaudio for easy GPU
acceleration and integration with a training loop.
"""

import torch
import torchaudio
import torchaudio.functional as F
from torchaudio.transforms import Resample, MelSpectrogram
from typing import List, Tuple, Optional

# ----------------------------------------------------------------------
# (1) RESAMPLING
# ----------------------------------------------------------------------

def resample_waveform(
    waveform: torch.Tensor,
    orig_freq: int,
    target_freq: int = 4000
) -> torch.Tensor:
    """
    Resample a waveform to the target sample rate.

    Many ICBHI papers resample to 4 kHz (Acharya & Basu, 2020;
    Nguyen & Pernkopf, 2022) because lung sounds have negligible
    energy above 2 kHz, and the lower rate reduces computational cost.

    Args:
        waveform (Tensor): (channels, time) or (time,) audio tensor.
        orig_freq (int): Original sample rate (Hz).
        target_freq (int): Desired sample rate (Hz). Default 4000.

    Returns:
        Tensor: Resampled waveform at target_freq.
    """
    if orig_freq == target_freq:
        return waveform

    resampler = Resample(orig_freq, target_freq)
    # Resample expects channels dimension: if waveform is 1D, add channel dim
    input_len = waveform.dim()
    if input_len == 1:
        waveform = waveform.unsqueeze(0)
    resampled = resampler(waveform)
    if input_len == 1:
        resampled = resampled.squeeze(0)
    return resampled


# ----------------------------------------------------------------------
# (2) BANDPASS FILTERING (50-2000 Hz)
# ----------------------------------------------------------------------

def bandpass_filter(
    waveform: torch.Tensor,
    sample_rate: int,
    low_hz: float = 50.0,
    high_hz: float = 2000.0
) -> torch.Tensor:
    """
    Apply a 2nd-order Butterworth bandpass filter.

    The default passband 50-2000 Hz is standard for lung sound analysis
    (Reichert et al. 2020, Mesquita et al. 2021). It attenuates heart
    sounds (< 50 Hz), muscle noise, and aliasing artefacts.

    The filter is implemented as a biquad cascade for efficient GPU
    operation.

    Args:
        waveform (Tensor): (channels, time) or (time,).
        sample_rate (int): Sample rate of the waveform (Hz).
        low_hz (float): Lower cutoff frequency (Hz).
        high_hz (float): Upper cutoff frequency (Hz).

    Returns:
        Tensor: Bandpass-filtered waveform.
    """
    # Compute centre frequency and Q factor for bandpass biquad
    # Q = centre_freq / (high - low)
    centre_freq = (low_hz * high_hz) ** 0.5
    Q = centre_freq / (high_hz - low_hz)

    # torchaudio's band_biquad expects (channels, time) or (time,)
    return F.bandpass_biquad(waveform, sample_rate, centre_freq, Q)


# ----------------------------------------------------------------------
# (3) LOG-MEL SPECTROGRAM
# ----------------------------------------------------------------------

def log_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 128,
    n_mels: int = 64,
    f_min: float = 50.0,
    f_max: float = 2000.0,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute log-mel spectrogram of a time-domain waveform.

    Log-mel spectrograms are the most common feature in recent ICBHI
    SOTA models (Fraiwan et al. 2021, Mouzali et al. 2022). The
    parameters here are chosen to provide high time-frequency resolution
    within the band of interest (50-2000 Hz).

    Args:
        waveform (Tensor): (time,) or (channels, time). If multi-channel,
            the transform is applied per channel.
        sample_rate (int): Sample rate of the waveform.
        n_fft (int): FFT window size. Default 1024.
        hop_length (int): Number of samples between frames. Default 128,
            giving ~8 ms overlap for 4 kHz audio.
        n_mels (int): Number of mel filterbank channels. Default 64.
        f_min (float): Lowest frequency in mel filterbank (Hz).
        f_max (float): Highest frequency in mel filterbank (Hz).
        eps (float): Small value added before log to avoid log(0).

    Returns:
        Tensor: (channels, n_mels, time_frames) or (n_mels, time_frames)
            log-mel spectrogram.
    """
    mel_transform = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=2.0,          # power spectrogram (squared magnitude)
        mel_scale='htk',    # standard mel scale
        norm='slaney',      # Slaney normalisation
        window_fn=torch.hamming_window,
    )
    # Compute mel spectrogram: (..., n_mels, time)
    mel_spec = mel_transform(waveform)
    # Force shape to (1, n_mels, time) regardless of torchaudio version
    if mel_spec.dim() == 4:
        mel_spec = mel_spec.squeeze(0)  # (1,1,n_mels,time) → (1,n_mels,time)
    if mel_spec.dim() == 2:
        mel_spec = mel_spec.unsqueeze(0)  # (n_mels,time) → (1,n_mels,time)
    # Convert to log scale: log(mel_spec + eps)
    return torch.log(mel_spec + eps)


# ----------------------------------------------------------------------
# (4) VARIABLE-LENGTH SEGMENTATION
# ----------------------------------------------------------------------

def segment_waveform(
    waveform: torch.Tensor,
    target_length: int,
    hop_length: int,
    pad_mode: str = 'constant',
    pad_value: float = 0.0
) -> torch.Tensor:
    """
    Convert a variable-length waveform into fixed-length segments.

    If the waveform is shorter than `target_length`, it is padded
    (on the right) to the target length, producing a single segment.
    If longer, a sliding window of length `target_length` is applied
    with the given `hop_length`. The returned tensor has shape
    (num_segments, target_length) or (num_segments, channels, target_length)
    if multi-channel.

    Args:
        waveform (Tensor): (time,) or (channels, time).
        target_length (int): Desired segment length in samples.
        hop_length (int): Step size in samples for overlapping segments.
        pad_mode (str): Padding mode (see torch.nn.functional.pad).
        pad_value (float): Fill value for constant padding.

    Returns:
        Tensor: (num_segments, target_length) or
                (num_segments, channels, target_length).
    """
    # Ensure waveform has shape (channels, time)
    expand_dim = False
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # (1, time)
        expand_dim = True

    channels, length = waveform.shape

    if length < target_length:
        # Pad to target_length
        pad_len = target_length - length
        waveform_padded = torch.nn.functional.pad(
            waveform,
            (0, pad_len),       # pad last dimension (time)
            mode=pad_mode,
            value=pad_value
        )
        segments = waveform_padded.unsqueeze(0)  # (1, channels, target_length)
        if expand_dim:
            segments = segments.squeeze(1)       # (1, target_length)
        return segments

    # Sliding window extraction
    num_segments = (length - target_length) // hop_length + 1
    segments = []
    for i in range(num_segments):
        start = i * hop_length
        segment = waveform[:, start:start + target_length]
        segments.append(segment)
    segments = torch.stack(segments, dim=0)  # (num_segments, channels, target_length)
    if expand_dim:
        segments = segments.squeeze(-2)      # (num_segments, target_length)
    return segments


# ----------------------------------------------------------------------
# (5) FULL PREPROCESSING PIPELINE
# ----------------------------------------------------------------------

class LungSoundPreprocessor:
    """
    High-level preprocessing pipeline for ICBHI2017 lung sounds.

    Typical usage:

        preprocessor = LungSoundPreprocessor(orig_sample_rate)
        # Raw waveform -> log-mel spectrogram patches
        specs = preprocessor(waveform)   # returns (num_segments, n_mels, time_frames)

    The pipeline:
        1. Resample to target_sample_rate (default 4 kHz).
        2. Bandpass filter (50-2000 Hz).
        3. Segment/pad to segment_length_samples.
        4. Compute log-mel spectrogram for each segment.

    All parameters are configurable to permit rapid experimentation.
    """

    def __init__(
        self,
        orig_sample_rate: int,
        target_sample_rate: int = 4000,
        low_cutoff_hz: float = 50.0,
        high_cutoff_hz: float = 2000.0,
        segment_length_seconds: float = 2.0,
        segment_hop_seconds: float = 1.0,
        n_fft: int = 1024,
        hop_length: int = 128,
        n_mels: int = 64,
        f_min: float = 50.0,
        f_max: float = 2000.0,
        eps: float = 1e-6
    ):
        """
        Args:
            orig_sample_rate (int): Sample rate of input recordings (Hz).
            target_sample_rate (int): Desired sample rate after resampling (Hz).
            low_cutoff_hz (float): Low cutoff of bandpass filter (Hz).
            high_cutoff_hz (float): High cutoff of bandpass filter (Hz).
            segment_length_seconds (float): Duration of each segment (s).
            segment_hop_seconds (float): Hop length for overlapping segments (s).
            n_fft (int): FFT window size for mel spectrogram.
            hop_length (int): Hop length (in samples) for STFT.
            n_mels (int): Number of mel filter banks.
            f_min (float): Minimum frequency for mel filterbank (Hz).
            f_max (float): Maximum frequency for mel filterbank (Hz).
            eps (float): Value added before log() for numerical stability.
        """
        self.orig_sample_rate = orig_sample_rate
        self.target_sample_rate = target_sample_rate
        self.low_cutoff = low_cutoff_hz
        self.high_cutoff = high_cutoff_hz

        # Convert seconds to samples
        self.seg_len_samples = int(segment_length_seconds * target_sample_rate)
        self.seg_hop_samples = int(segment_hop_seconds * target_sample_rate)

        # Spectrogram parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.eps = eps

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply full preprocessing pipeline to a single-channel waveform.

        Args:
            waveform (Tensor): (time,) raw waveform at orig_sample_rate.

        Returns:
            Tensor: (num_segments, n_mels, time_frames) log-mel
                spectrogram patches, ready for model input.
        """
        # (1) Resample
        resampled = resample_waveform(
            waveform,
            self.orig_sample_rate,
            self.target_sample_rate
        )

        # (2) Bandpass filter
        filtered = bandpass_filter(
            resampled,
            self.target_sample_rate,
            self.low_cutoff,
            self.high_cutoff
        )

        # (3) Segment
        segments = segment_waveform(
            filtered,
            self.seg_len_samples,
            self.seg_hop_samples
        )  # (num_segments, time)

        # (4) Log-mel spectrogram per segment
        specs = []
        for seg in segments:
            spec = log_mel_spectrogram(
                seg,
                sample_rate=self.target_sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                f_min=self.f_min,
                f_max=self.f_max,
                eps=self.eps
            )  # (n_mels, time_frames)
            specs.append(spec)
        return torch.stack(specs, dim=0)  # (num_segments, n_mels, time_frames)


# ----------------------------------------------------------------------
# (6) OPTIONAL BATCH INTERFACE (for DataLoader)
# ----------------------------------------------------------------------

def preprocess_batch(
    waveforms: List[torch.Tensor],
    preprocessor: LungSoundPreprocessor,
    max_segments_per_recording: Optional[int] = None
) -> Tuple[torch.Tensor, List[int]]:
    """
    Preprocess a batch of variable-length recordings.

    Each recording is processed by the given preprocessor, producing
    a variable number of spectrogram patches. These are concatenated
    along the batch dimension, and a list of recording indices is
    returned so that each patch can be traced back to its original
    recording (useful for sequence-level or aggregated predictions).

    Args:
        waveforms (List[Tensor]): Each tensor is a raw waveform (time,).
        preprocessor (LungSoundPreprocessor): Configured pipeline.
        max_segments_per_recording (int, optional): If given, only the
            first `max_segments_per_recording` patches are kept per
            recording (to limit memory in long recordings).

    Returns:
        specs (Tensor): (total_patches, n_mels, time_frames) stacked.
        rec_ids (List[int]): Recording index for each patch.
    """
    all_specs = []
    rec_ids = []
    for rec_idx, wav in enumerate(waveforms):
        specs = preprocessor(wav)  # (num_seg, n_mels, time)
        if max_segments_per_recording:
            specs = specs[:max_segments_per_recording]
        all_specs.append(specs)
        rec_ids.extend([rec_idx] * specs.size(0))

    if all_specs:
        return torch.cat(all_specs, dim=0), rec_ids
    else:
        # No segments (should not happen if segment always returns at least 1)
        return torch.empty(0, preprocessor.n_mels, 1), []
