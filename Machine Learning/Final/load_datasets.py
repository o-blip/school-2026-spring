import os
import re
from typing import Optional

import numpy as np
import noisereduce as nr
from pydub import AudioSegment
from scipy.signal import find_peaks
import librosa


def convert_m4a_to_wav(source_dir: str, target_sr: int = 48000) -> list[str]:
    """Convert all .m4a files in source_dir to .wav and save them in source_dir/wav/.

    Files already converted are skipped. Training files are named
    {loading}F{flange}A{area}.m4a; experimental files are F{flange}A{area}.m4a.

    Returns a sorted list of absolute paths to the .wav files.
    """
    m4a_dir = os.path.join(source_dir, "m4a")
    wav_dir = os.path.join(source_dir, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    m4a_files = sorted(f for f in os.listdir(m4a_dir) if f.lower().endswith(".m4a"))

    wav_paths = []
    for fname in m4a_files:
        stem     = os.path.splitext(fname)[0]
        wav_path = os.path.join(wav_dir, stem + ".wav")
        wav_paths.append(wav_path)

        if os.path.exists(wav_path):
            continue  # already converted

        audio = AudioSegment.from_file(os.path.join(m4a_dir, fname), format="m4a")
        audio = audio.set_frame_rate(target_sr).set_channels(1)
        audio.export(wav_path, format="wav")
        print(f"  Converted {fname} -> wav")

    print(f"wav files ready: {len(wav_paths)} files in {wav_dir}")
    return wav_paths


def apply_spectral_gate(audio: np.ndarray, sr: int, noise_sample_duration: float = 0.2) -> np.ndarray:
    """Apply spectral gating noise reduction to a waveform.

    The last `noise_sample_duration` seconds of the recording are assumed to be
    ambient noise and are used as the noise profile for the gate. This mirrors
    the recording protocol where a short quiet period follows the last tap.

    Parameters
    ----------
    audio                : 1-D float32 array (samples)
    sr                   : sample rate in Hz
    noise_sample_duration: seconds at the end to use as noise profile

    Returns
    -------
    clean : 1-D float32 array, same length as audio
    """
    noise_sample = audio[-int(noise_sample_duration * sr):]
    return nr.reduce_noise(y=audio, sr=sr, y_noise=noise_sample)


def split_training_data_into_datasets(
    wav_paths: list[str],
    clean_audio: list[np.ndarray],
) -> dict:
    """Group cleaned recordings into a nested dict keyed by flange number.

    Parses filenames of the form {loading_condition}F{flange}A{area}.wav and
    returns:
        { flange (int): { loading (str): { area (str): audio (ndarray) } } }

    Each dataset (one flange) contains 3 loading conditions × 4 areas = 12 recordings,
    yielding ~240 tap samples after splicing (~20 taps per file).

    Parameters
    ----------
    wav_paths   : list of .wav paths from convert_m4a_to_wav
    clean_audio : list of denoised arrays from denoise_all, same order as wav_paths

    Returns
    -------
    datasets : dict  { int: { str: { str: np.ndarray } } }
    """
    pattern  = re.compile(r"(\d+ftlb)F(\d)A(\d)\.wav", re.IGNORECASE)
    datasets = {}

    for path, audio in zip(wav_paths, clean_audio):
        fname = os.path.basename(path)
        match = pattern.match(fname)
        if not match:
            print(f"  WARNING: could not parse filename {fname}, skipping")
            continue

        loading = match.group(1)
        flange  = int(match.group(2))
        area    = f"A{match.group(3)}"

        datasets.setdefault(flange, {}).setdefault(loading, {})[area] = audio

    for flange, conditions in sorted(datasets.items()):
        n_files = sum(len(areas) for areas in conditions.values())
        print(f"  Dataset {flange} (Flange {flange}): {len(conditions)} loading conditions, {n_files} files")

    return datasets


def zscore_normalize(splice: np.ndarray) -> np.ndarray:
    """Shift and scale splice to zero mean and unit variance. Returns unchanged if std is zero."""
    std = np.std(splice)
    return splice if std == 0 else (splice - np.mean(splice)) / std


def splice_audio(audio: np.ndarray, pre_peak: int = 50, post_peak: int = 2400) -> np.ndarray:
    """Detect tap peaks in a waveform and extract fixed-length windows around each.

    Peak detection uses a threshold of 40% of the max absolute amplitude with a
    minimum separation of 20000 samples (~0.4 s at 48 kHz) to avoid double-counting
    a single tap. Each window spans pre_peak samples before and post_peak samples
    after the peak (total: 2450 samples by default).

    Parameters
    ----------
    audio     : 1-D float32 array
    pre_peak  : samples to include before the peak
    post_peak : samples to include after the peak

    Returns
    -------
    splices : (n_peaks, pre_peak + post_peak) float32 array
    """
    threshold = 0.4 * np.max(np.abs(audio))
    peaks, _  = find_peaks(np.abs(audio), height=threshold, distance=20000)
    valid     = [p for p in peaks if p >= pre_peak and p + post_peak <= len(audio)]
    return np.array([audio[p - pre_peak: p + post_peak] for p in valid])


def splice_datasets(
    datasets: dict,
    pre_peak: int = 50,
    post_peak: int = 2400,
) -> dict:
    """Apply splice_audio to every recording in the nested dataset dict.

    Replaces each full audio array with an ndarray of shape (n_splices, splice_len).
    Each recording yields approximately 20 taps (exact count varies). Each splice
    is z-score normalized to zero mean and unit variance.

    Parameters
    ----------
    datasets  : nested dict { flange: { loading: { area: audio } } }
    pre_peak  : passed through to splice_audio
    post_peak : passed through to splice_audio

    Returns
    -------
    spliced : same nested structure with audio arrays replaced by splice arrays
    """
    spliced = {}
    for flange, conditions in sorted(datasets.items()):
        spliced[flange] = {}
        for loading, areas in sorted(conditions.items()):
            spliced[flange][loading] = {}
            for area, audio in sorted(areas.items()):
                s = splice_audio(audio, pre_peak, post_peak)
                s = np.array([zscore_normalize(splice) for splice in s])
                spliced[flange][loading][area] = s
        total = sum(len(s) for cond in spliced[flange].values() for s in cond.values())
        print(f"  Dataset {flange}: {total} splices")
    return spliced


def denoise_all(
    wav_paths: list[str],
    sr: int,
    noise_sample_duration: float = 0.2,
    cache_path: Optional[str] = None,
) -> list[np.ndarray]:
    """Load and denoise every wav file in wav_paths via spectral gating.

    If cache_path is given and exists, loads pre-computed results from disk instead
    of re-running the denoiser. Delete the cache file to force a full reprocess.

    Returns a list of cleaned 1-D float32 arrays in the same order as wav_paths.
    """
    stems = [os.path.splitext(os.path.basename(p))[0] for p in wav_paths]

    if cache_path and os.path.exists(cache_path):
        print(f"Loading denoised audio from cache ({cache_path})...")
        data = np.load(cache_path)
        return [data[s] for s in stems]

    clean_list = []
    for path in wav_paths:
        audio, _ = librosa.load(path, sr=sr, mono=True)
        clean_list.append(apply_spectral_gate(audio, sr, noise_sample_duration))
        print(f"  Denoised {os.path.basename(path)}")

    if cache_path:
        np.savez(cache_path, **{s: a for s, a in zip(stems, clean_list)})
        print(f"Saved denoised audio to cache ({cache_path})")

    return clean_list