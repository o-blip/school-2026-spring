import os
import re

import numpy as np
import noisereduce as nr
from pydub import AudioSegment
from scipy.signal import find_peaks


def convert_m4a_to_wav(source_dir, target_sr=48000):
    """
    Convert all .m4a files in source_dir to .wav and save them in
    source_dir/wav/.  Files already converted are skipped.

    Works for both Training Data and Experimental Data — both follow the same
    folder structure.  Training files are prefixed with a loading condition
    ({0ftlb,25ftlb,50ftlb}F{flange}A{area}.m4a); Experimental files are 
    (F{flange}A{area}.m4a).

    Returns: list of absolute paths to the wav files (sorted).
    """
    m4a_dir = os.path.join(source_dir, "m4a")
    wav_dir = os.path.join(source_dir, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    m4a_files = sorted(
        f for f in os.listdir(m4a_dir) if f.lower().endswith(".m4a")
    )

    wav_paths = []
    for fname in m4a_files:
        stem = os.path.splitext(fname)[0]
        wav_path = os.path.join(wav_dir, stem + ".wav")
        wav_paths.append(wav_path)

        if os.path.exists(wav_path):
            continue

        audio = AudioSegment.from_file(os.path.join(m4a_dir, fname), format="m4a")
        audio = audio.set_frame_rate(target_sr).set_channels(1)
        audio.export(wav_path, format="wav")
        print(f"  Converted {fname} -> wav")

    print(f"wav files ready: {len(wav_paths)} files in {wav_dir}")
    return wav_paths

def apply_spectral_gate(audio, sr, noise_sample_duration=0.2):
    """
    Apply spectral gating noise reduction to a waveform.

    The first `noise_sample_duration` seconds of the recording are assumed to
    be silence and are used as the noise profile.  This mirrors the approach
    used when recording: a short quiet period at the start of each file before
    any tapping begins.

    Parameters
    ----------
    audio               : 1-D float32 numpy array (samples)
    sr                  : sample rate in Hz
    noise_sample_duration: seconds at the end of the recording to use as noise profile

    Returns
    -------
    clean : 1-D float32 numpy array, same length as audio
    """
    noise_sample = audio[-int(noise_sample_duration * sr):]
    return nr.reduce_noise(y=audio, sr=sr, y_noise=noise_sample)

def split_training_data_into_datasets(wav_paths, clean_audio):
    """
    Split training recordings into 4 datasets by flange number.

    Parses filenames of the form {loading_condition}F{flange}A{area}.wav and
    returns a nested dict:
        { flange_num (int): { loading_condition (str): { area (str): audio (ndarray) } } }

    Each dataset covers 3 loading conditions x 4 areas = 12 recordings,
    yielding 240 tap samples once spliced (20 per file, 80 per loading condition).

    Parameters
    ----------
    wav_paths   : list of paths returned by convert_m4a_to_wav
    clean_audio : list of denoised arrays returned by denoise_all, same order

    Returns
    -------
    datasets : dict  { int: { str: { str: np.ndarray } } }
    """
    pattern = re.compile(r"(\d+ftlb)F(\d)A(\d)\.wav", re.IGNORECASE)
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

def splice_audio(audio, pre_peak=50, post_peak=2400):
    """
    Detect tap peaks in a waveform and return fixed-length windows around each.

    Each window is pre_peak samples before the peak and post_peak samples after,
    giving a splice length of pre_peak + post_peak samples.

    Parameters
    ----------
    audio    : 1-D float32 numpy array
    pre_peak : samples to include before each peak
    post_peak: samples to include after each peak

    Returns
    -------
    splices : ndarray of shape (n_peaks, pre_peak + post_peak)
    """
    threshold = 0.4 * np.max(np.abs(audio))
    peaks, _ = find_peaks(np.abs(audio), height=threshold, distance=20000)
    valid = [p for p in peaks if p >= pre_peak and p + post_peak <= len(audio)]
    return np.array([audio[p - pre_peak: p + post_peak] for p in valid])


def splice_datasets(datasets, expected_per_file=20, pre_peak=50, post_peak=2400):
    """
    Apply splice_audio to every leaf in the nested dataset dict.

    Replaces each full audio array with an ndarray of shape (n_splices, splice_len).
    Prints a warning for any file that yields fewer splices than expected.

    Parameters
    ----------
    datasets         : nested dict { flange: { loading: { area: audio } } }
                       as returned by split_training_data_into_datasets
    expected_per_file: expected number of tap splices per recording (default 20)
    pre_peak         : passed through to splice_audio
    post_peak        : passed through to splice_audio

    Returns
    -------
    spliced : same nested structure with audio replaced by splice arrays
    """
    spliced = {}
    for flange, conditions in sorted(datasets.items()):
        spliced[flange] = {}
        for loading, areas in sorted(conditions.items()):
            spliced[flange][loading] = {}
            for area, audio in sorted(areas.items()):
                s = splice_audio(audio, pre_peak, post_peak)
                if len(s) != expected_per_file:
                    print(f"  WARNING: Flange {flange} {loading} {area}: "
                          f"{len(s)} splices (expected {expected_per_file})")
                spliced[flange][loading][area] = s
        total = sum(len(s) for cond in spliced[flange].values() for s in cond.values())
        print(f"  Dataset {flange}: {total} splices")
    return spliced


def denoise_all(wav_paths, sr, noise_sample_duration=0.2, cache_path=None):
    """
    Load and denoise every wav file in wav_paths.

    If cache_path is given and the file exists, loads from cache instead of
    re-running the denoising pipeline.  Delete the cache file to force a
    full reprocess.

    Returns a list of cleaned 1-D float32 arrays in the same order as wav_paths.
    """
    import librosa

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

