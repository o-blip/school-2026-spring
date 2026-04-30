import os
import re

import librosa
import numpy as np
import noisereduce as nr
from pydub import AudioSegment
from scipy.signal import find_peaks


def splice_audio(audio, pre_peak=100, post_peak=3000):
    """Detect tap peaks and return fixed-length windows around each: 100 samples before the peak and 3000 samples after the peak."""
    threshold = 0.2 * np.max(np.abs(audio))
    peaks, _ = find_peaks(np.abs(audio), height=threshold, distance=20000)
    valid = [p for p in peaks if p >= pre_peak and p + post_peak <= len(audio)]
    return np.array([audio[p - pre_peak : p + post_peak] for p in valid])


def is_double_tap(splice, pre_peak=100, threshold_ratio=0.6, min_separation=500):
    """
    Return True if the splice contains a second tap impact.

    After skipping `min_separation` samples past the main peak (to clear the
    natural impact decay), checks whether any secondary peak exceeds
    `threshold_ratio` × main peak amplitude.

    Parameters
    ----------
    splice          : 1-D array from splice_audio
    pre_peak        : samples before the main peak (matches splice_audio default)
    threshold_ratio : secondary peak must be at least this fraction of the main peak
    min_separation  : samples after the main peak to skip before searching
                      (~10 ms at 48 kHz with default 500)
    """
    main_amp = np.max(np.abs(splice))
    search_start = pre_peak + min_separation
    if search_start >= len(splice):
        return False
    secondary_peaks, _ = find_peaks(
        np.abs(splice[search_start:]),
        height=threshold_ratio * main_amp,
        distance=200,
    )
    return len(secondary_peaks) > 0


def filter_double_taps(splices, context=""):
    """Remove double-tap splices, printing a warning for each one dropped."""
    clean = []
    for i, s in enumerate(splices):
        if is_double_tap(s):
            print(f"  WARNING: {context} hit {i+1}: double tap detected — skipped")
        else:
            clean.append(s)
    return np.array(clean) if clean else np.empty((0, splices.shape[1]))


def rms_normalize(s):
    """Scale splice so its RMS amplitude equals 1."""
    rms = np.sqrt(np.mean(s**2))
    return s / rms if rms > 0 else s


def load_datasets(
    ml_recordings_dir, wav_dir, sr, noise_sample_duration, expected_hits={"g": 10, "b": 24}
):
    """
    Convert m4a → wav, apply spectral-gating noise reduction, splice taps, RMS-normalize.

    Returns
    -------
    healthy_hits   : np.ndarray  shape (N, splice_len)
    unhealthy_hits : np.ndarray  shape (M, splice_len)
    noise_plot_data: dict  {raw, clean, sr, ds_num, num, label, noise_sample_duration}
                     First sample encountered — used for the noise-comparison plot.
    """
    os.makedirs(wav_dir, exist_ok=True)
    datasets = {}

    # ── Convert m4a → wav ────────────────────────────────────────────────────
    for dataset_folder in sorted(os.listdir(ml_recordings_dir)):
        dataset_path = os.path.join(ml_recordings_dir, dataset_folder)
        if not os.path.isdir(dataset_path):
            continue
        ds_match = re.search(r"(\d+)", dataset_folder)
        if not ds_match:
            continue
        ds_num = int(ds_match.group(1))
        wav_ds_dir = os.path.join(wav_dir, f"dataset_{ds_num}")
        os.makedirs(wav_ds_dir, exist_ok=True)

        recordings = {}
        for fname in sorted(
            os.listdir(dataset_path), key=lambda x: int(re.search(r"\d+", x).group())
        ):
            match = re.match(r"s_(\d+)_(g|b)\.m4a", fname)
            if match:
                num, label = int(match.group(1)), match.group(2)
                m4a_path = os.path.join(dataset_path, fname)
                wav_path = os.path.join(wav_ds_dir, f"s_{num}_{label}.wav")
                AudioSegment.from_file(m4a_path, format="m4a").export(
                    wav_path, format="wav"
                )
                recordings[num] = {"label": label, "wav_path": wav_path}

        datasets[ds_num] = recordings
        n_good = sum(1 for v in recordings.values() if v["label"] == "g")
        n_bad = sum(1 for v in recordings.values() if v["label"] == "b")
        print(
            f"Dataset {ds_num}: {len(recordings)} files  (good={n_good}, bad={n_bad})"
        )

    print(f"\nLoaded {len(datasets)} datasets")

    # ── Spectral-gating noise reduction ──────────────────────────────────────
    noise_plot_data = None
    for ds_num, recordings in datasets.items():
        for num, info in recordings.items():
            audio, _ = librosa.load(info["wav_path"], sr=sr, mono=True)
            noise_sample = audio[: int(noise_sample_duration * sr)]
            clean = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_sample)
            datasets[ds_num][num]["audio"] = clean
            if noise_plot_data is None:
                noise_plot_data = {
                    "raw": audio,
                    "clean": clean,
                    "sr": sr,
                    "ds_num": ds_num,
                    "num": num,
                    "label": info["label"],
                    "noise_sample_duration": noise_sample_duration,
                }
        print(f"  Dataset {ds_num}: denoised {len(recordings)} recordings")

    # ── Splice taps and RMS-normalize ────────────────────────────────────────
    healthy_hits, unhealthy_hits = [], []
    for ds_num, recordings in datasets.items():
        for num, info in recordings.items():
            splices = splice_audio(info["audio"], sr)
            if splices.size == 0:
                print(
                    f"  WARNING: Dataset {ds_num} s_{num} ({info['label']}): no peaks found"
                )
                continue
            if len(splices) < expected_hits[info["label"]]:
                print(
                    f"  WARNING: Dataset {ds_num} s_{num} ({info['label']}): "
                    f"{len(splices)} hits (expected {expected_hits[info['label']]})"
                )
            # TODO: re-enable filter_double_taps once detection is tuned
            (healthy_hits if info["label"] == "g" else unhealthy_hits).append(splices)

    healthy_hits = np.array(
        [rms_normalize(s) for s in np.concatenate(healthy_hits, axis=0)]
    )
    unhealthy_hits = np.array(
        [rms_normalize(s) for s in np.concatenate(unhealthy_hits, axis=0)]
    )

    print(f"Healthy splices:   {healthy_hits.shape[0]}")
    print(f"Unhealthy splices: {unhealthy_hits.shape[0]}")
    print(
        f"Splice length:     {healthy_hits.shape[1]} samples "
        f"= {healthy_hits.shape[1] / sr * 1000:.1f} ms"
    )

    return healthy_hits, unhealthy_hits, noise_plot_data


def load_hw2_test_set(hw2_dir, sr, noise_sample_duration, expected_hits={"g": 10, "b": 24}):
    """
    Load, denoise, splice, and RMS-normalize HW2 recordings.

    Returns
    -------
    hw2_healthy   : np.ndarray  shape (N, splice_len)
    hw2_unhealthy : np.ndarray  shape (M, splice_len)
    """
    hw2_healthy, hw2_unhealthy = [], []
    for fname in sorted(
        os.listdir(hw2_dir), key=lambda x: int(re.search(r"\d+", x).group())
    ): # searches the directory for the 
        match = re.match(r"s_(\d+)_(g|b)\.wav", fname)
        if not match:
            continue
        num, label = int(match.group(1)), match.group(2)
        audio, _ = librosa.load(os.path.join(hw2_dir, fname), sr=sr, mono=True)
        noise_sample = audio[: int(noise_sample_duration * sr)]
        audio = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_sample)
        splices = splice_audio(audio, sr)
        if splices.size == 0:
            print(f"  WARNING: hw2 s_{num} ({label}): no peaks found")
            continue
        if len(splices) < expected_hits[label]:
            print(
                f"  WARNING: hw2 s_{num} ({label}): {len(splices)} hits (expected {expected_hits[label]})"
            )
        # TODO: re-enable filter_double_taps once detection is tuned
        splices = np.array([rms_normalize(s) for s in splices])
        (hw2_healthy if label == "g" else hw2_unhealthy).append(splices)

    hw2_healthy = np.concatenate(hw2_healthy, axis=0)
    hw2_unhealthy = np.concatenate(hw2_unhealthy, axis=0)
    print(
        f"HW2 test set: {len(hw2_healthy)} healthy, {len(hw2_unhealthy)} unhealthy splices"
    )
    return hw2_healthy, hw2_unhealthy
