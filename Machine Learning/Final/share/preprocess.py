import os
import numpy as np
from sklearn.model_selection import train_test_split
from load_datasets import convert_m4a_to_wav, denoise_all, split_training_data_into_datasets, splice_datasets
from dataclasses import dataclass

@dataclass
class PreprocessedData:
    # Raw splice arrays (N, splice_len) and corresponding labels
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray        # loading condition: 0=0ftlb, 1=25ftlb, 2=50ftlb
    y_test: np.ndarray
    y_train_flange: np.ndarray # flange number: 1–4
    y_test_flange: np.ndarray
    y_train_area: np.ndarray   # area number: 1–4
    y_test_area: np.ndarray
    datasets: dict             # nested dict { flange: { loading: { area: splices } } }
    experimental_clean: list   # denoised experimental waveforms (not split, used separately)
    datasets_raw: dict  # full-length waveforms before splicing

def preprocess(
    training_dir="Training Data",
    experimental_dir="Experimental Data",
    cache_dir="cache",
    sr=48000,
    noise_sample_duration=0.2,  # seconds at end of each recording used as noise profile
    normalization='rms',         # per-splice normalization: 'rms', 'zscore', or None
    test_size=0.3,
    random_state=42,
):
    """
    Full preprocessing pipeline from raw .m4a recordings to train/test arrays.

    Steps:
      1. Convert .m4a files to .wav (skips files already converted)
      2. Denoise each recording via spectral gating (cached to disk after first run)
      3. Group recordings by flange ID into a nested dataset dict
      4. Detect tap peaks and splice each recording into fixed-length windows
      5. Apply per-splice normalization (rms or zscore)
      6. Flatten all splices into (X, y) arrays and do a 70/30 train/test split

    Returns a PreprocessedData dataclass with train/test arrays and labels.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    training_dir = os.path.join(BASE_DIR, training_dir)
    cache_dir = os.path.join(BASE_DIR, cache_dir)
    experimental_dir = os.path.join(BASE_DIR, experimental_dir)

    os.makedirs(cache_dir, exist_ok=True)

    training_wav_paths = convert_m4a_to_wav(training_dir, target_sr=sr)
    experimental_wav_paths = convert_m4a_to_wav(experimental_dir, target_sr=sr)

    training_clean     = denoise_all(training_wav_paths,     sr, noise_sample_duration,
                                     cache_path=os.path.join(cache_dir, "training_denoised.npz"))
    experimental_clean = denoise_all(experimental_wav_paths, sr, noise_sample_duration,
                                     cache_path=os.path.join(cache_dir, "experimental_denoised.npz"))

    datasets = split_training_data_into_datasets(training_wav_paths, training_clean)
    spliced_datasets_norm = splice_datasets(datasets, normalization=normalization)

    (X, y_loading, y_flange, y_area) = flatten_datasets(spliced_datasets_norm)
    (X_train, X_test, y_train, y_test,
     y_train_flange, y_test_flange,
     y_train_area,   y_test_area) = train_test_split(
        X, y_loading, y_flange, y_area,
        test_size=test_size, shuffle=True, random_state=random_state
    )

    return PreprocessedData(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        y_train_flange=y_train_flange, y_test_flange=y_test_flange,
        y_train_area=y_train_area, y_test_area=y_test_area,
        datasets=spliced_datasets_norm,
        experimental_clean=experimental_clean,
        datasets_raw=datasets
    )


def independent_split(datasets, left_out_flange):
    """
    Leave-one-flange-out split for generalization testing.

    Trains on all flanges except left_out_flange, tests on left_out_flange only.
    This evaluates whether the model generalizes to a bolt it has never seen.

    Returns X_train, X_test, y_train, y_test.
    """
    all_flanges = set(datasets.keys())
    train_flanges = all_flanges - {left_out_flange}

    X_train, y_train, _, _ = flatten_datasets(datasets, train_flanges)
    X_test,  y_test,  _, _ = flatten_datasets(datasets, {left_out_flange})

    return X_train, X_test, y_train, y_test


def flatten_datasets(spliced_datasets, flanges=None):
    """
    Flatten the nested { flange: { loading: { area: splices } } } dict into arrays.

    flanges : set of flange IDs to include, or None to include all.
              Used by independent_split to build per-flange train/test sets.

    Returns (X, y_loading, y_flange, y_area) as numpy arrays.
    """
    LOADING_MAP = {"0ftlb": 0, "25ftlb": 1, "50ftlb": 2}
    X_list, y_loading_list, y_flange_list, y_area_list = [], [], [], []

    for flange, conditions in spliced_datasets.items():
        if flanges is not None and flange not in flanges:
            continue
        for loading, areas in conditions.items():
            for area, splices in areas.items():
                n = len(splices)
                X_list.append(splices)
                y_loading_list.extend([LOADING_MAP[loading]] * n)
                y_flange_list.extend([flange] * n)
                y_area_list.extend([int(area[1])] * n)

    return (
        np.concatenate(X_list, axis=0),
        np.array(y_loading_list),
        np.array(y_flange_list),
        np.array(y_area_list),
    )