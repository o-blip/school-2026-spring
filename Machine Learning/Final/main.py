# %%
import os

import numpy as np
import matplotlib.pyplot as plt
import librosa

import pywt
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from load_datasets import convert_m4a_to_wav, denoise_all, split_training_data_into_datasets, splice_datasets

# =============================================================================
# CONSTANTS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR    = os.path.join(BASE_DIR, "Training Data")
EXPERIMENTAL_DIR = os.path.join(BASE_DIR, "Experimental Data")
CACHE_DIR       = os.path.join(BASE_DIR, "cache")
SR = 48000
NOISE_SAMPLE_DURATION = 0.2  # seconds at the end of each recording used as noise profile

os.makedirs(CACHE_DIR, exist_ok=True)

# =============================================================================
# PART 1: LOAD DATA & NOISE REDUCTION
# =============================================================================

# %% ── 1.1  Convert m4a → wav ─────────────────────────────────────────────────
training_wav_paths = convert_m4a_to_wav(TRAINING_DIR, target_sr=SR)
experimental_wav_paths = convert_m4a_to_wav(EXPERIMENTAL_DIR,target_sr=SR)
# %% ── 1.2  Apply spectral gating to all recordings ─────────────────────────
training_clean     = denoise_all(training_wav_paths,     SR, NOISE_SAMPLE_DURATION,
                                 cache_path=os.path.join(CACHE_DIR, "training_denoised.npz"))
experimental_clean = denoise_all(experimental_wav_paths, SR, NOISE_SAMPLE_DURATION,
                                 cache_path=os.path.join(CACHE_DIR, "experimental_denoised.npz"))

# %% ── 1.3  Combine cleaned waveforms into datasets according to flange ID ─────────────────────────

datasets = split_training_data_into_datasets(training_wav_paths, training_clean)
spliced_datasets = splice_datasets(datasets)

# %% ── 1.4  Combine all spliced datasets into one flat dataset ───────────────

LOADING_MAP = {"0ftlb": 0, "25ftlb": 1, "50ftlb": 2}

X_list, y_loading_list, y_flange_list, y_area_list = [], [], [], []

for flange, conditions in spliced_datasets.items():
    for loading, areas in conditions.items():
        for area, splices in areas.items():
            n = len(splices)
            X_list.append(splices)
            y_loading_list.extend([LOADING_MAP[loading]] * n)
            y_flange_list.extend([flange] * n)
            y_area_list.extend([int(area[1])] * n)

X         = np.concatenate(X_list, axis=0)           # (N, splice_len)
y_loading = np.array(y_loading_list)                  # 0=0ftlb, 1=25ftlb, 2=50ftlb
y_flange  = np.array(y_flange_list)                   # 1–4
y_area    = np.array(y_area_list)                     # 1–4

print(f"Combined dataset: {X.shape[0]} splices, splice length {X.shape[1]} samples")
print(f"  Loading conditions — 0ftlb: {(y_loading==0).sum()}, "
      f"25ftlb: {(y_loading==1).sum()}, 50ftlb: {(y_loading==2).sum()}")
print("  Flanges — " + ", ".join(f"F{f}: {(y_flange==f).sum()}" for f in range(1, 5)))

# %% ── 1.5  Shuffle and split 70 / 30 train / test ───────────────────────────
(X_train, X_test,
 y_train, y_test,
 y_train_flange, y_test_flange,
 y_train_area,   y_test_area) = train_test_split(
    X, y_loading, y_flange, y_area,
    test_size=0.3, shuffle=True, random_state=42
)

print(f"Train: {len(X_train)} splices  |  Test: {len(X_test)} splices")
print(f"  Train — 0ftlb: {(y_train==0).sum()}, 25ftlb: {(y_train==1).sum()}, 50ftlb: {(y_train==2).sum()}")
print(f"  Test  — 0ftlb: {(y_test==0).sum()},  25ftlb: {(y_test==1).sum()},  50ftlb: {(y_test==2).sum()}")



# %% 2.1 Feature Extraction - discrete wavelet transform
# db4 wavelet, 4-level decomposition
# wavedec returns [cA4, cD4, cD3, cD2, cD1] — approximation + detail coeffs per level

WAVELET = "db4"
LEVEL   = 4

def extract_dwt_features(X):
    features = []
    for splice in X:
        coeffs = pywt.wavedec(splice, WAVELET, level=LEVEL)
        features.append(np.concatenate(coeffs))
    return np.array(features)

X_train_dwt = extract_dwt_features(X_train)
X_test_dwt  = extract_dwt_features(X_test)

print(f"DWT feature shape — train: {X_train_dwt.shape}, test: {X_test_dwt.shape}")
print("Coefficient breakdown (one splice):")
coeffs_sample = pywt.wavedec(X_train[0], WAVELET, level=LEVEL)
labels = [f"cA{LEVEL}"] + [f"cD{LEVEL - i}" for i in range(LEVEL)]
for label, c in zip(labels, coeffs_sample):
    print(f"  {label}: {len(c)} coefficients")







# %% SANITY CHECKS

def plot_peak_check(audio, sr, label, pre_peak=50, post_peak=2400):
    t = np.arange(len(audio)) / sr
    threshold = 0.4 * np.max(np.abs(audio))
    all_peaks, _ = find_peaks(np.abs(audio), height=threshold, distance=20000)
    valid_peaks   = [p for p in all_peaks if p >= pre_peak and p + post_peak <= len(audio)]
    dropped_peaks = [p for p in all_peaks if p not in valid_peaks]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, audio, linewidth=0.5, color="tab:blue")
    ax.vlines(np.array(valid_peaks) / sr, ymin=audio.min(), ymax=audio.max(),
              color="red", linewidth=1, label=f"{len(valid_peaks)} valid splices")
    if dropped_peaks:
        ax.vlines(np.array(dropped_peaks) / sr, ymin=audio.min(), ymax=audio.max(),
                  color="gray", linewidth=1, linestyle="--",
                  label=f"{len(dropped_peaks)} dropped (edge)")
    ax.set_title(f"{label} — denoised waveform with detected peaks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid()
    plt.tight_layout()
    plt.show()

# ── Flange 4 50ftlb A2: 23 splices (expected 20) ─────────────────────────────
plot_peak_check(datasets[4]["50ftlb"]["A2"], SR, "Flange 4 50ftlb A2")

# ── Flange 4 50ftlb A4 ────────────────────────────────────────────────────────
plot_peak_check(datasets[4]["50ftlb"]["A4"], SR, "Flange 4 50ftlb A4")

# %%
