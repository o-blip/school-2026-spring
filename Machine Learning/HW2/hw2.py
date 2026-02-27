# %%
import os
import re
import librosa
import numpy as np
import noisereduce as nr
import matplotlib.pyplot as plt
from pydub import AudioSegment
from matplotlib.patches import Patch
from scipy.signal import butter, sosfilt, find_peaks, periodogram
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# HW2: Machine Learning 
# Piero Risi Mortola
# Claude used in writing the code
# =============================================================================
# 1. LOAD & CONVERT RECORDINGS
# =============================================================================
# %%
RECORDINGS_DIR = "Recordings"
WAV_DIR = "wav"
os.makedirs(WAV_DIR, exist_ok=True)
IMAGES_DIR = "Images"
os.makedirs(IMAGES_DIR, exist_ok=True)

recordings = {}
for fname in sorted(
    os.listdir(RECORDINGS_DIR), key=lambda x: int(re.search(r"\d+", x).group())
):
    match = re.match(r"s_(\d+)_(g|b)\.m4a", fname)
    if match:
        num = int(match.group(1))
        label = match.group(2)
        m4a_path = os.path.join(RECORDINGS_DIR, fname)
        wav_path = os.path.join(WAV_DIR, f"s_{num}_{label}.wav")
        AudioSegment.from_file(m4a_path, format="m4a").export(wav_path, format="wav")
        recordings[num] = {"label": label, "wav_path": wav_path}

print(f"Converted {len(recordings)} files")
print(f"  Good: {sum(1 for v in recordings.values() if v['label'] == 'g')}")
print(f"  Bad:  {sum(1 for v in recordings.values() if v['label'] == 'b')}")


# =============================================================================
# 2. NOISE REDUCTION
# =============================================================================
# %%
SR = 48000
NOISE_SAMPLE_DURATION = (
    0.5  # seconds - assumes recording starts with silence before first tap
)


def bandpass_filter(audio, sr, low_hz=200, high_hz=20000, order=5):
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
    return sosfilt(sos, audio)


for num, info in recordings.items():
    audio, _ = librosa.load(info["wav_path"], sr=SR, mono=True)
    noise_sample = audio[: int(NOISE_SAMPLE_DURATION * SR)]
    audio_nr = nr.reduce_noise(y=audio, sr=SR, y_noise=noise_sample)
    audio_clean = bandpass_filter(audio_nr, SR)
    recordings[num]["audio"] = audio_clean

print(f"Denoised {len(recordings)} recordings")

# Before/after comparison on first recording
sample_num = next(iter(recordings))
raw, _ = librosa.load(recordings[sample_num]["wav_path"], sr=SR, mono=True)
clean = recordings[sample_num]["audio"]
t = np.arange(len(raw)) / SR

fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
ax[0].plot(t, raw, linewidth=0.5)
ax[0].set_title(f"Raw - s_{sample_num} ({recordings[sample_num]['label']})")
ax[0].set_ylabel("Amplitude")
ax[0].grid()
ax[1].plot(t, clean, linewidth=0.5, color="tab:orange")
ax[1].set_title("After spectral gating + bandpass (200 Hz - 20 kHz)")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Amplitude")
ax[1].grid()
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "01_noise_reduction_comparison.png"), dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# 3. SPLICE INDIVIDUAL TAPS
# =============================================================================
# %%
def splice_audio(audio, sr, pre_peak=100, post_peak=3000):
    threshold = 0.3 * np.max(np.abs(audio))
    peaks, _ = find_peaks(audio, height=threshold, distance=30000)
    valid = [p for p in peaks if p >= pre_peak and p + post_peak <= len(audio)]
    splices = np.array([audio[p - pre_peak: p + post_peak] for p in valid])
    return splices


healthy_hits = []
unhealthy_hits = []

EXPECTED_HITS = 10  # expected number of taps per recording

for num, info in recordings.items():
    splices = splice_audio(info["audio"], SR)
    if splices.size == 0:
        print(f"  WARNING: no peaks found in s_{num} ({info['label']})")
        continue
    if len(splices) < EXPECTED_HITS:
        print(f"  WARNING: s_{num} ({info['label']}) only has {len(splices)} hits (expected {EXPECTED_HITS})")
    if info["label"] == "g":
        healthy_hits.append(splices)
    else:
        unhealthy_hits.append(splices)

healthy_hits = np.concatenate(healthy_hits, axis=0)
unhealthy_hits = np.concatenate(unhealthy_hits, axis=0)

print(f"Healthy hits:   {healthy_hits.shape}  ({healthy_hits.shape[0]} splices)")
print(f"Unhealthy hits: {unhealthy_hits.shape}  ({unhealthy_hits.shape[0]} splices)")
print(
    f"Splice length:  {healthy_hits.shape[1]} samples = {healthy_hits.shape[1]/SR*1000:.1f} ms"
)


# =============================================================================
# 4. AVERAGE PSD (full dataset) + TRAIN/TEST SPLIT
# =============================================================================
# %%
def peak_normalize(s):
    peak = np.max(np.abs(s))
    return s / peak if peak > 0 else s


f_h, _ = periodogram(peak_normalize(healthy_hits[0]), fs=SR)
avg_psd_healthy = np.mean(
    [periodogram(peak_normalize(s), fs=SR)[1] for s in healthy_hits], axis=0
)
avg_psd_unhealthy = np.mean(
    [periodogram(peak_normalize(s), fs=SR)[1] for s in unhealthy_hits], axis=0
)

# Feature bands identified from full dataset (kHz, for plotting)
bands = [
    (0.8, 1.1, "F1", False),
    (1.9, 2.2, "F2", False),
    (2.2, 2.5, "F3", False),
    (3.3, 3.6, "F4", False),
    (4.0, 4.5, "F5", False),
    (5.1, 5.3, "F6", False),
    (5.3, 5.6, "F7", False),
    (6.0, 6.5, "F8", False),
]


def add_bands(axis):
    ylim = axis.get_ylim()
    for lo, hi, label, healthy_only in bands:
        color = "green" if healthy_only else "gray"
        axis.axvspan(lo, hi, alpha=0.15, color=color)
        axis.axvline(lo, color=color, linewidth=0.8, linestyle="--", alpha=0.6)
        axis.axvline(hi, color=color, linewidth=0.8, linestyle="--", alpha=0.6)
        axis.text(
            (lo + hi) / 2,
            ylim[1] * 0.92,
            label,
            ha="center",
            va="top",
            fontsize=8,
            color=color,
            fontweight="bold",
        )


fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax[0].plot(f_h / 1000, avg_psd_healthy)
ax[0].set_title(f"Average PSD: Healthy (n={len(healthy_hits)} splices, full dataset)")
ax[0].set_ylabel("PSD")
ax[0].grid()
ax[0].set_xlim(0, 8)
add_bands(ax[0])

ax[1].plot(f_h / 1000, avg_psd_unhealthy, color="tab:orange")
ax[1].set_title(
    f"Average PSD: Unhealthy (n={len(unhealthy_hits)} splices, full dataset)"
)
ax[1].set_ylabel("PSD")
ax[1].set_xlabel("Frequency (kHz)")
ax[1].grid()
ax[1].set_xlim(0, 8)
add_bands(ax[1])

ax[1].xaxis.set_major_locator(plt.MultipleLocator(0.5))

ax[0].legend(
    handles=[
        Patch(facecolor="gray", alpha=0.3, label="Feature bands"),
    ],
    fontsize=8,
    loc="upper right",
)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "02_average_psd_bands.png"), dpi=150, bbox_inches="tight")
plt.show()

# Train / test split (shuffled to avoid cell-ordering bias)
X_all = np.concatenate([healthy_hits, unhealthy_hits], axis=0)
y_all = np.array([0] * len(healthy_hits) + [1] * len(unhealthy_hits))

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.3, shuffle=True, random_state=42, stratify=y_all
)

print(f"Train - total: {len(X_train)}  ({(y_train == 0).sum()} healthy, {(y_train == 1).sum()} unhealthy)")
print(f"Test  - total: {len(X_test)}  ({(y_test == 0).sum()} healthy, {(y_test == 1).sum()} unhealthy)")


# =============================================================================
# 5. FEATURE EXTRACTION  (MFCC and PSD band energies)
# =============================================================================
# %%
N_MFCC = 13


def extract_mfcc(splices, sr, n_mfcc=N_MFCC):
    features = []
    for splice in splices:
        mfcc = librosa.feature.mfcc(y=peak_normalize(splice), sr=sr, n_mfcc=n_mfcc)
        features.append(np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]))
    return np.array(features)



# Band energies in Hz matching the identified feature bands
bands_hz = [
    (800, 1100),  # F1
    (1700, 1900),  # F2
    (1900, 2500),  # F3
    (2800, 3100),  # F4
    (3300, 3600),  # F5
    (4000, 4500),  # F6
    (5100, 5600),  # F7
    (6000, 6500),  # F8 - healthy only
    (7000, 7500),  # F9 - healthy only
]


def extract_psd_bands(splices, sr):
    features = []
    for splice in splices:
        f, Pxx = periodogram(peak_normalize(splice), fs=sr)  # normalize before PSD
        band_energies = [np.sum(Pxx[(f >= lo) & (f <= hi)]) for lo, hi in bands_hz]
        features.append(band_energies)
    return np.array(features)


X_train_mfcc = extract_mfcc(X_train, SR)
X_test_mfcc = extract_mfcc(X_test, SR)
X_train_psd = extract_psd_bands(X_train, SR)
X_test_psd = extract_psd_bands(X_test, SR)

print(f"MFCC features - train: {X_train_mfcc.shape}, test: {X_test_mfcc.shape}")
print(f"PSD features  - train: {X_train_psd.shape},  test: {X_test_psd.shape}")


# =============================================================================
# 6. K-NEAREST NEIGHBOURS CLASSIFIER
# =============================================================================
# %%
k_values = [1, 3, 5, 10]
feature_sets = [
    ("MFCC", X_train_mfcc, X_test_mfcc),
    ("PSD Bands", X_train_psd, X_test_psd),
]

for feat_name, X_tr, X_te in feature_sets:
    fig, axes = plt.subplots(2, len(k_values), figsize=(14, 7))
    for col, k in enumerate(k_values):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_tr, y_train)
        for row, (X_eval, y_eval, split_name) in enumerate([
            (X_tr, y_train, "Train"),
            (X_te, y_test,  "Test"),
        ]):
            y_pred = knn.predict(X_eval)
            acc = accuracy_score(y_eval, y_pred)
            cm = confusion_matrix(y_eval, y_pred)
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=["Healthy", "Unhealthy"]
            )
            disp.plot(ax=axes[row, col], cmap="Blues", colorbar=False)
            axes[row, col].set_title(f"k={k} | {split_name}\nAcc: {acc:.2%}", fontsize=9)
    plt.suptitle(f"KNN — {feat_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, f"03_knn_{feat_name.lower().replace(' ', '_')}_confusion.png"), dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# 7. DECISION TREE CLASSIFIER
# =============================================================================
# %%
trained_trees = {}

fig, axes = plt.subplots(2, len(feature_sets), figsize=(10, 7))
for col, (feat_name, X_tr, X_te) in enumerate(feature_sets):
    dt = DecisionTreeClassifier(random_state=42, max_depth=3)
    dt.fit(X_tr, y_train)
    trained_trees[feat_name] = dt
    for row, (X_eval, y_eval, split_name) in enumerate([
        (X_tr, y_train, "Train"),
        (X_te, y_test,  "Test"),
    ]):
        y_pred = dt.predict(X_eval)
        acc = accuracy_score(y_eval, y_pred)
        cm = confusion_matrix(y_eval, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Healthy", "Unhealthy"]
        )
        disp.plot(ax=axes[row, col], cmap="Greens", colorbar=False)
        axes[row, col].set_title(f"{feat_name} | {split_name}\nAcc: {acc:.2%}")

plt.suptitle("Decision Tree: MFCC vs PSD Band Features", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "04_decision_tree_confusion.png"), dpi=150, bbox_inches="tight")
plt.show()

# Tree diagrams
# %%
mfcc_feature_names = [f"MFCC{i}_mean" for i in range(N_MFCC)] + [
    f"MFCC{i}_std" for i in range(N_MFCC)
]
psd_feature_names = [f"F{i+1} ({lo}-{hi} Hz)" for i, (lo, hi) in enumerate(bands_hz)]

feature_names_map = {
    "MFCC": mfcc_feature_names,
    "PSD Bands": psd_feature_names,
}

for feat_name, dt in trained_trees.items():
    fig, ax = plt.subplots(figsize=(30, 12))
    plot_tree(
        dt,
        feature_names=feature_names_map[feat_name],
        class_names=["Healthy", "Unhealthy"],
        filled=True,
        rounded=True,
        fontsize=7,
        ax=ax,
    )
    ax.set_title(f"Decision Tree - {feat_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, f"05_tree_diagram_{feat_name.lower().replace(' ', '_')}.png"), dpi=150, bbox_inches="tight")
    plt.show()
