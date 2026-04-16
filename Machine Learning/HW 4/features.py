import numpy as np
import librosa
from scipy.signal import periodogram

# ── Frequency bands used for PSD feature extraction ──────────────────────────
bands_hz = [
    (0, 250, "F1"),
    (750, 1100, "F2"),
    (1800, 2100, "F3"),
    (2100, 2500, "F4"),
    (3300, 3600, "F5"),
    (3900, 4300, "F6"),
    (5000, 5250, "F7"),
    (5250, 5500, "F8"),
    (6000, 6400, "F9"),
]

# Number of MFCC coefficients to extract (common choice for speech/audio tasks)
N_MFCC = 13


# ── Plotting helper ──────────────────────────────────────────────────────
def add_bands(axis):
    """Shade and label PSD frequency bands on a kHz-scaled axis."""
    ylim = axis.get_ylim()
    for lo, hi, label in bands_hz:
        lo_k, hi_k = lo / 1000, hi / 1000
        axis.axvspan(lo_k, hi_k, alpha=0.15, color="gray")
        axis.axvline(lo_k, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        axis.axvline(hi_k, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        axis.text(
            (lo_k + hi_k) / 2,
            ylim[1] * 0.92,
            label,
            ha="center",
            va="top",
            fontsize=8,
            color="gray",
            fontweight="bold",
        )


# ── Classical (flat) features — used by BPNN, KNN, DT, LR, SVM ──────────────
def extract_psd_bands(splices, sr):
    """Relative energy in each frequency band (normalized by total PSD)."""
    features = []
    for splice in splices:
        f, Pxx = periodogram(splice, fs=sr)
        total = np.sum(Pxx)
        features.append(
            [np.sum(Pxx[(f >= lo) & (f <= hi)]) / total for lo, hi, *_ in bands_hz]
        )
    return np.array(features)


def extract_mfcc_flat(splices, sr, n_mfcc=N_MFCC):
    """Mean + std of MFCC coefficients across time frames → flat vector."""
    features = []
    for splice in splices:
        mfcc = librosa.feature.mfcc(y=splice, sr=sr, n_mfcc=n_mfcc)
        features.append(np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]))
    return np.array(features)


def add_bias(X):
    """Prepend a column of ones for logistic-regression bias term."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


# ── 2-D features — used by CNN ──────────────────────────────────────────────
def extract_logmel(splices, sr, n_mels=64, n_fft=512, hop_length=128):
    """
    Log-mel spectrogram per splice → shape (N, n_mels, time_frames).
    All spectrograms are zero-padded / truncated to the same time_frames.
    """
    specs = []
    for splice in splices:
        mel = librosa.feature.melspectrogram(
            y=splice, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        specs.append(librosa.power_to_db(mel, ref=np.max))
    # Uniform width: pad/truncate to the most common frame count
    min_frames = min(s.shape[1] for s in specs)
    return np.array([s[:, :min_frames] for s in specs])  # (N, n_mels, time_frames)


# ── Sequence features — used by RNN ──────────────────────────────────────────
def extract_mfcc_seq(splices, sr, n_mfcc=N_MFCC, hop_length=128):
    """
    MFCC sequence per splice → shape (N, time_frames, n_mfcc).
    Columns are time steps, suitable for an LSTM/GRU input.
    """
    seqs = []
    for splice in splices:
        mfcc = librosa.feature.mfcc(
            y=splice, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length
        )
        seqs.append(mfcc.T)  # (time_frames, n_mfcc)
    min_frames = min(s.shape[0] for s in seqs)
    return np.array([s[:min_frames, :] for s in seqs])  # (N, time_frames, n_mfcc)
