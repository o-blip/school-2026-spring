import numpy as np
import pywt
from scipy.stats import kurtosis, skew
from typing import Literal
import librosa

# =============================================================================
# Feature Extraction
# Two feature sets:
#   DWT  — Discrete Wavelet Transform subband statistics (db4, 4-level decomposition)
#   MFCC — Mel-Frequency Cepstral Coefficients, as fixed vectors or time sequences
# =============================================================================

# --- DWT subband helpers ---

def subband_energy(coeffs: np.ndarray) -> float:
    """L2 energy (sum of squared coefficients) for one DWT subband."""
    return np.sum(coeffs ** 2)


def subband_entropy(coeffs: np.ndarray) -> float:
    """Shannon entropy of the energy distribution within one DWT subband.

    High entropy means energy is spread evenly across coefficients (noise-like).
    Low entropy means energy is concentrated in a few coefficients (impulsive tap).
    """
    energy = coeffs ** 2
    total = np.sum(energy)
    if total == 0:
        return 0.0
    p = energy / total
    return -np.sum(p * np.log2(p + 1e-12))  # 1e-12 avoids log(0) on zero-energy bins


def extract_dwt_features(X: np.ndarray, wavelet: str = 'db4', level: int = 4) -> np.ndarray:
    """Extract DWT subband statistics from each waveform splice.

    wavedec with level=4 returns [cA4, cD4, cD3, cD2, cD1]:
    one approximation subband + 4 detail subbands at increasing temporal resolution.

    Per subband: energy, entropy, mean, variance, max absolute value, kurtosis, skewness
    Feature vector length: 7 stats × 5 subbands = 35

    Parameters
    ----------
    X       : (N, splice_len) array of waveform splices
    wavelet : PyWavelets wavelet family name
    level   : decomposition depth

    Returns
    -------
    features : (N, 35) float64 array
    """
    features = []
    for splice in X:
        coeffs = pywt.wavedec(splice, wavelet, level=level)
        vec = []
        for c in coeffs:
            vec.extend([
                subband_energy(c),
                subband_entropy(c),
                np.mean(c),
                np.var(c),
                np.max(np.abs(c)),
                kurtosis(c),
                skew(c),
            ])
        features.append(vec)
    return np.array(features)


# --- MFCC helpers ---

def extract_mfcc_sequence(X: np.ndarray, sr: int = 48000, n_mfcc: int = 13, hop_length: int = 512) -> np.ndarray:
    """Extract MFCC sequences suitable for LSTM input.

    Each splice becomes a (time_frames, n_mfcc) matrix, where
    time_frames = ceil(splice_len / hop_length).

    Parameters
    ----------
    X          : (N, splice_len) array of waveform splices
    sr         : sample rate in Hz (iPhone recordings are 48 kHz)
    n_mfcc     : number of cepstral coefficients
    hop_length : hop size in samples (e.g., 512 samples ≈ 10.7 ms at 48 kHz)

    Returns
    -------
    sequences : (N, time_frames, n_mfcc) float32 array N: number of splices, time_frames depends on splice_len and hop_length, n_mfcc is default 13
    """
    sequences = []
    for splice in X:
        mfcc = librosa.feature.mfcc(y=splice, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        sequences.append(mfcc.T)  # (n_mfcc, frames) → (frames, n_mfcc) transpose for LSTM input
    return np.array(sequences)


def extract_mfcc_features(X: np.ndarray, sr: int = 48000, n_mfcc: int = 13, hop_length: int = 512) -> np.ndarray:
    """Extract a fixed-length MFCC feature vector from each splice.

    Each of the 13 coefficients is summarized by 4 statistics:
    mean, variance, kurtosis, skewness → 13 × 4 = 52-dimensional vector.

    Parameters
    ----------
    X          : (N, splice_len) array of waveform splices
    sr         : sample rate in Hz
    n_mfcc     : number of cepstral coefficients
    hop_length : hop size in samples (e.g., 512 samples ≈ 10.7 ms at 48 kHz)

    Returns
    -------
    features : (N, 52) float64 array
    """
    features = []
    for splice in X:
        mfcc = librosa.feature.mfcc(y=splice, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        vec = []
        for coeff in mfcc:
            vec.extend([np.mean(coeff), np.var(coeff), kurtosis(coeff), skew(coeff)])
        features.append(vec)
    return np.array(features)


def build_features(X: np.ndarray, feature_type: Literal['DWT', 'MFCC', 'Both']) -> np.ndarray:
    """Dispatch to the appropriate feature extractor based on feature_type.

    Returns
    -------
    'DWT'  → (N, 35)
    'MFCC' → (N, 52)
    'Both' → (N, 87)  — DWT and MFCC concatenated column-wise
    """
    if feature_type == 'DWT':
        return extract_dwt_features(X)
    elif feature_type == 'MFCC':
        return extract_mfcc_features(X)
    elif feature_type == 'Both':
        return np.hstack([extract_dwt_features(X), extract_mfcc_features(X)])