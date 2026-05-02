import numpy as np
import pywt
from scipy.stats import kurtosis, skew
import librosa

## Wavelet transform features
# db4 wavelet, 4-level decomposition
# wavedec returns [cA4, cD4, cD3, cD2, cD1] — approximation + detail coeffs per level

def subband_energy(coeffs):
    return np.sum(coeffs ** 2)

def subband_entropy(coeffs):
    # Shannon entropy of the normalized energy distribution across coefficients
    energy = coeffs ** 2
    total = np.sum(energy)
    if total == 0:
        return 0.0
    p = energy / total
    return -np.sum(p * np.log2(p + 1e-12))

def extract_dwt_features(X, wavelet='db4', level=4):
    # Per subband: energy, entropy, mean, variance, max absolute value, kurtosis, skewness
    # Feature vector length: 7 features * (level + 1) subbands = 7 * 5 = 35
    features = []
    for splice in X:
        coeffs = pywt.wavedec(splice, wavelet, level=level)
        vec = []
        for c in coeffs:
            vec.append(subband_energy(c))
            vec.append(subband_entropy(c))
            vec.append(np.mean(c))
            vec.append(np.var(c))
            vec.append(np.max(np.abs(c)))
            vec.append(kurtosis(c))
            vec.append(skew(c))
        features.append(vec)
    return np.array(features)

# MFCC Feature sequences for LSTM

def extract_mfcc_sequence(X, sr=48000, n_mfcc=13, hop_length=512):
    sequences = []
    for splice in X:
        mfcc = librosa.feature.mfcc(y=splice, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        sequences.append(mfcc.T)  # (time_frames, n_mfcc)
    return np.array(sequences)

# 1D MFCC feature vectors, 13 coefficients and 4 stats -> 52 feature vector
def extract_mfcc_features(X, sr=48000, n_mfcc=13, hop_length=512):
    features = []
    for splice in X:
        mfcc = librosa.feature.mfcc(y=splice, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        vec = []
        for coeff in mfcc:
            vec.append(np.mean(coeff))
            vec.append(np.var(coeff))
            vec.append(kurtosis(coeff))
            vec.append(skew(coeff))
        features.append(vec)
    return np.array(features)