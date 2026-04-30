import os

import numpy as np
import matplotlib.pyplot as plt

from preprocessing import load_datasets


def load_training_data(recordings_dir, wav_dir, sr, noise_sample_duration, expected_hits, cache_path, images_dir):
    """
    Load (or process and cache) training splices.
    Training files follow the naming convention {loading_condition}F{flange}A{area}.m4a,
    where loading_condition is 0ftlb, 25ftlb, or 50ftlb.

    Returns: (healthy_hits, unhealthy_hits, noise_plot_data)
        healthy_hits    — ndarray (N, samples)
        unhealthy_hits  — ndarray (N, samples)
        noise_plot_data — dict with raw/clean waveforms for plotting, or None if loaded from cache
    """
    if os.path.exists(cache_path):
        print("Loading training splices from cache…")
        data = np.load(cache_path)
        healthy_hits = data["healthy"]
        unhealthy_hits = data["unhealthy"]
        noise_plot_data = None
        print(f"  Healthy:   {healthy_hits.shape[0]}  |  Unhealthy: {unhealthy_hits.shape[0]}")
    else:
        print("Processing training recordings (first run — this may take a while)…")
        healthy_hits, unhealthy_hits, noise_plot_data = load_datasets(
            recordings_dir, wav_dir, sr, noise_sample_duration, expected_hits
        )
        np.savez(cache_path, healthy=healthy_hits, unhealthy=unhealthy_hits)
        print(f"Saved training splices to {cache_path}")

    if noise_plot_data is not None:
        raw = noise_plot_data["raw"]
        clean = noise_plot_data["clean"]
        t = np.arange(len(raw)) / sr
        nd = noise_plot_data["noise_sample_duration"]
        ds = noise_plot_data["ds_num"]
        snum = noise_plot_data["num"]
        lbl = noise_plot_data["label"]

        fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        ax[0].plot(t, raw, linewidth=0.5)
        ax[0].axvspan(0, nd, alpha=0.2, color="red", label=f"Noise sample ({nd}s)")
        ax[0].set_title(f"Raw — Dataset {ds}, s_{snum} ({lbl})")
        ax[0].set_ylabel("Amplitude")
        ax[0].legend(fontsize=8, loc="upper right")
        ax[0].grid()
        ax[1].plot(t, clean, linewidth=0.5, color="tab:orange")
        ax[1].axvspan(0, nd, alpha=0.2, color="red", label=f"Noise sample ({nd}s)")
        ax[1].set_title("After spectral gating")
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Amplitude")
        ax[1].legend(fontsize=8, loc="upper right")
        ax[1].grid()
        ax[1].set_xlim(0, t[-1])
        plt.tight_layout()
        plt.savefig(
            os.path.join(images_dir, "01_noise_reduction_comparison.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()

    return healthy_hits, unhealthy_hits, noise_plot_data


def load_testing_data(testing_dir, sr, noise_sample_duration):
    """
    Load unlabelled testing splices.
    Testing files follow the naming convention F{flange}A{area}.m4a (no loading condition).

    Returns: hits — ndarray (N, samples)
    """
    # TODO: implement
    pass
