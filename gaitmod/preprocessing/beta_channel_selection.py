import json
import logging
import os
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.time_frequency import psd_array_welch

from gaitmod.utils.utils import load_pkl


BETA_BAND = (13.0, 30.0)


def compute_beta_power(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Compute average beta power for each channel.

    Args:
        data: Array of shape (n_epochs, n_channels, n_times)
        sfreq: Sampling frequency

    Returns:
        Mean beta power per channel (shape: n_channels)
    """
    n_times = data.shape[-1]
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    fft_vals = np.fft.rfft(data, axis=-1)
    psd = np.abs(fft_vals) ** 2
    beta_mask = (freqs >= BETA_BAND[0]) & (freqs <= BETA_BAND[1])
    beta_power = psd[..., beta_mask].mean(axis=-1)
    mean_power = beta_power.mean(axis=0)
    mean_power = np.maximum(mean_power, 1e-12)
    return 10 * np.log10(mean_power)


def analyze_beta_activity(pickle_path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    patient_epochs: Dict[str, mne.Epochs] = load_pkl(pickle_path)
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for subject, epochs in patient_epochs.items():
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        beta_per_channel = compute_beta_power(data, epochs.info['sfreq'])
        channel_names = epochs.ch_names
        channel_scores = {
            ch: float(score) for ch, score in zip(channel_names, beta_per_channel)
        }
        best_channel_name, best_score = max(channel_scores.items(), key=lambda item: item[1])
        best_channel_index = channel_names.index(best_channel_name)
        psd_values_all = []
        freqs = None
        for ch_idx in range(data.shape[1]):
            psd_vals, freqs = psd_array_welch(
                data[:, ch_idx, :],
                sfreq=epochs.info['sfreq'],
                fmin=1.0,
                fmax=100.0,
                average='mean',
                n_fft=data.shape[-1]
            )
            if psd_vals.ndim > 1:
                psd_vals = psd_vals.mean(axis=0)
            psd_vals = np.maximum(psd_vals, 1e-12)
            psd_vals_db = 10 * np.log10(psd_vals)
            psd_values_all.append(psd_vals_db.tolist())

        results[subject] = {
            'best_channel': best_channel_name,
            'best_channel_index': best_channel_index,
            'beta_scores': channel_scores,
            'psd_freqs': freqs.tolist() if freqs is not None else [],
            'psd_values': psd_values_all,
            'channels': channel_names
        }
        logging.info("Subject %s best beta channel: %s (index=%d)", subject, best_channel_name, best_channel_index)

    return results


def plot_beta_activity(results: Dict[str, Dict[str, Dict[str, float]]], output_dir: Path):
    subjects = sorted(results.keys())
    fig, axes = plt.subplots(2, len(subjects), figsize=(5 * len(subjects), 6))
    if len(subjects) == 1:
        axes = axes.reshape(2, 1)

    channel_color_map = {}
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    available_colors = iter(color_cycle * 3)

    def get_channel_color(channel: str) -> str:
        if channel not in channel_color_map:
            channel_color_map[channel] = next(available_colors)
        return channel_color_map[channel]

    for col, subject in enumerate(subjects):
        subj_data = results[subject]
        # PSD subplot (all channels)
        freqs = np.asarray(subj_data['psd_freqs'])
        for ch_name, psd_vals in zip(subj_data['channels'], subj_data['psd_values']):
            color = get_channel_color(ch_name)
            lw = 2.5 if ch_name == subj_data['best_channel'] else 1.5
            axes[0, col].plot(freqs, psd_vals, color=color, linewidth=lw, alpha=0.8)
        axes[0, col].set_title(f"{subject} – PSD (all channels)")
        axes[0, col].set_ylabel("Power (dB)")
        axes[0, col].set_xlim(*BETA_BAND)
        axes[0, col].grid(True, linestyle='--', alpha=0.5)

        # Beta bar subplot
        channel_scores = subj_data['beta_scores']
        channels = list(channel_scores.keys())
        values = [channel_scores[ch] for ch in channels]
        colors = [get_channel_color(ch) for ch in channels]
        edge_colors = [
            'black' if ch == subj_data['best_channel'] else get_channel_color(ch)
            for ch in channels
        ]
        linewidths = [2.0 if ch == subj_data['best_channel'] else 1.0 for ch in channels]
        x = np.arange(len(channels))
        bars = axes[1, col].bar(x, values, color=colors)
        for bar, edge_color, lw in zip(bars, edge_colors, linewidths):
            bar.set_edgecolor(edge_color)
            bar.set_linewidth(lw)
        axes[1, col].set_xticks(x)
        axes[1, col].set_xticklabels(channels, rotation=45, ha='right')
        axes[1, col].set_ylabel("Mean beta power (dB)")
        axes[1, col].grid(True, linestyle='--', alpha=0.5)

    axes[-1, -1].set_xlabel("Channel")
    fig.tight_layout()
    fig.suptitle("Beta-band PSD and mean power per channel (raw epochs)", fontsize=16)
    fig.subplots_adjust(top=0.88)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "beta_activity_raw.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    logging.info("Saved plot to %s", plot_path)


def save_results(results: Dict[str, Dict[str, Dict[str, float]]], output_dir: Path, combined_path: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    standalone_path = output_dir / "beta_channel_selection.json"
    with open(standalone_path, 'w', encoding='utf-8') as fp:
        json.dump(results, fp, indent=2)
    logging.info("Saved beta selection summary to %s", standalone_path)

    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined = {}
    if combined_path.exists():
        with open(combined_path, 'r', encoding='utf-8') as fp:
            combined = json.load(fp)
    new_combined = {'beta_channel_selection': results}
    for key, value in combined.items():
        if key != 'beta_channel_selection':
            new_combined[key] = value
    with open(combined_path, 'w', encoding='utf-8') as fp:
        json.dump(new_combined, fp, indent=2)
    logging.info("Updated combined channel selection file at %s", combined_path)

def main():
    logging.basicConfig(level=logging.INFO)
    pickle_path = os.path.join("results", "pickles", "4646epochs_patients_epochs.pickle")
    output_dir = Path("results/beta_channel_selection")
    combined_json = Path("results/channel_selection_summary.json")

    results = analyze_beta_activity(pickle_path)
    save_results(results, output_dir, combined_json)
    plot_beta_activity(results, output_dir)


if __name__ == "__main__":
    main()
