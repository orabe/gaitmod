import json
import logging
import os
from pathlib import Path
import re
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
    if not subjects:
        logging.warning("No beta results available; skipping raw beta figure generation.")
        return

    max_cols = 3
    title_fontsize = 18
    subject_title_fontsize = 14
    axis_label_fontsize = 16
    tick_fontsize = 13
    mean_best_color = "#2BB673"
    mean_other_color = "#B7BDC7"
    mean_line_color = "#4B5563"
    mean_panel_bg = "#FAFAFA"

    channel_color_map = {}
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    available_colors = iter(color_cycle * 3)

    def get_channel_color(channel: str) -> str:
        if channel not in channel_color_map:
            channel_color_map[channel] = next(available_colors)
        return channel_color_map[channel]

    n_cols = min(max_cols, len(subjects))
    n_rows = int(np.ceil(len(subjects) / float(n_cols)))

    def _padded_limits(values: np.ndarray, frac: float = 0.08, min_pad: float = 0.5):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return -1.0, 1.0
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if np.isclose(vmin, vmax):
            return vmin - 1.0, vmax + 1.0
        pad = max(min_pad, frac * (vmax - vmin))
        return vmin - pad, vmax + pad

    beta_freq_psd_values = []
    mean_beta_values = []
    for subject in subjects:
        subj_data = results[subject]
        freqs = np.asarray(subj_data['psd_freqs'], dtype=float)
        beta_mask = (freqs >= BETA_BAND[0]) & (freqs <= BETA_BAND[1])
        for psd_vals in subj_data['psd_values']:
            psd_vals = np.asarray(psd_vals, dtype=float)
            if beta_mask.shape == psd_vals.shape and np.any(beta_mask):
                beta_freq_psd_values.extend(psd_vals[beta_mask].tolist())
            else:
                beta_freq_psd_values.extend(psd_vals.tolist())
        mean_beta_values.extend(list(subj_data['beta_scores'].values()))

    psd_ymin, psd_ymax = _padded_limits(np.asarray(beta_freq_psd_values, dtype=float))
    mean_ymin, mean_ymax = _padded_limits(np.asarray(mean_beta_values, dtype=float))
    global_channel_order = []
    for subject in subjects:
        for channel in results[subject].get('channels', []):
            if channel not in global_channel_order:
                global_channel_order.append(channel)

    fig_psd, axes_psd = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.2 * n_cols, 4.6 * n_rows),
        squeeze=False,
    )
    axes_psd = axes_psd.ravel()

    for idx, subject in enumerate(subjects):
        ax = axes_psd[idx]
        subj_data = results[subject]
        freqs = np.asarray(subj_data['psd_freqs'], dtype=float)
        beta_mask = (freqs >= BETA_BAND[0]) & (freqs <= BETA_BAND[1])
        plot_freqs = freqs[beta_mask] if np.any(beta_mask) else freqs

        for ch_name, psd_vals in zip(subj_data['channels'], subj_data['psd_values']):
            color = get_channel_color(ch_name)
            psd_arr = np.asarray(psd_vals, dtype=float)
            if beta_mask.shape == psd_arr.shape and np.any(beta_mask):
                psd_arr = psd_arr[beta_mask]
            is_best = ch_name == subj_data['best_channel']
            ax.plot(
                plot_freqs,
                psd_arr,
                color=color,
                linewidth=2.8 if is_best else 1.5,
                alpha=0.95 if is_best else 0.75,
                zorder=3 if is_best else 1,
            )

        ax.set_title(subject, fontsize=subject_title_fontsize, fontweight='bold')
        ax.set_xlim(*BETA_BAND)
        ax.set_ylim(psd_ymin, psd_ymax)
        ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.4)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.set_xlabel("Frequency (Hz)", fontsize=axis_label_fontsize)
        if idx % n_cols == 0:
            ax.set_ylabel("Beta power (dB)", fontsize=axis_label_fontsize)

    for idx in range(len(subjects), len(axes_psd)):
        fig_psd.delaxes(axes_psd[idx])

    fig_psd.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    fig_psd.suptitle(
        "Beta Power Spectra by Subject (13-30 Hz)",
        fontsize=title_fontsize,
        fontweight='bold',
    )

    fig_mean, axes_mean = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.2 * n_cols, 4.8 * n_rows),
        squeeze=False,
    )
    axes_mean = axes_mean.ravel()

    for idx, subject in enumerate(subjects):
        ax = axes_mean[idx]
        subj_data = results[subject]
        channel_scores = subj_data['beta_scores']
        ordered_channels = list(global_channel_order)
        ordered_values = np.asarray([channel_scores.get(ch, np.nan) for ch in ordered_channels], dtype=float)
        x = np.arange(len(global_channel_order))
        best_channel = subj_data['best_channel']
        if best_channel in ordered_channels:
            best_idx = ordered_channels.index(best_channel)
        else:
            finite_idx = np.where(np.isfinite(ordered_values))[0]
            best_idx = int(finite_idx[np.argmax(ordered_values[finite_idx])]) if finite_idx.size else 0

        bar_colors = [mean_best_color if i == best_idx else mean_other_color for i in range(len(ordered_channels))]
        ax.set_facecolor(mean_panel_bg)
        ax.bar(
            x,
            ordered_values,
            color=bar_colors,
            alpha=0.9,
            edgecolor='white',
            linewidth=0.8,
            zorder=2,
        )
        ax.plot(
            x,
            ordered_values,
            color=mean_line_color,
            linewidth=1.8,
            marker='o',
            markersize=4.2,
            markerfacecolor='white',
            markeredgecolor=mean_line_color,
            markeredgewidth=1.1,
            zorder=3,
        )
        ax.scatter(
            [best_idx],
            [ordered_values[best_idx]],
            color=mean_best_color,
            edgecolors='black',
            linewidths=1.4,
            s=120,
            zorder=4,
        )

        ax.text(
            0.02,
            0.96,
            subject,
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=subject_title_fontsize,
            fontweight='normal',
            color=mean_line_color,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([_clean_channel_label(ch) for ch in ordered_channels], rotation=30, ha='right')
        ax.set_ylim(mean_ymin, mean_ymax)
        ax.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.45, zorder=1)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.1)
        ax.spines['bottom'].set_linewidth(1.1)
        ax.set_xlabel("Channel", fontsize=axis_label_fontsize)
        if idx % n_cols == 0:
            ax.set_ylabel("Mean beta power (dB)", fontsize=axis_label_fontsize)

    for idx in range(len(subjects), len(axes_mean)):
        fig_mean.delaxes(axes_mean[idx])

    fig_mean.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])

    output_dir.mkdir(parents=True, exist_ok=True)
    psd_plot_path = output_dir / "beta_power_db_raw.png"
    mean_plot_path = output_dir / "mean_beta_power_raw.png"
    fig_psd.savefig(psd_plot_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    fig_mean.savefig(mean_plot_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig_psd)
    plt.close(fig_mean)
    logging.info("Saved beta PSD plot to %s", psd_plot_path)
    logging.info("Saved mean beta power plot to %s", mean_plot_path)


def _clean_channel_label(channel: str) -> str:
    label = str(channel)
    label = re.sub(r"(?i)LFP[_-]*", "", label)
    label = re.sub(r"__+", "_", label)
    label = label.strip(" _-")
    return label or str(channel)


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
    pickle_path = os.path.join("results", "pickles", "6296epochs_patients_epochs.pickle")
    output_dir = Path("results/beta_channel_selection")
    combined_json = Path("results/channel_selection_summary.json")

    results = analyze_beta_activity(pickle_path)
    save_results(results, output_dir, combined_json)
    plot_beta_activity(results, output_dir)


if __name__ == "__main__":
    main()
