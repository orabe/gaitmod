import json
import logging
import os
from pathlib import Path
import re
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
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

        # Beta line subplot
        channel_scores = subj_data['beta_scores']
        channels = list(channel_scores.keys())
        values = np.asarray([channel_scores[ch] for ch in channels], dtype=float)
        x = np.arange(len(channels))
        axes[1, col].plot(x, values, color='dimgray', linewidth=2.0, alpha=0.85, zorder=1)
        for idx_ch, ch_name in enumerate(channels):
            point_color = get_channel_color(ch_name)
            is_best = ch_name == subj_data['best_channel']
            axes[1, col].scatter(
                x[idx_ch],
                values[idx_ch],
                color=point_color,
                edgecolors='black' if is_best else point_color,
                linewidths=1.8 if is_best else 0.9,
                s=70 if is_best else 35,
                zorder=3,
            )
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


def _clean_channel_label(channel: str) -> str:
    label = str(channel)
    label = re.sub(r"(?i)LFP[_-]*", "", label)
    label = re.sub(r"__+", "_", label)
    label = label.strip(" _-")
    return label or str(channel)


def _radar_factory(num_vars: int):
    """Create a polygon-frame radar projection and return (theta, projection_name)."""
    theta = np.linspace(0.0, 2.0 * np.pi, num_vars, endpoint=False)
    projection_name = f"radar_{num_vars}"
    unit_vertices = np.column_stack([np.cos(theta), np.sin(theta)])

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return MplPath(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = projection_name
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("E")

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(*args, closed=closed, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                x_data, y_data = line.get_data()
                if len(x_data) > 0 and x_data[0] != x_data[-1]:
                    x_data = np.append(x_data, x_data[0])
                    y_data = np.append(y_data, y_data[0])
                    line.set_data(x_data, y_data)
            return lines

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            verts_axes = np.column_stack([
                0.5 + 0.5 * unit_vertices[:, 0],
                0.5 + 0.5 * unit_vertices[:, 1],
            ])
            return Polygon(verts_axes, closed=True, edgecolor='gray', facecolor='none')

        def _gen_axes_spines(self):
            spine_path = MplPath(np.vstack([unit_vertices, unit_vertices[0]]))
            spine = Spine(axes=self, spine_type='circle', path=spine_path)
            spine.set_transform(Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta, projection_name


def plot_beta_activity_radar_per_subject(results: Dict[str, Dict[str, Dict[str, float]]], output_dir: Path):
    subjects = sorted(results.keys())
    if not subjects:
        logging.warning("No beta results available; skipping per-subject radar chart generation.")
        return

    channel_order = []
    for subject in subjects:
        for channel in results[subject].get('channels', []):
            if channel not in channel_order:
                channel_order.append(channel)
    if not channel_order:
        logging.warning("No channels found in beta results; skipping per-subject radar chart generation.")
        return

    all_scores = []
    for subject in subjects:
        all_scores.extend(results[subject].get('beta_scores', {}).values())
    finite_scores = np.asarray([x for x in all_scores if np.isfinite(x)], dtype=float)
    if finite_scores.size == 0:
        logging.warning("No finite beta scores found; skipping per-subject radar chart generation.")
        return

    radial_min = float(np.min(finite_scores))
    radial_max = float(np.max(finite_scores))
    if np.isclose(radial_min, radial_max):
        radial_min -= 1.0
        radial_max += 1.0
    else:
        pad = 0.1 * (radial_max - radial_min)
        radial_min -= pad
        radial_max += pad

    n_channels = len(channel_order)
    angles, projection_name = _radar_factory(n_channels)
    angles_closed = np.concatenate([angles, angles[:1]])
    channel_labels = [_clean_channel_label(ch) for ch in channel_order]

    n_cols = min(4, max(1, len(subjects)))
    n_rows = int(np.ceil(len(subjects) / float(n_cols)))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4.5 * n_rows),
        subplot_kw={'projection': projection_name},
    )
    axes = np.atleast_1d(axes).ravel()
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    radial_ticks = np.linspace(radial_min, radial_max, 5)

    for idx, subject in enumerate(subjects):
        ax = axes[idx]
        color = color_cycle[idx % len(color_cycle)]
        subject_scores = results[subject].get('beta_scores', {})
        values = np.asarray([subject_scores.get(ch, np.nan) for ch in channel_order], dtype=float)
        values = np.where(np.isfinite(values), values, radial_min)
        values_closed = np.concatenate([values, values[:1]])

        best_channel = results[subject].get('best_channel')
        best_idx = channel_order.index(best_channel) if best_channel in channel_order else int(np.argmax(values))

        ax.plot(angles_closed, values_closed, color=color, linewidth=2.0, alpha=0.95)
        ax.fill(angles_closed, values_closed, color=color, alpha=0.12)
        ax.scatter([angles[best_idx]], [values[best_idx]], color=color, edgecolors='black', linewidths=0.7, s=40, zorder=5)

        ax.set_rlabel_position(0)  # Radial value labels on horizontal axis.
        ax.set_varlabels(channel_labels)
        ax.tick_params(axis='x', pad=14)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
            label.set_clip_on(False)
        ax.set_yticks(radial_ticks)
        ax.set_yticklabels([f"{tick:.0f}" for tick in radial_ticks], fontsize=8)
        ax.tick_params(axis='y', pad=4)
        for label in ax.get_yticklabels():
            label.set_rotation(0)
            label.set_horizontalalignment('left')
            label.set_verticalalignment('center')
        ax.set_ylim(radial_min, radial_max)
        ax.grid(True, linestyle='--', alpha=0.55)
        ax.set_title(f"{subject}\nBest: {_clean_channel_label(best_channel)}", fontsize=10, va='bottom')

    for idx in range(len(subjects), len(axes)):
        fig.delaxes(axes[idx])

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.suptitle("Beta-Band Power Spiderweb (One Plot Per Subject)", fontsize=16)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.9, bottom=0.06, wspace=0.35, hspace=0.45)
    plot_path = output_dir / "beta_activity_radar_per_subject.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)
    logging.info("Saved per-subject radar chart to %s", plot_path)


def plot_beta_activity_radar(results: Dict[str, Dict[str, Dict[str, float]]], output_dir: Path):
    subjects = sorted(results.keys())
    if not subjects:
        logging.warning("No beta results available; skipping radar chart generation.")
        return

    channel_order = []
    for subject in subjects:
        for channel in results[subject].get('channels', []):
            if channel not in channel_order:
                channel_order.append(channel)
    if not channel_order:
        logging.warning("No channels found in beta results; skipping radar chart generation.")
        return

    all_scores = []
    for subject in subjects:
        all_scores.extend(results[subject].get('beta_scores', {}).values())
    finite_scores = np.asarray([x for x in all_scores if np.isfinite(x)], dtype=float)
    if finite_scores.size == 0:
        logging.warning("No finite beta scores found; skipping radar chart generation.")
        return

    radial_min = float(np.min(finite_scores))
    radial_max = float(np.max(finite_scores))
    if np.isclose(radial_min, radial_max):
        radial_min -= 1.0
        radial_max += 1.0
    else:
        pad = 0.1 * (radial_max - radial_min)
        radial_min -= pad
        radial_max += pad

    n_channels = len(channel_order)
    angles, projection_name = _radar_factory(n_channels)
    angles_closed = np.concatenate([angles, angles[:1]])
    channel_labels = [_clean_channel_label(ch) for ch in channel_order]
    fig, ax = plt.subplots(
        figsize=(9, 8),
        subplot_kw={'projection': projection_name},
    )
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    radial_ticks = np.linspace(radial_min, radial_max, 6)
    for idx, subject in enumerate(subjects):
        color = color_cycle[idx % len(color_cycle)]
        subject_scores = results[subject].get('beta_scores', {})
        values = np.asarray([subject_scores.get(ch, np.nan) for ch in channel_order], dtype=float)
        values = np.where(np.isfinite(values), values, radial_min)
        values_closed = np.concatenate([values, values[:1]])

        best_channel = results[subject].get('best_channel')
        if best_channel in channel_order:
            best_idx = channel_order.index(best_channel)
        else:
            best_idx = int(np.argmax(values))

        ax.plot(angles_closed, values_closed, color=color, linewidth=2.0, alpha=0.9, label=subject)
        ax.fill(angles_closed, values_closed, color=color, alpha=0.08)
        ax.scatter(
            [angles[best_idx]],
            [values[best_idx]],
            color=color,
            edgecolors='black',
            linewidths=0.7,
            s=40,
            zorder=5,
        )

    ax.set_rlabel_position(0)  # Radial value labels on horizontal axis.
    ax.set_varlabels(channel_labels)
    ax.tick_params(axis='x', pad=14)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_clip_on(False)
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f"{tick:.0f}" for tick in radial_ticks], fontsize=9)
    ax.tick_params(axis='y', pad=4)
    for label in ax.get_yticklabels():
        label.set_rotation(0)
        label.set_horizontalalignment('left')
        label.set_verticalalignment('center')
    ax.set_ylim(radial_min, radial_max)
    ax.grid(True, linestyle='--', alpha=0.55)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.suptitle("Beta-Band Power Spiderweb (All Subjects Overlay)", fontsize=16)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.05), frameon=True)
    fig.subplots_adjust(left=0.08, right=0.8, top=0.9, bottom=0.08)
    plot_path = output_dir / "beta_activity_radar.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)
    logging.info("Saved radar chart to %s", plot_path)


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
    plot_beta_activity_radar_per_subject(results, output_dir)
    plot_beta_activity_radar(results, output_dir)


if __name__ == "__main__":
    main()
