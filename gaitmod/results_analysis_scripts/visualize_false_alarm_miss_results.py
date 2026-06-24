#!/usr/bin/env python3
"""
Create visualizations for false alarms/misses results.

Outputs:
1) Per-subject grouped bar plot for false alarms/misses rates.
2) Per-trial distributions per subject (box + jitter) for both rates.
3) Per-trial distributions per subject (box + jitter) for inverse metrics
   (unit-adaptive *_per_false_alarm, *_per_miss).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler

PUBLICATION_DPI = 600
FANCY_PALETTE = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#2E91E5",
    "#00A6A6",
    "#8E6C8A",
    "#F2A104",
]


def _resolve_existing_path(candidates: List[Path], label: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    checked = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(f"Missing {label}. Checked:\n{checked}")


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.linewidth": 1.3,
            "lines.linewidth": 2.0,
            "savefig.dpi": PUBLICATION_DPI,
            "axes.prop_cycle": cycler(color=FANCY_PALETTE),
        }
    )


def infer_rate_columns(df_subject: pd.DataFrame) -> Tuple[str, str, str]:
    fa_candidates = [
        c
        for c in df_subject.columns
        if c.startswith("false_alarms_per_") and not c.endswith("_mean_subject")
    ]
    if not fa_candidates:
        raise ValueError("Could not find a false alarm rate column in subject CSV.")

    for fa_col in sorted(fa_candidates):
        unit = fa_col.replace("false_alarms_per_", "")
        miss_col = f"misses_per_{unit}"
        if miss_col in df_subject.columns:
            return fa_col, miss_col, unit

    raise ValueError("Could not find matching false alarm/miss rate columns in subject CSV.")


def unit_plural(unit: str) -> str:
    mapping = {"second": "seconds", "minute": "minutes", "hour": "hours"}
    if unit not in mapping:
        raise ValueError(f"Unsupported unit: {unit}")
    return mapping[unit]


def load_trial_dataframe(per_trial_json: Path) -> pd.DataFrame:
    payload = json.loads(per_trial_json.read_text(encoding="utf-8"))
    rows: List[Dict[str, float]] = []
    for subject, subject_blob in payload.get("subjects", {}).items():
        for trial in subject_blob.get("trials", []):
            row = {"subject": subject}
            row.update(trial)
            rows.append(row)
    if not rows:
        raise ValueError(f"No trial rows found in {per_trial_json}")
    return pd.DataFrame(rows)


def _subject_colors(subjects: List[str]) -> Dict[str, str]:
    colors = {}
    for i, subject in enumerate(subjects):
        colors[subject] = FANCY_PALETTE[i % len(FANCY_PALETTE)]
    return colors


def plot_subject_rates(
    df_subject: pd.DataFrame,
    fa_col: str,
    miss_col: str,
    unit: str,
    out_path: Path,
) -> None:
    plot_df = df_subject.copy()
    plot_df["combined"] = plot_df[fa_col] + plot_df[miss_col]
    plot_df = plot_df.sort_values("combined", ascending=False).reset_index(drop=True)

    x = np.arange(len(plot_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    ax.bar(x - width / 2, plot_df[fa_col], width, label=fa_col, color=FANCY_PALETTE[0])
    ax.bar(x + width / 2, plot_df[miss_col], width, label=miss_col, color=FANCY_PALETTE[1])

    # Show each subject's covered duration to contextualize rates.
    for i, (_, row) in enumerate(plot_df.iterrows()):
        y_top = max(row[fa_col], row[miss_col])
        ax.text(
            i,
            y_top + (0.02 * max(1.0, float(plot_df["combined"].max()))),
            f"{row['covered_duration_seconds']:.1f}s",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=90,
            alpha=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["subject"], rotation=25, ha="right")
    ax.set_ylabel(f"Events per {unit}")
    ax.set_xlabel("Subject")
    ax.set_title("Per-Subject False Alarm and Miss Rates")
    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.8, axis="y")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_trial_distributions(
    df_trials: pd.DataFrame,
    fa_col: str,
    miss_col: str,
    unit: str,
    out_path: Path,
) -> None:
    subjects = sorted(df_trials["subject"].unique().tolist())
    positions = np.arange(1, len(subjects) + 1)
    offset = 0.11
    width = 0.20
    fa_color = FANCY_PALETTE[0]
    miss_color = FANCY_PALETTE[1]

    fig, ax = plt.subplots(figsize=(16.0, 8.4))
    fa_data = [df_trials.loc[df_trials["subject"] == s, fa_col].to_numpy() for s in subjects]
    miss_data = [df_trials.loc[df_trials["subject"] == s, miss_col].to_numpy() for s in subjects]

    bp_fa = ax.boxplot(
        fa_data,
        positions=positions - offset,
        widths=width,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.3},
    )
    for patch in bp_fa["boxes"]:
        patch.set_facecolor(fa_color)
        patch.set_alpha(0.35)
        patch.set_linewidth(1.1)

    bp_miss = ax.boxplot(
        miss_data,
        positions=positions + offset,
        widths=width,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.3},
    )
    for patch in bp_miss["boxes"]:
        patch.set_facecolor(miss_color)
        patch.set_alpha(0.35)
        patch.set_linewidth(1.1)

    rng = np.random.default_rng(7)
    for i, subject in enumerate(subjects, start=1):
        fa_vals = df_trials.loc[df_trials["subject"] == subject, fa_col].to_numpy()
        miss_vals = df_trials.loc[df_trials["subject"] == subject, miss_col].to_numpy()
        fa_jitter = rng.uniform(-0.04, 0.04, size=len(fa_vals))
        miss_jitter = rng.uniform(-0.04, 0.04, size=len(miss_vals))
        ax.scatter(
            np.full_like(fa_vals, i - offset, dtype=float) + fa_jitter,
            fa_vals,
            s=14,
            alpha=0.45,
            color=fa_color,
            edgecolors="none",
        )
        ax.scatter(
            np.full_like(miss_vals, i + offset, dtype=float) + miss_jitter,
            miss_vals,
            s=14,
            alpha=0.45,
            color=miss_color,
            edgecolors="none",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(subjects, rotation=30, ha="right")
    ax.set_ylabel(f"Events per {unit}")
    ax.set_xlabel("Subject")
    ax.set_title("Per-Trial Rate Distributions by Subject (Overlay)")
    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.8, axis="y")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=fa_color, lw=8, alpha=0.45, label=fa_col),
            plt.Line2D([0], [0], color=miss_color, lw=8, alpha=0.45, label=miss_col),
        ],
        loc="upper right",
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_inverse_trial_distributions(
    df_trials: pd.DataFrame,
    fa_col: str,
    miss_col: str,
    interval_unit_label: str,
    out_path: Path,
) -> None:
    subjects = sorted(df_trials["subject"].unique().tolist())
    positions = np.arange(1, len(subjects) + 1)
    offset = 0.11
    width = 0.20
    fa_color = FANCY_PALETTE[0]
    miss_color = FANCY_PALETTE[1]
    fig, ax = plt.subplots(figsize=(16.0, 8.4))
    fa_data = [
        df_trials.loc[df_trials["subject"] == s, fa_col].dropna().to_numpy() for s in subjects
    ]
    miss_data = [
        df_trials.loc[df_trials["subject"] == s, miss_col].dropna().to_numpy() for s in subjects
    ]

    bp_fa = ax.boxplot(
        fa_data,
        positions=positions - offset,
        widths=width,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.3},
    )
    for patch in bp_fa["boxes"]:
        patch.set_facecolor(fa_color)
        patch.set_alpha(0.35)
        patch.set_linewidth(1.1)

    bp_miss = ax.boxplot(
        miss_data,
        positions=positions + offset,
        widths=width,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.3},
    )
    for patch in bp_miss["boxes"]:
        patch.set_facecolor(miss_color)
        patch.set_alpha(0.35)
        patch.set_linewidth(1.1)

    rng = np.random.default_rng(11)
    for i, subject in enumerate(subjects, start=1):
        fa_vals = df_trials.loc[df_trials["subject"] == subject, fa_col].dropna().to_numpy()
        miss_vals = df_trials.loc[df_trials["subject"] == subject, miss_col].dropna().to_numpy()
        fa_jitter = rng.uniform(-0.04, 0.04, size=len(fa_vals))
        miss_jitter = rng.uniform(-0.04, 0.04, size=len(miss_vals))
        ax.scatter(
            np.full_like(fa_vals, i - offset, dtype=float) + fa_jitter,
            fa_vals,
            s=14,
            alpha=0.45,
            color=fa_color,
            edgecolors="none",
        )
        ax.scatter(
            np.full_like(miss_vals, i + offset, dtype=float) + miss_jitter,
            miss_vals,
            s=14,
            alpha=0.45,
            color=miss_color,
            edgecolors="none",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(subjects, rotation=30, ha="right")
    ax.set_ylabel(f"{interval_unit_label.capitalize()} between events")
    ax.set_xlabel("Subject")
    ax.set_title("Per-Trial Inverse Metrics by Subject (Overlay)")
    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.8, axis="y")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=fa_color, lw=8, alpha=0.45, label=fa_col),
            plt.Line2D([0], [0], color=miss_color, lw=8, alpha=0.45, label=miss_col),
        ],
        loc="upper right",
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # -------------------------------------------------------------------------
    # User-editable settings (no CLI).
    # -------------------------------------------------------------------------
    model_name = "Seq2SeqCNNLSTM_raw_betaChs"
    repo_root = Path(__file__).resolve().parents[2]
    logs_root = repo_root / "logs" / model_name
    summary_dir = logs_root / "summary"
    subject_csv = summary_dir / "false_alarm_miss_rates_by_subject.csv"
    per_trial_json = summary_dir / "false_alarm_miss_rates_per_trial_per_subject.json"
    # -------------------------------------------------------------------------

    # Resolve files robustly across common layouts and working directories.
    subject_candidates = [
        subject_csv,
        repo_root / "logs" / "results" / model_name / "summary" / subject_csv.name,
        Path.cwd() / "logs" / model_name / "summary" / subject_csv.name,
        Path.cwd() / "logs" / "results" / model_name / "summary" / subject_csv.name,
    ]
    trial_candidates = [
        per_trial_json,
        repo_root / "logs" / "results" / model_name / "summary" / per_trial_json.name,
        Path.cwd() / "logs" / model_name / "summary" / per_trial_json.name,
        Path.cwd() / "logs" / "results" / model_name / "summary" / per_trial_json.name,
    ]

    subject_csv = _resolve_existing_path(subject_candidates, "subject CSV")
    per_trial_json = _resolve_existing_path(trial_candidates, "per-trial JSON")
    summary_dir = subject_csv.parent

    apply_publication_style()

    df_subject = pd.read_csv(subject_csv)
    df_trials = load_trial_dataframe(per_trial_json)
    fa_col, miss_col, unit = infer_rate_columns(df_subject)

    if fa_col not in df_trials.columns or miss_col not in df_trials.columns:
        raise ValueError(
            f"Trial JSON does not contain expected columns '{fa_col}' and '{miss_col}'."
        )
    if "covered_duration_seconds" not in df_trials.columns:
        raise ValueError("Trial JSON is missing 'covered_duration_seconds'.")
    interval_unit = unit_plural(unit)
    inverse_fa_col = f"{interval_unit}_per_false_alarm"
    inverse_miss_col = f"{interval_unit}_per_miss"
    if inverse_fa_col not in df_trials.columns or inverse_miss_col not in df_trials.columns:
        # Backward compatibility with older outputs that used seconds.
        inverse_fa_col = "seconds_per_false_alarm"
        inverse_miss_col = "seconds_per_miss"
    for inverse_col in (inverse_fa_col, inverse_miss_col):
        if inverse_col not in df_trials.columns:
            raise ValueError(f"Trial JSON is missing '{inverse_col}'.")

    out_subject = summary_dir / f"false_alarm_miss_rates_subject_bars_per_{unit}.png"
    out_dist = summary_dir / f"false_alarm_miss_rates_trial_distributions_per_{unit}.png"
    out_inverse = summary_dir / f"false_alarm_miss_inverse_trial_distributions_{interval_unit}.png"

    plot_subject_rates(df_subject, fa_col, miss_col, unit, out_subject)
    plot_trial_distributions(df_trials, fa_col, miss_col, unit, out_dist)
    plot_inverse_trial_distributions(
        df_trials=df_trials,
        fa_col=inverse_fa_col,
        miss_col=inverse_miss_col,
        interval_unit_label=interval_unit,
        out_path=out_inverse,
    )

    print(f"Saved: {out_subject}")
    print(f"Saved: {out_dist}")
    print(f"Saved: {out_inverse}")


if __name__ == "__main__":
    main()
