#!/usr/bin/env python3
"""
Plot selected-feature matrices after running the feature-selection pipeline.

What this script does:
- Loads HCTSA data for preferred subject-channel mappings.
- Runs `FeatureSelector` over a parameter grid (same style as
  `02_report_hctsa_correlation_filter.py`).
- For each valid parameter combination, plots the selected-feature matrix using
  the same visualization functions from `08_plot_feature_matrix.py`.

Required input:
- HCTSA data root directory (default: `data/hctsa`).
- Channel mapping from `CHANNEL_METHODS` in this file.
- Grid-search settings in `main()`:
  `variance_thresholds`, `selection_methods`, `correlation_thresholds`,
  `n_features_list`.

Generated output:
- One matrix figure per successful parameter combination in:
  `results/hctsa_segments_datamatrix/`
- Optional colorGroups-style figure (if enabled) per combination.
- A run log:
  `results/hctsa_segments_datamatrix/feature_matrix_after_fs.log`
"""
from __future__ import annotations

import importlib.util
import logging
import re
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from gaitmod.feature_selection import FeatureSelector
from gaitmod.preprocessing.hctsa_segments import parse_segment_identifier
from gaitmod.utils.utils import load_hctsa_data


# Numbered module filename cannot be imported via normal dotted import.
_PLOT_MODULE_PATH = Path(__file__).with_name("08_plot_feature_matrix.py")
_PLOT_SPEC = importlib.util.spec_from_file_location("stats_plot_feature_matrix", _PLOT_MODULE_PATH)
if _PLOT_SPEC is None or _PLOT_SPEC.loader is None:
    raise ImportError(f"Failed to load plotting module from {_PLOT_MODULE_PATH}")
_PLOT_MODULE = importlib.util.module_from_spec(_PLOT_SPEC)
_PLOT_SPEC.loader.exec_module(_PLOT_MODULE)

plot_data_matrix = _PLOT_MODULE.plot_data_matrix
plot_data_matrix_color_groups = _PLOT_MODULE.plot_data_matrix_color_groups
robust_sigmoid_0_1 = _PLOT_MODULE._robust_sigmoid_0_1
minmax_scale_0_1 = _PLOT_MODULE._minmax_scale_0_1


CHANNEL_METHODS = {
    "beta": {
        "PW_EM59": "channel_2",
        "PW_FH57": "channel_2",
        "PW_HK59": "channel_2",
        "PW_HZ58": "channel_2",
        "PW_SN61": "channel_2",
        "PW_SN66": "channel_5",
        "PW_US68": "channel_1",
    },
    "logRegF1": {
        "PW_EM59": "channel_0",
        "PW_FH57": "channel_1",
        "PW_HK59": "channel_0",
        "PW_HZ58": "channel_1",
        "PW_SN61": "channel_2",
        "PW_SN66": "channel_0",
        "PW_US68": "channel_4",
    },
}


def _canonical_channel_label(channel_name: str | None) -> str | None:
    if channel_name is None:
        return None
    channel_name = str(channel_name)
    match = re.match(r"(channel_\d+)", channel_name)
    if match:
        return match.group(1)
    if channel_name.startswith("ch") and channel_name[2:].isdigit():
        return f"channel_{channel_name[2:]}"
    if channel_name.isdigit():
        return f"channel_{channel_name}"
    return channel_name


def _resolve_channel_dir(data_root: Path, channel_label: str) -> Path:
    candidates = sorted(data_root.glob(f"{channel_label}*"))
    if not candidates:
        raise SystemExit(f"No HCTSA directory found for channel {channel_label} under {data_root}")
    return candidates[0]


def _load_preferred_channel_data(
    data_root: Path,
    preferred_map: dict[str, str],
    variant: str,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    ts_parts: list[pd.DataFrame] = []
    operations_ref = None

    for channel_label in sorted({_canonical_channel_label(v) for v in preferred_map.values()}):
        if channel_label is None:
            continue
        channel_dir = _resolve_channel_dir(data_root, channel_label)
        ts_data_mat, timeseries, operations, labels = load_hctsa_data(
            str(channel_dir), data_variant=variant, verbose=False
        )
        if operations_ref is None:
            operations_ref = operations
        elif len(operations_ref) != len(operations):
            raise SystemExit("Operations metadata mismatch across channels.")

        labels = np.asarray(labels)
        subject_mask = []
        for name in timeseries["Name"].astype(str):
            parsed = parse_segment_identifier(name)
            subject = parsed.get("subject")
            preferred_raw = preferred_map.get(subject)
            if not preferred_raw:
                raise SystemExit(f"No preferred channel mapping for subject: {subject}")
            preferred_channel = _canonical_channel_label(preferred_raw)
            subject_mask.append(preferred_channel == channel_label)
        subject_mask = np.asarray(subject_mask, dtype=bool)
        if not np.any(subject_mask):
            continue

        x_parts.append(ts_data_mat[subject_mask])
        y_parts.append(labels[subject_mask])
        ts_parts.append(timeseries.loc[subject_mask].reset_index(drop=True))

    if not x_parts or operations_ref is None:
        raise SystemExit("No samples matched preferred channel mapping.")

    x = np.vstack(x_parts)
    y = np.concatenate(y_parts)
    timeseries_all = pd.concat(ts_parts, ignore_index=True)
    return x, timeseries_all, operations_ref.reset_index(drop=True), y


def _parse_row_metadata(timeseries_df: pd.DataFrame) -> pd.DataFrame:
    parsed = [parse_segment_identifier(name) for name in timeseries_df["Name"].astype(str)]
    meta = pd.DataFrame(parsed)
    meta["name"] = timeseries_df["Name"].astype(str).to_numpy()
    group_series = (
        timeseries_df["Group"].astype(str) if "Group" in timeseries_df.columns
        else pd.Series([""] * len(timeseries_df), dtype="string")
    )
    rename = {
        "gait_modulation": "Gait Modulation",
        "normal_walking": "Steady-State Walking",
    }
    meta["group"] = group_series.apply(lambda g: rename.get(str(g).strip().lower(), str(g))).to_numpy()
    return meta


def main() -> None:
    # ------------------------------------------------------------------
    # Fixed configuration (edit here for your permutations)
    # ------------------------------------------------------------------
    data_root = Path("data/hctsa")
    variant = ""
    channel_method = "beta"  # "beta" or "logRegF1"

    # Parameter grid (same style as script 02)
    variance_thresholds = [0.0001]
    selection_methods = [
        # "anova",
        "mann_whitney",
        # "roc_auc",
        # "pr_auc",
        # "cliffs_delta",
        # "brunner_munzel",
        # "mutual_info",
    ]
    correlation_thresholds = [0.3]
    n_features_list = [20]

    # Plot configuration
    normalize_0_1 = True
    normalization_method = "robust_sigmoid"  # "robust_sigmoid" | "minmax"
    discrete_step = 0.1  # None for continuous colormap
    feature_tick_step = None  # show feature IDs every N columns (None disables)
    show_group_strip = False
    show_legend = False
    cluster_rows = True
    cluster_cols = True
    show_feature_names = True
    save_color_groups_figure = True

    output_dir = Path("results/hctsa_segments_datamatrix")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "feature_matrix_after_fs.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_file, "w"), logging.StreamHandler()],
    )

    preferred_map = CHANNEL_METHODS.get(channel_method, {})
    if not preferred_map:
        raise SystemExit(f"No subjects found for channel method '{channel_method}'.")
    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}")

    logging.info("Loading HCTSA data...")
    x, timeseries_df, operations_df, y = _load_preferred_channel_data(
        data_root=data_root,
        preferred_map=preferred_map,
        variant=variant,
    )
    logging.info("Loaded data: samples=%d, features=%d", x.shape[0], x.shape[1])

    # Discard invalid features (NaN/Inf in any sample) to match script 02 behavior.
    valid_mask = np.isfinite(x).all(axis=0)
    x = x[:, valid_mask]
    operations_df = operations_df.iloc[valid_mask].reset_index(drop=True)
    logging.info("After invalid-feature filtering: samples=%d, features=%d", x.shape[0], x.shape[1])

    meta_all = _parse_row_metadata(timeseries_df)

    param_combinations = list(product(
        variance_thresholds,
        selection_methods,
        n_features_list,
        correlation_thresholds,
    ))
    logging.info("\n%s", "=" * 90)
    logging.info("Plotting selected matrices for %d parameter combinations", len(param_combinations))
    logging.info("%s\n", "=" * 90)

    processed = 0
    for idx, (var_thr, method, n_feat, ct) in enumerate(param_combinations, 1):
        logging.info(
            "[%d/%d] variance_threshold=%s, method=%s, n_features=%s, corr_threshold=%s",
            idx,
            len(param_combinations),
            var_thr,
            method,
            n_feat,
            ct,
        )

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="invalid value encountered in divide")
                selector = FeatureSelector(
                    n_features=int(n_feat),
                    variance_threshold=float(var_thr),
                    correlation_threshold=float(ct),
                    selection_method=str(method),
                    enabled=True,
                )
                selector.fit(x, y)

            selected_indices = selector.selected_features_ or []
            if not selected_indices:
                logging.warning("  Skipped (no selected features).")
                continue

            x_sel = x[:, selected_indices]
            operations_sel = operations_df.iloc[selected_indices].reset_index(drop=True)

            if normalize_0_1:
                method = (normalization_method or "robust_sigmoid").strip().lower()
                if method == "robust_sigmoid":
                    x_plot = robust_sigmoid_0_1(x_sel, axis=0)
                elif method == "minmax":
                    x_plot = minmax_scale_0_1(x_sel, axis=0)
                else:
                    raise ValueError(f"Unknown normalization_method: {normalization_method}")
            else:
                x_plot = x_sel

            base_name = (
                f"{method}_var{var_thr}_nfeat{n_feat}_ct{ct}_selected_matrix"
            )
            title = (
                "HCTSA Matrix After Feature Selection | "
                f"method={method}, variance_threshold={var_thr}, "
                f"correlation_threshold={ct}, top_k={n_feat}, selected={x_sel.shape[1]}"
            )

            out_path = output_dir / f"{base_name}.png"
            plot_data_matrix(
                x_plot,
                meta_all,
                operations_sel,
                title=title,
                output_path=out_path,
                cluster_rows=bool(cluster_rows),
                cluster_cols=bool(cluster_cols),
                vmin=0.0 if normalize_0_1 else None,
                vmax=1.0 if normalize_0_1 else None,
                discrete_step=discrete_step if normalize_0_1 else None,
                feature_tick_step=feature_tick_step,
                show_group_strip=bool(show_group_strip),
                show_legend=bool(show_legend),
                show_feature_names=bool(show_feature_names),
            )
            logging.info("  Saved: %s", out_path.name)

            if save_color_groups_figure and normalize_0_1:
                out_path_groups = output_dir / f"{base_name}_colorGroups.png"
                plot_data_matrix_color_groups(
                    x_plot,
                    meta_all,
                    operations_sel,
                    title=title + " | colorGroups",
                    output_path=out_path_groups,
                    cluster_rows=bool(cluster_rows),
                    cluster_cols=bool(cluster_cols),
                    discrete_step=discrete_step if normalize_0_1 else None,
                    feature_tick_step=feature_tick_step,
                )
                logging.info("  Saved: %s", out_path_groups.name)

            processed += 1
        except Exception as exc:
            logging.error("  Failed: %s", exc)

    logging.info("\n%s", "=" * 90)
    logging.info("Done. Successful matrix plots: %d / %d", processed, len(param_combinations))
    logging.info("Figures saved to: %s", output_dir)
    logging.info("%s\n", "=" * 90)


if __name__ == "__main__":
    main()
