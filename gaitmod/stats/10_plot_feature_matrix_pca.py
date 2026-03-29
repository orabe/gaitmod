#!/usr/bin/env python3
"""
Plot PCA overview for the HCTSA feature matrix in a dedicated script.

What this script does:
- Loads beta-selected channel data from the HCTSA segment cache.
- Applies the same optional row/feature subsetting and normalization as matrix plotting.
- Generates one PCA overview figure containing:
  1) PC1 vs PC2 scatter colored by class labels.
  2) Explained-variance scree plot (individual + cumulative).

Required input:
- Segment cache directory (default: `data/hctsa_segments`).
- Beta channel-selection JSON
  (default: `results/beta_channel_selection/beta_channel_selection.json`).

Generated output:
- `results/hctsa_segments_datamatrix/datamatrix_beta_selected_pca_overview.png`
"""
from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


# Numbered module filename cannot be imported via normal dotted import.
_MATRIX_MODULE_PATH = Path(__file__).with_name("08_plot_feature_matrix.py")
_MATRIX_SPEC = importlib.util.spec_from_file_location("stats_plot_feature_matrix", _MATRIX_MODULE_PATH)
if _MATRIX_SPEC is None or _MATRIX_SPEC.loader is None:
    raise ImportError(f"Failed to load module from {_MATRIX_MODULE_PATH}")
_MATRIX_MODULE = importlib.util.module_from_spec(_MATRIX_SPEC)
_MATRIX_SPEC.loader.exec_module(_MATRIX_MODULE)

HCTSASegmentCache = _MATRIX_MODULE.HCTSASegmentCache
load_beta_best_channel_map = _MATRIX_MODULE._load_beta_best_channel_map
resolve_subject_channel_map = _MATRIX_MODULE._resolve_subject_channel_map
subset_rows = _MATRIX_MODULE._subset_rows
subset_features_by_variance = _MATRIX_MODULE._subset_features_by_variance
minmax_scale_0_1 = _MATRIX_MODULE._minmax_scale_0_1
robust_sigmoid_0_1 = _MATRIX_MODULE._robust_sigmoid_0_1
filter_invalid_features = _MATRIX_MODULE._filter_invalid_features

logger = logging.getLogger(__name__)

DEFAULT_GROUP_COLORS_HEX = ["#298c8c", "#f1a226"]


def _hex_to_rgb01(value: str) -> tuple[float, float, float]:
    s = value.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"Invalid hex color: {value}")
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return (r, g, b)


def _impute_for_pca(X: np.ndarray) -> np.ndarray:
    if not np.isnan(X).any():
        return X
    col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    out = np.array(X, copy=True)
    nan_mask = np.isnan(out)
    out[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    return out


def plot_pca_overview(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    output_path: Path,
    max_components: int = 20,
) -> None:
    """Plot PC1/PC2 class scatter and scree plot in one figure."""
    try:
        from sklearn.decomposition import PCA
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for PCA plotting.") from exc

    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels).ravel()
    if X.ndim != 2:
        raise ValueError("X must be 2D for PCA plotting.")
    if X.shape[0] != labels.shape[0]:
        raise ValueError("labels length must match number of rows in X.")

    X_pca = _impute_for_pca(X)
    X_pca = np.nan_to_num(X_pca, nan=0.0, posinf=0.0, neginf=0.0)

    n_comp = int(min(max(2, max_components), X_pca.shape[0], X_pca.shape[1]))
    pca = PCA(n_components=n_comp, random_state=42)
    X_emb = pca.fit_transform(X_pca)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    ax_scatter, ax_scree = axes

    unique_labels = np.unique(labels)
    if unique_labels.size <= 2:
        palette = [_hex_to_rgb01(c) for c in DEFAULT_GROUP_COLORS_HEX[: max(1, unique_labels.size)]]
    else:
        palette = [plt.get_cmap("tab10")(i % 10)[:3] for i in range(unique_labels.size)]

    label_name_map = {
        0: "Steady-State Walking",
        1: "Gait Modulation",
    }
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = label_name_map.get(int(lbl), f"Class {int(lbl)}")
        ax_scatter.scatter(
            X_emb[mask, 0],
            X_emb[mask, 1],
            s=18,
            alpha=0.68,
            color=palette[i],
            edgecolors="none",
            label=name,
        )

    ax_scatter.set_xlabel(f"PC1 ({explained[0] * 100:.2f}\\%)", fontsize=12)
    ax_scatter.set_ylabel(f"PC2 ({explained[1] * 100:.2f}\\%)", fontsize=12)
    ax_scatter.tick_params(labelsize=10)
    ax_scatter.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax_scatter.legend(fontsize=9, frameon=True, loc="best")

    idx = np.arange(1, n_comp + 1)
    ax_scree.bar(
        idx,
        explained,
        color="#7aa5d6",
        edgecolor="#4d78a8",
        linewidth=0.8,
        alpha=0.9,
        label="Explained variance",
    )
    ax_scree.plot(
        idx,
        cumulative,
        color="#d95f02",
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label="Cumulative variance",
    )
    ax_scree.set_xlabel("Principal component index", fontsize=12)
    ax_scree.set_ylabel("Explained variance ratio", fontsize=12)
    ax_scree.set_ylim(0.0, 1.02)
    ax_scree.tick_params(labelsize=10)
    ax_scree.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax_scree.legend(fontsize=9, frameon=True, loc="best")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    # -------------------- configuration (edit here) --------------------
    segment_cache_dir = Path("data/hctsa_segments")
    beta_selection_json = Path("results/beta_channel_selection/beta_channel_selection.json")
    outdir = Path("results/hctsa_segments_datamatrix")
    feature_type = "hctsa"  # "hctsa" or "raw" (used in output filename)

    subjects: Optional[List[str]] = None  # e.g. ["PW_HK59", "PW_EM59"]
    groups: Optional[List[str]] = None  # e.g. ["normal_walking", "gait_modulation"]
    max_rows: Optional[int] = None
    max_features: Optional[int] = None

    normalize_0_1 = True
    normalization_method = "robust_sigmoid"  # "robust_sigmoid" | "minmax"
    min_finite_fraction = 1.0
    min_feature_std: Optional[float] = None

    pca_max_components = 20
    strict = True
    verbose = 1
    # ---------------------------------------------------------------

    logging.basicConfig(level=logging.INFO if verbose >= 1 else logging.WARNING)

    cache = HCTSASegmentCache(segment_cache_dir)
    best_channel_map = load_beta_best_channel_map(beta_selection_json)
    subject_channel_map = resolve_subject_channel_map(
        cache,
        best_channel_map=best_channel_map,
        subjects=subjects,
        strict=bool(strict),
    )

    outdir.mkdir(parents=True, exist_ok=True)
    map_path = outdir / "subject_channel_map_beta.json"
    with open(map_path, "w", encoding="utf-8") as fp:
        json.dump(subject_channel_map, fp, indent=2)
    logger.info("Saved resolved subject->channel map to %s", map_path)

    X, timeseries_df, operations_df, labels = cache.load_subject_channel_data(subject_channel_map)
    logger.info("Loaded beta-selected data matrix: %s", X.shape)

    X, timeseries_df, labels = subset_rows(
        X,
        timeseries_df,
        labels,
        groups=groups,
        max_rows=max_rows,
    )
    X, operations_df = subset_features_by_variance(X, operations_df, max_features=max_features)

    if normalize_0_1:
        method = (normalization_method or "robust_sigmoid").strip().lower()
        if method == "robust_sigmoid":
            X = robust_sigmoid_0_1(X, axis=0)
        elif method == "minmax":
            X = minmax_scale_0_1(X, axis=0)
        else:
            raise ValueError(f"Unknown normalization_method: {normalization_method}")

    n_features_before = int(X.shape[1])
    X, operations_df = filter_invalid_features(
        X,
        operations_df,
        min_finite_fraction=min_finite_fraction,
        drop_inf=True,
        drop_constant=min_feature_std is not None,
        min_std=float(min_feature_std) if min_feature_std is not None else 0.0,
    )
    n_dropped = n_features_before - int(X.shape[1])
    logger.info("Dropped features: %d (kept %d).", n_dropped, int(X.shape[1]))

    title = (
        f"PCA of HCTSA Feature Matrix (beta-selected channels) | "
        f"rows={X.shape[0]} cols={X.shape[1]}"
    )
    if groups:
        title += f" | groups={','.join(groups)}"
    if normalize_0_1:
        title += f" | {normalization_method}01"

    feature_type_tag = str(feature_type).strip().lower().replace(" ", "_")
    if feature_type_tag not in {"hctsa", "raw"}:
        raise ValueError(f"feature_type must be 'hctsa' or 'raw', got: {feature_type}")
    output_path = outdir / f"datamatrix_beta_selected_{feature_type_tag}_pca_overview.png"
    plot_pca_overview(
        X,
        labels,
        title=title,
        output_path=output_path,
        max_components=int(pca_max_components),
    )
    logger.info("Saved PCA overview figure to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
