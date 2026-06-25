#!/usr/bin/env python3
"""
Compare PCA explained-variance profiles between HCTSA and raw segment matrices.

What this script does:
- Loads beta-selected channel data from both caches:
  1) `4646_data/hctsa_segments`
  2) `4646_data/raw_segments`
- Applies the same preprocessing to both matrices.
- Computes PCA on each matrix.
- Saves a single figure with two aligned panels:
  1) Explained variance ratio vs PC index
  2) Cumulative explained variance vs PC index

Generated output:
- `results/feature_space_analysis/pca_variance_comparison_raw_vs_hctsa.png`
"""
from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Numbered module filename cannot be imported via normal dotted import.
_MATRIX_MODULE_PATH = Path(__file__).with_name("08_plot_feature_matrix.py")
_MATRIX_SPEC = importlib.util.spec_from_file_location("stats_plot_feature_matrix", _MATRIX_MODULE_PATH)
if _MATRIX_SPEC is None or _MATRIX_SPEC.loader is None:
    raise ImportError(f"Failed to load module from {_MATRIX_MODULE_PATH}")
_MATRIX_MODULE = importlib.util.module_from_spec(_MATRIX_SPEC)
_MATRIX_SPEC.loader.exec_module(_MATRIX_MODULE)

_PCA_MODULE_PATH = Path(__file__).with_name("10_plot_pca.py")
_PCA_SPEC = importlib.util.spec_from_file_location("stats_plot_pca", _PCA_MODULE_PATH)
if _PCA_SPEC is None or _PCA_SPEC.loader is None:
    raise ImportError(f"Failed to load module from {_PCA_MODULE_PATH}")
_PCA_MODULE = importlib.util.module_from_spec(_PCA_SPEC)
_PCA_SPEC.loader.exec_module(_PCA_MODULE)

HCTSASegmentCache = _MATRIX_MODULE.HCTSASegmentCache
load_beta_best_channel_map = _MATRIX_MODULE._load_beta_best_channel_map
resolve_subject_channel_map = _MATRIX_MODULE._resolve_subject_channel_map
subset_rows = _MATRIX_MODULE._subset_rows
subset_features_by_variance = _MATRIX_MODULE._subset_features_by_variance
minmax_scale_0_1 = _MATRIX_MODULE._minmax_scale_0_1
robust_sigmoid_0_1 = _MATRIX_MODULE._robust_sigmoid_0_1
filter_invalid_features = _MATRIX_MODULE._filter_invalid_features
impute_for_pca = _PCA_MODULE._impute_for_pca

logger = logging.getLogger(__name__)


def _prepare_matrix(
    *,
    segment_cache_dir: Path,
    beta_selection_json: Path,
    subjects: Optional[List[str]],
    groups: Optional[List[str]],
    max_rows: Optional[int],
    max_features: Optional[int],
    normalize_0_1: bool,
    normalization_method: str,
    min_finite_fraction: float,
    min_feature_std: Optional[float],
    strict: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    cache = HCTSASegmentCache(segment_cache_dir)
    best_channel_map = load_beta_best_channel_map(beta_selection_json)
    subject_channel_map = resolve_subject_channel_map(
        cache,
        best_channel_map=best_channel_map,
        subjects=subjects,
        strict=bool(strict),
    )

    X, timeseries_df, operations_df, labels = cache.load_subject_channel_data(subject_channel_map)
    logger.info("[%s] loaded matrix shape=%s", segment_cache_dir.name, X.shape)

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

    X, operations_df = filter_invalid_features(
        X,
        operations_df,
        min_finite_fraction=min_finite_fraction,
        drop_inf=True,
        drop_constant=min_feature_std is not None,
        min_std=float(min_feature_std) if min_feature_std is not None else 0.0,
    )
    logger.info("[%s] prepared matrix shape=%s", segment_cache_dir.name, X.shape)
    return X, labels


def _pca_variance_curves(X: np.ndarray, max_components: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.decomposition import PCA
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for PCA plotting.") from exc

    X_pca = impute_for_pca(np.asarray(X, dtype=np.float64))
    X_pca = np.nan_to_num(X_pca, nan=0.0, posinf=0.0, neginf=0.0)

    n_comp = int(min(max(2, max_components), X_pca.shape[0], X_pca.shape[1]))
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(X_pca)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    return explained, cumulative


def _plot_comparison(
    *,
    explained_hctsa: np.ndarray,
    cumulative_hctsa: np.ndarray,
    explained_raw: np.ndarray,
    cumulative_raw: np.ndarray,
    output_path: Path,
) -> None:
    n = min(len(explained_hctsa), len(explained_raw))
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.2, 7.2))
    ax.set_box_aspect(1.0)

    bar_width = 0.42
    ax.bar(
        x - bar_width / 2,
        explained_hctsa[:n],
        width=bar_width,
        color="#1f77b4",
        edgecolor="#164f85",
        linewidth=0.8,
        alpha=0.9,
        label="Explained variance (HCTSA)",
    )
    ax.bar(
        x + bar_width / 2,
        explained_raw[:n],
        width=bar_width,
        color="#ff7f0e",
        edgecolor="#b45a09",
        linewidth=0.8,
        alpha=0.9,
        label="Explained variance (Raw)",
    )
    ax.plot(
        x,
        cumulative_hctsa[:n],
        marker="o",
        markersize=4.0,
        linewidth=2.0,
        color="#1f77b4",
        linestyle="-",
        label="Cumulative variance (HCTSA)",
    )
    ax.plot(
        x,
        cumulative_raw[:n],
        marker="o",
        markersize=4.0,
        linewidth=2.0,
        color="#ff7f0e",
        linestyle="-",
        label="Cumulative variance (Raw)",
    )
    ax.set_xlabel("Principal component", fontsize=12)
    ax.set_ylabel("Variance ratio", fontsize=12)
    ax.set_xticks(x)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.8)
    ax.tick_params(labelsize=10)
    ax.legend(loc="upper left", fontsize=9, frameon=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    # -------------------- configuration (edit here) --------------------
    hctsa_cache_dir = Path("4646_data/hctsa_segments")
    raw_cache_dir = Path("4646_data/raw_segments")
    beta_selection_json = Path("results/beta_channel_selection/beta_channel_selection.json")
    outdir = Path("results/feature_space_analysis")
    output_figure_name = "pca_variance_comparison_raw_vs_hctsa.png"

    subjects: Optional[List[str]] = None
    groups: Optional[List[str]] = None
    max_rows: Optional[int] = None
    max_features: Optional[int] = None

    normalize_0_1 = True
    normalization_method = "robust_sigmoid"  # "robust_sigmoid" | "minmax"
    min_finite_fraction = 1.0
    min_feature_std: Optional[float] = None

    pca_max_components = 10
    strict = True
    verbose = 1
    # ---------------------------------------------------------------

    logging.basicConfig(level=logging.INFO if verbose >= 1 else logging.WARNING)

    X_hctsa, _labels_hctsa = _prepare_matrix(
        segment_cache_dir=hctsa_cache_dir,
        beta_selection_json=beta_selection_json,
        subjects=subjects,
        groups=groups,
        max_rows=max_rows,
        max_features=max_features,
        normalize_0_1=normalize_0_1,
        normalization_method=normalization_method,
        min_finite_fraction=min_finite_fraction,
        min_feature_std=min_feature_std,
        strict=bool(strict),
    )
    X_raw, _labels_raw = _prepare_matrix(
        segment_cache_dir=raw_cache_dir,
        beta_selection_json=beta_selection_json,
        subjects=subjects,
        groups=groups,
        max_rows=max_rows,
        max_features=max_features,
        normalize_0_1=normalize_0_1,
        normalization_method=normalization_method,
        min_finite_fraction=min_finite_fraction,
        min_feature_std=min_feature_std,
        strict=bool(strict),
    )

    explained_hctsa, cumulative_hctsa = _pca_variance_curves(X_hctsa, max_components=int(pca_max_components))
    explained_raw, cumulative_raw = _pca_variance_curves(X_raw, max_components=int(pca_max_components))

    output_path = outdir / output_figure_name
    _plot_comparison(
        explained_hctsa=explained_hctsa,
        cumulative_hctsa=cumulative_hctsa,
        explained_raw=explained_raw,
        cumulative_raw=cumulative_raw,
        output_path=output_path,
    )

    summary = {
        "hctsa_cache_dir": str(hctsa_cache_dir),
        "raw_cache_dir": str(raw_cache_dir),
        "beta_selection_json": str(beta_selection_json),
        "normalization": normalization_method if normalize_0_1 else None,
        "n_components_hctsa": int(len(explained_hctsa)),
        "n_components_raw": int(len(explained_raw)),
        "pc1_hctsa": float(explained_hctsa[0]),
        "pc1_raw": float(explained_raw[0]),
        "pc2_hctsa": float(explained_hctsa[1]),
        "pc2_raw": float(explained_raw[1]),
    }
    summary_path = outdir / "pca_variance_comparison_raw_vs_hctsa_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    logger.info("Saved figure: %s", output_path)
    logger.info("Saved summary: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
