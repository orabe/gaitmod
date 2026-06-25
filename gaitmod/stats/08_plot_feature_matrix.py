import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from gaitmod.preprocessing.hctsa_segments import HCTSASegmentCache, parse_segment_identifier

logger = logging.getLogger(__name__)

TITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 32
TICK_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 12
LEGEND_TITLE_FONTSIZE = 13
CBAR_LABEL_FONTSIZE = 14

DEFAULT_GROUP_COLORS_HEX = ["#298c8c", "#f1a226"]

GROUP_LABEL_RENAME = {
    "gait_modulation": "Gait Modulation",
    "normal_walking": "Steady-State Walking",
}


def _hex_to_rgb01(value: str) -> Tuple[float, float, float]:
    s = value.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"Invalid hex color: {value}")
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return (r, g, b)


def _load_beta_best_channel_map(beta_json_path: Path) -> Dict[str, str]:
    with open(beta_json_path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    best_map: Dict[str, str] = {}
    for subject, entry in payload.items():
        best = entry.get("best_channel")
        if not best:
            continue
        best_map[str(subject)] = str(best)
    return best_map


def _resolve_subject_channel_map(
    cache: HCTSASegmentCache,
    best_channel_map: Dict[str, str],
    subjects: Optional[Sequence[str]] = None,
    strict: bool = False,
) -> Dict[str, str]:
    index_df = cache.load_index()
    if index_df.empty:
        raise ValueError(f"Segment cache index is empty at {cache.index_file}")

    requested: Iterable[str] = subjects if subjects else best_channel_map.keys()

    resolved: Dict[str, str] = {}
    missing: List[str] = []
    no_match: List[str] = []

    for subject in requested:
        subject = str(subject)
        best = best_channel_map.get(subject)
        if best is None:
            missing.append(subject)
            continue
        subject_df = index_df[index_df["subject"] == subject]
        if subject_df.empty:
            no_match.append(subject)
            continue

        # Cache layout uses channel folders like: "channel_2_LFP_L0-2"
        match_df = subject_df[subject_df["channel"].astype(str).str.contains(best, na=False)]
        if match_df.empty:
            no_match.append(subject)
            continue

        canonical = match_df["channel_canonical"].value_counts().idxmax()
        resolved[subject] = str(canonical)

    if strict and (missing or no_match):
        raise ValueError(
            "Unable to resolve beta-selected channels for some subjects. "
            f"missing_in_json={missing}, missing_in_cache_or_no_match={no_match}"
        )

    if missing:
        logger.warning("Subjects missing from beta selection JSON: %s", ", ".join(sorted(missing)))
    if no_match:
        logger.warning(
            "Subjects with no matching cached channel for beta selection: %s",
            ", ".join(sorted(no_match)),
        )

    if not resolved:
        raise ValueError("No subjects resolved to a cached beta-selected channel.")

    return resolved


def _parse_row_metadata(timeseries_df: pd.DataFrame) -> pd.DataFrame:
    parsed = [parse_segment_identifier(name) for name in timeseries_df["Name"].astype(str)]
    meta = pd.DataFrame(parsed)
    meta["name"] = timeseries_df["Name"].astype(str).to_numpy()
    raw_groups = timeseries_df.get("Group", pd.Series([""] * len(timeseries_df))).astype(str)
    meta["group"] = raw_groups.apply(
        lambda g: GROUP_LABEL_RENAME.get(str(g).strip().lower(), str(g))
    ).to_numpy()
    return meta


def _subset_rows(
    X: np.ndarray,
    timeseries_df: pd.DataFrame,
    labels: np.ndarray,
    groups: Optional[Sequence[str]],
    max_rows: Optional[int],
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    mask = np.ones(X.shape[0], dtype=bool)
    if groups:
        groups_set = set(str(g) for g in groups)
        mask &= timeseries_df["Group"].astype(str).isin(groups_set).to_numpy()
    if not mask.all():
        X = X[mask]
        timeseries_df = timeseries_df.loc[mask].reset_index(drop=True)
        labels = labels[mask]
    if max_rows is not None and X.shape[0] > max_rows:
        X = X[:max_rows]
        timeseries_df = timeseries_df.iloc[:max_rows].reset_index(drop=True)
        labels = labels[:max_rows]
    return X, timeseries_df, labels


def _subset_features_by_variance(
    X: np.ndarray,
    operations_df: pd.DataFrame,
    max_features: Optional[int],
) -> Tuple[np.ndarray, pd.DataFrame]:
    if max_features is None or X.shape[1] <= max_features:
        return X, operations_df

    variances = np.nanvar(X, axis=0)
    order = np.argsort(variances)[::-1][:max_features]
    X = X[:, order]
    operations_df = operations_df.iloc[order].reset_index(drop=True)
    return X, operations_df


def _minmax_scale_0_1(
    X: np.ndarray,
    *,
    axis: int = 0,
    epsilon: float = 1e-12,
    clip: bool = True,
) -> np.ndarray:
    """
    Min-max normalize to [0, 1] along a given axis (default: per feature / column).

    NaNs are preserved.
    """
    X = np.asarray(X, dtype=np.float64)
    minv = np.nanmin(X, axis=axis, keepdims=True)
    maxv = np.nanmax(X, axis=axis, keepdims=True)
    denom = maxv - minv
    denom = np.where(np.isfinite(denom) & (denom > epsilon), denom, 1.0)
    out = (X - minv) / denom
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out


def _robust_sigmoid_0_1(
    X: np.ndarray,
    *,
    axis: int = 0,
    epsilon: float = 1e-12,
    clip_z: float = 20.0,
) -> np.ndarray:
    """
    Robust sigmoidal normalization (hctsa-like spirit):
    - robust center: median
    - robust scale: IQR (converted to sigma using /1.349)
    - logistic sigmoid -> (0, 1)

    NaNs are preserved.
    """
    X = np.asarray(X, dtype=np.float64)
    med = np.nanmedian(X, axis=axis, keepdims=True)
    q25 = np.nanpercentile(X, 25, axis=axis, keepdims=True)
    q75 = np.nanpercentile(X, 75, axis=axis, keepdims=True)
    iqr = q75 - q25
    robust_sigma = iqr / 1.349
    robust_sigma = np.where(np.isfinite(robust_sigma) & (robust_sigma > epsilon), robust_sigma, 1.0)
    z = (X - med) / robust_sigma
    if clip_z is not None:
        z = np.clip(z, -float(clip_z), float(clip_z))
    return 1.0 / (1.0 + np.exp(-z))


def _filter_invalid_features(
    X: np.ndarray,
    operations_df: pd.DataFrame,
    *,
    min_finite_fraction: float = 1.0,
    drop_inf: bool = True,
    drop_constant: bool = True,
    min_std: float = 1e-12,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Drop invalid feature columns from X and keep operations_df aligned.

    - Non-finite-heavy columns (finite fraction < min_finite_fraction)
    - Columns containing +/-inf (optional)
    - (Near-)constant columns (optional; std <= min_std on finite values)
    """
    if X.ndim != 2:
        raise ValueError("Expected X to be 2D (n_samples, n_features).")
    if X.shape[1] != len(operations_df):
        raise ValueError("operations_df must have one row per feature column in X.")

    min_finite_fraction = float(min_finite_fraction)
    if not (0.0 < min_finite_fraction <= 1.0):
        raise ValueError("min_finite_fraction must be in (0, 1].")

    finite = np.isfinite(X)
    finite_counts = finite.sum(axis=0)
    finite_frac = finite_counts / max(X.shape[0], 1)

    valid = (finite_counts > 0) & (finite_frac >= min_finite_fraction)

    if drop_inf:
        valid &= ~np.isinf(X).any(axis=0)

    if drop_constant:
        col_std = np.nanstd(X, axis=0)
        col_std = np.where(np.isfinite(col_std), col_std, 0.0)
        valid &= col_std > float(min_std)

    valid_idx = np.where(valid)[0]
    dropped = int(X.shape[1] - valid_idx.size)
    if dropped > 0:
        logger.info(
            "Dropped %d/%d invalid features (min_finite_fraction=%.2f, drop_inf=%s, drop_constant=%s).",
            dropped,
            X.shape[1],
            min_finite_fraction,
            bool(drop_inf),
            bool(drop_constant),
        )

    X = X[:, valid_idx]
    operations_df = operations_df.iloc[valid_idx].reset_index(drop=True)
    return X, operations_df


def _impute_for_clustering(X: np.ndarray) -> np.ndarray:
    if not np.isnan(X).any():
        return X
    col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    out = np.array(X, copy=True)
    nan_mask = np.isnan(out)
    out[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    return out


def _cluster_order(X: np.ndarray, axis: int) -> Optional[np.ndarray]:
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist
    except Exception:  # pragma: no cover
        return None

    def _nan_aware_corr_distance_condensed(items: np.ndarray) -> np.ndarray:
        """
        Condensed distance matrix using 1 - corr(x,y) with pairwise NaN ignoring.

        items: (n_items, n_features)
        """
        n_items = int(items.shape[0])
        m = n_items * (n_items - 1) // 2
        out = np.empty(m, dtype=np.float64)
        k = 0
        for i in range(n_items - 1):
            xi = items[i]
            for j in range(i + 1, n_items):
                xj = items[j]
                mask = np.isfinite(xi) & np.isfinite(xj)
                if mask.sum() < 2:
                    corr = 0.0
                else:
                    a = xi[mask]
                    b = xj[mask]
                    a = a - a.mean()
                    b = b - b.mean()
                    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
                    if denom <= 1e-12:
                        corr = 0.0
                    else:
                        corr = float(np.sum(a * b) / denom)
                        if not np.isfinite(corr):
                            corr = 0.0
                        corr = float(np.clip(corr, -1.0, 1.0))
                out[k] = 1.0 - corr
                k += 1
        return out

    if axis == 1:
        X = X.T

    # hctsa-like defaults:
    # - Rows (time series): euclidean
    # - Cols (operations): correlation with NaN-ignoring (corr_fast-like)
    if axis == 0:
        # Euclidean can't handle NaNs; impute for clustering only.
        Xc = _impute_for_clustering(X)
        Z = linkage(Xc, method="average", metric="euclidean")
        return leaves_list(Z).astype(int, copy=False)

    # axis==1 -> operations/features
    # Prefer SciPy's fast correlation pdist when fully finite; otherwise fall back.
    if np.isfinite(X).all():
        # Correlation distance is undefined for constant vectors (zero variance),
        # which can happen after min-max scaling (features with min==max -> all zeros).
        row_std = np.std(X, axis=1)
        non_constant = np.isfinite(row_std) & (row_std > 1e-12)
        if int(non_constant.sum()) < 2:
            return np.arange(X.shape[0], dtype=int)
        Z = linkage(X[non_constant], method="average", metric="correlation")
        order_non_constant = leaves_list(Z).astype(int, copy=False)
        non_constant_idx = np.where(non_constant)[0]
        clustered_idx = non_constant_idx[order_non_constant]
        constant_idx = np.where(~non_constant)[0]
        return np.concatenate([clustered_idx, constant_idx]).astype(int, copy=False)

    # NaN-aware correlation distance (corr_fast-like). This is O(n_items^2) and can be large.
    n_items = int(X.shape[0])
    if n_items > 3000:
        raise RuntimeError(
            f"NaN-aware correlation clustering requested for {n_items} features; "
            "set max_features to a smaller value or increase min_finite_fraction to drop NaNs."
        )
    d = _nan_aware_corr_distance_condensed(X)
    Z = linkage(d, method="average")
    return leaves_list(Z).astype(int, copy=False)


def plot_data_matrix(
    X: np.ndarray,
    meta: pd.DataFrame,
    operations_df: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    cluster_rows: bool = False,
    cluster_cols: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    discrete_step: Optional[float] = None,
    feature_tick_step: Optional[int] = None,
    show_group_strip: bool = True,
    show_legend: bool = True,
    show_feature_names: bool = False,
    figure_size: Optional[Tuple[float, float]] = None,
    show_colorbar_label: bool = True,
):
    row_order = None
    col_order = None
    if cluster_rows:
        row_order = _cluster_order(X, axis=0)
        if row_order is None:
            raise RuntimeError("Row clustering requested, but SciPy is not available.")
    if cluster_cols:
        col_order = _cluster_order(X, axis=1)
        if col_order is None:
            raise RuntimeError("Column clustering requested, but SciPy is not available.")

    if row_order is not None:
        X = X[row_order]
        meta = meta.iloc[row_order].reset_index(drop=True)
    if col_order is not None:
        X = X[:, col_order]
        operations_df = operations_df.iloc[col_order].reset_index(drop=True)

    groups = meta["group"].astype(str).fillna("").to_list()
    unique_groups = [g for g in pd.Series(groups, dtype="string").unique().tolist() if g and g != ""]
    if not unique_groups:
        unique_groups = ["(unlabeled)"]
        groups = ["(unlabeled)"] * len(groups)

    if len(unique_groups) > 2:
        raise ValueError(
            f"Expected at most 2 groups for fixed colors, got {len(unique_groups)}: {unique_groups}"
        )
    group_to_color: Dict[str, Tuple[float, float, float, float]] = {}
    for i, g in enumerate(unique_groups):
        hex_color = DEFAULT_GROUP_COLORS_HEX[min(i, len(DEFAULT_GROUP_COLORS_HEX) - 1)]
        group_to_color[g] = (*_hex_to_rgb01(hex_color), 1.0)
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}
    group_ids = np.array([group_to_idx.get(g, 0) for g in groups], dtype=int)
    group_cmap = ListedColormap([group_to_color[g] for g in unique_groups])

    X_masked = np.ma.masked_invalid(X)
    # Match hctsa's default blue/yellow/red scheme (low=blue, high=red).
    base_cmap = plt.get_cmap("RdYlBu_r")

    if vmin is None:
        vmin = np.nanpercentile(X, 2)
    if vmax is None:
        vmax = np.nanpercentile(X, 98)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = None, None
    norm = None

    if discrete_step is not None and vmin is not None and vmax is not None:
        step = float(discrete_step)
        if step <= 0:
            raise ValueError("discrete_step must be > 0")
        # Ensure boundaries cover [vmin, vmax] inclusively.
        boundaries = np.arange(vmin, vmax + step * 0.5, step)
        if boundaries.size < 2 or boundaries[-1] < vmax:
            boundaries = np.append(boundaries, vmax)
        n_bins = int(boundaries.size - 1)
        value_range = float(vmax - vmin)
        # Choose colors by sampling the underlying continuous colormap at a small,
        # fixed offset into each bin so coarse binning still starts at the same
        # dark-blue end as finer binning (e.g., 0.1 steps).
        offset = min(step / 2.0, 0.05 * value_range)
        sample_points = boundaries[:-1] + offset
        centers = (sample_points - vmin) / value_range if value_range > 0 else np.linspace(0, 1, n_bins)
        centers = np.clip(centers, 0.0, 1.0)
        cmap = ListedColormap(base_cmap(centers))
        cmap.set_bad(color="black")
        norm = BoundaryNorm(boundaries, ncolors=n_bins, clip=True)
    else:
        cmap = base_cmap.copy()
        cmap.set_bad(color="black")

    if figure_size is None:
        fig_w = max(10.0, min(22.0, 8.0 + X.shape[1] / 350.0))
        fig_h = max(6.0, min(18.0, 4.0 + X.shape[0] / 120.0))
    else:
        fig_w, fig_h = float(figure_size[0]), float(figure_size[1])
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_group = None
    ax_leg = None
    if show_group_strip and show_legend:
        gs = GridSpec(nrows=1, ncols=4, figure=fig, width_ratios=[0.2, 4.8, 0.25, 0.9], wspace=0.05)
        ax_group = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[0, 1])
        ax_cb = fig.add_subplot(gs[0, 2])
        ax_leg = fig.add_subplot(gs[0, 3])
    elif show_group_strip and (not show_legend):
        gs = GridSpec(nrows=1, ncols=3, figure=fig, width_ratios=[0.2, 4.8, 0.25], wspace=0.05)
        ax_group = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[0, 1])
        ax_cb = fig.add_subplot(gs[0, 2])
    elif (not show_group_strip) and show_legend:
        gs = GridSpec(nrows=1, ncols=3, figure=fig, width_ratios=[4.8, 0.25, 0.9], wspace=0.05)
        ax = fig.add_subplot(gs[0, 0])
        ax_cb = fig.add_subplot(gs[0, 1])
        ax_leg = fig.add_subplot(gs[0, 2])
    else:
        gs = GridSpec(nrows=1, ncols=2, figure=fig, width_ratios=[4.8, 0.25], wspace=0.05)
        ax = fig.add_subplot(gs[0, 0])
        ax_cb = fig.add_subplot(gs[0, 1])

    if ax_group is not None:
        ax_group.imshow(group_ids.reshape(-1, 1), aspect="auto", cmap=group_cmap, interpolation="nearest")
        ax_group.set_xticks([])
        ax_group.set_yticks([])

    imshow_kwargs = dict(
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
    )
    if norm is not None:
        imshow_kwargs["norm"] = norm
    else:
        imshow_kwargs["vmin"] = vmin
        imshow_kwargs["vmax"] = vmax

    im = ax.imshow(X_masked, **imshow_kwargs)
    ax.set_ylabel("Segments", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlabel("Features", fontsize=AXIS_LABEL_FONTSIZE)

    if show_feature_names and X.shape[1] <= 60:
        ax.set_xticks(np.arange(X.shape[1]))
        ax.set_xticklabels(operations_df["Name"].astype(str).to_list(), rotation=90, fontsize=7)
    elif feature_tick_step is not None and int(feature_tick_step) > 0:
        step = int(feature_tick_step)
        xticks = np.arange(0, X.shape[1], step, dtype=int)
        ax.set_xticks(xticks)
        # Prefer 1-based operation IDs if available, otherwise 1-based indices.
        if "ID" in operations_df.columns:
            ids = operations_df["ID"].to_numpy()
            labels = [str(int(ids[i])) for i in xticks]
        else:
            labels = [str(int(i + 1)) for i in xticks]
        ax.set_xticklabels(labels, rotation=0, fontsize=TICK_LABEL_FONTSIZE)
    else:
        ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

    if norm is not None:
        cb = fig.colorbar(im, cax=ax_cb, ticks=norm.boundaries)
    else:
        cb = fig.colorbar(im, cax=ax_cb)
    if show_colorbar_label:
        cb.set_label("Feature value", fontsize=CBAR_LABEL_FONTSIZE)
    cb.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

    if ax_leg is not None:
        legend_patches = [Patch(facecolor=group_to_color[g], edgecolor="none", label=g) for g in unique_groups]
        ax_leg.axis("off")
        ax_leg.legend(
            handles=legend_patches,
            title="Groups",
            loc="center left",
            borderaxespad=0.0,
            fontsize=LEGEND_FONTSIZE,
            title_fontsize=LEGEND_TITLE_FONTSIZE,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_data_matrix_color_groups(
    X: np.ndarray,
    meta: pd.DataFrame,
    operations_df: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    cluster_rows: bool = False,
    cluster_cols: bool = False,
    group_color_map: Optional[Dict[str, Tuple[float, float, float]]] = None,
    discrete_step: Optional[float] = None,
    feature_tick_step: Optional[int] = None,
    figure_size: Optional[Tuple[float, float]] = None,
):
    """
    hctsa-like "colorGroups" visualization:
    - Same numeric matrix, but rows are rendered using group-specific palettes:
      values near 0 -> white; values near 1 -> dark group color.
    - Colorbar area is split into one colorbar per group.
    """
    row_order = None
    col_order = None
    if cluster_rows:
        row_order = _cluster_order(X, axis=0)
        if row_order is None:
            raise RuntimeError("Row clustering requested, but SciPy is not available.")
    if cluster_cols:
        col_order = _cluster_order(X, axis=1)
        if col_order is None:
            raise RuntimeError("Column clustering requested, but SciPy is not available.")

    if row_order is not None:
        X = X[row_order]
        meta = meta.iloc[row_order].reset_index(drop=True)
    if col_order is not None:
        X = X[:, col_order]
        operations_df = operations_df.iloc[col_order].reset_index(drop=True)

    group_labels = meta["group"].astype(str).fillna("").to_list()
    unique_groups = [g for g in pd.Series(group_labels, dtype="string").unique().tolist() if g and g != ""]
    if not unique_groups:
        unique_groups = ["(unlabeled)"]
        group_labels = ["(unlabeled)"] * len(group_labels)

    # Assign base colors per group (dark end of the palette).
    if group_color_map is None:
        if len(unique_groups) > 2:
            raise ValueError(
                f"Expected at most 2 groups for fixed colors, got {len(unique_groups)}: {unique_groups}"
            )
        group_color_map = {}
        for i, g in enumerate(unique_groups):
            hex_color = DEFAULT_GROUP_COLORS_HEX[min(i, len(DEFAULT_GROUP_COLORS_HEX) - 1)]
            group_color_map[g] = _hex_to_rgb01(hex_color)

    # Build per-group sequential colormaps: white -> base color.
    # If discrete_step is set, bin values into fixed intervals (hctsa-like discrete rendering).
    group_cmaps: Dict[str, object] = {}
    group_norms: Dict[str, Optional[BoundaryNorm]] = {}
    boundaries = None
    n_bins = None
    if discrete_step is not None:
        step = float(discrete_step)
        if step <= 0:
            raise ValueError("discrete_step must be > 0")
        boundaries = np.arange(0.0, 1.0 + step * 0.5, step)
        if boundaries.size < 2 or boundaries[-1] < 1.0:
            boundaries = np.append(boundaries, 1.0)
        boundaries[-1] = 1.0
        n_bins = int(boundaries.size - 1)

    for g in unique_groups:
        base = group_color_map.get(g, (0.2, 0.2, 0.2))
        if boundaries is None or n_bins is None:
            cm = LinearSegmentedColormap.from_list(f"group_{g}", [(1.0, 1.0, 1.0), base], N=256)
            cm.set_bad(color="black")
            group_cmaps[g] = cm
            group_norms[g] = None
        else:
            cont = LinearSegmentedColormap.from_list(f"group_{g}_cont", [(1.0, 1.0, 1.0), base], N=256)
            centers = (boundaries[:-1] + boundaries[1:]) / 2.0
            cm = ListedColormap(cont(centers))
            cm.set_bad(color="black")
            group_cmaps[g] = cm
            group_norms[g] = BoundaryNorm(boundaries, ncolors=n_bins, clip=True)

    # Reorder rows to create contiguous group blocks (like TS_PlotDataMatrix colorGroups).
    group_to_rows: Dict[str, List[int]] = {g: [] for g in unique_groups}
    other_rows: List[int] = []
    for i, g in enumerate(group_labels):
        if g in group_to_rows:
            group_to_rows[g].append(i)
        else:
            other_rows.append(i)

    row_blocks: List[Tuple[str, List[int]]] = [(g, group_to_rows[g]) for g in unique_groups if group_to_rows[g]]
    if other_rows:
        row_blocks.append(("(other)", other_rows))
        other_base = (0.3, 0.3, 0.3)
        if boundaries is None or n_bins is None:
            other_cm = LinearSegmentedColormap.from_list("group_other", [(1.0, 1.0, 1.0), other_base], N=256)
            other_cm.set_bad(color="black")
            group_cmaps["(other)"] = other_cm
            group_norms["(other)"] = None
        else:
            cont = LinearSegmentedColormap.from_list("group_other_cont", [(1.0, 1.0, 1.0), other_base], N=256)
            centers = (boundaries[:-1] + boundaries[1:]) / 2.0
            other_cm = ListedColormap(cont(centers))
            other_cm.set_bad(color="black")
            group_cmaps["(other)"] = other_cm
            group_norms["(other)"] = BoundaryNorm(boundaries, ncolors=n_bins, clip=True)
        if "(other)" not in unique_groups:
            unique_groups = unique_groups + ["(other)"]

    new_order = np.array([idx for _, rows in row_blocks for idx in rows], dtype=int)
    X = X[new_order]
    meta = meta.iloc[new_order].reset_index(drop=True)
    group_labels = [group_labels[i] if group_labels[i] in group_to_rows else "(other)" for i in new_order.tolist()]

    # Render group-specific colors into an RGBA image.
    img = np.zeros((X.shape[0], X.shape[1], 4), dtype=np.float32)
    start = 0
    boundaries: List[int] = []
    for g, rows in row_blocks:
        n = len(rows)
        block = X[start:start + n]
        cmap = group_cmaps[g]
        gnorm = group_norms.get(g)
        masked = np.ma.masked_invalid(block)
        if gnorm is None:
            rgba = cmap(Normalize(vmin=0.0, vmax=1.0, clip=True)(masked))
        else:
            rgba = cmap(gnorm(masked))
        img[start:start + n] = rgba
        start += n
        boundaries.append(start)

    if figure_size is None:
        fig_w = max(10.0, min(22.0, 8.0 + X.shape[1] / 350.0))
        fig_h = max(6.0, min(18.0, 4.0 + X.shape[0] / 120.0))
    else:
        fig_w, fig_h = float(figure_size[0]), float(figure_size[1])
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Heatmap + a stacked colorbar column (one per group), aligned exactly to the
    # corresponding row blocks (so each legend bar matches the matrix block height).
    cb_blocks = [pair for pair in row_blocks if pair[1]]
    gs = GridSpec(
        nrows=X.shape[0],
        ncols=2,
        figure=fig,
        width_ratios=[4.8, 0.25],
        wspace=0.05,
        hspace=0.0,
    )

    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_ylabel("Segments (group-blocked)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlabel("Features", fontsize=AXIS_LABEL_FONTSIZE)
    if feature_tick_step is not None and int(feature_tick_step) > 0:
        step = int(feature_tick_step)
        xticks = np.arange(0, X.shape[1], step, dtype=int)
        ax.set_xticks(xticks)
        if "ID" in operations_df.columns:
            ids = operations_df["ID"].to_numpy()
            labels = [str(int(ids[i])) for i in xticks]
        else:
            labels = [str(int(i + 1)) for i in xticks]
        ax.set_xticklabels(labels, rotation=0, fontsize=7)
    else:
        ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

    # Draw separators between group blocks.
    for y in boundaries[:-1]:
        ax.axhline(y - 0.5, color="k", linewidth=0.6, alpha=0.5)

    # One colorbar per group, stacked (legend split into halves for 2 groups).
    start_row = 0
    for g, rows in cb_blocks:
        end_row = start_row + len(rows)
        ax_cb = fig.add_subplot(gs[start_row:end_row, 1])
        gnorm = group_norms.get(g)
        if gnorm is None:
            sm = plt.cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0, clip=True), cmap=group_cmaps[g])
            cb = fig.colorbar(sm, cax=ax_cb)
            cb.set_ticks([0.0, 0.5, 1.0])
        else:
            sm = plt.cm.ScalarMappable(norm=gnorm, cmap=group_cmaps[g])
            cb = fig.colorbar(sm, cax=ax_cb, ticks=gnorm.boundaries)
        cb.set_label(g, rotation=90, labelpad=8, fontsize=CBAR_LABEL_FONTSIZE)
        cb.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
        start_row = end_row

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_data_matrix_color_groups_mixed(
    X: np.ndarray,
    meta: pd.DataFrame,
    operations_df: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    cluster_rows: bool = False,
    cluster_cols: bool = False,
    group_color_map: Optional[Dict[str, Tuple[float, float, float]]] = None,
    discrete_step: Optional[float] = None,
    feature_tick_step: Optional[int] = None,
):
    """
    Like hctsa 'colorGroups', but keep rows mixed (no group-block sorting).

    Rows/cols are clustered as usual, then each row is rendered with a group-specific
    palette (white->group color). This allows groups to be interleaved while still
    visually encoding class membership.
    """
    row_order = None
    col_order = None
    if cluster_rows:
        row_order = _cluster_order(X, axis=0)
        if row_order is None:
            raise RuntimeError("Row clustering requested, but SciPy is not available.")
    if cluster_cols:
        col_order = _cluster_order(X, axis=1)
        if col_order is None:
            raise RuntimeError("Column clustering requested, but SciPy is not available.")

    if row_order is not None:
        X = X[row_order]
        meta = meta.iloc[row_order].reset_index(drop=True)
    if col_order is not None:
        X = X[:, col_order]
        operations_df = operations_df.iloc[col_order].reset_index(drop=True)

    group_labels = meta["group"].astype(str).fillna("").to_list()
    unique_groups = [g for g in pd.Series(group_labels, dtype="string").unique().tolist() if g and g != ""]
    if not unique_groups:
        unique_groups = ["(unlabeled)"]
        group_labels = ["(unlabeled)"] * len(group_labels)

    if group_color_map is None:
        if len(unique_groups) > 2:
            raise ValueError(
                f"Expected at most 2 groups for fixed colors, got {len(unique_groups)}: {unique_groups}"
            )
        group_color_map = {}
        for i, g in enumerate(unique_groups):
            hex_color = DEFAULT_GROUP_COLORS_HEX[min(i, len(DEFAULT_GROUP_COLORS_HEX) - 1)]
            group_color_map[g] = _hex_to_rgb01(hex_color)

    group_cmaps: Dict[str, object] = {}
    group_norms: Dict[str, Optional[BoundaryNorm]] = {}
    boundaries = None
    n_bins = None
    if discrete_step is not None:
        step = float(discrete_step)
        if step <= 0:
            raise ValueError("discrete_step must be > 0")
        boundaries = np.arange(0.0, 1.0 + step * 0.5, step)
        if boundaries.size < 2 or boundaries[-1] < 1.0:
            boundaries = np.append(boundaries, 1.0)
        boundaries[-1] = 1.0
        n_bins = int(boundaries.size - 1)

    for g in unique_groups:
        base = group_color_map.get(g, (0.2, 0.2, 0.2))
        if boundaries is None or n_bins is None:
            cm = LinearSegmentedColormap.from_list(f"group_{g}", [(1.0, 1.0, 1.0), base], N=256)
            cm.set_bad(color="black")
            group_cmaps[g] = cm
            group_norms[g] = None
        else:
            cont = LinearSegmentedColormap.from_list(f"group_{g}_cont", [(1.0, 1.0, 1.0), base], N=256)
            centers = (boundaries[:-1] + boundaries[1:]) / 2.0
            cm = ListedColormap(cont(centers))
            cm.set_bad(color="black")
            group_cmaps[g] = cm
            group_norms[g] = BoundaryNorm(boundaries, ncolors=n_bins, clip=True)

    # Render group-specific colors into an RGBA image, keeping row order as-is.
    img = np.zeros((X.shape[0], X.shape[1], 4), dtype=np.float32)
    norm_cont = Normalize(vmin=0.0, vmax=1.0, clip=True)
    masked_all = np.ma.masked_invalid(X)

    for g in unique_groups:
        rows = np.where(pd.Series(group_labels, dtype="string").astype(str).to_numpy() == g)[0]
        if rows.size == 0:
            continue
        cmap = group_cmaps[g]
        gnorm = group_norms.get(g)
        block = masked_all[rows]
        if gnorm is None:
            img[rows] = cmap(norm_cont(block))
        else:
            img[rows] = cmap(gnorm(block))

    fig_w = max(10.0, min(22.0, 8.0 + X.shape[1] / 350.0))
    fig_h = max(6.0, min(18.0, 4.0 + X.shape[0] / 120.0))
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Heatmap + stacked colorbar column (heights proportional to class size).
    group_sizes = {g: int(np.sum(np.asarray(group_labels, dtype=object) == g)) for g in unique_groups}
    cb_groups = [g for g in unique_groups if group_sizes.get(g, 0) > 0]
    height_ratios = [max(1, group_sizes[g]) for g in cb_groups]

    gs = GridSpec(
        nrows=len(cb_groups),
        ncols=2,
        figure=fig,
        width_ratios=[4.8, 0.25],
        height_ratios=height_ratios,
        wspace=0.05,
        hspace=0.25,
    )

    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_ylabel("Segments", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlabel("Features", fontsize=AXIS_LABEL_FONTSIZE)
    if feature_tick_step is not None and int(feature_tick_step) > 0:
        step = int(feature_tick_step)
        xticks = np.arange(0, X.shape[1], step, dtype=int)
        ax.set_xticks(xticks)
        if "ID" in operations_df.columns:
            ids = operations_df["ID"].to_numpy()
            labels = [str(int(ids[i])) for i in xticks]
        else:
            labels = [str(int(i + 1)) for i in xticks]
        ax.set_xticklabels(labels, rotation=0, fontsize=7)
    else:
        ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

    for r, g in enumerate(cb_groups):
        ax_cb = fig.add_subplot(gs[r, 1])
        gnorm = group_norms.get(g)
        if gnorm is None:
            sm = plt.cm.ScalarMappable(norm=norm_cont, cmap=group_cmaps[g])
            cb = fig.colorbar(sm, cax=ax_cb)
            cb.set_ticks([0.0, 0.5, 1.0])
        else:
            sm = plt.cm.ScalarMappable(norm=gnorm, cmap=group_cmaps[g])
            cb = fig.colorbar(sm, cax=ax_cb, ticks=gnorm.boundaries)
        cb.set_label(g, rotation=90, labelpad=8, fontsize=CBAR_LABEL_FONTSIZE)
        cb.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    # -------------------- configuration (edit here) --------------------
    segment_cache_dir = Path("4646_data/hctsa_segments")
    beta_selection_json = Path("results/beta_channel_selection/beta_channel_selection.json")
    outdir = Path("results/hctsa_segments_datamatrix")

    subjects: Optional[List[str]] = None  # e.g. ["PW_HK59", "PW_EM59"]
    groups: Optional[List[str]] = None  # e.g. ["normal_walking", "freezing"]
    max_rows: Optional[int] = None
    max_features: Optional[int] = None

    normalize_0_1 = True  # robust sigmoid per feature (column) to ~[0, 1]
    normalization_method = "robust_sigmoid"  # "robust_sigmoid" | "minmax"
    discrete_step = 0.1  # set to None for continuous colormap
    feature_tick_step: Optional[int] = None  # e.g. 200 shows feature IDs every 200 columns; None disables
    show_group_strip = False  # disable left group strip for large matrices
    show_legend = False  # disable legend for large matrices
    min_finite_fraction = 1.0  # set < 1.0 to allow some NaNs per feature
    min_feature_std: Optional[float] = None  # set to None to disable std-based filtering

    save_color_groups_figure = True  # TS_PlotDataMatrix(...,'colorGroups',true)-like
    save_color_groups_mixed_figure = False  # colorGroups, but keep rows mixed

    cluster_rows = True  # requires SciPy
    cluster_cols = True  # requires SciPy
    strict = True
    show_feature_names = False
    verbose = 1
    # ---------------------------------------------------------------

    logging.basicConfig(level=logging.INFO if verbose >= 1 else logging.WARNING)

    cache = HCTSASegmentCache(segment_cache_dir)
    best_channel_map = _load_beta_best_channel_map(beta_selection_json)
    subject_channel_map = _resolve_subject_channel_map(
        cache,
        best_channel_map=best_channel_map,
        subjects=subjects,
        strict=bool(strict),
    )

    outdir.mkdir(parents=True, exist_ok=True)
    map_path = outdir / "subject_channel_map_beta.json"
    with open(map_path, "w", encoding="utf-8") as fp:
        json.dump(subject_channel_map, fp, indent=2)
    logger.info("Saved resolved subject→channel map to %s", map_path)

    X, timeseries_df, operations_df, labels = cache.load_subject_channel_data(subject_channel_map)
    logger.info("Loaded beta-selected data matrix: %s", X.shape)

    X, timeseries_df, labels = _subset_rows(
        X,
        timeseries_df,
        labels,
        groups=groups,
        max_rows=max_rows,
    )
    X, operations_df = _subset_features_by_variance(X, operations_df, max_features=max_features)
    if normalize_0_1:
        method = (normalization_method or "robust_sigmoid").strip().lower()
        if method == "robust_sigmoid":
            X = _robust_sigmoid_0_1(X, axis=0)
        elif method == "minmax":
            X = _minmax_scale_0_1(X, axis=0)
        else:
            raise ValueError(f"Unknown normalization_method: {normalization_method}")
    n_features_before = int(X.shape[1])
    X, operations_df = _filter_invalid_features(
        X,
        operations_df,
        min_finite_fraction=min_finite_fraction,
        drop_inf=True,
        drop_constant=min_feature_std is not None,
        min_std=float(min_feature_std) if min_feature_std is not None else 0.0,
    )
    n_dropped = n_features_before - int(X.shape[1])
    logger.info("Dropped features: %d (kept %d).", n_dropped, int(X.shape[1]))
    print(f"Dropped features: {n_dropped} (kept {int(X.shape[1])})")

    meta = _parse_row_metadata(timeseries_df)
    title = ""

    if save_color_groups_figure and normalize_0_1:
        plot_data_matrix_color_groups(
            X,
            meta,
            operations_df,
            title=title,
            output_path=outdir / "datamatrix_beta_selected_colorGroups.png",
            cluster_rows=bool(cluster_rows),
            cluster_cols=bool(cluster_cols),
            discrete_step=discrete_step,
            feature_tick_step=feature_tick_step,
        )

    # (optional) mixed colorGroups figure disabled by default

    output_path = outdir / "datamatrix_beta_selected.png"
    plot_data_matrix(
        X,
        meta,
        operations_df,
        title=title,
        output_path=output_path,
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
    logger.info("Saved data matrix figure to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
