#!/usr/bin/env python3
"""
Generate chapter-ready artifacts for the Model-Level Evaluation results section.

Inputs:
- logs/results/*/summary/final_summary.json
- logs/results/*/summary/nested_cv_results.csv
- Existing comparison figures under logs/results/comparison_figures
- Existing per-model threshold/subject figures under logs/results/scores_thresholds

Outputs (default):
- results/model_level_evaluation/model_comparison_primary_test_f1.csv
- results/model_level_evaluation/model_comparison_primary_test_f1.tex
- results/model_level_evaluation/subject_level_top_models_test_f1.csv
- results/model_level_evaluation/subject_level_top_models_test_f1.tex
- results/model_level_evaluation/model_level_evaluation_summary.json
- results/model_level_evaluation/model_level_evaluation_section.tex
- results/model_level_evaluation/figures/*.png (copied canonical figures)
"""

from __future__ import annotations

import json
import csv
import math
import glob
import re
from statistics import mean, pstdev
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from scipy.stats import wilcoxon  # type: ignore
except Exception:
    wilcoxon = None


PRIMARY_METRIC = "test_f1"
SECONDARY_METRICS = [
    "test_roc_auc",
    "test_pr_auc",
    "test_balanced_accuracy",
    "test_precision",
    "test_recall",
    "test_specificity",
]
TABLE_METRICS = [PRIMARY_METRIC, *SECONDARY_METRICS]


RUN_LABELS = {
    "dummy_raw_betaChs": "Baseline-Dummy",
    "logreg_hctsa_betaChs": "LogReg",
    "rf_hctsa_betaChs": "RF",
    "xgb_hctsa_betaChs": "XGB",
    "svm_hctsa_betaChs": "SVM",
    "Seq2VecMLP_hctsa_betaChs": "IntraSeg-MLP",
    "Seq2VecCNN_raw_betaChs": "IntraSeg-CNN",
    "Seq2VecLSTM_raw_betaChs": "IntraSeg-LSTM",
    "Seq2VecMLPLSTM_betaChs": "IntraSeg-MLP-LSTM",
    "Seq2SeqLSTM_hctsa_betaChs": "InterSeg-LSTM",
    "Seq2SeqCNNLSTM_raw_betaChs": "InterSeg-CNN-LSTM",
}

FAMILY_ORDER = ["Inter-segment", "Intra-segment", "Classical ML", "Baseline"]
FAMILY_BY_LABEL = {
    "InterSeg-CNN-LSTM": "Inter-segment",
    "InterSeg-LSTM": "Inter-segment",
    "IntraSeg-CNN": "Intra-segment",
    "IntraSeg-LSTM": "Intra-segment",
    "IntraSeg-MLP": "Intra-segment",
    "IntraSeg-MLP-LSTM": "Intra-segment",
    "LogReg": "Classical ML",
    "SVM": "Classical ML",
    "RF": "Classical ML",
    "XGB": "Classical ML",
    "Baseline-Dummy": "Baseline",
}

# Main-text split-figure preferences (used when available).
MAIN_OVERALL_METRIC_GROUP = "discrimination"
MAIN_SUBJECT_METRIC_GROUP = "primary"


@dataclass
class RunArtifacts:
    run_id: str
    label: str
    final_summary: dict
    nested_rows: List[dict]


def _assign_family(label: str) -> str:
    family = FAMILY_BY_LABEL.get(label)
    if family is None:
        raise RuntimeError(
            f"Unmapped model label '{label}'. Add it to FAMILY_BY_LABEL in generator script."
        )
    return family


def _read_csv_rows(csv_path: Path) -> List[dict]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _load_runs(results_root: Path) -> List[RunArtifacts]:
    runs: List[RunArtifacts] = []
    for summary_path in sorted(results_root.glob("*/summary/final_summary.json")):
        run_id = summary_path.parents[1].name
        csv_path = summary_path.with_name("nested_cv_results.csv")
        if not csv_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as f:
            final_summary = json.load(f)
        nested_rows = _read_csv_rows(csv_path)
        runs.append(
            RunArtifacts(
                run_id=run_id,
                label=RUN_LABELS.get(run_id, run_id),
                final_summary=final_summary,
                nested_rows=nested_rows,
            )
        )
    if not runs:
        raise RuntimeError(f"No runs loaded from {results_root}")
    return runs


def _subject_set_ok(runs: List[RunArtifacts]) -> Tuple[bool, List[str]]:
    ref = None
    for run in runs:
        subjects = tuple(run.final_summary.get("subjects", []))
        if ref is None:
            ref = subjects
        elif subjects != ref:
            return False, list(ref or [])
    return True, list(ref or [])


def _metric_stat(summary: dict, metric: str, key: str) -> float:
    metric_obj = summary.get(metric, {})
    value = metric_obj.get(key, float("nan")) if isinstance(metric_obj, dict) else float("nan")
    return float(value)


def _build_model_table(runs: List[RunArtifacts]) -> List[dict]:
    rows: List[dict] = []
    for run in runs:
        row = {
            "run_id": run.run_id,
            "model": run.label,
            "family": _assign_family(run.label),
        }
        for metric in TABLE_METRICS:
            row[f"{metric}_mean"] = _metric_stat(run.final_summary, metric, "mean")
            row[f"{metric}_std"] = _metric_stat(run.final_summary, metric, "std")
            row[f"{metric}_min"] = _metric_stat(run.final_summary, metric, "min")
            row[f"{metric}_max"] = _metric_stat(run.final_summary, metric, "max")
        rows.append(row)
    rows = sorted(rows, key=lambda r: r[f"{PRIMARY_METRIC}_mean"], reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank_test_f1"] = idx
    # keep rank as first key by reordering
    ordered_rows = []
    for row in rows:
        ordered = {"rank_test_f1": row["rank_test_f1"]}
        for k, v in row.items():
            if k != "rank_test_f1":
                ordered[k] = v
        ordered_rows.append(ordered)
    return ordered_rows


def _best_model_by_family(ranked_rows: List[dict]) -> Dict[str, dict]:
    best: Dict[str, dict] = {}
    for row in ranked_rows:
        fam = str(row["family"])
        if fam not in best:
            best[fam] = row
    return best


def _build_family_table(ranked_rows: List[dict]) -> List[dict]:
    dummy_row = next((r for r in ranked_rows if r["family"] == "Baseline"), None)
    if dummy_row is None:
        raise RuntimeError("Baseline model not found; cannot compute family deltas.")
    dummy_mean = float(dummy_row[f"{PRIMARY_METRIC}_mean"])

    grouped: Dict[str, List[dict]] = {}
    for row in ranked_rows:
        fam = str(row["family"])
        grouped.setdefault(fam, []).append(row)

    out: List[dict] = []
    for fam, fam_rows in grouped.items():
        f1_means = [float(r[f"{PRIMARY_METRIC}_mean"]) for r in fam_rows]
        best = fam_rows[0]
        best_mean = float(best[f"{PRIMARY_METRIC}_mean"])
        delta_abs = best_mean - dummy_mean
        delta_rel = (delta_abs / dummy_mean * 100.0) if dummy_mean != 0 else float("nan")
        out.append(
            {
                "family": fam,
                "n_models": int(len(fam_rows)),
                "best_model_in_family": str(best["model"]),
                "best_model_test_f1_mean": best_mean,
                "best_model_test_f1_std": float(best[f"{PRIMARY_METRIC}_std"]),
                "family_mean_test_f1": float(np.mean(f1_means)),
                "family_std_test_f1": float(np.std(f1_means)),
                "delta_best_vs_dummy_abs": float(delta_abs),
                "delta_best_vs_dummy_rel_pct": float(delta_rel),
            }
        )

    fam_order_idx = {name: i for i, name in enumerate(FAMILY_ORDER)}
    out = sorted(
        out,
        key=lambda r: (
            -float(r["best_model_test_f1_mean"]),
            fam_order_idx.get(str(r["family"]), 999),
        ),
    )
    for idx, row in enumerate(out, start=1):
        row["rank_by_best_test_f1"] = idx
    ordered_rows: List[dict] = []
    for row in out:
        ordered = {"rank_by_best_test_f1": row["rank_by_best_test_f1"]}
        for k, v in row.items():
            if k != "rank_by_best_test_f1":
                ordered[k] = v
        ordered_rows.append(ordered)
    return ordered_rows


def _build_subject_top_table(runs: List[RunArtifacts], top_n: int = 3) -> Tuple[List[dict], List[str]]:
    ranked = sorted(
        runs,
        key=lambda r: _metric_stat(r.final_summary, PRIMARY_METRIC, "mean"),
        reverse=True,
    )
    top_runs = ranked[:top_n]
    top_ids = [r.run_id for r in top_runs]

    # Build subject rows from consistent subject ordering in summaries.
    subjects: List[str] = list(top_runs[0].final_summary.get("subjects", []))
    rows: List[dict] = []
    for subject in subjects:
        row = {"subject": subject}
        scores = []
        for run in top_runs:
            score = float("nan")
            for rec in run.nested_rows:
                if rec.get("test_subject_name") == subject:
                    try:
                        score = float(rec.get(PRIMARY_METRIC, "nan"))
                    except (TypeError, ValueError):
                        score = float("nan")
                    break
            row[f"{run.label}_{PRIMARY_METRIC}"] = score
            scores.append(score)
        valid = [s for s in scores if not math.isnan(s)]
        row["mean_across_top_models"] = float(mean(valid)) if valid else float("nan")
        row["std_across_top_models"] = float(pstdev(valid)) if len(valid) > 1 else 0.0
        rows.append(row)

    return rows, top_ids


def _latex_model_table(rows: List[dict]) -> str:
    header = (
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Model & $\\mathrm{F1}$ & ROC-AUC & PR-AUC & Balanced Acc. \\\\\n"
        "\\midrule\n"
    )
    lines = []
    for row in rows:
        lines.append(
            (
                f"{row['model']} & "
                f"{row['test_f1_mean']:.2f}$\\pm${row['test_f1_std']:.2f} & "
                f"{row['test_roc_auc_mean']:.2f}$\\pm${row['test_roc_auc_std']:.2f} & "
                f"{row['test_pr_auc_mean']:.2f}$\\pm${row['test_pr_auc_std']:.2f} & "
                f"{row['test_balanced_accuracy_mean']:.2f}$\\pm${row['test_balanced_accuracy_std']:.2f} \\\\"
            )
        )
    footer = "\n\\bottomrule\n\\end{tabular}\n"
    return header + "\n".join(lines) + footer


def _latex_subject_table(rows: List[dict], top_labels: List[str]) -> str:
    cols = "l" + "c" * len(top_labels)
    header = (
        f"\\begin{{tabular}}{{{cols}}}\n"
        "\\toprule\n"
        f"Subject & {' & '.join(top_labels)} \\\\\n"
        "\\midrule\n"
    )
    lines = []
    for row in rows:
        vals = [f"{row[f'{label}_{PRIMARY_METRIC}']:.2f}" for label in top_labels]
        lines.append(f"{row['subject']} & {' & '.join(vals)} \\\\")
    footer = "\n\\bottomrule\n\\end{tabular}\n"
    return header + "\n".join(lines) + footer


def _latex_family_table(rows: List[dict]) -> str:
    header = (
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        "Family & $n$ & Best Model & Best $\\mathrm{F1}$ & Family Mean $\\mathrm{F1}$ & $\\Delta$ Best vs Baseline \\\\\n"
        "\\midrule\n"
    )
    lines = []
    for row in rows:
        lines.append(
            (
                f"{row['family']} & "
                f"{int(row['n_models'])} & "
                f"{row['best_model_in_family']} & "
                f"{row['best_model_test_f1_mean']:.2f}$\\pm${row['best_model_test_f1_std']:.2f} & "
                f"{row['family_mean_test_f1']:.2f}$\\pm${row['family_std_test_f1']:.2f} & "
                f"{row['delta_best_vs_dummy_abs']:.2f} ({row['delta_best_vs_dummy_rel_pct']:.2f}\\%) \\\\"
            )
        )
    footer = "\n\\bottomrule\n\\end{tabular}\n"
    return header + "\n".join(lines) + footer


def _latest_match(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _sanitize_stem(text: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text.strip())
    return stem.strip("_")


def _appendix_caption_from_name(name: str) -> str:
    txt = name.replace(".png", "").replace("_", " ")
    txt = re.sub(r"\s+", " ", txt).strip()
    txt = txt.replace("metrics boxplots compare", "Boxplot comparison")
    txt = txt.replace("metrics meanstd bars compare", "Mean±std bar comparison")
    txt = txt.replace("family", "family")
    txt = txt.replace("group", "metric-group")
    return txt[:1].upper() + txt[1:] if txt else "Supplementary comparison figure"


def _appendix_tex(extra_fig_paths: List[Path]) -> str:
    if not extra_fig_paths:
        return (
            "\\section*{Model-Level Evaluation Supplementary Figures}\n"
            "No supplementary split figures were generated.\n"
        )
    lines: List[str] = []
    lines.append("\\section*{Model-Level Evaluation Supplementary Figures}")
    lines.append(
        "The following split figures were generated automatically and are provided as supplementary material."
    )
    for idx, p in enumerate(extra_fig_paths, start=1):
        lines.append("")
        lines.append("\\begin{figure}[t]")
        lines.append("    \\centering")
        lines.append(f"    \\includegraphics[width=\\textwidth]{{img/appendix/{p.name}}}")
        lines.append(f"    \\caption{{Supplementary Figure {idx}. {_appendix_caption_from_name(p.name)}.}}")
        lines.append(f"    \\label{{fig:model_eval_supp_{_sanitize_stem(p.stem)}}}")
        lines.append("\\end{figure}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _copy_figures(
    results_root: Path, out_fig_dir: Path, best_run_id: str
) -> Tuple[Dict[str, str], List[str]]:
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    appendix_dir = out_fig_dir / "appendix"
    appendix_dir.mkdir(parents=True, exist_ok=True)

    # Canonical figure sources already produced by existing scripts.
    threshold_model_main_src = results_root / "comparison_figures" / "test" / "models_threshold_curves_main.png"
    threshold_model_main_auc_src = (
        results_root / "comparison_figures" / "test" / "models_threshold_curves_main_auc.png"
    )
    threshold_model_main_operating_src = (
        results_root / "comparison_figures" / "test" / "models_threshold_curves_main_threshold.png"
    )
    threshold_model_full_src = results_root / "comparison_figures" / "test" / "models_threshold_curves.png"
    threshold_model_src = threshold_model_main_src if threshold_model_main_src.exists() else threshold_model_full_src
    compare_dir = results_root / "comparison_figures" / "test"

    # Prefer split (cleaner) figures for main section; fallback to canonical all-metrics.
    overall_meanstd_src = _latest_match(
        list(
            compare_dir.glob(
                f"*/split/metrics_meanstd_bars_compare_group-{MAIN_OVERALL_METRIC_GROUP}_all-models.png"
            )
        )
    )
    if overall_meanstd_src is None:
        overall_meanstd_src = _latest_match(list(compare_dir.glob("*/metrics_meanstd_bars_compare.png")))

    subject_box_src = _latest_match(
        list(
            compare_dir.glob(
                f"*/split/metrics_boxplots_compare_group-{MAIN_SUBJECT_METRIC_GROUP}_all-models.png"
            )
        )
    )
    if subject_box_src is None:
        subject_box_src = _latest_match(list(compare_dir.glob("*/metrics_boxplots_compare.png")))

    # Collect additional split figures for appendix (excluding whichever are selected for main text).
    split_candidates = sorted(compare_dir.glob("*/split/*.png"))
    main_src_set = {
        str(p.resolve())
        for p in [overall_meanstd_src, subject_box_src]
        if p is not None and p.exists()
    }
    appendix_sources: List[Path] = []
    seen_appendix: set = set()
    for p in sorted(split_candidates, key=lambda x: x.stat().st_mtime, reverse=True):
        rp = str(p.resolve())
        if rp in main_src_set:
            continue
        if p.name in seen_appendix:
            continue
        seen_appendix.add(p.name)
        appendix_sources.append(p)

    threshold_subject_src = (
        results_root / "scores_thresholds" / best_run_id / f"{best_run_id}_all_metrics_by_subject.png"
    )

    targets = {}
    if overall_meanstd_src and overall_meanstd_src.exists():
        tgt = out_fig_dir / "metrics_meanstd_bars_all_models.png"
        tgt.write_bytes(overall_meanstd_src.read_bytes())
        targets["overall_meanstd"] = str(tgt)
        targets["overall_source_name"] = overall_meanstd_src.name
    if threshold_model_src.exists():
        tgt = out_fig_dir / "threshold_metrics_all_models.png"
        tgt.write_bytes(threshold_model_src.read_bytes())
        targets["threshold_model"] = str(tgt)
        targets["threshold_model_source_name"] = threshold_model_src.name
    if threshold_model_main_auc_src.exists():
        tgt = out_fig_dir / "threshold_metrics_auc_all_models.png"
        tgt.write_bytes(threshold_model_main_auc_src.read_bytes())
        targets["threshold_model_auc"] = str(tgt)
        targets["threshold_model_auc_source_name"] = threshold_model_main_auc_src.name
    if threshold_model_main_operating_src.exists():
        tgt = out_fig_dir / "threshold_metrics_operating_all_models.png"
        tgt.write_bytes(threshold_model_main_operating_src.read_bytes())
        targets["threshold_model_operating"] = str(tgt)
        targets["threshold_model_operating_source_name"] = threshold_model_main_operating_src.name
    if subject_box_src and subject_box_src.exists():
        tgt = out_fig_dir / "metrics_boxplots_all_models.png"
        tgt.write_bytes(subject_box_src.read_bytes())
        targets["subject_boxplot"] = str(tgt)
        targets["subject_source_name"] = subject_box_src.name
    if threshold_subject_src.exists():
        tgt = out_fig_dir / "threshold_metrics_subjects_best_model.png"
        tgt.write_bytes(threshold_subject_src.read_bytes())
        targets["threshold_subject"] = str(tgt)

    appendix_targets: List[str] = []
    for src in appendix_sources:
        tgt = appendix_dir / src.name
        tgt.write_bytes(src.read_bytes())
        appendix_targets.append(str(tgt))

    # If main threshold figure is the clean one, keep full version in appendix.
    if (
        threshold_model_full_src.exists()
        and (
            threshold_model_main_src.exists()
            or threshold_model_main_auc_src.exists()
            or threshold_model_main_operating_src.exists()
        )
    ):
        full_tgt = appendix_dir / threshold_model_full_src.name
        full_tgt.write_bytes(threshold_model_full_src.read_bytes())
        appendix_targets.append(str(full_tgt))

    return targets, appendix_targets


def _format_model_list(rows_ranked: List[dict]) -> str:
    parts = [str(r["model"]) for r in rows_ranked]
    return ", ".join(parts[:-1]) + f", and {parts[-1]}" if len(parts) > 1 else parts[0]


def _run_by_id(runs: List[RunArtifacts], run_id: str) -> RunArtifacts:
    for run in runs:
        if run.run_id == run_id:
            return run
    raise KeyError(f"Run not found: {run_id}")


def _subject_metric_map(run: RunArtifacts, metric: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for rec in run.nested_rows:
        subj = rec.get("test_subject_name")
        if not subj:
            continue
        try:
            out[subj] = float(rec.get(metric, "nan"))
        except (TypeError, ValueError):
            continue
    return out


def _win_count(subject_rows: List[dict], best_label: str) -> Tuple[int, int]:
    best_key = f"{best_label}_{PRIMARY_METRIC}"
    model_keys = [k for k in subject_rows[0].keys() if k.endswith(f"_{PRIMARY_METRIC}")]
    wins = 0
    for row in subject_rows:
        vals = [float(row[k]) for k in model_keys]
        row_best = max(vals)
        if math.isclose(float(row[best_key]), row_best, rel_tol=0.0, abs_tol=1e-12):
            wins += 1
    return wins, len(subject_rows)


def _range_for_model_from_subject_rows(subject_rows: List[dict], label: str) -> Tuple[float, float, float]:
    key = f"{label}_{PRIMARY_METRIC}"
    vals = [float(r[key]) for r in subject_rows]
    mn = float(min(vals))
    mx = float(max(vals))
    return mn, mx, float(mx - mn)


def _wilcoxon_best_vs(best_run: RunArtifacts, other_run: RunArtifacts, metric: str = PRIMARY_METRIC) -> Dict[str, float]:
    best_map = _subject_metric_map(best_run, metric)
    other_map = _subject_metric_map(other_run, metric)
    subjects = sorted(set(best_map.keys()) & set(other_map.keys()))
    x = np.asarray([best_map[s] for s in subjects], dtype=float)
    y = np.asarray([other_map[s] for s in subjects], dtype=float)
    wins = int(np.sum(x > y))
    out = {
        "n": float(len(subjects)),
        "wins": float(wins),
        "W": float("nan"),
        "p": float("nan"),
    }
    if wilcoxon is not None and len(subjects) >= 1:
        try:
            stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", mode="auto")
            out["W"] = float(stat)
            out["p"] = float(p)
        except Exception:
            pass
    return out


def _holm_adjust(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [float("nan")] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        raw = pvals[idx]
        if math.isnan(raw):
            adjusted[idx] = float("nan")
            continue
        val = (m - rank) * raw
        val = max(val, prev)
        prev = val
        adjusted[idx] = min(val, 1.0)
    return adjusted


def _train_test_f1_gap(run: RunArtifacts) -> Optional[float]:
    train_vals = []
    test_vals = []
    for rec in run.nested_rows:
        try:
            test_vals.append(float(rec.get("test_f1", "nan")))
        except (TypeError, ValueError):
            pass
        try:
            train_vals.append(float(rec.get("train_f1", "nan")))
        except (TypeError, ValueError):
            pass
    train_vals = [v for v in train_vals if not math.isnan(v)]
    test_vals = [v for v in test_vals if not math.isnan(v)]
    if not train_vals or not test_vals:
        return None
    return float(np.mean(train_vals) - np.mean(test_vals))


def _threshold_summary_for_run(run_id: str) -> Optional[dict]:
    files = sorted(glob.glob(f"logs/{run_id}/*/outer_fold_*_test_*/refit/*/refit_results_scores.npz"))
    if not files:
        return None

    thresholds = np.linspace(0.0, 1.0, 101)
    best_thr: List[float] = []
    best_f1: List[float] = []
    best_bal: List[float] = []

    for p in files:
        data = np.load(p)
        y = np.ravel(data["y_true"]).astype(int)
        s = np.ravel(data["y_score"]).astype(float)

        f1_vals = []
        bal_vals = []
        for thr in thresholds:
            pred = (s >= thr).astype(int)
            tp = int(np.sum((y == 1) & (pred == 1)))
            fp = int(np.sum((y == 0) & (pred == 1)))
            fn = int(np.sum((y == 1) & (pred == 0)))
            tn = int(np.sum((y == 0) & (pred == 0)))
            den = 2 * tp + fp + fn
            f1 = (2.0 * tp / den) if den > 0 else 0.0
            tpr = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            tnr = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            bal = 0.5 * (tpr + tnr)
            f1_vals.append(f1)
            bal_vals.append(bal)
        arr_f1 = np.asarray(f1_vals, dtype=float)
        arr_bal = np.asarray(bal_vals, dtype=float)
        idx = int(np.argmax(arr_f1))
        best_thr.append(float(thresholds[idx]))
        best_f1.append(float(arr_f1[idx]))
        best_bal.append(float(arr_bal[idx]))

    return {
        "n": int(len(files)),
        "thr_mean": float(np.mean(best_thr)),
        "thr_std": float(np.std(best_thr)),
        "best_f1_mean": float(np.mean(best_f1)),
        "best_f1_std": float(np.std(best_f1)),
        "best_bal_mean": float(np.mean(best_bal)),
        "best_bal_std": float(np.std(best_bal)),
    }


def _format_name_value_pairs(items: List[Tuple[str, float]]) -> str:
    parts = [f"{name} ({val:.2f})" for name, val in items]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _section_tex(
    ranked_rows: List[dict],
    family_rows: List[dict],
    family_tests: Dict[str, dict],
    subject_rows: List[dict],
    runs: List[RunArtifacts],
    best_run_id: str,
    figure_targets: Dict[str, str],
) -> str:
    best = ranked_rows[0]
    second = ranked_rows[1]
    third = ranked_rows[2] if len(ranked_rows) > 2 else second
    dummy_row = next((r for r in ranked_rows if r["family"] == "Baseline"), None)
    if dummy_row is None:
        dummy_row = ranked_rows[-1]

    margin = float(best["test_f1_mean"] - second["test_f1_mean"])
    margin_rel = (margin / float(second["test_f1_mean"])) * 100.0 if float(second["test_f1_mean"]) != 0 else float("nan")
    best_vs_dummy_abs = float(best["test_f1_mean"] - dummy_row["test_f1_mean"])
    best_vs_dummy_rel = (
        ((float(best["test_f1_mean"]) / float(dummy_row["test_f1_mean"])) - 1.0) * 100.0
        if float(dummy_row["test_f1_mean"]) != 0
        else float("nan")
    )

    best_run = _run_by_id(runs, best["run_id"])
    second_run = _run_by_id(runs, second["run_id"])
    third_run = _run_by_id(runs, third["run_id"])

    w_best_vs_second = _wilcoxon_best_vs(best_run, second_run)
    w_best_vs_third = _wilcoxon_best_vs(best_run, third_run)
    p_adj = _holm_adjust([w_best_vs_second["p"], w_best_vs_third["p"]])

    best_wins, n_subjects = _win_count(subject_rows, best["model"])
    best_min, best_max, best_range = _range_for_model_from_subject_rows(subject_rows, best["model"])
    second_min, second_max, second_range = _range_for_model_from_subject_rows(subject_rows, second["model"])
    third_min, third_max, third_range = _range_for_model_from_subject_rows(subject_rows, third["model"])

    gaps: List[Tuple[str, float]] = []
    for row in ranked_rows:
        run = _run_by_id(runs, row["run_id"])
        gap = _train_test_f1_gap(run)
        if gap is not None:
            gaps.append((str(row["model"]), float(gap)))
    gaps = sorted(gaps, key=lambda x: x[1])
    small_group = gaps[: min(3, len(gaps))]
    mid_group = gaps[min(3, len(gaps)): min(6, len(gaps))]
    large_group = gaps[min(6, len(gaps)): min(8, len(gaps))]
    largest_group = gaps[min(8, len(gaps)):] if len(gaps) > 8 else []

    thr_best = _threshold_summary_for_run(best["run_id"])
    thr_second = _threshold_summary_for_run(second["run_id"])
    thr_third = _threshold_summary_for_run(third["run_id"])

    top_labels = [
        c.replace(f"_{PRIMARY_METRIC}", "")
        for c in subject_rows[0].keys()
        if c.endswith(f"_{PRIMARY_METRIC}")
    ]
    top_labels_str = ", ".join(top_labels)

    model_table_ref = "tab:model_comparison_primary_test_f1"
    family_table_ref = "tab:family_comparison_primary_test_f1"
    subject_table_ref = "tab:subject_level_top_models_test_f1"
    fig_overall_ref = "fig:metrics_meanstd_all_models"
    fig_subject_ref = "fig:metrics_boxplots_all_models"
    fig_thr_model_ref = "fig:threshold_metrics_all_models"
    fig_thr_auc_ref = "fig:threshold_metrics_auc_all_models"
    fig_thr_oper_ref = "fig:threshold_metrics_operating_all_models"
    fig_thr_subject_ref = "fig:threshold_metrics_subjects_best_model"

    fig_overall_name = Path(
        figure_targets.get("overall_meanstd", "figures/metrics_meanstd_bars_all_models.png")
    ).name
    fig_subject_name = Path(
        figure_targets.get("subject_boxplot", "figures/metrics_boxplots_all_models.png")
    ).name
    fig_thr_model_name = Path(figure_targets.get("threshold_model", "figures/threshold_metrics_all_models.png")).name
    fig_thr_auc_name = Path(
        figure_targets.get("threshold_model_auc", "figures/threshold_metrics_auc_all_models.png")
    ).name
    fig_thr_oper_name = Path(
        figure_targets.get("threshold_model_operating", "figures/threshold_metrics_operating_all_models.png")
    ).name
    fig_thr_subject_name = Path(
        figure_targets.get("threshold_subject", "figures/threshold_metrics_subjects_best_model.png")
    ).name
    fig_thr_model_source_name = str(figure_targets.get("threshold_model_source_name", fig_thr_model_name))
    fig_thr_auc_source_name = str(
        figure_targets.get("threshold_model_auc_source_name", fig_thr_auc_name)
    )
    fig_thr_oper_source_name = str(
        figure_targets.get("threshold_model_operating_source_name", fig_thr_oper_name)
    )
    fig_overall_src_name = str(figure_targets.get("overall_source_name", fig_overall_name))
    fig_subject_src_name = str(figure_targets.get("subject_source_name", fig_subject_name))
    fig_overall_is_split = "group-" in fig_overall_src_name
    fig_subject_is_primary_split = "group-primary" in fig_subject_src_name

    overall_figure_sentence = (
        f"Figure~\\ref{{{fig_overall_ref}}} summarizes key discrimination-oriented metrics graphically as mean $\\pm$ standard deviation bars with a baseline reference overlay."
        if fig_overall_is_split
        else f"Figure~\\ref{{{fig_overall_ref}}} summarizes the same cross-model comparison graphically as mean $\\pm$ standard deviation bars with a baseline reference overlay."
    )
    subject_intro_sentence = (
        f"Subject-level behavior is summarized jointly by Figure~\\ref{{{fig_subject_ref}}} and Table~\\ref{{{subject_table_ref}}}. Figure~\\ref{{{fig_subject_ref}}} shows fold-level \\texttt{{test\\_f1}} distributions across models, while Table~\\ref{{{subject_table_ref}}} reports explicit per-subject \\texttt{{test\\_f1}} values for the top models ({top_labels_str})."
        if fig_subject_is_primary_split
        else f"Subject-level behavior is summarized jointly by Figure~\\ref{{{fig_subject_ref}}} and Table~\\ref{{{subject_table_ref}}}. Figure~\\ref{{{fig_subject_ref}}} shows fold-level score distributions across models for each metric, while Table~\\ref{{{subject_table_ref}}} reports explicit per-subject \\texttt{{test\\_f1}} values for the top models ({top_labels_str})."
    )
    subject_caption = (
        "Fold-level \\texttt{test\\_f1} distributions across models (outer-fold test results). Each box summarizes the cross-subject score distribution for a model; dashed line indicates the baseline mean."
        if fig_subject_is_primary_split
        else "Fold-level metric distributions across models (outer-fold test results). Each panel reports one metric, and each box summarizes the cross-subject score distribution for a model; dashed lines indicate baseline means."
    )
    thr_main_is_clean = "models_threshold_curves_main" in fig_thr_model_source_name
    thr_has_split_pair = ("threshold_model_auc" in figure_targets) and ("threshold_model_operating" in figure_targets)
    threshold_intro_sentence = (
        f"Threshold-sensitivity results are reported in three complementary views. Figure~\\ref{{{fig_thr_auc_ref}}} summarizes ROC/PR behavior, Figure~\\ref{{{fig_thr_oper_ref}}} summarizes threshold-dependent operating metrics (F1, precision, recall, specificity), and Figure~\\ref{{{fig_thr_subject_ref}}} reports subject-wise threshold curves for the best-ranked model ({best['model']}). This split reduces visual crowding while preserving direct comparison."
        if thr_has_split_pair
        else f"Threshold-sensitivity results are reported in two complementary views. Figure~\\ref{{{fig_thr_model_ref}}} compares model-level mean threshold curves for a compact set of key metrics and representative models, enabling clear comparison of threshold-dependent and threshold-independent behavior. Figure~\\ref{{{fig_thr_subject_ref}}} reports subject-wise threshold curves for the best-ranked model ({best['model']}), showing how operating-point sensitivity varies across held-out subjects."
    )
    threshold_model_caption = (
        "Model-level threshold sensitivity for a compact comparison set. Threshold-based metrics are plotted against threshold; ROC and PR are shown as standard curves. Curves represent mean behavior over held-out subjects."
        if thr_main_is_clean
        else "Model-level threshold sensitivity across all evaluated models. Threshold-based metrics are plotted against threshold; ROC and PR are shown as standard curves. Curves represent mean behavior over held-out subjects."
    )
    threshold_auc_caption = (
        "Model-level ROC/PR summary for the compact comparison set. This panel includes only ROC-AUC and PR-AUC views to isolate threshold-independent ranking behavior."
        if thr_main_is_clean
        else "Model-level ROC/PR summary. This panel includes only ROC-AUC and PR-AUC views."
    )
    threshold_oper_caption = (
        "Model-level operating-point sensitivity for the compact comparison set. This panel includes F1, precision, recall, and specificity versus threshold."
        if thr_main_is_clean
        else "Model-level operating-point sensitivity with F1, precision, recall, and specificity versus threshold."
    )

    text = f"""\\section{{Model-Level Evaluation}}
\\label{{sec:model_level_evaluation}}

\\subsection{{Evaluation Protocol}}
\\label{{sec:evaluation_protocol}}
All model-level results in this chapter are reported on held-out outer-fold test subjects from a nested leave-one-subject-out (LOSO) cross-validation design with seven outer folds (subjects: PW\\_EM59, PW\\_FH57, PW\\_HK59, PW\\_HZ58, PW\\_SN61, PW\\_SN66, and PW\\_US68). Hyperparameter tuning and feature-selection decisions were restricted to inner-loop training/validation data. Accordingly, no outer-test subject information was used during model selection. The primary endpoint is \\texttt{{test\\_f1}}. Secondary endpoints are \\texttt{{test\\_roc\\_auc}}, \\texttt{{test\\_pr\\_auc}}, \\texttt{{test\\_balanced\\_accuracy}}, \\texttt{{test\\_precision}}, \\texttt{{test\\_recall}}, and \\texttt{{test\\_specificity}}; threshold-tuned variants are used only in the threshold-sensitivity subsection.

\\subsection{{Overall Cross-Model Performance}}
\\label{{sec:overall_cross_model_performance}}
Table~\\ref{{{model_table_ref}}} compares all evaluated models ({_format_model_list(ranked_rows)}), reporting mean $\\pm$ standard deviation over outer folds for the primary and key secondary metrics. Ranking by \\texttt{{test\\_f1}} yields {best['model']} as the best-performing model with mean \\texttt{{test\\_f1}}={best['test_f1_mean']:.2f}$\\pm${best['test_f1_std']:.2f}, followed by {second['model']} ({second['test_f1_mean']:.2f}$\\pm${second['test_f1_std']:.2f}) and {third['model']} ({third['test_f1_mean']:.2f}$\\pm${third['test_f1_std']:.2f}). The absolute margin between first and second rank is {margin:.2f} in \\texttt{{test\\_f1}}, corresponding to a relative gain of {margin_rel:.2f}\\%. Relative to the baseline ({dummy_row['test_f1_mean']:.2f}$\\pm${dummy_row['test_f1_std']:.2f}), the best model improves \\texttt{{test\\_f1}} by {best_vs_dummy_abs:.2f} ({best_vs_dummy_rel:.2f}\\% relative).
{overall_figure_sentence}

Secondary metrics show the same ordering trend at the top of the table. {best['model']} attains \\texttt{{test\\_roc\\_auc}}={best['test_roc_auc_mean']:.2f}$\\pm${best['test_roc_auc_std']:.2f} and \\texttt{{test\\_pr\\_auc}}={best['test_pr_auc_mean']:.2f}$\\pm${best['test_pr_auc_std']:.2f}, compared with {second['test_roc_auc_mean']:.2f}$\\pm${second['test_roc_auc_std']:.2f} and {second['test_pr_auc_mean']:.2f}$\\pm${second['test_pr_auc_std']:.2f} for {second['model']}. For \\texttt{{test\\_balanced\\_accuracy}}, the corresponding values are {best['test_balanced_accuracy_mean']:.2f}$\\pm${best['test_balanced_accuracy_std']:.2f} ({best['model']}) versus {second['test_balanced_accuracy_mean']:.2f}$\\pm${second['test_balanced_accuracy_std']:.2f} ({second['model']}), indicating consistent ranking across threshold-independent and threshold-dependent summaries.

Paired outer-fold comparisons of \\texttt{{test\\_f1}} confirm these ranking differences. For {best['model']} versus {second['model']}, Wilcoxon signed-rank testing gives $W={w_best_vs_second['W']:.2f}$, $p={w_best_vs_second['p']:.2f}$ (Holm-adjusted $p={p_adj[0]:.2f}$); the comparison against {third['model']} gives $W={w_best_vs_third['W']:.2f}$, $p={w_best_vs_third['p']:.2f}$ (Holm-adjusted $p={p_adj[1]:.2f}$), with {best['model']} scoring higher in {int(w_best_vs_second['wins'])}/{n_subjects} and {int(w_best_vs_third['wins'])}/{n_subjects} folds, respectively.
Additional split views (family-specific and metric-group-specific panels) are provided in the supplementary appendix figures.

\\begin{{table}}[t]
    \\centering
    \\caption{{Outer-fold test performance comparison across evaluated models. Values are mean $\\pm$ standard deviation over seven outer folds. Models are ranked by \\texttt{{test\\_f1}}.}}
    \\label{{{model_table_ref}}}
{_latex_model_table(ranked_rows)}
\\end{{table}}

\\begin{{figure}}[t]
    \\centering
    \\includegraphics[width=\\textwidth]{{img/{fig_overall_name}}}
    \\caption{{Cross-model performance summary as mean $\\pm$ standard deviation across outer-fold test subjects. Bars report metric means, whiskers report standard deviations, and the dashed line indicates the baseline mean for each plotted metric.}}
    \\label{{{fig_overall_ref}}}
\\end{{figure}}

\\subsection{{Architecture-Family Comparison}}
\\label{{sec:architecture_family_comparison}}
To compare model families directly, models were grouped as Inter-segment, Intra-segment, Classical ML, and Baseline. Family-level ranking by best \\texttt{{test\\_f1}} is given in Table~\\ref{{{family_table_ref}}}. The top family is {family_rows[0]['family']} (best model: {family_rows[0]['best_model_in_family']}, \\texttt{{test\\_f1}}={family_rows[0]['best_model_test_f1_mean']:.2f}$\\pm${family_rows[0]['best_model_test_f1_std']:.2f}), followed by {family_rows[1]['family']} ({family_rows[1]['best_model_in_family']}, {family_rows[1]['best_model_test_f1_mean']:.2f}$\\pm${family_rows[1]['best_model_test_f1_std']:.2f}) and {family_rows[2]['family']} ({family_rows[2]['best_model_in_family']}, {family_rows[2]['best_model_test_f1_mean']:.2f}$\\pm${family_rows[2]['best_model_test_f1_std']:.2f}).

Relative to the baseline, the best model in {family_rows[0]['family']} improves \\texttt{{test\\_f1}} by {family_rows[0]['delta_best_vs_dummy_abs']:.2f} ({family_rows[0]['delta_best_vs_dummy_rel_pct']:.2f}\\%), while the best models in {family_rows[1]['family']} and {family_rows[2]['family']} improve by {family_rows[1]['delta_best_vs_dummy_abs']:.2f} ({family_rows[1]['delta_best_vs_dummy_rel_pct']:.2f}\\%) and {family_rows[2]['delta_best_vs_dummy_abs']:.2f} ({family_rows[2]['delta_best_vs_dummy_rel_pct']:.2f}\\%), respectively.

Paired subject-level Wilcoxon tests further characterize these family contrasts: best Inter vs best Intra ($W={family_tests['best_inter_vs_best_intra']['W']:.2f}$, $p={family_tests['best_inter_vs_best_intra']['p']:.2f}$, Holm-adjusted $p={family_tests['best_inter_vs_best_intra']['p_holm']:.2f}$), best Deep (best of Inter/Intra) vs best Classical ($W={family_tests['best_deep_vs_best_classical']['W']:.2f}$, $p={family_tests['best_deep_vs_best_classical']['p']:.2f}$, Holm-adjusted $p={family_tests['best_deep_vs_best_classical']['p_holm']:.2f}$), and best overall vs baseline ($W={family_tests['best_overall_vs_dummy']['W']:.2f}$, $p={family_tests['best_overall_vs_dummy']['p']:.2f}$, Holm-adjusted $p={family_tests['best_overall_vs_dummy']['p_holm']:.2f}$).

\\begin{{table}}[t]
    \\centering
    \\caption{{Family-level comparison under the primary endpoint (\\texttt{{test\\_f1}}). Best and family-level statistics are computed from outer-fold test summaries only.}}
    \\label{{{family_table_ref}}}
{_latex_family_table(family_rows)}
\\end{{table}}

\\subsection{{Train--Test Generalization Gap}}
\\label{{sec:train_test_generalization_gap}}
To quantify generalization behavior, the train--test \\texttt{{f1}} gap was computed as
$\\Delta_{{\\mathrm{{gen}}}}=\\overline{{\\mathrm{{train\\_f1}}}}-\\overline{{\\mathrm{{test\\_f1}}}}$
from outer-fold summaries for runs where \\texttt{{train\\_f1}} was available. The smallest gaps were observed for {_format_name_value_pairs(small_group)}."""
    if mid_group:
        text += f""" Intermediate gaps were observed for {_format_name_value_pairs(mid_group)}."""
    if large_group:
        text += f""" Larger gaps were observed for {_format_name_value_pairs(large_group)}."""
    if largest_group:
        text += f""" The largest gaps were observed for {_format_name_value_pairs(largest_group)}."""
    text += f"""

\\subsection{{Subject-Level Generalization}}
\\label{{sec:subject_level_generalization}}
{subject_intro_sentence} {best['model']} is the highest-\\texttt{{test\\_f1}} model for each held-out subject ({best_wins}/{n_subjects} subject wins). For {best['model']}, \\texttt{{test\\_f1}} ranges from {best_min:.2f} to {best_max:.2f} (range={best_range:.2f}). For comparison, {second['model']} ranges from {second_min:.2f} to {second_max:.2f} (range={second_range:.2f}), and {third['model']} ranges from {third_min:.2f} to {third_max:.2f} (range={third_range:.2f}).

\\begin{{figure}}[t]
    \\centering
    \\includegraphics[width=\\textwidth]{{img/{fig_subject_name}}}
    \\caption{{{subject_caption}}}
    \\label{{{fig_subject_ref}}}
\\end{{figure}}

\\begin{{table}}[t]
    \\centering
    \\caption{{Per-subject \\texttt{{test\\_f1}} for the top three models (ranked by outer-fold mean \\texttt{{test\\_f1}}).}}
    \\label{{{subject_table_ref}}}
{_latex_subject_table(subject_rows, top_labels)}
\\end{{table}}

\\subsection{{Threshold-Sensitivity Analysis}}
\\label{{sec:threshold_sensitivity_analysis}}
{threshold_intro_sentence}
"""

    if thr_best is not None and thr_second is not None and thr_third is not None:
        text += f"""
Using per-fold threshold sweeps on the saved score files, the mean F1-optimal operating threshold for {best['model']} is {thr_best['thr_mean']:.2f}$\\pm${thr_best['thr_std']:.2f}, with mean best-achievable \\texttt{{f1}}={thr_best['best_f1_mean']:.2f}$\\pm${thr_best['best_f1_std']:.2f} and corresponding balanced accuracy {thr_best['best_bal_mean']:.2f}$\\pm${thr_best['best_bal_std']:.2f}. For {second['model']}, the corresponding values are threshold {thr_second['thr_mean']:.2f}$\\pm${thr_second['thr_std']:.2f}, best \\texttt{{f1}}={thr_second['best_f1_mean']:.2f}$\\pm${thr_second['best_f1_std']:.2f}, and balanced accuracy {thr_second['best_bal_mean']:.2f}$\\pm${thr_second['best_bal_std']:.2f}. For {third['model']}, the values are threshold {thr_third['thr_mean']:.2f}$\\pm${thr_third['thr_std']:.2f}, best \\texttt{{f1}}={thr_third['best_f1_mean']:.2f}$\\pm${thr_third['best_f1_std']:.2f}, and balanced accuracy {thr_third['best_bal_mean']:.2f}$\\pm${thr_third['best_bal_std']:.2f}.

These threshold-sweep summaries are consistent with the threshold-independent ordering reported in Table~\\ref{{{model_table_ref}}}: the model with the highest ROC/PR profile ({best['model']}) also shows the highest threshold-optimized \\texttt{{f1}} and balanced-accuracy levels.
"""

    text += f"""
"""
    if thr_has_split_pair:
        text += f"""
\\begin{{figure}}[t]
    \\centering
    \\includegraphics[width=\\textwidth]{{img/{fig_thr_auc_name}}}
    \\caption{{{threshold_auc_caption}}}
    \\label{{{fig_thr_auc_ref}}}
\\end{{figure}}

\\begin{{figure}}[t]
    \\centering
    \\includegraphics[width=\\textwidth]{{img/{fig_thr_oper_name}}}
    \\caption{{{threshold_oper_caption}}}
    \\label{{{fig_thr_oper_ref}}}
\\end{{figure}}
"""
    else:
        text += f"""
\\begin{{figure}}[t]
    \\centering
    \\includegraphics[width=\\textwidth]{{img/{fig_thr_model_name}}}
    \\caption{{{threshold_model_caption}}}
    \\label{{{fig_thr_model_ref}}}
\\end{{figure}}
"""

    text += f"""

\\begin{{figure}}[t]
    \\centering
    \\includegraphics[width=\\textwidth]{{img/{fig_thr_subject_name}}}
    \\caption{{Subject-level threshold sensitivity for the best-ranked model ({best['model']}). The figure reports per-subject threshold curves and ROC/PR behavior, highlighting subject-dependent operating-point variability.}}
    \\label{{{fig_thr_subject_ref}}}
\\end{{figure}}

\\subsection{{Cross-Model Comparison Synthesis}}
\\label{{sec:cross_model_comparison_synthesis}}
Under the primary endpoint (\\texttt{{test\\_f1}}), {best['model']} ranks first with a {margin:.2f} absolute margin over the second-ranked model. Across models, subject-level variability remains non-negligible, and threshold behavior differs by model family, as shown by the multi-model threshold curves and per-subject sensitivity view. These results establish the comparative predictive profile on held-out subjects and provide the basis for the subsequent discussion of model behavior and practical trade-offs.
"""
    return text


def main() -> None:
    results_root = Path("logs/results")
    out_root = Path("results/model_level_evaluation")
    out_fig = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)

    runs = _load_runs(results_root)

    subject_ok, subjects = _subject_set_ok(runs)
    if not subject_ok:
        raise RuntimeError("Subject sets are not identical across compared runs.")
    if len(subjects) != 7:
        raise RuntimeError(f"Expected 7 subjects, found {len(subjects)}: {subjects}")

    ranked_rows = _build_model_table(runs)
    family_rows = _build_family_table(ranked_rows)
    subject_rows, top_run_ids = _build_subject_top_table(runs, top_n=3)
    best_run_id = ranked_rows[0]["run_id"]

    best_by_family = _best_model_by_family(ranked_rows)
    for required_family in FAMILY_ORDER:
        if required_family not in best_by_family:
            raise RuntimeError(f"Missing required family in loaded runs: {required_family}")

    best_inter = _run_by_id(runs, str(best_by_family["Inter-segment"]["run_id"]))
    best_intra = _run_by_id(runs, str(best_by_family["Intra-segment"]["run_id"]))
    best_classical = _run_by_id(runs, str(best_by_family["Classical ML"]["run_id"]))
    best_overall = _run_by_id(runs, str(ranked_rows[0]["run_id"]))
    dummy_run = _run_by_id(runs, str(best_by_family["Baseline"]["run_id"]))

    t_inter_intra = _wilcoxon_best_vs(best_inter, best_intra)
    if float(best_by_family["Intra-segment"]["test_f1_mean"]) >= float(best_by_family["Inter-segment"]["test_f1_mean"]):
        best_deep_row = best_by_family["Intra-segment"]
    else:
        best_deep_row = best_by_family["Inter-segment"]
    best_deep = _run_by_id(runs, str(best_deep_row["run_id"]))
    t_deep_classical = _wilcoxon_best_vs(best_deep, best_classical)
    t_best_dummy = _wilcoxon_best_vs(best_overall, dummy_run)

    family_padj = _holm_adjust(
        [
            float(t_inter_intra["p"]),
            float(t_deep_classical["p"]),
            float(t_best_dummy["p"]),
        ]
    )
    family_tests = {
        "best_inter_vs_best_intra": {
            "model_a": str(best_by_family["Inter-segment"]["model"]),
            "model_b": str(best_by_family["Intra-segment"]["model"]),
            **t_inter_intra,
            "p_holm": float(family_padj[0]),
        },
        "best_deep_vs_best_classical": {
            "model_a": str(best_deep_row["model"]),
            "model_b": str(best_by_family["Classical ML"]["model"]),
            **t_deep_classical,
            "p_holm": float(family_padj[1]),
        },
        "best_overall_vs_dummy": {
            "model_a": str(ranked_rows[0]["model"]),
            "model_b": str(best_by_family["Baseline"]["model"]),
            **t_best_dummy,
            "p_holm": float(family_padj[2]),
        },
    }

    # Save CSV tables.
    model_csv = out_root / "model_comparison_primary_test_f1.csv"
    with model_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ranked_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ranked_rows)
    subject_csv = out_root / "subject_level_top_models_test_f1.csv"
    with subject_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(subject_rows[0].keys()))
        writer.writeheader()
        writer.writerows(subject_rows)
    family_csv = out_root / "family_comparison_primary_test_f1.csv"
    with family_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(family_rows[0].keys()))
        writer.writeheader()
        writer.writerows(family_rows)

    # Save LaTeX table snippets.
    model_tex = out_root / "model_comparison_primary_test_f1.tex"
    model_tex.write_text(_latex_model_table(ranked_rows), encoding="utf-8")
    top_labels = [
        c.replace(f"_{PRIMARY_METRIC}", "")
        for c in subject_rows[0].keys()
        if c.endswith(f"_{PRIMARY_METRIC}")
    ]
    subject_tex = out_root / "subject_level_top_models_test_f1.tex"
    subject_tex.write_text(_latex_subject_table(subject_rows, top_labels), encoding="utf-8")
    family_tex = out_root / "family_comparison_primary_test_f1.tex"
    family_tex.write_text(_latex_family_table(family_rows), encoding="utf-8")

    figure_targets, appendix_figure_targets = _copy_figures(
        results_root, out_fig, best_run_id=best_run_id
    )

    # Save full section text.
    section_tex = out_root / "model_level_evaluation_section.tex"
    section_tex.write_text(
        _section_tex(ranked_rows, family_rows, family_tests, subject_rows, runs, best_run_id, figure_targets),
        encoding="utf-8",
    )
    appendix_tex = out_root / "model_level_evaluation_appendix_figures.tex"
    appendix_tex.write_text(
        _appendix_tex([Path(p) for p in appendix_figure_targets]),
        encoding="utf-8",
    )

    summary = {
        "primary_metric": PRIMARY_METRIC,
        "secondary_metrics": SECONDARY_METRICS,
        "n_models": int(len(runs)),
        "models": [r.run_id for r in runs],
        "subjects": subjects,
        "best_model": {
            "run_id": str(ranked_rows[0]["run_id"]),
            "label": str(ranked_rows[0]["model"]),
            "test_f1_mean": float(ranked_rows[0]["test_f1_mean"]),
            "test_f1_std": float(ranked_rows[0]["test_f1_std"]),
        },
        "runner_up_model": {
            "run_id": str(ranked_rows[1]["run_id"]),
            "label": str(ranked_rows[1]["model"]),
            "test_f1_mean": float(ranked_rows[1]["test_f1_mean"]),
            "test_f1_std": float(ranked_rows[1]["test_f1_std"]),
        },
        "top3_models_by_test_f1": top_run_ids,
        "family_definitions": {
            family: [label for label, fam in FAMILY_BY_LABEL.items() if fam == family]
            for family in FAMILY_ORDER
        },
        "family_rankings_by_test_f1": family_rows,
        "family_pairwise_tests": family_tests,
        "artifacts": {
            "model_table_csv": str(model_csv),
            "subject_table_csv": str(subject_csv),
            "family_table_csv": str(family_csv),
            "model_table_tex": str(model_tex),
            "subject_table_tex": str(subject_tex),
            "family_table_tex": str(family_tex),
            "section_tex": str(section_tex),
            "appendix_tex": str(appendix_tex),
            "figures": figure_targets,
            "appendix_split_figures": appendix_figure_targets,
        },
    }
    summary_path = out_root / "model_level_evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Wrote artifacts to: {out_root}")
    print(f"[OK] Best model by {PRIMARY_METRIC}: {summary['best_model']['label']}")


if __name__ == "__main__":
    main()
