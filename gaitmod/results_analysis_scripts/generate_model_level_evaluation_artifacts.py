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

FAMILY_INTER_DL = "Inter-segment (DL)"
FAMILY_INTRA_DL = "Intra-segment (DL)"
FAMILY_INTRA_CLASSICAL = "Intra-segment (Classical ML)"
FAMILY_DUMMY_BASELINE = "Dummy (Baseline)"

FAMILY_ORDER = [
    FAMILY_INTER_DL,
    FAMILY_INTRA_DL,
    FAMILY_INTRA_CLASSICAL,
    FAMILY_DUMMY_BASELINE,
]
FAMILY_BY_LABEL = {
    "InterSeg-CNN-LSTM": FAMILY_INTER_DL,
    "InterSeg-LSTM": FAMILY_INTER_DL,
    "IntraSeg-CNN": FAMILY_INTRA_DL,
    "IntraSeg-LSTM": FAMILY_INTRA_DL,
    "IntraSeg-MLP": FAMILY_INTRA_DL,
    "IntraSeg-MLP-LSTM": FAMILY_INTRA_DL,
    "LogReg": FAMILY_INTRA_CLASSICAL,
    "SVM": FAMILY_INTRA_CLASSICAL,
    "RF": FAMILY_INTRA_CLASSICAL,
    "XGB": FAMILY_INTRA_CLASSICAL,
    "Baseline-Dummy": FAMILY_DUMMY_BASELINE,
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


def _safe_float(value: object) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _mlplstm_gap_from_callbacks(run_id: str) -> Optional[float]:
    """
    Exceptional fallback for Seq2VecMLPLSTM when nested_cv_results.csv does not
    contain train_f1. Uses per-fold final-training callback CSVs:
      - train: lstm_head_weighted_sum_f1_score
      - test:  test_f1_score
    """
    pattern = f"logs/{run_id}/*/outer_fold_*_test_*/refit/*/final_training/callbacks/training_*.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        return None

    train_vals: List[float] = []
    test_vals: List[float] = []

    for fp in files:
        rows = _read_csv_rows(Path(fp))
        if not rows:
            continue
        last = rows[-1]
        tr = _safe_float(last.get("lstm_head_weighted_sum_f1_score", "nan"))
        te = _safe_float(last.get("test_f1_score", "nan"))
        if not math.isnan(tr) and not math.isnan(te):
            train_vals.append(tr)
            test_vals.append(te)

    if not train_vals or not test_vals:
        return None
    return float(np.mean(train_vals) - np.mean(test_vals))


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
    dummy_row = next((r for r in ranked_rows if r["family"] == FAMILY_DUMMY_BASELINE), None)
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


def _build_subject_top_table(
    runs: List[RunArtifacts], top_n: int = 3, include_dummy: bool = True
) -> Tuple[List[dict], List[str]]:
    ranked = sorted(
        runs,
        key=lambda r: _metric_stat(r.final_summary, PRIMARY_METRIC, "mean"),
        reverse=True,
    )
    top_runs = ranked[:top_n]
    top_ids = [r.run_id for r in top_runs]

    table_runs: List[RunArtifacts] = list(top_runs)
    if include_dummy:
        dummy_run = next((r for r in ranked if r.label == "Baseline-Dummy"), None)
        if dummy_run is None:
            raise RuntimeError(
                "Baseline-Dummy run not found; cannot include dummy model in subject-level table."
            )
        if dummy_run.run_id not in {r.run_id for r in table_runs}:
            table_runs.append(dummy_run)

    # Build subject rows from consistent subject ordering in summaries.
    subjects: List[str] = list(top_runs[0].final_summary.get("subjects", []))
    rows: List[dict] = []
    for subject in subjects:
        row = {"subject": subject}
        scores = []
        for run in table_runs:
            score_f1 = float("nan")
            score_roc = float("nan")
            for rec in run.nested_rows:
                if rec.get("test_subject_name") == subject:
                    try:
                        score_f1 = float(rec.get(PRIMARY_METRIC, "nan"))
                    except (TypeError, ValueError):
                        score_f1 = float("nan")
                    try:
                        score_roc = float(rec.get("test_roc_auc", "nan"))
                    except (TypeError, ValueError):
                        score_roc = float("nan")
                    break
            if math.isnan(score_f1):
                raise RuntimeError(
                    f"Missing/invalid {PRIMARY_METRIC} for subject='{subject}', model='{run.label}'."
                )
            if math.isnan(score_roc):
                raise RuntimeError(
                    f"Missing/invalid test_roc_auc for subject='{subject}', model='{run.label}'."
                )
            row[f"{run.label}_{PRIMARY_METRIC}"] = score_f1
            row[f"{run.label}_test_roc_auc"] = score_roc
            scores.append(score_f1)
        valid = [s for s in scores if not math.isnan(s)]
        row["mean_across_models_shown"] = float(mean(valid)) if valid else float("nan")
        row["std_across_models_shown"] = float(pstdev(valid)) if len(valid) > 1 else 0.0
        rows.append(row)

    return rows, top_ids


def _latex_model_table(rows: List[dict]) -> str:
    def _is_close(a: float, b: float, tol: float = 1e-12) -> bool:
        return abs(a - b) <= tol

    def _fmt_pm(mean_v: float, std_v: float, bold: bool = False) -> str:
        cell = f"{mean_v:.2f}$\\pm${std_v:.2f}"
        return f"\\textbf{{{cell}}}" if bold else cell

    max_f1 = max(float(r["test_f1_mean"]) for r in rows)
    max_roc = max(float(r["test_roc_auc_mean"]) for r in rows)
    max_pr = max(float(r["test_pr_auc_mean"]) for r in rows)
    max_bal = max(float(r["test_balanced_accuracy_mean"]) for r in rows)

    header = (
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccc}\n"
        "\\toprule\n"
        "Model & $\\mathrm{F1}$ & ROC-AUC & PR-AUC & Balanced Acc. \\\\\n"
        "\\midrule\n"
    )
    grouped_rows: Dict[str, List[dict]] = {fam: [] for fam in FAMILY_ORDER}
    for row in rows:
        grouped_rows.setdefault(str(row["family"]), []).append(row)

    lines = []
    first_group = True
    for fam in FAMILY_ORDER:
        fam_rows = grouped_rows.get(fam, [])
        if not fam_rows:
            continue
        if not first_group:
            lines.append("\\midrule")
        for row in fam_rows:
            model_cell = _latex_escape(row["model"])
            if int(row.get("rank_test_f1", 9999)) == 1:
                model_cell = f"\\textbf{{{model_cell}}}"

            lines.append(
                (
                    f"{model_cell} & "
                    f"{_fmt_pm(float(row['test_f1_mean']), float(row['test_f1_std']), _is_close(float(row['test_f1_mean']), max_f1))} & "
                    f"{_fmt_pm(float(row['test_roc_auc_mean']), float(row['test_roc_auc_std']), _is_close(float(row['test_roc_auc_mean']), max_roc))} & "
                    f"{_fmt_pm(float(row['test_pr_auc_mean']), float(row['test_pr_auc_std']), _is_close(float(row['test_pr_auc_mean']), max_pr))} & "
                    f"{_fmt_pm(float(row['test_balanced_accuracy_mean']), float(row['test_balanced_accuracy_std']), _is_close(float(row['test_balanced_accuracy_mean']), max_bal))} \\\\"
                )
            )
        first_group = False
    footer = "\n\\bottomrule\n\\end{tabular*}\n"
    return header + "\n".join(lines) + footer


def _latex_subject_table(rows: List[dict], top_labels: List[str]) -> str:
    def _is_close(a: float, b: float, tol: float = 1e-12) -> bool:
        return abs(a - b) <= tol

    header_label_map = {
        "InterSeg-CNN-LSTM": "\\shortstack{InterSeg-\\\\CNN-\\\\LSTM}",
        "IntraSeg-CNN": "\\shortstack{IntraSeg-\\\\CNN}",
        "InterSeg-LSTM": "\\shortstack{InterSeg-\\\\LSTM}",
        "Baseline-Dummy": "\\shortstack{Baseline-\\\\Dummy}",
    }

    def _header_lbl(lbl: str) -> str:
        return header_label_map.get(lbl, f"\\shortstack{{{_latex_escape(lbl)}}}")

    cols = "l" + "cc" * len(top_labels)
    metric_top = (
        f"\\multicolumn{{{len(top_labels)}}}{{c}}{{\\textbf{{F1}}}}"
        f" & \\multicolumn{{{len(top_labels)}}}{{c}}{{\\textbf{{ROC-AUC}}}}"
    )
    f1_cmid = f"\\cmidrule(lr){{2-{1 + len(top_labels)}}}"
    roc_cmid = f"\\cmidrule(lr){{{2 + len(top_labels)}-{1 + 2*len(top_labels)}}}"
    model_subcols = " & ".join(_header_lbl(lbl) for lbl in top_labels)
    model_subcols_both = f"{model_subcols} & {model_subcols}"
    header = (
        "\\begingroup\n"
        "\\scriptsize\n"
        "\\setlength{\\tabcolsep}{2pt}\n"
        "\\renewcommand{\\arraystretch}{1.05}\n"
        f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{cols}}}\n"
        "\\toprule\n"
        f" & {metric_top} \\\\\n"
        f"{f1_cmid}{roc_cmid}\n"
        f"Subject & {model_subcols_both} \\\\\n"
        "\\midrule\n"
    )
    lines = []
    for row in rows:
        f1_scores = [float(row[f"{label}_{PRIMARY_METRIC}"]) for label in top_labels]
        roc_scores = [float(row[f"{label}_test_roc_auc"]) for label in top_labels]
        row_max = max(f1_scores)
        roc_max = max(roc_scores)
        f1_vals: List[str] = []
        roc_vals: List[str] = []
        for label, f1_score in zip(top_labels, f1_scores):
            roc_score = float(row[f"{label}_test_roc_auc"])
            sval = f"{f1_score:.2f}"
            if _is_close(f1_score, row_max):
                sval = f"\\textbf{{{sval}}}"
            rstr = f"{roc_score:.2f}"
            if _is_close(roc_score, roc_max):
                rstr = f"\\textbf{{{rstr}}}"
            f1_vals.append(sval)
            roc_vals.append(rstr)
        vals = f1_vals + roc_vals
        lines.append(f"{_latex_escape(row['subject'])} & {' & '.join(vals)} \\\\")
    footer = "\n\\bottomrule\n\\end{tabular*}\n\\endgroup\n"
    return header + "\n".join(lines) + footer


def _latex_family_table(rows: List[dict]) -> str:
    def _is_close(a: float, b: float, tol: float = 1e-12) -> bool:
        return abs(a - b) <= tol

    def _fmt_pm(mean_v: float, std_v: float, bold: bool = False) -> str:
        cell = f"{mean_v:.2f}$\\pm${std_v:.2f}"
        return f"\\textbf{{{cell}}}" if bold else cell

    max_best_f1 = max(float(r["best_model_test_f1_mean"]) for r in rows)
    max_family_mean = max(float(r["family_mean_test_f1"]) for r in rows)
    max_delta_abs = max(float(r["delta_best_vs_dummy_abs"]) for r in rows)

    header = (
        "\\begingroup\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\renewcommand{\\arraystretch}{1.03}\n"
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}p{0.23\\textwidth}c p{0.19\\textwidth}cc p{0.17\\textwidth}}\n"
        "\\toprule\n"
        "Family & $n$ & Best Model & Best F1 & Family Mean F1 & $\\Delta$ vs Dummy \\\\\n"
        "\\midrule\n"
    )
    lines = []
    for row in rows:
        best_f1_mean = float(row["best_model_test_f1_mean"])
        best_f1_std = float(row["best_model_test_f1_std"])
        fam_mean = float(row["family_mean_test_f1"])
        fam_std = float(row["family_std_test_f1"])
        delta_abs = float(row["delta_best_vs_dummy_abs"])
        delta_rel = float(row["delta_best_vs_dummy_rel_pct"])

        family_cell = _latex_escape(row["family"])
        best_model_cell = _latex_escape(row["best_model_in_family"])
        if _is_close(best_f1_mean, max_best_f1):
            family_cell = f"\\textbf{{{family_cell}}}"
            best_model_cell = f"\\textbf{{{best_model_cell}}}"

        delta_cell = f"{delta_abs:.2f} ({delta_rel:.2f}\\%)"
        if _is_close(delta_abs, max_delta_abs):
            delta_cell = f"\\textbf{{{delta_cell}}}"

        lines.append(
            (
                f"{family_cell} & "
                f"{int(row['n_models'])} & "
                f"{best_model_cell} & "
                f"{_fmt_pm(best_f1_mean, best_f1_std, _is_close(best_f1_mean, max_best_f1))} & "
                f"{_fmt_pm(fam_mean, fam_std, _is_close(fam_mean, max_family_mean))} & "
                f"{delta_cell} \\\\"
            )
        )
    footer = "\n\\bottomrule\n\\end{tabular*}\n\\endgroup\n"
    return header + "\n".join(lines) + footer


def _latest_match(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _sanitize_stem(text: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text.strip())
    return stem.strip("_")


def _latex_escape(text: object) -> str:
    s = str(text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def _fig_hyperref(label: str, panel: str = "") -> str:
    """Return 'Figure <clickable ref>' with optional panel suffix (e.g., 'A')."""
    return f"Figure~\\hyperref[{label}]{{\\ref*{{{label}}}{panel}}}"


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
        lines.append("\\begin{figure}[!t]")
        lines.append("    \\centering")
        lines.append(
            f"    \\includegraphics[width=\\textwidth,height=0.72\\textheight,keepaspectratio]{{img/appendix/{p.name}}}"
        )
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
    compare_dir = results_root / "comparison_figures" / "test"

    # Strict mode: require split (cleaner) figures from a single consistent run folder.
    split_dirs = sorted(
        [p for p in compare_dir.glob("*/split") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not split_dirs:
        raise RuntimeError(
            f"Missing required split figure directories under: {compare_dir}. "
            "Expected pattern: logs/results/comparison_figures/test/*/split/"
        )

    selected_split_dir: Optional[Path] = None
    overall_meanstd_src: Optional[Path] = None
    subject_box_src: Optional[Path] = None

    for split_dir in split_dirs:
        cand_overall = (
            split_dir
            / f"metrics_meanstd_bars_compare_group-{MAIN_OVERALL_METRIC_GROUP}_all-models.png"
        )
        cand_subject = (
            split_dir
            / f"metrics_boxplots_compare_group-{MAIN_SUBJECT_METRIC_GROUP}_all-models.png"
        )
        if cand_overall.exists() and cand_subject.exists():
            selected_split_dir = split_dir
            overall_meanstd_src = cand_overall
            subject_box_src = cand_subject
            break

    if selected_split_dir is None or overall_meanstd_src is None or subject_box_src is None:
        raise RuntimeError(
            "Missing required split main figures in a consistent run folder. "
            f"Expected both files in the same split dir: "
            f"'metrics_meanstd_bars_compare_group-{MAIN_OVERALL_METRIC_GROUP}_all-models.png' and "
            f"'metrics_boxplots_compare_group-{MAIN_SUBJECT_METRIC_GROUP}_all-models.png'."
        )

    if not threshold_model_main_auc_src.exists():
        raise RuntimeError(
            f"Missing required model AUC threshold figure: {threshold_model_main_auc_src}"
        )
    if not threshold_model_main_operating_src.exists():
        raise RuntimeError(
            f"Missing required model operating threshold figure: {threshold_model_main_operating_src}"
        )

    # Collect additional split figures for appendix (excluding whichever are selected for main text).
    split_candidates = sorted(selected_split_dir.glob("*.png"))
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

    threshold_subject_threshold_src = (
        results_root / "scores_thresholds" / best_run_id / f"{best_run_id}_threshold_metrics_by_subject.png"
    )
    threshold_subject_auc_src = (
        results_root / "scores_thresholds" / best_run_id / f"{best_run_id}_auc_metrics_by_subject.png"
    )
    best_model_epoch_curves_src = out_fig_dir / "interseg_cnn_lstm_epoch_curves_train_test.png"
    if not threshold_subject_threshold_src.exists():
        raise RuntimeError(
            f"Missing required subject threshold figure: {threshold_subject_threshold_src}"
        )
    if not threshold_subject_auc_src.exists():
        raise RuntimeError(
            f"Missing required subject AUC figure: {threshold_subject_auc_src}"
        )
    if not best_model_epoch_curves_src.exists():
        raise RuntimeError(
            "Missing required best-model epoch-curves figure: "
            f"{best_model_epoch_curves_src}. "
            "Generate it first, then rerun this script."
        )
    threshold_subject_src = threshold_subject_threshold_src

    targets = {}
    appendix_targets: List[str] = []
    if overall_meanstd_src and overall_meanstd_src.exists():
        # Keep mean±std bars as supplementary (appendix) figure.
        tgt = appendix_dir / "metrics_meanstd_bars_all_models.png"
        tgt.write_bytes(overall_meanstd_src.read_bytes())
        appendix_targets.append(str(tgt))
        # Keep key for backward compatibility, but path now points to appendix.
        targets["overall_meanstd"] = str(tgt)
        targets["overall_source_name"] = overall_meanstd_src.name
    if threshold_model_main_src.exists():
        tgt = out_fig_dir / "threshold_metrics_all_models.png"
        tgt.write_bytes(threshold_model_main_src.read_bytes())
        targets["threshold_model"] = str(tgt)
        targets["threshold_model_source_name"] = threshold_model_main_src.name
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
        targets["threshold_subject_source_name"] = threshold_subject_src.name
    if threshold_subject_auc_src.exists():
        tgt = appendix_dir / "threshold_metrics_subjects_auc_best_model.png"
        tgt.write_bytes(threshold_subject_auc_src.read_bytes())
        appendix_targets.append(str(tgt))
        # keep key in summary for traceability
        targets["threshold_subject_auc_appendix"] = str(tgt)
        targets["threshold_subject_auc_source_name"] = threshold_subject_auc_src.name
    targets["best_model_epoch_curves"] = str(best_model_epoch_curves_src)
    targets["best_model_epoch_curves_source_name"] = best_model_epoch_curves_src.name

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
        test_vals.append(_safe_float(rec.get("test_f1", "nan")))
        train_vals.append(_safe_float(rec.get("train_f1", "nan")))
    train_vals = [v for v in train_vals if not math.isnan(v)]
    test_vals = [v for v in test_vals if not math.isnan(v)]
    if train_vals and test_vals:
        return float(np.mean(train_vals) - np.mean(test_vals))

    # Exceptional fallback requested for Seq2VecMLPLSTM.
    if run.run_id == "Seq2VecMLPLSTM_betaChs":
        return _mlplstm_gap_from_callbacks(run.run_id)

    return None


def _threshold_summary_for_run(run_id: str) -> Optional[dict]:
    files = sorted(glob.glob(f"logs/{run_id}/*/outer_fold_*_test_*/refit/*/refit_results_scores.npz"))
    if not files:
        return None

    thresholds = np.linspace(0.0, 1.0, 101)
    best_thr: List[float] = []
    best_f1: List[float] = []
    best_bal: List[float] = []
    best_prec: List[float] = []
    best_rec: List[float] = []
    best_spec: List[float] = []
    fixed_f1: List[float] = []
    fixed_bal: List[float] = []
    fixed_prec: List[float] = []
    fixed_rec: List[float] = []
    fixed_spec: List[float] = []
    delta_f1: List[float] = []
    delta_bal: List[float] = []
    delta_prec: List[float] = []
    delta_rec: List[float] = []
    delta_spec: List[float] = []

    for p in files:
        data = np.load(p)
        y = np.ravel(data["y_true"]).astype(int)
        s = np.ravel(data["y_score"]).astype(float)

        f1_vals = []
        bal_vals = []
        prec_vals = []
        rec_vals = []
        spec_vals = []
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
            prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            bal = 0.5 * (tpr + tnr)
            f1_vals.append(f1)
            bal_vals.append(bal)
            prec_vals.append(prec)
            rec_vals.append(tpr)
            spec_vals.append(tnr)
        arr_f1 = np.asarray(f1_vals, dtype=float)
        arr_bal = np.asarray(bal_vals, dtype=float)
        arr_prec = np.asarray(prec_vals, dtype=float)
        arr_rec = np.asarray(rec_vals, dtype=float)
        arr_spec = np.asarray(spec_vals, dtype=float)
        idx = int(np.argmax(arr_f1))
        idx_050 = int(np.argmin(np.abs(thresholds - 0.50)))
        best_thr.append(float(thresholds[idx]))
        best_f1.append(float(arr_f1[idx]))
        best_bal.append(float(arr_bal[idx]))
        best_prec.append(float(arr_prec[idx]))
        best_rec.append(float(arr_rec[idx]))
        best_spec.append(float(arr_spec[idx]))
        fixed_f1.append(float(arr_f1[idx_050]))
        fixed_bal.append(float(arr_bal[idx_050]))
        fixed_prec.append(float(arr_prec[idx_050]))
        fixed_rec.append(float(arr_rec[idx_050]))
        fixed_spec.append(float(arr_spec[idx_050]))
        delta_f1.append(float(arr_f1[idx] - arr_f1[idx_050]))
        delta_bal.append(float(arr_bal[idx] - arr_bal[idx_050]))
        delta_prec.append(float(arr_prec[idx] - arr_prec[idx_050]))
        delta_rec.append(float(arr_rec[idx] - arr_rec[idx_050]))
        delta_spec.append(float(arr_spec[idx] - arr_spec[idx_050]))

    return {
        "n": int(len(files)),
        "thr_mean": float(np.mean(best_thr)),
        "thr_std": float(np.std(best_thr)),
        "thr_min": float(np.min(best_thr)),
        "thr_max": float(np.max(best_thr)),
        "best_f1_mean": float(np.mean(best_f1)),
        "best_f1_std": float(np.std(best_f1)),
        "best_f1_min": float(np.min(best_f1)),
        "best_f1_max": float(np.max(best_f1)),
        "best_bal_mean": float(np.mean(best_bal)),
        "best_bal_std": float(np.std(best_bal)),
        "best_bal_min": float(np.min(best_bal)),
        "best_bal_max": float(np.max(best_bal)),
        "best_prec_mean": float(np.mean(best_prec)),
        "best_prec_std": float(np.std(best_prec)),
        "best_rec_mean": float(np.mean(best_rec)),
        "best_rec_std": float(np.std(best_rec)),
        "best_spec_mean": float(np.mean(best_spec)),
        "best_spec_std": float(np.std(best_spec)),
        "fixed_thr": 0.50,
        "f1_at_050_mean": float(np.mean(fixed_f1)),
        "f1_at_050_std": float(np.std(fixed_f1)),
        "bal_at_050_mean": float(np.mean(fixed_bal)),
        "bal_at_050_std": float(np.std(fixed_bal)),
        "prec_at_050_mean": float(np.mean(fixed_prec)),
        "prec_at_050_std": float(np.std(fixed_prec)),
        "rec_at_050_mean": float(np.mean(fixed_rec)),
        "rec_at_050_std": float(np.std(fixed_rec)),
        "spec_at_050_mean": float(np.mean(fixed_spec)),
        "spec_at_050_std": float(np.std(fixed_spec)),
        "delta_f1_best_vs_050_mean": float(np.mean(delta_f1)),
        "delta_f1_best_vs_050_std": float(np.std(delta_f1)),
        "delta_bal_best_vs_050_mean": float(np.mean(delta_bal)),
        "delta_bal_best_vs_050_std": float(np.std(delta_bal)),
        "delta_prec_best_vs_050_mean": float(np.mean(delta_prec)),
        "delta_prec_best_vs_050_std": float(np.std(delta_prec)),
        "delta_rec_best_vs_050_mean": float(np.mean(delta_rec)),
        "delta_rec_best_vs_050_std": float(np.std(delta_rec)),
        "delta_spec_best_vs_050_mean": float(np.mean(delta_spec)),
        "delta_spec_best_vs_050_std": float(np.std(delta_spec)),
    }


def _format_name_value_pairs(items: List[Tuple[str, float]]) -> str:
    if not items:
        return "none"
    parts = [f"{name} ({val:.2f})" for name, val in items]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _format_name_list(items: List[str]) -> str:
    if not items:
        return "none"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _split_gap_bands(gaps: List[Tuple[str, float]]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    Deterministic gap bands for train-test F1 generalization gaps.
    - low:       gap < 0.10
    - moderate:  0.10 <= gap < 0.20
    - high:      gap >= 0.20
    """
    low = [(name, g) for name, g in gaps if g < 0.10]
    moderate = [(name, g) for name, g in gaps if 0.10 <= g < 0.20]
    high = [(name, g) for name, g in gaps if g >= 0.20]
    return low, moderate, high


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
    dummy_row = next((r for r in ranked_rows if r["family"] == FAMILY_DUMMY_BASELINE), None)
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
    gaps_excluded: List[str] = []
    for row in ranked_rows:
        run = _run_by_id(runs, row["run_id"])
        gap = _train_test_f1_gap(run)
        if gap is None:
            gaps_excluded.append(str(row["model"]))
            continue
        gaps.append((str(row["model"]), float(gap)))
    if not gaps:
        raise RuntimeError(
            "No models with both train_f1 and test_f1 were found; cannot compute Train--Test generalization gap."
        )
    gaps = sorted(gaps, key=lambda x: x[1])
    low_gap_group, moderate_gap_group, high_gap_group = _split_gap_bands(gaps)

    thr_best = _threshold_summary_for_run(best["run_id"])
    thr_second = _threshold_summary_for_run(second["run_id"])
    thr_third = _threshold_summary_for_run(third["run_id"])
    if thr_best is None or thr_second is None or thr_third is None:
        raise RuntimeError(
            "Missing threshold sweep score files for one or more top-ranked models. "
            f"Required run_ids: {best['run_id']}, {second['run_id']}, {third['run_id']}."
        )

    top_labels = [
        c.replace(f"_{PRIMARY_METRIC}", "")
        for c in subject_rows[0].keys()
        if c.endswith(f"_{PRIMARY_METRIC}")
    ]
    top_labels_str = ", ".join(top_labels)

    model_table_ref = "tab:model_comparison_primary_test_f1"
    family_table_ref = "tab:family_comparison_primary_test_f1"
    subject_table_ref = "tab:subject_level_top_models_test_f1"
    fig_overall_appendix_ref = "fig:model_eval_supp_metrics_meanstd_bars_all_models"
    fig_subject_ref = "fig:metrics_boxplots_all_models"
    fig_thr_model_ref = "fig:threshold_metrics_all_models"
    fig_thr_auc_ref = "fig:threshold_metrics_auc_all_models"
    fig_thr_oper_ref = "fig:threshold_metrics_operating_all_models"
    fig_thr_subject_ref = "fig:threshold_metrics_subjects_best_model"
    fig_best_epoch_ref = "fig:interseg_cnn_lstm_epoch_curves_train_test"
    fig_overall_appendix_txt = _fig_hyperref(fig_overall_appendix_ref)
    fig_subject_txt = _fig_hyperref(fig_subject_ref)
    fig_thr_model_txt = _fig_hyperref(fig_thr_model_ref)
    fig_thr_auc_txt = _fig_hyperref(fig_thr_auc_ref)
    fig_thr_auc_a_txt = _fig_hyperref(fig_thr_auc_ref, "A")
    fig_thr_auc_b_txt = _fig_hyperref(fig_thr_auc_ref, "B")
    fig_thr_oper_txt = _fig_hyperref(fig_thr_oper_ref)
    fig_thr_subject_txt = _fig_hyperref(fig_thr_subject_ref)
    fig_best_epoch_txt = _fig_hyperref(fig_best_epoch_ref)

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
    has_best_epoch_figure = "best_model_epoch_curves" in figure_targets
    fig_best_epoch_name = (
        Path(str(figure_targets["best_model_epoch_curves"])).name if has_best_epoch_figure else ""
    )
    fig_thr_model_source_name = str(figure_targets.get("threshold_model_source_name", fig_thr_model_name))
    fig_thr_auc_source_name = str(
        figure_targets.get("threshold_model_auc_source_name", fig_thr_auc_name)
    )
    fig_thr_oper_source_name = str(
        figure_targets.get("threshold_model_operating_source_name", fig_thr_oper_name)
    )
    fig_overall_src_name = str(
        figure_targets.get("overall_source_name", "metrics_meanstd_bars_all_models.png")
    )
    fig_subject_src_name = str(figure_targets.get("subject_source_name", fig_subject_name))
    fig_overall_is_split = "group-" in fig_overall_src_name
    fig_subject_is_primary_split = "group-primary" in fig_subject_src_name

    overall_figure_sentence = (
        f"A complementary mean $\\pm$ standard deviation bar summary with baseline overlay is provided in the supplementary appendix ({fig_overall_appendix_txt})."
        if fig_overall_is_split
        else f"A complementary full mean $\\pm$ standard deviation bar summary with baseline overlay is provided in the supplementary appendix ({fig_overall_appendix_txt})."
    )
    subject_intro_sentence = (
        f"Subject-level behavior is summarized jointly by {fig_subject_txt} and Table~\\ref{{{subject_table_ref}}}. {fig_subject_txt} shows fold-level Test F1 distributions across models, while Table~\\ref{{{subject_table_ref}}} reports per-subject Test F1 and Test ROC-AUC using a hierarchical header (top row: metrics; second row: models) for the selected models ({top_labels_str})."
        if fig_subject_is_primary_split
        else f"Subject-level behavior is summarized jointly by {fig_subject_txt} and Table~\\ref{{{subject_table_ref}}}. {fig_subject_txt} shows fold-level score distributions across models for each metric, while Table~\\ref{{{subject_table_ref}}} reports per-subject Test F1 and Test ROC-AUC using a hierarchical header (top row: metrics; second row: models) for the selected models ({top_labels_str})."
    )
    subject_caption = (
        "Distribution of outer-fold Test F1 across models in nested LOSO evaluation (7 held-out subjects). Each box summarizes per-subject test scores for one model; the dashed line marks the Baseline-Dummy mean Test F1."
        if fig_subject_is_primary_split
        else "Distribution of outer-fold test metrics across models in nested LOSO evaluation. Each panel reports one metric, each box summarizes per-subject test scores for one model, and dashed lines indicate Baseline-Dummy means."
    )
    thr_main_is_clean = "models_threshold_curves_main" in fig_thr_model_source_name
    thr_has_split_pair = ("threshold_model_auc" in figure_targets) and ("threshold_model_operating" in figure_targets)
    threshold_intro_sentence = (
        f"Threshold-sensitivity results are reported in three complementary views. {fig_thr_auc_a_txt} and {fig_thr_auc_b_txt} summarize ROC and PR behavior, {fig_thr_oper_txt} summarizes threshold-dependent operating metrics (F1, Precision, Recall, Specificity), and {fig_thr_subject_txt} reports subject-wise threshold curves for the best-ranked model ({best['model']}). This split reduces visual crowding while preserving direct comparison."
        if thr_has_split_pair
        else f"Threshold-sensitivity results are reported in two complementary views. {fig_thr_model_txt} compares model-level mean threshold curves for a compact set of key metrics and representative models, enabling clear comparison of threshold-dependent and threshold-independent behavior. {fig_thr_subject_txt} reports subject-wise threshold curves for the best-ranked model ({best['model']}), showing how operating-point sensitivity varies across held-out subjects."
    )
    threshold_model_caption = (
        "Model-level threshold sensitivity for a compact comparison set. Threshold-based metrics are plotted against threshold; ROC and PR are shown as standard curves. Curves represent mean behavior over held-out subjects."
        if thr_main_is_clean
        else "Model-level threshold sensitivity across all evaluated models. Threshold-based metrics are plotted against threshold; ROC and PR are shown as standard curves. Curves represent mean behavior over held-out subjects."
    )
    threshold_auc_caption = (
        "Model-level threshold-independent discrimination summary. ROC and PR curves are shown for the compact comparison set; higher curves indicate stronger ranking performance across held-out subjects."
        if thr_main_is_clean
        else "Model-level threshold-independent discrimination summary. This panel includes ROC and PR curves to compare ranking performance across models."
    )
    threshold_oper_caption = (
        "Model-level operating-point sensitivity for the compact comparison set. Curves report F1, Precision, Recall, and Specificity as functions of decision threshold."
        if thr_main_is_clean
        else "Model-level operating-point sensitivity. Curves report F1, Precision, Recall, and Specificity as functions of decision threshold."
    )
    auc_quant_ref = fig_thr_auc_ref if thr_has_split_pair else fig_thr_model_ref
    oper_quant_ref = fig_thr_oper_ref if thr_has_split_pair else fig_thr_model_ref
    auc_quant_txt = _fig_hyperref(auc_quant_ref)
    oper_quant_txt = _fig_hyperref(oper_quant_ref)

    text = f"""\\section{{Model-Level Evaluation}}
\\label{{sec:model_level_evaluation}}

% Local float-tuning for dense results layout.
\\setcounter{{topnumber}}{{3}}
\\setcounter{{bottomnumber}}{{1}}
\\setcounter{{totalnumber}}{{4}}
\\renewcommand{{\\topfraction}}{{0.95}}
\\renewcommand{{\\bottomfraction}}{{0.80}}
\\renewcommand{{\\textfraction}}{{0.05}}
\\renewcommand{{\\floatpagefraction}}{{0.75}}
\\setlength{{\\textfloatsep}}{{8pt plus 2pt minus 2pt}}
\\setlength{{\\floatsep}}{{8pt plus 2pt minus 2pt}}
\\setlength{{\\intextsep}}{{8pt plus 2pt minus 2pt}}

\\subsection{{Evaluation Protocol}}
\\label{{sec:evaluation_protocol}}
All model-level results in this chapter are reported on held-out outer-fold test subjects from a nested leave-one-subject-out (LOSO) cross-validation design with seven outer folds (subjects: PW\\_EM59, PW\\_FH57, PW\\_HK59, PW\\_HZ58, PW\\_SN61, PW\\_SN66, and PW\\_US68). Hyperparameter tuning and feature-selection decisions were restricted to inner-loop training/validation data. Accordingly, no outer-test subject information was used during model selection. The primary endpoint is Test F1. Secondary endpoints are Test ROC-AUC, Test PR-AUC, Test Balanced Accuracy, Test Precision, Test Recall, and Test Specificity; threshold-tuned variants are used only in the threshold-sensitivity subsection.

\\subsection{{Overall Cross-Model Performance}}
\\label{{sec:overall_cross_model_performance}}
Table~\\ref{{{model_table_ref}}} reports mean $\\pm$ standard deviation over outer folds for the primary and key secondary metrics across all evaluated models. Ranking by Test F1 yields {best['model']} as the best-performing model with mean Test F1={best['test_f1_mean']:.2f}$\\pm${best['test_f1_std']:.2f}, followed by {second['model']} ({second['test_f1_mean']:.2f}$\\pm${second['test_f1_std']:.2f}) and {third['model']} ({third['test_f1_mean']:.2f}$\\pm${third['test_f1_std']:.2f}). The absolute margin between first and second rank is {margin:.2f} in Test F1, corresponding to a relative gain of {margin_rel:.2f}\\% with respect to {second['model']} (denominator: {second['model']} Test F1). Relative to Baseline-Dummy ({dummy_row['test_f1_mean']:.2f}$\\pm${dummy_row['test_f1_std']:.2f}), the best model improves Test F1 by {best_vs_dummy_abs:.2f} ({best_vs_dummy_rel:.2f}\\%).
{overall_figure_sentence}

Secondary metrics show the same ordering trend at the top of the table. {best['model']} attains Test ROC-AUC={best['test_roc_auc_mean']:.2f}$\\pm${best['test_roc_auc_std']:.2f} and Test PR-AUC={best['test_pr_auc_mean']:.2f}$\\pm${best['test_pr_auc_std']:.2f}, compared with {second['test_roc_auc_mean']:.2f}$\\pm${second['test_roc_auc_std']:.2f} and {second['test_pr_auc_mean']:.2f}$\\pm${second['test_pr_auc_std']:.2f} for {second['model']}. For Test Balanced Accuracy, the corresponding values are {best['test_balanced_accuracy_mean']:.2f}$\\pm${best['test_balanced_accuracy_std']:.2f} ({best['model']}) versus {second['test_balanced_accuracy_mean']:.2f}$\\pm${second['test_balanced_accuracy_std']:.2f} ({second['model']}), indicating consistent ranking across threshold-independent and threshold-dependent summaries.

Statistical comparisons were performed on outer-fold test results at subject level. For each model pair, paired Test F1 values were formed across the same held-out subjects (\\(n={n_subjects}\\)), and a two-sided Wilcoxon signed-rank test was used to test whether the median paired difference was zero. Because multiple pairwise comparisons were evaluated, raw p-values were adjusted with the Holm procedure; significance was assessed at adjusted \\(p<0.05\\).
(Plain-language interpretation: the null hypothesis states that, across subjects, neither model tends to score higher; rejecting it indicates a consistent directional advantage.)

Paired outer-fold comparisons of Test F1 confirm these ranking differences. For {best['model']} versus {second['model']}, Wilcoxon signed-rank testing gives $W={w_best_vs_second['W']:.2f}$, $p={w_best_vs_second['p']:.2f}$ (Holm-adjusted $p={p_adj[0]:.2f}$); the comparison against {third['model']} gives $W={w_best_vs_third['W']:.2f}$, $p={w_best_vs_third['p']:.2f}$ (Holm-adjusted $p={p_adj[1]:.2f}$), with {best['model']} scoring higher in {int(w_best_vs_second['wins'])}/{n_subjects} and {int(w_best_vs_third['wins'])}/{n_subjects} folds, respectively.
Family-aggregated contrasts are reported in Section~\\ref{{sec:architecture_family_comparison}}. Supplementary appendix figures provide split views by family and by metric group (threshold-independent vs threshold-dependent).

\\begin{{table}}[!t]
    \\centering
    \\caption{{Cross-model performance on held-out outer-fold test subjects (nested LOSO, 7 subjects). Values are mean $\\pm$ standard deviation across folds; higher values indicate better performance. Models are grouped by family ({FAMILY_INTER_DL}, {FAMILY_INTRA_DL}, {FAMILY_INTRA_CLASSICAL}, {FAMILY_DUMMY_BASELINE}) and ordered by Test F1 within each family. Bold entries mark the best mean value in each metric column, and the top-ranked model name is boldfaced.}}
    \\label{{{model_table_ref}}}
{_latex_model_table(ranked_rows)}
\\end{{table}}

\\subsection{{Architecture-Family Comparison}}
\\label{{sec:architecture_family_comparison}}
Using the predefined family groups from Section~\\ref{{sec:evaluation_protocol}}, Table~\\ref{{{family_table_ref}}} summarizes family-level aggregation under the primary endpoint. {family_rows[0]['family']} ranks first, followed by {family_rows[1]['family']} and {family_rows[2]['family']}. Model-level rankings remain in Table~\\ref{{{model_table_ref}}}; this subsection reports only family-level contrasts.

Relative to Baseline-Dummy (Dummy baseline), the best model in {family_rows[0]['family']} improves Test F1 by {family_rows[0]['delta_best_vs_dummy_abs']:.2f} ({family_rows[0]['delta_best_vs_dummy_rel_pct']:.2f}\\% relative to baseline), while the best models in {family_rows[1]['family']} and {family_rows[2]['family']} improve by {family_rows[1]['delta_best_vs_dummy_abs']:.2f} ({family_rows[1]['delta_best_vs_dummy_rel_pct']:.2f}\\% relative to baseline) and {family_rows[2]['delta_best_vs_dummy_abs']:.2f} ({family_rows[2]['delta_best_vs_dummy_rel_pct']:.2f}\\% relative to baseline), respectively.

Using the same paired Wilcoxon + Holm procedure described above, family-level contrasts show: best {FAMILY_INTER_DL} vs best {FAMILY_INTRA_DL} ($W={family_tests['best_inter_vs_best_intra']['W']:.2f}$, $p={family_tests['best_inter_vs_best_intra']['p']:.2f}$, Holm-adjusted $p={family_tests['best_inter_vs_best_intra']['p_holm']:.2f}$), best DL (best of {FAMILY_INTER_DL}/{FAMILY_INTRA_DL}) vs best {FAMILY_INTRA_CLASSICAL} ($W={family_tests['best_deep_vs_best_classical']['W']:.2f}$, $p={family_tests['best_deep_vs_best_classical']['p']:.2f}$, Holm-adjusted $p={family_tests['best_deep_vs_best_classical']['p_holm']:.2f}$), and best overall vs {FAMILY_DUMMY_BASELINE} ($W={family_tests['best_overall_vs_dummy']['W']:.2f}$, $p={family_tests['best_overall_vs_dummy']['p']:.2f}$, Holm-adjusted $p={family_tests['best_overall_vs_dummy']['p_holm']:.2f}$).

\\begin{{table}}[!t]
    \\centering
    \\caption{{Architecture-family comparison under the primary endpoint (Test F1). ``Best Model'' values are outer-fold mean $\\pm$ standard deviation; ``Family Mean'' summarizes model-level means within each family (not fold-level uncertainty). The final column reports gain of each family's best model relative to Baseline-Dummy (dummy classifier baseline), with percentages computed against Baseline-Dummy Test F1. Bold entries mark the leading family and best values in key columns.}}
    \\label{{{family_table_ref}}}
{_latex_family_table(family_rows)}
\\end{{table}}

\\subsection{{Train--Test Generalization Gap}}
\\label{{sec:train_test_generalization_gap}}
To quantify generalization behavior, the Train--Test F1 gap was computed as
$F1_{{\\text{{train}}}} - F1_{{\\text{{test}}}}$
from outer-fold summaries. Low-gap models were {_format_name_value_pairs(low_gap_group)}. Moderate-gap models were {_format_name_value_pairs(moderate_gap_group)}. High-gap models were {_format_name_value_pairs(high_gap_group)}."""
    if gaps_excluded:
        text += f""" Models excluded from this gap analysis (missing Train F1 in fold-level summaries) were {_format_name_list(gaps_excluded)}."""
    if has_best_epoch_figure:
        text += f"""

To complement this gap summary, {fig_best_epoch_txt} reports epoch-wise train and test trajectories for the best-performing model ({best['model']}) across F1, ROC-AUC, and loss. Curves summarize mean behavior across held-out subjects with standard-deviation shading, enabling direct inspection of train--test divergence over epochs.

\\begin{{figure}}[!t]
    \\centering
    \\includegraphics[width=\\textwidth,height=0.72\\textheight,keepaspectratio]{{img/{fig_best_epoch_name}}}
    \\caption{{Epoch-wise training and test dynamics for {best['model']} across F1, ROC-AUC, and loss. Solid curves show mean trajectories across subjects; shaded regions indicate standard deviation.}}
    \\label{{{fig_best_epoch_ref}}}
\\end{{figure}}
"""
    text += f"""

\\subsection{{Subject-Level Generalization}}
\\label{{sec:subject_level_generalization}}
{subject_intro_sentence} {best['model']} is the highest Test F1 model for each held-out subject ({best_wins}/{n_subjects} subject wins). For {best['model']}, Test F1 ranges from {best_min:.2f} to {best_max:.2f} (range={best_range:.2f}). For comparison, {second['model']} ranges from {second_min:.2f} to {second_max:.2f} (range={second_range:.2f}), and {third['model']} ranges from {third_min:.2f} to {third_max:.2f} (range={third_range:.2f}).

\\begin{{figure}}[!t]
    \\centering
    \\includegraphics[width=\\textwidth,height=0.72\\textheight,keepaspectratio]{{img/{fig_subject_name}}}
    \\caption{{{subject_caption}}}
    \\label{{{fig_subject_ref}}}
\\end{{figure}}

\\begin{{table}}[!t]
    \\centering
    \\caption{{Per-subject Test F1 and Test ROC-AUC for the top three models plus Baseline-Dummy. The header is hierarchical: top row groups metrics (F1, ROC-AUC), and the second row lists model names under each metric group. Values are held-out outer-fold test scores; bold values indicate the highest value within each metric block (F1 and ROC-AUC) for each subject row.}}
    \\label{{{subject_table_ref}}}
{_latex_subject_table(subject_rows, top_labels)}
\\end{{table}}

\\subsection{{Threshold-Sensitivity Analysis}}
\\label{{sec:threshold_sensitivity_analysis}}
{threshold_intro_sentence}
"""

    text += f"""
Here, the decision threshold \\(t\\) converts continuous model scores to class labels via \\(\\hat{{y}}=1\\) if score \\(\\ge t\\), else \\(\\hat{{y}}=0\\). Threshold tuning denotes choosing \\(t\\) to optimize an operating metric; in this subsection, threshold sweeps are applied to held-out score files as a sensitivity analysis.

Using per-fold threshold sweeps on the saved score files, the mean F1-optimal operating threshold for {best['model']} is \\({thr_best['thr_mean']:.2f}\\pm{thr_best['thr_std']:.2f}\\), with mean best-achievable F1 \\(={thr_best['best_f1_mean']:.2f}\\pm{thr_best['best_f1_std']:.2f}\\) (\\(\\Delta\\) vs \\(t=0.50\\): \\({thr_best['delta_f1_best_vs_050_mean']:+.2f}\\pm{thr_best['delta_f1_best_vs_050_std']:.2f}\\)) and corresponding Balanced Accuracy \\(={thr_best['best_bal_mean']:.2f}\\pm{thr_best['best_bal_std']:.2f}\\) (\\(\\Delta\\): \\({thr_best['delta_bal_best_vs_050_mean']:+.2f}\\pm{thr_best['delta_bal_best_vs_050_std']:.2f}\\)). For {second['model']}, the corresponding values are threshold \\({thr_second['thr_mean']:.2f}\\pm{thr_second['thr_std']:.2f}\\), best F1 \\(={thr_second['best_f1_mean']:.2f}\\pm{thr_second['best_f1_std']:.2f}\\) (\\(\\Delta\\): \\({thr_second['delta_f1_best_vs_050_mean']:+.2f}\\pm{thr_second['delta_f1_best_vs_050_std']:.2f}\\)), and Balanced Accuracy \\(={thr_second['best_bal_mean']:.2f}\\pm{thr_second['best_bal_std']:.2f}\\) (\\(\\Delta\\): \\({thr_second['delta_bal_best_vs_050_mean']:+.2f}\\pm{thr_second['delta_bal_best_vs_050_std']:.2f}\\)). For {third['model']}, the values are threshold \\({thr_third['thr_mean']:.2f}\\pm{thr_third['thr_std']:.2f}\\), best F1 \\(={thr_third['best_f1_mean']:.2f}\\pm{thr_third['best_f1_std']:.2f}\\) (\\(\\Delta\\): \\({thr_third['delta_f1_best_vs_050_mean']:+.2f}\\pm{thr_third['delta_f1_best_vs_050_std']:.2f}\\)), and Balanced Accuracy \\(={thr_third['best_bal_mean']:.2f}\\pm{thr_third['best_bal_std']:.2f}\\) (\\(\\Delta\\): \\({thr_third['delta_bal_best_vs_050_mean']:+.2f}\\pm{thr_third['delta_bal_best_vs_050_std']:.2f}\\)).

At the tuned operating points, {best['model']} reaches Precision \\(={thr_best['best_prec_mean']:.2f}\\pm{thr_best['best_prec_std']:.2f}\\), Recall \\(={thr_best['best_rec_mean']:.2f}\\pm{thr_best['best_rec_std']:.2f}\\), and Specificity \\(={thr_best['best_spec_mean']:.2f}\\pm{thr_best['best_spec_std']:.2f}\\); for {second['model']}: \\({thr_second['best_prec_mean']:.2f}\\pm{thr_second['best_prec_std']:.2f}\\), \\({thr_second['best_rec_mean']:.2f}\\pm{thr_second['best_rec_std']:.2f}\\), \\({thr_second['best_spec_mean']:.2f}\\pm{thr_second['best_spec_std']:.2f}\\); and for {third['model']}: \\({thr_third['best_prec_mean']:.2f}\\pm{thr_third['best_prec_std']:.2f}\\), \\({thr_third['best_rec_mean']:.2f}\\pm{thr_third['best_rec_std']:.2f}\\), \\({thr_third['best_spec_mean']:.2f}\\pm{thr_third['best_spec_std']:.2f}\\). Relative to \\(t=0.50\\), the corresponding changes for {best['model']} are Precision \\({thr_best['delta_prec_best_vs_050_mean']:+.2f}\\pm{thr_best['delta_prec_best_vs_050_std']:.2f}\\), Recall \\({thr_best['delta_rec_best_vs_050_mean']:+.2f}\\pm{thr_best['delta_rec_best_vs_050_std']:.2f}\\), and Specificity \\({thr_best['delta_spec_best_vs_050_mean']:+.2f}\\pm{thr_best['delta_spec_best_vs_050_std']:.2f}\\).

{auc_quant_txt} provides the threshold-independent ranking view: {best['model']} attains Test ROC-AUC \\(={best['test_roc_auc_mean']:.2f}\\pm{best['test_roc_auc_std']:.2f}\\) and Test PR-AUC \\(={best['test_pr_auc_mean']:.2f}\\pm{best['test_pr_auc_std']:.2f}\\), compared with \\({second['test_roc_auc_mean']:.2f}\\pm{second['test_roc_auc_std']:.2f}\\), \\({second['test_pr_auc_mean']:.2f}\\pm{second['test_pr_auc_std']:.2f}\\) for {second['model']} and \\({third['test_roc_auc_mean']:.2f}\\pm{third['test_roc_auc_std']:.2f}\\), \\({third['test_pr_auc_mean']:.2f}\\pm{third['test_pr_auc_std']:.2f}\\) for {third['model']}.

{oper_quant_txt} shows why threshold sweeps matter for operating metrics: the best-F1 threshold is not fixed at \\(0.50\\) and varies across folds ({best['model']}: \\({thr_best['thr_min']:.2f}\\) to \\({thr_best['thr_max']:.2f}\\)). At subject level ({fig_thr_subject_txt}), this variability appears as distinct threshold-response curves, reinforcing that operating-point choice changes F1/precision/recall/specificity trade-offs across held-out subjects.

Together, these threshold figures quantify both ranking quality (ROC/PR) and operating-point sensitivity (F1/Precision/Recall/Specificity), which are complementary for deployment-relevant model assessment.
"""

    text += f"""
"""
    if thr_has_split_pair:
        text += f"""
\\begin{{figure}}[!t]
    \\centering
    \\includegraphics[width=\\textwidth,height=0.72\\textheight,keepaspectratio]{{img/{fig_thr_auc_name}}}
    \\caption{{{threshold_auc_caption}}}
    \\label{{{fig_thr_auc_ref}}}
\\end{{figure}}

\\begin{{figure}}[!t]
    \\centering
    \\includegraphics[width=\\textwidth,height=0.72\\textheight,keepaspectratio]{{img/{fig_thr_oper_name}}}
    \\caption{{{threshold_oper_caption}}}
    \\label{{{fig_thr_oper_ref}}}
\\end{{figure}}
"""
    else:
        text += f"""
\\begin{{figure}}[!t]
    \\centering
    \\includegraphics[width=\\textwidth,height=0.72\\textheight,keepaspectratio]{{img/{fig_thr_model_name}}}
    \\caption{{{threshold_model_caption}}}
    \\label{{{fig_thr_model_ref}}}
\\end{{figure}}
"""

    text += f"""

\\begin{{figure}}[!t]
    \\centering
    \\includegraphics[width=\\textwidth,height=0.72\\textheight,keepaspectratio]{{img/{fig_thr_subject_name}}}
    \\caption{{Subject-level operating-point sensitivity for the best-ranked model ({best['model']}). Per-subject curves show how threshold-dependent metrics vary with the decision threshold on held-out test subjects, highlighting between-subject operating-point variability.}}
    \\label{{{fig_thr_subject_ref}}}
\\end{{figure}}

\\subsection{{Cross-Model Comparison Synthesis}}
\\label{{sec:cross_model_comparison_synthesis}}
Under the primary endpoint (Test F1), model-level ranking, family-level aggregation, subject-level spread, and threshold sensitivity are consistent in identifying {best['model']} as the leading configuration on held-out subjects. Family-aggregated contrasts indicate strongest performance for {family_rows[0]['family']}, while threshold analyses show that operating-point choice materially affects threshold-dependent metrics. These results establish the comparative predictive profile and set up interpretation of mechanisms and trade-offs in the Discussion.
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

    best_inter = _run_by_id(runs, str(best_by_family[FAMILY_INTER_DL]["run_id"]))
    best_intra = _run_by_id(runs, str(best_by_family[FAMILY_INTRA_DL]["run_id"]))
    best_classical = _run_by_id(runs, str(best_by_family[FAMILY_INTRA_CLASSICAL]["run_id"]))
    best_overall = _run_by_id(runs, str(ranked_rows[0]["run_id"]))
    dummy_run = _run_by_id(runs, str(best_by_family[FAMILY_DUMMY_BASELINE]["run_id"]))

    t_inter_intra = _wilcoxon_best_vs(best_inter, best_intra)
    if float(best_by_family[FAMILY_INTRA_DL]["test_f1_mean"]) >= float(best_by_family[FAMILY_INTER_DL]["test_f1_mean"]):
        best_deep_row = best_by_family[FAMILY_INTRA_DL]
    else:
        best_deep_row = best_by_family[FAMILY_INTER_DL]
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
            "model_a": str(best_by_family[FAMILY_INTER_DL]["model"]),
            "model_b": str(best_by_family[FAMILY_INTRA_DL]["model"]),
            **t_inter_intra,
            "p_holm": float(family_padj[0]),
        },
        "best_deep_vs_best_classical": {
            "model_a": str(best_deep_row["model"]),
            "model_b": str(best_by_family[FAMILY_INTRA_CLASSICAL]["model"]),
            **t_deep_classical,
            "p_holm": float(family_padj[1]),
        },
        "best_overall_vs_dummy": {
            "model_a": str(ranked_rows[0]["model"]),
            "model_b": str(best_by_family[FAMILY_DUMMY_BASELINE]["model"]),
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
