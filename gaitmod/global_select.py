#!/usr/bin/env python3
import argparse
import json
import math
import statistics
from pathlib import Path


def _parse_list(value):
    if not value:
        return []
    parts = [item.strip() for item in value.split(",")]
    return [item for item in parts if item]


def _to_float(value):
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return float(value)
        return float(value)
    except (TypeError, ValueError):
        return None


def _aggregate(values, method):
    clean = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not clean:
        return None
    if method == "median":
        return float(statistics.median(clean))
    return float(statistics.mean(clean))


def _load_json(path):
    with path.open("r") as f:
        return json.load(f)


def _find_inner_fold_results(log_roots):
    for root in log_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for path in root_path.rglob("evaluation_results.json"):
            yield path


def _param_key(params):
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser(description="Aggregate inner-fold scores to select global hyperparameters.")
    parser.add_argument(
        "--hyperparams-config",
        type=str,
        default=None,
        help="Path to hyperparameter JSON config with global_selection settings.",
    )
    parser.add_argument(
        "--log-roots",
        type=str,
        default=None,
        help="Comma-separated list of log roots to scan (e.g., logs/ExpA/PW_SN66,logs/ExpA/PW_US68).",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default=None,
        help="Metric key to aggregate (defaults to selection_score_metric in metadata).",
    )
    parser.add_argument(
        "--inner-aggregation",
        type=str,
        choices=("median", "mean"),
        default=None,
        help="Aggregation for inner-fold scores within each outer fold.",
    )
    parser.add_argument(
        "--outer-aggregation",
        type=str,
        choices=("median", "mean"),
        default=None,
        help="Aggregation across outer folds to choose global best params.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write global selection JSON.",
    )
    args = parser.parse_args()

    config_selection_metric = None
    config_inner_agg = None
    config_outer_agg = None
    config_output = None
    config_log_roots = []
    config_subjects = []
    config_experiment_name = None
    if args.hyperparams_config:
        config_path = Path(args.hyperparams_config).expanduser()
        if not config_path.is_file():
            raise SystemExit(f"Hyperparams config not found: {config_path}")
        config_data = _load_json(config_path)
        global_settings = config_data.get("global_settings", {})
        config_experiment_name = global_settings.get("experiment_name")
        config_global = global_settings.get("global_selection", {}) or {}
        config_inner_agg = config_global.get("inner_aggregation")
        config_outer_agg = config_global.get("outer_aggregation")
        config_selection_metric = config_global.get("selection_metric")
        config_output = config_global.get("output_path")
        config_subjects = config_global.get("subjects") or []

    log_roots = _parse_list(args.log_roots)
    if not log_roots and config_subjects and config_experiment_name:
        log_roots = [
            str(Path("logs") / config_experiment_name / subject)
            for subject in config_subjects
        ]
    if not log_roots:
        raise SystemExit("No log roots provided (use --log-roots or global_selection.subjects in config).")

    selection_metric = args.selection_metric or config_selection_metric
    inner_aggregation = args.inner_aggregation or config_inner_agg or "median"
    outer_aggregation = args.outer_aggregation or config_outer_agg or "median"
    output_path = args.output or config_output
    if not output_path:
        if config_experiment_name:
            output_path = str(Path("logs") / config_experiment_name)
        else:
            output_path = "logs"
    output_path = Path(output_path)
    if output_path.suffix != ".json":
        output_path = output_path / "global_selection.json"

    per_param_fold_scores = {}
    param_store = {}

    for path in _find_inner_fold_results(log_roots):
        data = _load_json(path)
        metadata = data.get("metadata", {})
        if metadata.get("refit") is True:
            continue
        if "inner_fold" not in metadata:
            continue

        metric_scores = data.get("evaluation_results", {}).get("metric_scores", {})
        if selection_metric is None:
            selection_metric = (
                metadata.get("selection_parameters", {}) or {}
            ).get("selection_score_metric")
        metric_key = selection_metric or "val_tuned_f1"
        score = _to_float(metric_scores.get(metric_key))
        if score is None:
            score = 0.0

        outer_fold = metadata.get("outer_fold")
        params = metadata.get("hyperparameters", {})
        if not isinstance(params, dict):
            continue

        key = _param_key(params)
        param_store[key] = params
        per_param_fold_scores.setdefault(key, {}).setdefault(outer_fold, []).append(score)

    if not per_param_fold_scores:
        raise SystemExit("No inner-fold evaluation results found in provided log roots.")

    per_param_summary = []
    for key, fold_scores in per_param_fold_scores.items():
        per_fold_agg = {}
        for fold_id, scores in fold_scores.items():
            agg_score = _aggregate(scores, inner_aggregation)
            if agg_score is not None:
                per_fold_agg[str(fold_id)] = agg_score

        outer_scores = list(per_fold_agg.values())
        outer_agg = _aggregate(outer_scores, outer_aggregation)
        outer_mean = _aggregate(outer_scores, "mean")

        per_param_summary.append(
            {
                "param_key": key,
                "params": param_store[key],
                "outer_fold_scores": per_fold_agg,
                "outer_agg_score": outer_agg,
                "outer_mean_score": outer_mean,
                "outer_fold_count": len(per_fold_agg),
            }
        )

    def _rank_key(item):
        outer_score = item.get("outer_agg_score")
        outer_score = -float("inf") if outer_score is None else outer_score
        mean_score = item.get("outer_mean_score")
        mean_score = -float("inf") if mean_score is None else mean_score
        fold_count = item.get("outer_fold_count", 0)
        return (outer_score, mean_score, fold_count)

    best = max(per_param_summary, key=_rank_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "selection_metric": selection_metric or "val_tuned_f1",
        "inner_aggregation": inner_aggregation,
        "outer_aggregation": outer_aggregation,
        "log_roots": log_roots,
        "global_best_params": best.get("params", {}),
        "global_best_score": best.get("outer_agg_score"),
        "param_summaries": per_param_summary,
    }

    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote global selection to {output_path}")


if __name__ == "__main__":
    main()
