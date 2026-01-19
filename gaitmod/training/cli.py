import argparse

from gaitmod.training.trainer import TrainConfig, Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gaitmod training CLI")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["train", "evaluate", "cv", "tune"],
        default="train",
        help="Training subcommand (default: train)",
    )
    parser.add_argument(
        "--hyperparams-config",
        "--config",
        dest="hyperparams_config",
        required=True,
        help="Path to the hyperparameter JSON config.",
    )
    parser.add_argument(
        "--global-params",
        default=None,
        help="Path to JSON with global_best_params to skip inner CV and reuse fixed hyperparameters.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Classifier to train (overrides config model_type).",
    )
    parser.add_argument(
        "--outer-subjects",
        type=str,
        default=None,
        help="Comma-separated list of outer subjects to run (e.g., 'PW_EM59,PW_SN61')",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional string inserted into log directory names.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=3,
        help="Verbosity level (0-3).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs to use where applicable.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "evaluate" and not args.global_params:
        parser.error("--global-params is required for evaluate.")

    config = TrainConfig.from_args(args)
    trainer = Trainer(config)

    if args.command in ("train", "cv", "tune"):
        return trainer.fit()
    if args.command == "evaluate":
        return trainer.evaluate()

    raise ValueError(f"Unknown command: {args.command}")


__all__ = ["main", "build_parser"]
