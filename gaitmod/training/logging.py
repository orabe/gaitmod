"""Logging utilities for training."""

import logging
import os
import re
import sys
import time
from typing import Optional

import psutil


# Prefixes that mark highly detailed log lines. These are filtered out of the console
# unless the user explicitly enables verbose output.
DETAILED_CONSOLE_PREFIXES = (
    "[BUILD_MODEL]",
    "[BUILD_PIPELINE]",
    "[CALLBACKS]",
    "[CV_SKLEARN]",
    "[FEATURE_SELECTOR]",
    "[FILTER]",
    "[FIT]",
    "[GROUP]",
    "[HISTORY]",
    "[HPARAMS]",
    "[LSTM FIT]",
    "[MASK SEARCH]",
    "[PAD]",
    "[PARAM_GRID]",
    "[PARSE]",
    "[X_data_mask]",
)


class ConsoleVerbosityFilter(logging.Filter):
    """Filter out noisy informational logs unless verbose mode is requested."""

    def __init__(self, verbose_level: int):
        super().__init__()
        self.verbose_level = verbose_level

    def filter(self, record: logging.LogRecord) -> bool:
        if self.verbose_level >= 3:
            return True
        if record.levelno >= logging.WARNING:
            return True

        message = (record.getMessage() or "").lstrip()
        if not message:
            return True

        return not any(message.startswith(prefix) for prefix in DETAILED_CONSOLE_PREFIXES)


def log_memory_usage() -> None:
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / (1024**3)
    logging.info("[MEMORY] Current RAM usage: %.2f GB", mem_gb)


def setup_logging(verbose_level: int = 2, log_dir: Optional[str] = None) -> Optional[str]:
    """
    Configure logging with different verbosity levels and optional file logging.

    Args:
        verbose_level (int):
            0 = ERROR only (quiet)
            1 = WARNING and above
            2 = INFO and above (default - normal output)
            3 = DEBUG and above (most verbose)
        log_dir (str, optional): Directory for log file. If None, console only.

    Returns:
        str: Path to log file if log_dir provided, None otherwise
    """
    log_levels = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }

    level = log_levels.get(verbose_level, logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Remove any existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    if verbose_level < 3:
        console_handler.addFilter(ConsoleVerbosityFilter(verbose_level))
    logging.root.addHandler(console_handler)

    log_file = None
    # Setup file handler if log directory specified
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"seq_model_training_{timestamp}.log")

        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        if verbose_level < 3:
            file_handler.addFilter(ConsoleVerbosityFilter(verbose_level))
        logging.root.addHandler(file_handler)

        logging.info("Logging initialized. Log file: %s", log_file)

    # Configure root logger
    logging.root.setLevel(level)

    # Suppress TensorFlow logging unless in debug mode
    if verbose_level < 3:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf_logger = logging.getLogger('tensorflow')
        tf_logger.setLevel(logging.ERROR)

    return log_file


def parse_outer_subject_selection(selection_str: Optional[str]):
    """
    Parse a comma-separated string of outer test subject names.

    Args:
        selection_str (str or None): e.g., "PW_EM59,PW_SN61" to run only those subjects.

    Returns:
        list[str] or None: List of trimmed subject names (as provided), or None if not provided.
    """
    if not selection_str:
        return None

    filters = [token.strip() for token in selection_str.split(',') if token.strip()]
    return filters if filters else None


def sanitize_path_component(component: Optional[str]) -> Optional[str]:
    """Make a string filesystem-friendly. Returns None if nothing valid remains."""
    if component is None:
        return None
    text = str(component).strip()
    if not text:
        return None
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    sanitized = re.sub(r"_{2,}", "_", sanitized).strip("_")
    return sanitized or None


__all__ = [
    "DETAILED_CONSOLE_PREFIXES",
    "ConsoleVerbosityFilter",
    "log_memory_usage",
    "setup_logging",
    "parse_outer_subject_selection",
    "sanitize_path_component",
]
