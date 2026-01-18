from __future__ import annotations

import ast
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from PySide6 import QtCore, QtGui, QtWidgets


STYLE_SHEET = """
QMainWindow {
    background: #ffffff;
    color: #111111;
    font-family: "SF Pro Text", "Inter", "Segoe UI", sans-serif;
    font-size: 13px;
}
QWidget {
    color: #111111;
}
QFrame#NavBar {
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
}
QLabel[brand="true"],
QAbstractButton[brand="true"] {
    font-size: 16px;
    font-weight: 700;
}
QAbstractButton[brand="true"] {
    background: transparent;
    border: none;
    padding: 0;
}
QAbstractButton[brand="true"]:hover {
    color: #6400ff;
}
QPushButton {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 6px 12px;
}
QPushButton:hover {
    background: #e2e8f0;
}
QPushButton[nav="true"] {
    background: transparent;
    border: none;
    color: #111111;
    padding: 6px 8px;
}
QPushButton[nav="true"]:hover {
    color: #6400ff;
}
QPushButton[navActive="true"] {
    color: #6400ff;
}
QPushButton[accent="true"] {
    background: #6400ff;
    color: #ffffff;
    border: 1px solid #6400ff;
    padding: 8px 16px;
}
QPushButton[accent="true"]:hover {
    background: #5200d6;
}
QPushButton[warning="true"] {
    background: #fef2f2;
    color: #b91c1c;
    border: 1px solid #fecaca;
}
QTabWidget::pane {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    background: #ffffff;
    padding-top: 8px;
}
QTabBar::tab {
    background: #ffffff;
    padding: 8px 14px;
    margin-right: 6px;
    border: 1px solid #e2e8f0;
    border-bottom: 2px solid transparent;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}
QTabBar::tab:selected {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-bottom: 2px solid #6400ff;
    color: #111111;
}
QGroupBox {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    margin-top: 10px;
    background: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #111111;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QListWidget {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 6px 8px;
    background: #ffffff;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {
    border: 1px solid #6400ff;
}
QListWidget[sidebar="true"] {
    background: #f8fafc;
}
QScrollArea {
    border: none;
}
QLabel[heroTitle="true"] {
    font-size: 28px;
    font-weight: 700;
}
QLabel[heroSub="true"] {
    color: #4b5563;
}
QLabel[sectionTitle="true"] {
    font-size: 18px;
    font-weight: 600;
}
QLabel[sectionTitleLarge="true"] {
    font-size: 22px;
    font-weight: 700;
}
QLabel[muted="true"] {
    color: #6b7280;
    font-size: 11px;
}
QLabel[error="true"] {
    color: #b00020;
    font-size: 11px;
}
QFrame#HeroCard {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
}
QFrame[card="true"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
}
QSplitter::handle {
    background: #e2e8f0;
}
QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #cbd5f5;
    min-height: 24px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background: #6400ff;
}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    background: none;
}
QScrollBar:horizontal {
    background: transparent;
    height: 10px;
    margin: 2px;
}
QScrollBar::handle:horizontal {
    background: #cbd5f5;
    min-width: 24px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal:hover {
    background: #6400ff;
}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0px;
}
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {
    background: none;
}
QWidget[invalid="true"] {
    background: #fff1f1;
    border: 1px solid #fecaca;
    border-radius: 8px;
}
QWidget[highlight="true"] {
    background: #f1edff;
    border: 1px solid #c9b6ff;
    border-radius: 8px;
}
"""



Segment = Union[str, int]


@dataclass
class FieldBinding:
    path: List[Segment]
    getter: Callable[[], Any]
    validator: Callable[[Any], Optional[str]]
    hint_builder: Callable[[Any], str]
    row_widget: QtWidgets.QWidget
    error_label: QtWidgets.QLabel
    hint_label: QtWidgets.QLabel
    path_str: str
    section: str


class ConfigStats:
    def __init__(self) -> None:
        self.values: Dict[str, set] = defaultdict(set)
        self.list_values: Dict[str, set] = defaultdict(set)
        self.numeric: Dict[str, Dict[str, Optional[float]]] = defaultdict(
            lambda: {"min": None, "max": None, "is_int": True}
        )

    def record_value(self, path_norm: str, value: Any) -> None:
        self.values[path_norm].add(value)
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            self._record_numeric(path_norm, float(value), is_int=True)
        elif isinstance(value, float):
            self._record_numeric(path_norm, value, is_int=False)

    def record_list_value(self, path_norm: str, value: Any) -> None:
        self.list_values[path_norm].add(value)
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            self._record_numeric(path_norm, float(value), is_int=True)
        elif isinstance(value, float):
            self._record_numeric(path_norm, value, is_int=False)

    def _record_numeric(self, path_norm: str, value: float, is_int: bool) -> None:
        record = self.numeric[path_norm]
        record["is_int"] = record["is_int"] and is_int
        record["min"] = value if record["min"] is None else min(record["min"], value)
        record["max"] = value if record["max"] is None else max(record["max"], value)

    def options_for(self, path_norm: str, current: Any) -> List[Any]:
        values = set(self.values.get(path_norm, set()))
        if current is not None and _is_hashable(current):
            values.add(current)
        return sorted(values, key=lambda item: str(item))

    def list_options_for(self, path_norm: str, current: Iterable[Any]) -> List[Any]:
        values = set(self.list_values.get(path_norm, set()))
        for item in current:
            values.add(item)
        return sorted(values, key=lambda item: str(item))

    def numeric_range_for(self, path_norm: str, current: float) -> Tuple[float, float, bool]:
        record = self.numeric.get(path_norm, {})
        min_val = record.get("min")
        max_val = record.get("max")
        is_int = bool(record.get("is_int", False))
        if min_val is None or max_val is None:
            span = max(abs(current), 1.0) * 10
            return current - span, current + span, is_int
        return min_val, max_val, is_int


class LocalConfigStore:
    def __init__(self, config_dir: Path) -> None:
        self.config_dir = config_dir

    def set_config_dir(self, path: Path) -> None:
        self.config_dir = path

    def list_configs(self) -> List[str]:
        if not self.config_dir.is_dir():
            return []
        return sorted([p.name for p in self.config_dir.glob("*.json") if p.is_file()])

    def get_config_path(self, name: str) -> Path:
        if not name:
            raise RuntimeError("No config selected")
        path = (self.config_dir / name).resolve()
        if not path.is_file():
            raise RuntimeError(f"Config not found: {path}")
        return path

    def load_config(self, name: str) -> dict:
        path = self.get_config_path(name)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_config(self, name: str, payload: dict) -> None:
        path = self.get_config_path(name)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")


def _normalize_path(segments: List[Segment]) -> str:
    parts = []
    for segment in segments:
        if isinstance(segment, int):
            parts.append("[]")
        else:
            parts.append(segment)
    return ".".join(parts)


def _format_path(segments: List[Segment]) -> str:
    parts: List[str] = []
    for segment in segments:
        if isinstance(segment, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{segment}]"
            else:
                parts.append(f"[{segment}]")
        else:
            parts.append(segment)
    return ".".join(parts)


def _is_hashable(value: Any) -> bool:
    try:
        hash(value)
        return True
    except TypeError:
        return False


def _is_feature_params_path(path: List[Segment]) -> bool:
    for segment in path:
        if isinstance(segment, str) and segment == "feature_params":
            return True
    return False


def _truncate(value: str, max_len: int = 40) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _describe_value(value: Any) -> Tuple[str, str]:
    if isinstance(value, dict):
        return "object", ""
    if isinstance(value, list):
        if not value:
            return "list[empty]", ""
        item_types = sorted({type(item).__name__ for item in value})
        type_label = ",".join(item_types)
        return f"list[{type_label}] len={len(value)}", ""
    if value is None:
        return "null", ""
    if isinstance(value, bool):
        return "bool", str(value)
    if isinstance(value, (int, float)):
        return "number", str(value)
    if isinstance(value, str):
        return "string", _truncate(value)
    return type(value).__name__, _truncate(str(value))


def _populate_schema(tree: QtWidgets.QTreeWidget, payload: Any) -> None:
    tree.clear()
    if isinstance(payload, dict):
        for key, value in payload.items():
            tree.addTopLevelItem(_schema_item(key, value))
    else:
        tree.addTopLevelItem(_schema_item("(root)", payload))
    tree.expandToDepth(1)


def _schema_item(key: str, value: Any) -> QtWidgets.QTreeWidgetItem:
    type_label, example = _describe_value(value)
    item = QtWidgets.QTreeWidgetItem([key, type_label, example])
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            item.addChild(_schema_item(child_key, child_value))
    elif isinstance(value, list) and value:
        sample = value[0]
        sample_type, _ = _describe_value(sample)
        sample_item = QtWidgets.QTreeWidgetItem(["item[0]", sample_type, ""])
        if isinstance(sample, dict):
            for child_key, child_value in sample.items():
                sample_item.addChild(_schema_item(child_key, child_value))
        else:
            _, example = _describe_value(sample)
            sample_item.setText(2, example)
        item.addChild(sample_item)
    return item


def _walk_payload(payload: Any, path: Optional[List[Segment]], stats: ConfigStats) -> None:
    path = path or []
    if isinstance(payload, dict):
        for key, value in payload.items():
            _walk_payload(value, path + [key], stats)
    elif isinstance(payload, list):
        if all(isinstance(item, dict) for item in payload):
            for index, item in enumerate(payload):
                _walk_payload(item, path + [index], stats)
        elif all(not isinstance(item, (dict, list)) for item in payload):
            path_norm = _normalize_path(path)
            for item in payload:
                stats.record_list_value(path_norm, item)
        else:
            path_norm = _normalize_path(path)
            stats.record_value(path_norm, payload)
    else:
        path_norm = _normalize_path(path)
        stats.record_value(path_norm, payload)


def _load_supported_model_types(train_path: Path) -> List[str]:
    if not train_path.is_file():
        return []
    text = train_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"SUPPORTED_MODEL_TYPES:[^=]*=\s*\((.*?)\)", text, re.S)
    if not match:
        return []
    tuple_src = "(" + match.group(1) + ")"
    try:
        values = ast.literal_eval(tuple_src)
    except (SyntaxError, ValueError):
        return []
    return [item for item in values if isinstance(item, str)]


def _format_model_label(model_name: str) -> str:
    replacements = {
        "rf": "Random Forest",
        "svm": "SVM",
        "xgb": "XGBoost",
        "logreg": "Logistic Regression",
        "lda": "LDA",
        "knn": "k-NN",
        "dummy": "Dummy",
    }
    return replacements.get(model_name, model_name)


def _model_description(model_name: str) -> str:
    descriptions = {
        "Seq2SeqLSTM": "Encoder-decoder LSTM for sequence-to-sequence learning.",
        "Seq2VecLSTM": "LSTM encoder producing fixed-length embeddings.",
        "Seq2VecMLP": "MLP encoder over aggregated features.",
        "Seq2VecCNN": "Temporal CNN blocks with dense projection.",
        "Seq2VecMLPLSTM": "MLP projection followed by LSTM encoding.",
        "rf": "Random Forest ensemble on feature vectors.",
        "svm": "Support Vector Machine classifier.",
        "xgb": "Gradient-boosted trees (XGBoost-style).",
        "logreg": "Logistic regression linear classifier.",
        "lda": "Linear Discriminant Analysis baseline.",
        "knn": "k-Nearest Neighbors baseline.",
        "dummy": "Dummy baseline for pipeline checks.",
    }
    return descriptions.get(model_name, "Model option for experiment tracking.")


def _model_detail_description(model_name: str) -> str:
    details = {
        "Seq2SeqLSTM": (
            "Encoder-decoder LSTM for sequence-to-sequence learning. Suited for "
            "temporal modeling when outputs depend on the full input history."
        ),
        "Seq2VecLSTM": (
            "LSTM encoder that compresses sequences into fixed-length embeddings "
            "for classification and benchmarking."
        ),
        "Seq2VecMLP": (
            "MLP encoder over aggregated features. Simple and fast for quick "
            "iterations and baselines."
        ),
        "Seq2VecCNN": (
            "Temporal convolution blocks with dense projection. Efficient for "
            "windowed signal classification."
        ),
        "Seq2VecMLPLSTM": (
            "Hybrid MLP projection followed by LSTM encoding. Balances feature "
            "learning with temporal modeling."
        ),
        "rf": "Random Forest ensemble on feature vectors; strong non-linear baseline.",
        "svm": "Support Vector Machine classifier with margin-based decision boundaries.",
        "xgb": "Gradient-boosted trees (XGBoost-style) for tabular performance.",
        "logreg": "Logistic regression linear classifier with calibrated outputs.",
        "lda": "Linear Discriminant Analysis baseline with Gaussian assumptions.",
        "knn": "k-Nearest Neighbors baseline using distance-based voting.",
        "dummy": "Dummy baseline for pipeline sanity checks.",
    }
    return details.get(model_name, "Model option for experiment tracking.")


def _model_category(model_name: str) -> str:
    deep_models = {
        "Seq2SeqLSTM",
        "Seq2VecLSTM",
        "Seq2VecMLP",
        "Seq2VecCNN",
        "Seq2VecMLPLSTM",
    }
    classical_models = {"rf", "svm", "xgb", "logreg", "lda", "knn", "dummy"}
    if model_name in deep_models:
        return "Deep Learning"
    if model_name in classical_models:
        return "Classical ML"
    return "Other"


def _collect_stats(config_dir: Path, train_path: Path) -> ConfigStats:
    stats = ConfigStats()
    if config_dir.is_dir():
        for config_path in config_dir.glob("*.json"):
            try:
                with config_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            _walk_payload(payload, [], stats)
    model_types = _load_supported_model_types(train_path)
    if model_types:
        for value in model_types:
            stats.record_value("global_settings.model_type", value)
    return stats


def _set_value(root: Any, path: List[Segment], value: Any) -> None:
    ref = root
    for segment in path[:-1]:
        ref = ref[segment]
    ref[path[-1]] = value


def _is_path_key(key: str) -> bool:
    lowered = key.lower()
    return "path" in lowered or "dir" in lowered


def _split_subjects(raw: str) -> List[str]:
    return [item.strip() for item in raw.replace("\n", ",").split(",") if item.strip()]


def _format_number(value: float) -> str:
    return f"{value:.6g}"


class ConfigEditor(QtWidgets.QWidget):
    def __init__(self, store: LocalConfigStore, project_root: Path) -> None:
        super().__init__()
        self.store = store
        self.project_root = project_root
        self.stats = _collect_stats(self.store.config_dir, self.project_root / "gaitmod" / "train.py")
        self.field_bindings: List[FieldBinding] = []
        self.field_index: Dict[str, FieldBinding] = {}
        self.section_scrolls: Dict[str, QtWidgets.QScrollArea] = {}
        self.section_indices: Dict[str, int] = {}
        self.current_config: Optional[dict] = None
        self.validation_timer = QtCore.QTimer(self)
        self.validation_timer.setSingleShot(True)
        self.validation_timer.timeout.connect(self._validate_fields)
        self._build_ui()
        self.refresh_configs()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(16)
        header = QtWidgets.QLabel("Configuration")
        header.setStyleSheet("font-weight: 600; font-size: 16px;")
        layout.addWidget(header)

        dir_row = QtWidgets.QHBoxLayout()
        dir_row.addWidget(QtWidgets.QLabel("Config dir"))
        self.dir_input = QtWidgets.QLineEdit(str(self.store.config_dir))
        browse_button = QtWidgets.QPushButton("Browse")
        refresh_button = QtWidgets.QPushButton("Refresh")
        browse_button.clicked.connect(self.choose_dir)
        refresh_button.clicked.connect(self.refresh_configs)
        dir_row.addWidget(self.dir_input, stretch=1)
        dir_row.addWidget(browse_button)
        dir_row.addWidget(refresh_button)
        layout.addLayout(dir_row)

        file_row = QtWidgets.QHBoxLayout()
        file_row.addWidget(QtWidgets.QLabel("Config"))
        self.config_combo = QtWidgets.QComboBox()
        file_row.addWidget(self.config_combo, stretch=1)
        layout.addLayout(file_row)

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        sidebar = QtWidgets.QWidget()
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar)
        sidebar_layout.addWidget(QtWidgets.QLabel("Search"))
        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("Type to filter fields...")
        sidebar_layout.addWidget(self.search_input)
        sidebar_layout.addWidget(QtWidgets.QLabel("Sections"))
        self.section_list = QtWidgets.QListWidget()
        self.section_list.setProperty("sidebar", True)
        sidebar_layout.addWidget(self.section_list, stretch=1)
        main_splitter.addWidget(sidebar)

        self.section_stack = QtWidgets.QStackedWidget()
        main_splitter.addWidget(self.section_stack)
        main_splitter.setStretchFactor(1, 4)
        layout.addWidget(main_splitter, stretch=1)

        action_row = QtWidgets.QHBoxLayout()
        self.save_button = QtWidgets.QPushButton("Save")
        validate_button = QtWidgets.QPushButton("Validate")
        self.save_button.setProperty("accent", True)
        self.save_button.clicked.connect(self.save_config)
        validate_button.clicked.connect(self._validate_fields)
        action_row.addWidget(self.save_button)
        action_row.addWidget(validate_button)
        action_row.addStretch()
        layout.addLayout(action_row)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setProperty("muted", True)
        layout.addWidget(self.status_label)

        self.config_combo.currentTextChanged.connect(self.load_config)
        self.section_list.currentTextChanged.connect(self._on_section_selected)
        self.search_input.textChanged.connect(self._apply_filter)

    def choose_dir(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select config directory", str(self.store.config_dir)
        )
        if not selected:
            return
        self.store.set_config_dir(Path(selected))
        self.dir_input.setText(selected)
        self.refresh_configs()

    def refresh_configs(self) -> None:
        self.stats = _collect_stats(self.store.config_dir, self.project_root / "gaitmod" / "train.py")
        configs = self.store.list_configs()
        current = self.config_combo.currentText()
        self.config_combo.blockSignals(True)
        self.config_combo.clear()
        self.config_combo.addItems(configs)
        self.config_combo.blockSignals(False)
        if current in configs:
            self.config_combo.setCurrentText(current)
        elif configs:
            self.config_combo.setCurrentIndex(0)
            self.load_config(self.config_combo.currentText())
        self._set_status("Loaded list.")

    def load_config(self, name: str) -> None:
        if not name:
            return
        try:
            payload = self.store.load_config(name)
        except (RuntimeError, json.JSONDecodeError) as exc:
            self._set_status(f"Load failed: {exc}")
            return
        self.current_config = payload
        self._rebuild_form(payload)
        self._validate_fields()
        self._set_status("Loaded.")

    def save_config(self) -> None:
        if self.current_config is None:
            self._set_status("No config loaded.")
            return
        if not self._validate_fields():
            return
        name = self.config_combo.currentText()
        if not name:
            return
        try:
            for binding in self.field_bindings:
                _set_value(self.current_config, binding.path, binding.getter())
            self.store.save_config(name, self.current_config)
            self._set_status("Saved.")
        except (RuntimeError, json.JSONDecodeError) as exc:
            self._set_status(f"Save failed: {exc}")

    def _rebuild_form(self, payload: dict) -> None:
        self.field_bindings = []
        self.field_index = {}
        self.section_scrolls = {}
        self.section_indices = {}

        while self.section_stack.count():
            widget = self.section_stack.widget(0)
            self.section_stack.removeWidget(widget)
            widget.deleteLater()

        self.section_list.clear()

        for section_key, section_value in payload.items():
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            container = QtWidgets.QWidget()
            container_layout = QtWidgets.QVBoxLayout(container)
            scroll.setWidget(container)

            if isinstance(section_value, dict):
                for child_key, child_value in section_value.items():
                    child_widget = self._build_node(child_key, child_value, [section_key, child_key])
                    if child_widget is not None:
                        container_layout.addWidget(child_widget)
            else:
                child_widget = self._build_node(section_key, section_value, [section_key])
                if child_widget is not None:
                    container_layout.addWidget(child_widget)

            container_layout.addStretch()

            index = self.section_stack.addWidget(scroll)
            self.section_scrolls[section_key] = scroll
            self.section_indices[section_key] = index
            self.section_list.addItem(section_key)

        if self.section_list.count() > 0:
            self.section_list.setCurrentRow(0)

        self._apply_filter()

    def _build_node(self, key: str, value: Any, path: List[Segment]) -> Optional[QtWidgets.QWidget]:
        if isinstance(value, dict):
            group, content_layout = self._collapsible_group(key)
            for child_key, child_value in value.items():
                child_widget = self._build_node(child_key, child_value, path + [child_key])
                if child_widget is not None:
                    content_layout.addWidget(child_widget)
            return group
        if isinstance(value, list):
            return self._build_list_widget(key, value, path)
        return self._build_field_widget(key, value, path)

    def _collapsible_group(self, title: str) -> Tuple[QtWidgets.QGroupBox, QtWidgets.QVBoxLayout]:
        group = QtWidgets.QGroupBox(title)
        group.setCheckable(True)
        group.setChecked(True)
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        layout = QtWidgets.QVBoxLayout(group)
        layout.addWidget(content)
        group.toggled.connect(content.setVisible)
        return group, content_layout

    def _build_list_widget(self, key: str, value: list, path: List[Segment]) -> Optional[QtWidgets.QWidget]:
        if len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
            return self._build_range_widget(key, value, path)

        if _is_feature_params_path(path) and all(isinstance(item, (int, float)) for item in value):
            return self._build_numeric_list_widget(key, value, path)

        if all(isinstance(item, dict) for item in value):
            group, content_layout = self._collapsible_group(key)
            for index, item in enumerate(value):
                item_group, item_layout = self._collapsible_group(f"{key}[{index}]")
                for child_key, child_value in item.items():
                    child_widget = self._build_node(child_key, child_value, path + [index, child_key])
                    if child_widget is not None:
                        item_layout.addWidget(child_widget)
                content_layout.addWidget(item_group)
            return group

        if not all(not isinstance(item, (dict, list)) for item in value):
            return self._build_field_widget(key, value, path)

        path_norm = _normalize_path(path)
        options = self.stats.list_options_for(path_norm, value) or list(value)

        list_widget = QtWidgets.QListWidget()
        option_map: List[Tuple[str, Any]] = []
        for option in options:
            label = str(option)
            option_map.append((label, option))
            item = QtWidgets.QListWidgetItem(label)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if option in value else QtCore.Qt.Unchecked)
            list_widget.addItem(item)
        list_widget.setMaximumHeight(160)

        def getter() -> List[Any]:
            selected: List[Any] = []
            for row in range(list_widget.count()):
                item = list_widget.item(row)
                if item.checkState() == QtCore.Qt.Checked:
                    selected.append(option_map[row][1])
            return selected

        validator = self._make_validator(key, value)
        hint_builder = self._make_hint_builder(path_norm, value, options=options)
        return self._register_field_row(
            key,
            list_widget,
            [list_widget],
            path,
            getter,
            validator,
            hint_builder,
        )

    def _build_range_widget(self, key: str, value: list, path: List[Segment]) -> QtWidgets.QWidget:
        path_norm = _normalize_path(path)
        min_val, max_val, is_int = self.stats.numeric_range_for(path_norm, float(value[0]))

        if is_int:
            min_spin = QtWidgets.QSpinBox()
            max_spin = QtWidgets.QSpinBox()
            min_spin.setRange(int(min_val), int(max_val))
            max_spin.setRange(int(min_val), int(max_val))
            min_spin.setValue(int(value[0]))
            max_spin.setValue(int(value[1]))
        else:
            min_spin = QtWidgets.QDoubleSpinBox()
            max_spin = QtWidgets.QDoubleSpinBox()
            min_spin.setDecimals(6)
            max_spin.setDecimals(6)
            min_spin.setRange(min_val, max_val)
            max_spin.setRange(min_val, max_val)
            min_spin.setValue(float(value[0]))
            max_spin.setValue(float(value[1]))

        def sync_min(val: float) -> None:
            if val > max_spin.value():
                max_spin.setValue(val)

        def sync_max(val: float) -> None:
            if val < min_spin.value():
                min_spin.setValue(val)

        min_spin.valueChanged.connect(sync_min)
        max_spin.valueChanged.connect(sync_max)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("Min"))
        layout.addWidget(min_spin)
        layout.addWidget(QtWidgets.QLabel("Max"))
        layout.addWidget(max_spin)
        layout.addStretch()

        def getter() -> List[float]:
            return [float(min_spin.value()), float(max_spin.value())]

        validator = self._make_validator(key, value)
        hint_builder = self._make_hint_builder(path_norm, value, numeric_range=(min_val, max_val))
        return self._register_field_row(
            key,
            container,
            [min_spin, max_spin],
            path,
            getter,
            validator,
            hint_builder,
        )

    def _build_numeric_list_widget(self, key: str, value: list, path: List[Segment]) -> QtWidgets.QWidget:
        path_norm = _normalize_path(path)
        min_val, max_val, is_int = self.stats.numeric_range_for(path_norm, float(value[0]))

        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        spins: List[Union[QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox]] = []
        for index, item in enumerate(value):
            if is_int and float(item).is_integer():
                spin = QtWidgets.QSpinBox()
                spin.setRange(int(min_val), int(max_val))
                spin.setValue(int(item))
            else:
                spin = QtWidgets.QDoubleSpinBox()
                spin.setDecimals(6)
                spin.setRange(min_val, max_val)
                spin.setValue(float(item))
            spins.append(spin)
            if len(value) > 1:
                layout.addWidget(QtWidgets.QLabel(f"[{index}]"))
            layout.addWidget(spin)

        layout.addStretch()

        def getter() -> List[float]:
            return [float(spin.value()) for spin in spins]

        validator = self._make_validator(key, value)
        hint_builder = self._make_hint_builder(path_norm, value, numeric_range=(min_val, max_val))
        return self._register_field_row(
            key,
            container,
            list(spins),
            path,
            getter,
            validator,
            hint_builder,
        )

    def _build_field_widget(self, key: str, value: Any, path: List[Segment]) -> QtWidgets.QWidget:
        path_norm = _normalize_path(path)
        validator = self._make_validator(key, value)

        if isinstance(value, bool):
            widget = QtWidgets.QCheckBox()
            widget.setChecked(value)
            hint_builder = self._make_hint_builder(path_norm, value)
            return self._register_field_row(
                key,
                widget,
                [widget],
                path,
                widget.isChecked,
                validator,
                hint_builder,
            )

        if isinstance(value, int) and not isinstance(value, bool):
            min_val, max_val, _ = self.stats.numeric_range_for(path_norm, float(value))
            spin = QtWidgets.QSpinBox()
            spin.setRange(int(min_val), int(max_val))
            spin.setValue(value)
            hint_builder = self._make_hint_builder(path_norm, value, numeric_range=(min_val, max_val))
            return self._register_field_row(
                key,
                spin,
                [spin],
                path,
                spin.value,
                validator,
                hint_builder,
            )

        if isinstance(value, float):
            min_val, max_val, _ = self.stats.numeric_range_for(path_norm, value)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(min_val, max_val)
            spin.setValue(value)
            hint_builder = self._make_hint_builder(path_norm, value, numeric_range=(min_val, max_val))
            return self._register_field_row(
                key,
                spin,
                [spin],
                path,
                spin.value,
                validator,
                hint_builder,
            )

        if isinstance(value, str) or value is None:
            if _is_path_key(key):
                widget, line = self._path_picker(value or "")
                hint_builder = self._make_hint_builder(path_norm, value)
                return self._register_field_row(
                    key,
                    widget,
                    [line],
                    path,
                    line.text,
                    validator,
                    hint_builder,
                )

            options = self.stats.options_for(path_norm, value)
            combo = QtWidgets.QComboBox()
            combo.addItems([str(option) for option in options])
            if value is not None:
                combo.setCurrentText(str(value))

            def getter() -> str:
                return combo.currentText()

            hint_builder = self._make_hint_builder(path_norm, value, options=options)
            return self._register_field_row(
                key,
                combo,
                [combo],
                path,
                getter,
                validator,
                hint_builder,
            )

        label = QtWidgets.QLabel(str(value))
        hint_builder = self._make_hint_builder(path_norm, value)
        return self._register_field_row(
            key,
            label,
            [],
            path,
            lambda: value,
            validator,
            hint_builder,
        )

    def _path_picker(self, value: str) -> Tuple[QtWidgets.QWidget, QtWidgets.QLineEdit]:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        line = QtWidgets.QLineEdit(value)
        line.setReadOnly(True)
        browse_button = QtWidgets.QPushButton("Browse")

        def browse() -> None:
            selected = ""
            if "dir" in (line.text().lower() or ""):
                selected = QtWidgets.QFileDialog.getExistingDirectory(self, "Select directory")
            if not selected:
                selected, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file")
            if selected:
                line.setText(selected)

        browse_button.clicked.connect(browse)
        layout.addWidget(line, stretch=1)
        layout.addWidget(browse_button)
        return container, line

    def _register_field_row(
        self,
        key: str,
        widget: QtWidgets.QWidget,
        watch_widgets: List[QtWidgets.QWidget],
        path: List[Segment],
        getter: Callable[[], Any],
        validator: Callable[[Any], Optional[str]],
        hint_builder: Callable[[Any], str],
    ) -> QtWidgets.QWidget:
        row_container = QtWidgets.QWidget()
        row_layout = QtWidgets.QVBoxLayout(row_container)
        row_layout.setContentsMargins(0, 0, 0, 0)

        top_row = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(key)
        label.setMinimumWidth(180)
        top_row.addWidget(label)
        top_row.addWidget(widget, stretch=1)
        row_layout.addLayout(top_row)

        hint_label = QtWidgets.QLabel(hint_builder(getter()))
        hint_label.setProperty("muted", True)
        row_layout.addWidget(hint_label)

        error_label = QtWidgets.QLabel("")
        error_label.setProperty("error", True)
        row_layout.addWidget(error_label)

        for watch in watch_widgets:
            self._connect_change_signal(watch)

        path_str = _format_path(path)
        section = path[0] if path else "(root)"
        binding = FieldBinding(
            path=path,
            getter=getter,
            validator=validator,
            hint_builder=hint_builder,
            row_widget=row_container,
            error_label=error_label,
            hint_label=hint_label,
            path_str=path_str,
            section=section,
        )
        self.field_bindings.append(binding)
        self.field_index[path_str] = binding

        type_label, example = _describe_value(getter())
        tooltip = f"Type: {type_label}"
        if example:
            tooltip += f"\nExample: {example}"
        label.setToolTip(tooltip)
        widget.setToolTip(tooltip)

        return row_container

    def _connect_change_signal(self, widget: QtWidgets.QWidget) -> None:
        if isinstance(widget, QtWidgets.QCheckBox):
            widget.toggled.connect(self._schedule_validate)
        elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            widget.valueChanged.connect(self._schedule_validate)
        elif isinstance(widget, QtWidgets.QComboBox):
            widget.currentTextChanged.connect(self._schedule_validate)
        elif isinstance(widget, QtWidgets.QLineEdit):
            widget.textChanged.connect(self._schedule_validate)
        elif isinstance(widget, QtWidgets.QListWidget):
            widget.itemChanged.connect(self._schedule_validate)

    def _make_hint_builder(
        self,
        path_norm: str,
        base_value: Any,
        options: Optional[List[Any]] = None,
        numeric_range: Optional[Tuple[float, float]] = None,
    ) -> Callable[[Any], str]:
        type_label, _ = _describe_value(base_value)

        def builder(current: Any) -> str:
            parts = [f"Type: {type_label}"]
            if numeric_range is not None:
                parts.append(
                    f"Observed: {_format_number(numeric_range[0])}–{_format_number(numeric_range[1])}"
                )
            elif isinstance(base_value, (int, float)):
                min_val, max_val, _ = self.stats.numeric_range_for(path_norm, float(base_value))
                parts.append(f"Observed: {_format_number(min_val)}–{_format_number(max_val)}")

            option_values = options or []
            if option_values and len(option_values) > 1:
                preview = ", ".join(str(opt) for opt in option_values[:6])
                if len(option_values) > 6:
                    preview += "..."
                parts.append(f"Options: {preview}")

            if isinstance(current, list):
                parts.append(f"Selected: {len(current)}")
            else:
                parts.append(f"Current: {_truncate(str(current))}")

            return " | ".join(parts)

        return builder

    def _make_validator(self, key: str, base_value: Any) -> Callable[[Any], Optional[str]]:
        required = False
        if isinstance(base_value, str) and base_value:
            required = True
        if isinstance(base_value, list) and base_value:
            required = True

        def validator(current: Any) -> Optional[str]:
            if required:
                if current is None:
                    return "Value required"
                if isinstance(current, str) and not current.strip():
                    return "Value required"
                if isinstance(current, list) and not current:
                    return "Select at least one item"

            if _is_path_key(key) and isinstance(current, str) and current.strip():
                candidate = Path(current).expanduser()
                if candidate.exists():
                    return None
                if not candidate.is_absolute():
                    fallback = self.project_root / candidate
                    if fallback.exists():
                        return None
                return "Path not found"

            return None

        return validator

    def _validate_fields(self) -> bool:
        errors = 0
        for binding in self.field_bindings:
            try:
                value = binding.getter()
                error = binding.validator(value)
            except Exception as exc:
                error = str(exc)
                value = ""
            binding.hint_label.setText(binding.hint_builder(value))
            binding.error_label.setText(error or "")
            binding.row_widget.setProperty("invalid", bool(error))
            binding.row_widget.style().unpolish(binding.row_widget)
            binding.row_widget.style().polish(binding.row_widget)
            if error:
                errors += 1

        ok = errors == 0 and self.current_config is not None
        self.save_button.setEnabled(ok)
        if errors:
            self._set_status(f"{errors} field(s) need attention.")
        return ok

    def _schedule_validate(self) -> None:
        self.validation_timer.start(150)

    def _on_section_selected(self, section: str) -> None:
        if section in self.section_indices:
            self.section_stack.setCurrentIndex(self.section_indices[section])
        self._apply_filter()

    def _apply_filter(self) -> None:
        query = self.search_input.text().strip().lower()
        current_section = self.section_list.currentItem().text() if self.section_list.currentItem() else ""
        for binding in self.field_bindings:
            in_section = binding.section == current_section
            matches = not query or query in binding.path_str.lower()
            binding.row_widget.setVisible(in_section and matches)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    @staticmethod
    def _clear_highlight(widget: QtWidgets.QWidget) -> None:
        widget.setProperty("highlight", False)
        widget.style().unpolish(widget)
        widget.style().polish(widget)


class TrainingRunner(QtWidgets.QWidget):
    def __init__(self, store: LocalConfigStore, project_root: Path) -> None:
        super().__init__()
        self.store = store
        self.project_root = project_root
        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._append_output)
        self.process.started.connect(self._on_started)
        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_error)
        self._build_ui()
        self.refresh_configs()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(16)
        header = QtWidgets.QLabel("Run training locally")
        header.setStyleSheet("font-weight: 600; font-size: 16px;")
        layout.addWidget(header)

        form = QtWidgets.QFormLayout()
        self.config_combo = QtWidgets.QComboBox()
        self.run_id_input = QtWidgets.QLineEdit()
        self.outer_subjects_input = QtWidgets.QLineEdit()
        self.global_params_input = QtWidgets.QLineEdit()
        form.addRow("Config", self.config_combo)
        form.addRow("Run ID (optional)", self.run_id_input)
        form.addRow("Outer subjects (optional)", self.outer_subjects_input)
        form.addRow("Global params path (optional)", self.global_params_input)
        layout.addLayout(form)

        action_row = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Run")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.run_button.setProperty("accent", True)
        self.stop_button.setProperty("warning", True)
        self.stop_button.setEnabled(False)
        refresh_button = QtWidgets.QPushButton("Refresh configs")
        self.run_button.clicked.connect(self.run_training)
        self.stop_button.clicked.connect(self.stop_training)
        refresh_button.clicked.connect(self.refresh_configs)
        action_row.addWidget(self.run_button)
        action_row.addWidget(self.stop_button)
        action_row.addWidget(refresh_button)
        action_row.addStretch()
        layout.addLayout(action_row)

        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output, stretch=1)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setProperty("muted", True)
        layout.addWidget(self.status_label)

    def refresh_configs(self) -> None:
        configs = self.store.list_configs()
        current = self.config_combo.currentText()
        self.config_combo.clear()
        self.config_combo.addItems(configs)
        if current in configs:
            self.config_combo.setCurrentText(current)

    def run_training(self) -> None:
        if self.process.state() != QtCore.QProcess.NotRunning:
            self._set_status("Training already running.")
            return
        try:
            config_path = self.store.get_config_path(self.config_combo.currentText())
        except RuntimeError as exc:
            self._set_status(str(exc))
            return

        args = [
            str(self.project_root / "gaitmod" / "train.py"),
            "--hyperparams-config",
            str(config_path),
        ]

        run_id = self.run_id_input.text().strip()
        if run_id:
            args += ["--run-id", run_id]

        outer_subjects = _split_subjects(self.outer_subjects_input.text())
        if outer_subjects:
            args += ["--outer-subjects", ",".join(outer_subjects)]

        global_params = self.global_params_input.text().strip()
        if global_params:
            args += ["--global-params", global_params]

        self.output.clear()
        self.output.appendPlainText("Running: " + " ".join([sys.executable] + args))

        self.process.setWorkingDirectory(str(self.project_root))
        self.process.start(sys.executable, args)

    def stop_training(self) -> None:
        if self.process.state() == QtCore.QProcess.NotRunning:
            return
        self.process.terminate()
        if not self.process.waitForFinished(2000):
            self.process.kill()
        self._set_status("Stopped.")

    def _append_output(self) -> None:
        data = self.process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        if data:
            self.output.appendPlainText(data.rstrip("\n"))

    def _on_started(self) -> None:
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._set_status("Running...")

    def _on_finished(self, exit_code: int, status: QtCore.QProcess.ExitStatus) -> None:
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        state = "finished" if status == QtCore.QProcess.NormalExit else "crashed"
        self._set_status(f"Training {state} (exit {exit_code}).")

    def _on_error(self, error: QtCore.QProcess.ProcessError) -> None:
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._set_status(f"Process error: {error}")

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)


class LandingPage(QtWidgets.QWidget):
    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        on_cta: Optional[Callable[[], None]] = None,
        on_catalog: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__()
        model_names = model_names or []
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(36, 28, 36, 36)
        layout.setSpacing(24)

        hero = QtWidgets.QFrame()
        hero.setObjectName("HeroCard")
        hero_layout = QtWidgets.QVBoxLayout(hero)
        hero_layout.setContentsMargins(24, 24, 24, 24)
        hero_layout.setSpacing(12)

        hero_title = QtWidgets.QLabel("Framework for neural data classification with Deep Learning")
        hero_title.setProperty("heroTitle", True)
        hero_desc = QtWidgets.QLabel(
            "Configure, train, and benchmark sequence models with clear, reproducible "
            "experiments. Keep everything auditable while moving from prototypes to "
            "production-grade runs."
        )
        hero_desc.setWordWrap(True)
        hero_desc.setProperty("heroSub", True)

        cta = QtWidgets.QPushButton("Explore models")
        cta.setProperty("accent", True)
        if on_cta:
            cta.clicked.connect(on_cta)

        hero_layout.addWidget(hero_title)
        hero_layout.addWidget(hero_desc)
        hero_layout.addWidget(cta, alignment=QtCore.Qt.AlignLeft)
        layout.addWidget(hero)

        section_title = QtWidgets.QLabel("Model Lineup")
        section_title.setProperty("sectionTitleLarge", True)
        layout.addWidget(section_title)

        if not model_names:
            model_names = ["Seq2VecCNN", "Seq2VecLSTM", "Seq2SeqLSTM"]

        categories: Dict[str, List[str]] = {"Deep Learning": [], "Classical ML": [], "Other": []}
        for model_name in model_names:
            categories[_model_category(model_name)].append(model_name)

        for category_label, models in categories.items():
            if not models:
                continue
            cat_title = QtWidgets.QLabel(category_label)
            cat_title.setProperty("sectionTitle", True)
            layout.addWidget(cat_title)

            cards = QtWidgets.QWidget()
            cards_layout = QtWidgets.QGridLayout(cards)
            cards_layout.setHorizontalSpacing(16)
            cards_layout.setVerticalSpacing(16)

            for index, model_name in enumerate(models):
                card = QtWidgets.QFrame()
                card.setProperty("card", True)
                card_layout = QtWidgets.QVBoxLayout(card)
                card_layout.setContentsMargins(16, 16, 16, 16)
                card_layout.setSpacing(8)

                card_title = QtWidgets.QLabel(_format_model_label(model_name))
                card_title.setProperty("sectionTitle", True)
                card_text = QtWidgets.QLabel(_model_description(model_name))
                card_text.setWordWrap(True)
                card_text.setProperty("heroSub", True)

                card_layout.addWidget(card_title)
                card_layout.addWidget(card_text)

                cards_layout.addWidget(card, index // 3, index % 3)

            layout.addWidget(cards)

        catalog_button = QtWidgets.QPushButton("View full model catalog")
        catalog_button.setProperty("accent", True)
        if on_catalog:
            catalog_button.clicked.connect(on_catalog)
        layout.addWidget(catalog_button, alignment=QtCore.Qt.AlignLeft)

        case_title = QtWidgets.QLabel("Use Cases")
        case_title.setProperty("sectionTitle", True)
        layout.addWidget(case_title)

        case_container = QtWidgets.QWidget()
        case_layout = QtWidgets.QVBoxLayout(case_container)
        case_layout.setContentsMargins(0, 0, 0, 0)
        case_layout.setSpacing(12)

        cases = [
            (
                "General-purpose signal classification",
                "Configurable toolkit for end-to-end signal modeling and evaluation.",
            ),
            (
                "Modular benchmarking framework",
                "Reusable components for fair comparisons across models and features.",
            ),
            (
                "Fast, clean configuration",
                "Simple Python API and JSON configs to spin up experiments quickly.",
            ),
            (
                "Flexible hyperparameter control",
                "Tune models, features, CV, thresholds, and preprocessing in one place.",
            ),
            (
                "Reproducible LOSO-CV workflows",
                "Built for leave-one-subject-out pipelines with consistent logging.",
            ),
            (
                "Modular & extensible architecture",
                "Add new models, feature extractors, and evaluators without rewiring the pipeline.",
            ),
            (
                "Multiple feature sets",
                "Supports Raw, PSD, HCTSA, and other signal representations.",
            ),
            (
                "Multiple model families",
                "Classical ML, LSTM, and extensible to CNN/Transformers.",
            ),
            (
                "Clear pipeline boundaries",
                "Separation of preprocessing, feature extraction, training, and evaluation.",
            ),
            (
                "Interactive experiment management",
                "TensorBoard-ready monitoring plus comprehensive metrics logging.",
            ),
        ]

        for headline, summary in cases:
            card = QtWidgets.QFrame()
            card.setProperty("card", True)
            card_layout = QtWidgets.QVBoxLayout(card)
            card_layout.setContentsMargins(16, 14, 16, 14)
            card_layout.setSpacing(6)

            title = QtWidgets.QLabel(headline)
            title.setProperty("sectionTitle", True)
            text = QtWidgets.QLabel(summary)
            text.setProperty("heroSub", True)
            text.setWordWrap(True)

            card_layout.addWidget(title)
            card_layout.addWidget(text)
            case_layout.addWidget(card)

        case_layout.addStretch()
        layout.addWidget(case_container)


class WorkspacePage(QtWidgets.QWidget):
    def __init__(self, store: LocalConfigStore, project_root: Path) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(16)

        title = QtWidgets.QLabel("Workspace")
        title.setProperty("sectionTitle", True)
        layout.addWidget(title)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(ConfigEditor(store, project_root), "Configuration")
        tabs.addTab(TrainingRunner(store, project_root), "Run training")
        layout.addWidget(tabs, stretch=1)


class ModelDetailsPage(QtWidgets.QWidget):
    def __init__(self, model_names: List[str], on_back: Optional[Callable[[], None]] = None) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(36, 28, 36, 36)
        layout.setSpacing(20)

        header_row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Model catalog")
        title.setProperty("sectionTitleLarge", True)
        header_row.addWidget(title)
        header_row.addStretch()
        back_button = QtWidgets.QPushButton("Back to Models")
        back_button.setProperty("accent", True)
        if on_back:
            back_button.clicked.connect(on_back)
        header_row.addWidget(back_button)
        layout.addLayout(header_row)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(16)

        if not model_names:
            model_names = ["Seq2VecCNN", "Seq2VecLSTM", "Seq2SeqLSTM"]

        for model_name in model_names:
            card = QtWidgets.QFrame()
            card.setProperty("card", True)
            card_layout = QtWidgets.QVBoxLayout(card)
            card_layout.setContentsMargins(16, 16, 16, 16)
            card_layout.setSpacing(8)

            label = QtWidgets.QLabel(_format_model_label(model_name))
            label.setProperty("sectionTitle", True)
            desc = QtWidgets.QLabel(_model_detail_description(model_name))
            desc.setWordWrap(True)
            desc.setProperty("heroSub", True)

            card_layout.addWidget(label)
            card_layout.addWidget(desc)
            container_layout.addWidget(card)

        container_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll, stretch=1)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, store: LocalConfigStore, project_root: Path) -> None:
        super().__init__()
        self.setWindowTitle("gaitmod studio")
        self.resize(1280, 820)

        root = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        nav = QtWidgets.QFrame()
        nav.setObjectName("NavBar")
        nav_layout = QtWidgets.QHBoxLayout(nav)
        nav_layout.setContentsMargins(24, 12, 24, 12)

        brand = QtWidgets.QPushButton("GaitMod")
        brand.setProperty("brand", True)
        brand.setCursor(QtCore.Qt.PointingHandCursor)
        brand.clicked.connect(lambda: self._navigate("Home"))
        nav_layout.addWidget(brand)
        nav_layout.addSpacing(24)

        self.nav_buttons: Dict[str, QtWidgets.QPushButton] = {}
        nav_labels = ["Home", "Models", "Workspace"]
        for label in nav_labels:
            button = QtWidgets.QPushButton(label)
            button.setProperty("nav", True)
            button.setCursor(QtCore.Qt.PointingHandCursor)
            button.clicked.connect(lambda _, name=label: self._navigate(name))
            nav_layout.addWidget(button)
            self.nav_buttons[label] = button

        docs_button = QtWidgets.QPushButton("Docs")
        docs_button.setProperty("nav", True)
        docs_button.setCursor(QtCore.Qt.PointingHandCursor)
        docs_button.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl("https://gaitmod.readthedocs.io/en/latest/")
        ))
        nav_layout.addWidget(docs_button)

        nav_layout.addStretch()

        root_layout.addWidget(nav)

        self.page_stack = QtWidgets.QStackedWidget()
        model_names = _load_supported_model_types(project_root / "gaitmod" / "train.py")
        landing = LandingPage(
            model_names=model_names,
            on_cta=lambda: self._navigate("Models"),
            on_catalog=self._show_model_catalog,
        )
        landing_scroll = QtWidgets.QScrollArea()
        landing_scroll.setWidgetResizable(True)
        landing_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        landing_scroll.setWidget(landing)

        self.workspace = WorkspacePage(store, project_root)
        self.model_catalog = ModelDetailsPage(model_names=model_names, on_back=lambda: self._navigate("Models"))
        self.page_stack.addWidget(landing_scroll)
        self.page_stack.addWidget(self.workspace)
        self.page_stack.addWidget(self.model_catalog)

        root_layout.addWidget(self.page_stack, stretch=1)
        self.setCentralWidget(root)

        self._navigate("Home")

    def _navigate(self, label: str) -> None:
        if label == "Workspace":
            self.page_stack.setCurrentIndex(1)
        elif label == "Models":
            self.page_stack.setCurrentIndex(2)
        else:
            self.page_stack.setCurrentIndex(0)
        self._set_active_nav(label)

    def _show_model_catalog(self) -> None:
        self.page_stack.setCurrentIndex(2)
        self._set_active_nav("Models")

    def _set_active_nav(self, label: str) -> None:
        for name, button in self.nav_buttons.items():
            active = name == label
            button.setProperty("navActive", active)
            button.style().unpolish(button)
            button.style().polish(button)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_dir = project_root / "gaitmod" / "configs" / "hparams_configs"
    store = LocalConfigStore(default_dir)

    app = QtWidgets.QApplication([])
    app.setApplicationName("gaitmod config editor")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE_SHEET)
    window = MainWindow(store, project_root)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
