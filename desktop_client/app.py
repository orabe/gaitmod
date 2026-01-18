from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from PySide6 import QtCore, QtGui, QtWidgets


@dataclass
class ApiConfig:
    base_url: str


class ApiClient:
    def __init__(self, config: ApiConfig) -> None:
        self.config = config
        self.session = requests.Session()

    def set_base_url(self, base_url: str) -> None:
        self.config.base_url = base_url.rstrip("/")

    def health(self) -> Dict[str, Any]:
        return self._json("GET", "/api/health")

    def list_configs(self) -> List[str]:
        payload = self._json("GET", "/api/configs")
        return payload.get("configs", [])

    def load_config(self, name: str) -> Any:
        return self._json("GET", f"/api/configs/{name}")

    def save_config(self, name: str, payload: Any) -> None:
        self._json("PUT", f"/api/configs/{name}", json=payload)

    def submit_jobs(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._json("POST", "/api/jobs/submit", json=payload)

    def get_status(self, run_id: str) -> List[Dict[str, Any]]:
        payload = self._json("GET", f"/api/jobs/status/{run_id}")
        return payload.get("jobs", [])

    def get_log(self, run_id: str, job_id: str, stream: str, tail: int) -> str:
        return self._text(
            "GET",
            f"/api/jobs/log/{run_id}/{job_id}",
            params={"stream": stream, "tail": tail},
        )

    def list_runs(self) -> List[Dict[str, Any]]:
        payload = self._json("GET", "/api/results/runs")
        return payload.get("runs", [])

    def get_results(self, run_id: str) -> Dict[str, Any]:
        return self._json("GET", f"/api/results/{run_id}")

    def file_url(self, run_id: str, path: str) -> str:
        return f"{self.config.base_url}/api/results/{run_id}/file?path={requests.utils.quote(path)}"

    def _json(self, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        response = self._request(method, path, **kwargs)
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError("Invalid JSON response") from exc

    def _text(self, method: str, path: str, **kwargs: Any) -> str:
        response = self._request(method, path, **kwargs)
        return response.text

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self.config.base_url}{path}"
        try:
            response = self.session.request(method, url, timeout=15, **kwargs)
        except requests.RequestException as exc:
            raise RuntimeError(f"API request failed: {exc}") from exc
        if not response.ok:
            detail = response.text.strip() or f"{response.status_code} {response.reason}"
            raise RuntimeError(detail)
        return response


class ConfigEditor(QtWidgets.QWidget):
    def __init__(self, api: ApiClient) -> None:
        super().__init__()
        self.api = api
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QLabel("Config editor")
        header.setStyleSheet("font-weight: 600; font-size: 16px;")
        layout.addWidget(header)

        row = QtWidgets.QHBoxLayout()
        self.config_combo = QtWidgets.QComboBox()
        refresh_button = QtWidgets.QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_configs)
        row.addWidget(QtWidgets.QLabel("Config"))
        row.addWidget(self.config_combo, stretch=1)
        row.addWidget(refresh_button)
        layout.addLayout(row)

        self.editor = QtWidgets.QPlainTextEdit()
        self.editor.setPlaceholderText("Select a config to load...")
        layout.addWidget(self.editor, stretch=1)

        action_row = QtWidgets.QHBoxLayout()
        save_button = QtWidgets.QPushButton("Save")
        format_button = QtWidgets.QPushButton("Format JSON")
        save_button.clicked.connect(self.save_config)
        format_button.clicked.connect(self.format_json)
        action_row.addWidget(save_button)
        action_row.addWidget(format_button)
        action_row.addStretch()
        layout.addLayout(action_row)

        self.status_label = QtWidgets.QLabel("")
        layout.addWidget(self.status_label)

        self.config_combo.currentTextChanged.connect(self.load_config)

    def refresh_configs(self) -> None:
        try:
            configs = self.api.list_configs()
        except RuntimeError as exc:
            self._set_status(str(exc))
            return
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
            payload = self.api.load_config(name)
            self.editor.setPlainText(json.dumps(payload, indent=2))
            self._set_status("Loaded.")
        except RuntimeError as exc:
            self._set_status(f"Load failed: {exc}")

    def save_config(self) -> None:
        name = self.config_combo.currentText()
        if not name:
            return
        try:
            payload = json.loads(self.editor.toPlainText())
            self.api.save_config(name, payload)
            self._set_status("Saved.")
        except (json.JSONDecodeError, RuntimeError) as exc:
            self._set_status(f"Save failed: {exc}")

    def format_json(self) -> None:
        try:
            payload = json.loads(self.editor.toPlainText())
            self.editor.setPlainText(json.dumps(payload, indent=2))
            self._set_status("Formatted.")
        except json.JSONDecodeError as exc:
            self._set_status(f"Format failed: {exc}")

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)


class JobSubmission(QtWidgets.QWidget):
    def __init__(self, api: ApiClient) -> None:
        super().__init__()
        self.api = api
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QLabel("Job submission")
        header.setStyleSheet("font-weight: 600; font-size: 16px;")
        layout.addWidget(header)

        form = QtWidgets.QFormLayout()
        self.config_combo = QtWidgets.QComboBox()
        self.run_id_input = QtWidgets.QLineEdit()
        self.global_params_input = QtWidgets.QLineEdit()
        self.combine_checkbox = QtWidgets.QCheckBox("Submit all subjects as one batch")
        form.addRow("Config", self.config_combo)
        form.addRow("Run ID (optional)", self.run_id_input)
        form.addRow("Global params path (optional)", self.global_params_input)
        form.addRow("Mode", self.combine_checkbox)
        layout.addLayout(form)

        layout.addWidget(QtWidgets.QLabel("Subjects (comma or newline separated)"))
        self.subjects_edit = QtWidgets.QPlainTextEdit()
        layout.addWidget(self.subjects_edit)

        submit_button = QtWidgets.QPushButton("Submit")
        submit_button.clicked.connect(self.submit)
        layout.addWidget(submit_button)

        self.status_label = QtWidgets.QLabel("")
        layout.addWidget(self.status_label)

        self.result_view = QtWidgets.QPlainTextEdit()
        self.result_view.setReadOnly(True)
        layout.addWidget(self.result_view, stretch=1)

    def refresh_configs(self) -> None:
        try:
            configs = self.api.list_configs()
        except RuntimeError as exc:
            self._set_status(str(exc))
            return
        current = self.config_combo.currentText()
        self.config_combo.clear()
        self.config_combo.addItems(configs)
        if current in configs:
            self.config_combo.setCurrentText(current)

    def submit(self) -> None:
        subjects = self._split_subjects(self.subjects_edit.toPlainText())
        payload = {
            "config_name": self.config_combo.currentText(),
            "subjects": subjects,
            "run_id": self.run_id_input.text().strip() or None,
            "global_params": self.global_params_input.text().strip() or None,
            "combine_subjects": self.combine_checkbox.isChecked(),
        }
        try:
            result = self.api.submit_jobs(payload)
            self.result_view.setPlainText(json.dumps(result, indent=2))
            self._set_status("Submitted.")
        except RuntimeError as exc:
            self._set_status(f"Submit failed: {exc}")

    @staticmethod
    def _split_subjects(raw: str) -> List[str]:
        return [item.strip() for item in raw.replace("\n", ",").split(",") if item.strip()]

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)


class JobStatus(QtWidgets.QWidget):
    def __init__(self, api: ApiClient) -> None:
        super().__init__()
        self.api = api
        self.current_jobs: List[Dict[str, Any]] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QLabel("Job status")
        header.setStyleSheet("font-weight: 600; font-size: 16px;")
        layout.addWidget(header)

        top_row = QtWidgets.QHBoxLayout()
        self.run_combo = QtWidgets.QComboBox()
        refresh_button = QtWidgets.QPushButton("Refresh runs")
        refresh_button.clicked.connect(self.refresh_runs)
        status_button = QtWidgets.QPushButton("Update status")
        status_button.clicked.connect(self.refresh_status)
        top_row.addWidget(QtWidgets.QLabel("Run"))
        top_row.addWidget(self.run_combo, stretch=1)
        top_row.addWidget(refresh_button)
        top_row.addWidget(status_button)
        layout.addLayout(top_row)

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Job", "ID", "State", "Reason", "Elapsed"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        layout.addWidget(self.table)

        log_row = QtWidgets.QHBoxLayout()
        self.stream_combo = QtWidgets.QComboBox()
        self.stream_combo.addItems(["out", "err"])
        self.tail_spin = QtWidgets.QSpinBox()
        self.tail_spin.setRange(10, 5000)
        self.tail_spin.setValue(200)
        load_log_button = QtWidgets.QPushButton("Load selected log")
        load_log_button.clicked.connect(self.load_log)
        log_row.addWidget(QtWidgets.QLabel("Stream"))
        log_row.addWidget(self.stream_combo)
        log_row.addWidget(QtWidgets.QLabel("Tail lines"))
        log_row.addWidget(self.tail_spin)
        log_row.addWidget(load_log_button)
        log_row.addStretch()
        layout.addLayout(log_row)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view, stretch=1)

        self.status_label = QtWidgets.QLabel("")
        layout.addWidget(self.status_label)

    def refresh_runs(self) -> None:
        try:
            runs = self.api.list_runs()
        except RuntimeError as exc:
            self._set_status(str(exc))
            return
        current = self.run_combo.currentText()
        self.run_combo.clear()
        self.run_combo.addItems([run["run_id"] for run in runs])
        if current:
            self.run_combo.setCurrentText(current)
        if self.run_combo.currentText():
            self.refresh_status()

    def refresh_status(self) -> None:
        run_id = self.run_combo.currentText()
        if not run_id:
            return
        try:
            jobs = self.api.get_status(run_id)
            self.current_jobs = jobs
            self._populate_table(jobs)
            self._set_status("Updated.")
        except RuntimeError as exc:
            self._set_status(f"Status failed: {exc}")

    def _populate_table(self, jobs: List[Dict[str, Any]]) -> None:
        self.table.setRowCount(0)
        for job in jobs:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(job.get("job_name", ""))))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(job.get("job_id", ""))))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(job.get("state", ""))))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(job.get("reason", ""))))
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(str(job.get("elapsed", ""))))

    def load_log(self) -> None:
        selected = self.table.currentRow()
        if selected < 0 or selected >= len(self.current_jobs):
            self._set_status("Select a job row first.")
            return
        run_id = self.run_combo.currentText()
        job = self.current_jobs[selected]
        job_id = str(job.get("job_id"))
        try:
            log_text = self.api.get_log(
                run_id,
                job_id,
                self.stream_combo.currentText(),
                self.tail_spin.value(),
            )
            self.log_view.setPlainText(log_text)
            self._set_status("Log loaded.")
        except RuntimeError as exc:
            self._set_status(f"Log failed: {exc}")

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)


class ResultsView(QtWidgets.QWidget):
    def __init__(self, api: ApiClient) -> None:
        super().__init__()
        self.api = api
        self.current_files: List[Dict[str, Any]] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QLabel("Results")
        header.setStyleSheet("font-weight: 600; font-size: 16px;")
        layout.addWidget(header)

        top_row = QtWidgets.QHBoxLayout()
        self.run_combo = QtWidgets.QComboBox()
        refresh_button = QtWidgets.QPushButton("Refresh runs")
        refresh_button.clicked.connect(self.refresh_runs)
        load_button = QtWidgets.QPushButton("Load")
        load_button.clicked.connect(self.load_results)
        open_button = QtWidgets.QPushButton("Open selected")
        open_button.clicked.connect(self.open_selected)
        top_row.addWidget(QtWidgets.QLabel("Run"))
        top_row.addWidget(self.run_combo, stretch=1)
        top_row.addWidget(refresh_button)
        top_row.addWidget(load_button)
        top_row.addWidget(open_button)
        layout.addLayout(top_row)

        self.meta_view = QtWidgets.QPlainTextEdit()
        self.meta_view.setReadOnly(True)
        self.meta_view.setPlaceholderText("Run metadata will appear here.")
        layout.addWidget(self.meta_view)

        layout.addWidget(QtWidgets.QLabel("Artifacts"))
        self.file_list = QtWidgets.QListWidget()
        layout.addWidget(self.file_list, stretch=1)

        self.status_label = QtWidgets.QLabel("")
        layout.addWidget(self.status_label)

    def refresh_runs(self) -> None:
        try:
            runs = self.api.list_runs()
        except RuntimeError as exc:
            self._set_status(str(exc))
            return
        current = self.run_combo.currentText()
        self.run_combo.clear()
        self.run_combo.addItems([run["run_id"] for run in runs])
        if current:
            self.run_combo.setCurrentText(current)
        if self.run_combo.currentText():
            self.load_results()

    def load_results(self) -> None:
        run_id = self.run_combo.currentText()
        if not run_id:
            return
        try:
            payload = self.api.get_results(run_id)
            self.current_files = payload.get("files", [])
            meta = payload.get("meta", {})
        except RuntimeError as exc:
            self._set_status(f"Load failed: {exc}")
            return
        self.meta_view.setPlainText(json.dumps(meta, indent=2))
        self.file_list.clear()
        for file in self.current_files:
            label = f"{file['path']} ({file['size']} bytes)"
            self.file_list.addItem(label)
        self._set_status("Loaded.")

    def open_selected(self) -> None:
        row = self.file_list.currentRow()
        if row < 0 or row >= len(self.current_files):
            self._set_status("Select a file first.")
            return
        run_id = self.run_combo.currentText()
        path = self.current_files[row].get("path", "")
        if not path:
            return
        url = self.api.file_url(run_id, path)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))
        self._set_status("Opened in browser.")

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, api: ApiClient) -> None:
        super().__init__()
        self.api = api
        self.setWindowTitle("gaitmod desktop client")
        self.resize(1200, 800)
        self._build_ui()

    def _build_ui(self) -> None:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)

        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("API base"))
        self.api_input = QtWidgets.QLineEdit(self.api.config.base_url)
        apply_button = QtWidgets.QPushButton("Apply")
        test_button = QtWidgets.QPushButton("Test")
        apply_button.clicked.connect(self.apply_api)
        test_button.clicked.connect(self.test_api)
        header.addWidget(self.api_input, stretch=1)
        header.addWidget(apply_button)
        header.addWidget(test_button)
        layout.addLayout(header)

        self.tabs = QtWidgets.QTabWidget()
        self.config_tab = ConfigEditor(self.api)
        self.submit_tab = JobSubmission(self.api)
        self.status_tab = JobStatus(self.api)
        self.results_tab = ResultsView(self.api)
        self.tabs.addTab(self.config_tab, "Config editor")
        self.tabs.addTab(self.submit_tab, "Job submission")
        self.tabs.addTab(self.status_tab, "Job status")
        self.tabs.addTab(self.results_tab, "Results")
        layout.addWidget(self.tabs, stretch=1)

        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)

        self.setCentralWidget(container)
        self.refresh_all()

    def apply_api(self) -> None:
        self.api.set_base_url(self.api_input.text().strip())
        self.status_bar.showMessage(f"API base set to {self.api.config.base_url}", 5000)
        self.refresh_all()

    def test_api(self) -> None:
        try:
            payload = self.api.health()
            self.status_bar.showMessage(f"API ok: {payload}", 5000)
        except RuntimeError as exc:
            self.status_bar.showMessage(f"API test failed: {exc}", 5000)

    def refresh_all(self) -> None:
        self.config_tab.refresh_configs()
        self.submit_tab.refresh_configs()
        self.status_tab.refresh_runs()
        self.results_tab.refresh_runs()


def main() -> None:
    base_url = os.getenv("GAITMOD_API_BASE", "http://localhost:8000").rstrip("/")
    config = ApiConfig(base_url=base_url)
    api = ApiClient(config)

    app = QtWidgets.QApplication([])
    app.setApplicationName("gaitmod desktop client")
    window = MainWindow(api)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
