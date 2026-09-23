from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from local_model_bench.models import BenchmarkCaseResult, BenchmarkReport, OllamaModel, ScoreBreakdown
from local_model_bench.ui import MainWindow


def test_benchmark_shows_every_installed_model_and_a_scrollable_table(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(MainWindow, "_load_settings", lambda self: {
        "storage_path": str(tmp_path / "results.sqlite"), "language": "ko",
    })
    for method in ("refresh_installed", "refresh_results", "refresh_usage",
                   "_enrich_installed_hf_metadata"):
        monkeypatch.setattr(MainWindow, method, lambda self, *_args: None)

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window.resize(1440, 920)
        window.live_reports = [BenchmarkReport(
            model_name="test-model-0:latest", source="Ollama", size_label="1 GB", quant="Q4",
            status="OK", cases=[BenchmarkCaseResult("smoke", True, 1, 1, 0, 51.2, 0, 0, "OK")],
            score=ScoreBreakdown(82, 71, 65, 80, 80, 80, {}, ["coding"]),
            avg_tokens_per_second=51.2, started_at="2026-09-23T00:00:00",
            finished_at="2026-09-23T00:00:01", run_settings={"tps_source": "ollama_eval"},
        )]
        window._on_installed([OllamaModel(name=f"test-model-{index}:latest") for index in range(13)])
        window.tabs.setCurrentIndex(1)
        window.show()
        app.processEvents()

        assert not hasattr(window, "benchmark_chart")
        assert window.live_table.rowCount() == 13
        row_for = lambda name: next(row for row in range(window.live_table.rowCount())
                                    if window.live_table.item(row, 0).text() == name)
        assert window.live_table.item(row_for("test-model-12:latest"), 1).text() == "미검증"
        assert window.live_table.columnCount() == 7
        assert window.live_table.item(row_for("test-model-0:latest"), 2).text() == "82"
        assert window.live_table.item(row_for("test-model-0:latest"), 5).text() == "51.20 tok/s"
        assert window.score_scope_label.text() == "보유 13개 · 측정 1개"
        assert window.live_table.height() >= 440
        assert window.tabs.widget(1).verticalScrollBar().maximum() > 0

        window.live_table.sortItems(2, Qt.SortOrder.DescendingOrder)
        window._select_score_row(12)
        assert window._selected_source_index(window.live_table) == 12
        assert "test-model-12:latest" in window.benchmark_detail.toPlainText()
    finally:
        window.close()
