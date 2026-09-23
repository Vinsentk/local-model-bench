from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QRectF, QThreadPool, QTimer, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QIcon, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QAbstractSpinBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .benchmark import BenchmarkRunner
from .hardware import HardwareProbe
from . import __version__
from .hf_search import HuggingFaceSearch
from .i18n import LANGUAGES, USE_CASES, format_use_cases, tr, use_case_label
from .models import BenchmarkCaseResult, BenchmarkReport, ModelRecommendation, OllamaModel, ScoreBreakdown
from .model_connections import connect_codex, delete_connected_model, register_opencodex
from .ollama_logs import OllamaLogImporter
from .ollama_client import OllamaClient
from .opencode_logs import OpenCodeLogImporter
from .store import ResultStore


TABLE_HEADERS = {
    "installed": ["Model", "ID", "Size", "Params", "Thinking", "Quant", "Last Status", "Total", "Speed", "1d Tokens", "7d Tokens", "Last Tested", "Modified"],
    "history": ["Tested", "Status", "Total", "Speed", "Quality", "Best Uses"],
    "live": ["Model", "Status", "Size", "Thinking", "Total", "Speed", "Quality", "CPU Peak", "GPU Peak", "VRAM Peak", "Best Uses"],
    "rankings": ["Use Case", "#1", "#2", "#3"],
    "recommend": ["Page", "Model", "Pull Name", "Quant", "Size", "Thinking", "Fit", "Score", "Uses", "Downloads", "Reason"],
    "usage_active": ["Model", "Size", "Processor", "Until", "Last Checked"],
    "usage_model": ["Model", "Source", "Calls", "Success", "Failures", "Failure %", "Prompt", "Eval", "Total", "Last Used"],
    "usage_source": ["Source", "Calls", "Success", "Failures", "Total Tokens"],
    "usage_failure": ["Failure Type", "Count"],
    "usage_recent": ["Time", "Model", "Source", "Operation", "Status", "Failure", "Prompt", "Eval"],
    "usage_overview": [
        "Model",
        "Source Detail",
        "Raw Source",
        "Operation",
        "Client",
        "Calls",
        "Success",
        "Failures",
        "Failure %",
        "Prompt",
        "Eval",
        "Known Total",
        "Token Coverage",
        "Unknown Token Calls",
        "Avg Sec",
        "Last Used",
        "Evidence",
        "Recent Failure",
    ],
    "results_base": [
        "Model",
        "Install",
        "Size",
        "Thinking",
        "Quant",
        "Status",
        "Total",
        "Speed",
        "Quality",
        "Stability",
        "Resource",
        "Uses",
        "Avg tok/s",
        "First token",
    ],
    "results_tail": ["Last Tested", "Error"],
}

FIT_EXPLANATIONS = {
    "en": {
        "fast_draft": "Short drafts and quick replies; based on measured speed.",
        "coding_assistant": "Small code edits; tool use and full agent work are unverified.",
        "code_review": "Basic bug spotting; complex reviews need more tests.",
        "long_analysis": "Simple multistep reasoning; long context is unverified.",
        "korean_summary": "Short Korean summaries.",
        "low_resource_fast": "Light local tasks with a favorable memory fit.",
        "high_quality_slow": "Quality-focused single responses where latency is acceptable.",
        "needs_more_tests": "Only a basic run was checked. Run Standard/Thinking to judge practical uses.",
    },
    "ko": {
        "fast_draft": "짧은 초안·빠른 응답. 측정된 생성 속도 기반.",
        "coding_assistant": "작은 코드 수정 보조. 도구 호출·에이전트 작업은 별도 검증 필요.",
        "code_review": "기초 버그 찾기. 복잡한 리뷰는 추가 검증 필요.",
        "long_analysis": "간단한 다단계 추론. 긴 문맥은 별도 검증 필요.",
        "korean_summary": "짧은 한국어 요약.",
        "low_resource_fast": "메모리 부담이 적은 로컬 작업.",
        "high_quality_slow": "응답 시간이 길어도 되는 단일 답변 작업.",
        "needs_more_tests": "기초 실행만 확인했습니다. 용도 판단에는 표준·추론 벤치가 필요합니다.",
    },
}

TABLE_HEADER_TRANSLATIONS = {
    "ko": {
        "installed": ["모델", "ID", "크기", "파라미터", "Thinking", "양자화", "최근 상태", "총점", "속도", "1일 토큰", "7일 토큰", "마지막 테스트", "수정일"],
        "history": ["테스트일", "상태", "총점", "속도", "품질", "추천 용도"],
        "live": ["모델", "상태", "크기", "Thinking", "총점", "속도", "품질", "CPU 최고", "GPU 최고", "VRAM 최고", "추천 용도"],
        "rankings": ["용도", "1위", "2위", "3위"],
        "recommend": ["페이지", "모델", "Pull 이름", "양자화", "크기", "Thinking", "적합도", "점수", "용도", "다운로드", "이유"],
        "usage_active": ["모델", "크기", "프로세서", "유지 시간", "확인 시간"],
        "usage_overview": ["모델", "소스 상세", "원본 소스", "작업", "클라이언트", "호출", "성공", "실패", "실패율", "프롬프트", "응답", "확인 토큰", "토큰 커버", "토큰 미확인", "평균 초", "마지막 사용", "근거", "최근 실패"],
        "results_base": ["모델", "설치", "크기", "Thinking", "양자화", "상태", "총점", "속도", "품질", "안정성", "리소스", "용도", "평균 tok/s", "첫 토큰"],
        "results_tail": ["마지막 테스트", "오류"],
    },
    "ja": {
        "installed": ["モデル", "ID", "サイズ", "パラメータ", "Thinking", "量子化", "最新状態", "総合", "速度", "1日トークン", "7日トークン", "最終テスト", "更新日"],
        "history": ["テスト日", "状態", "総合", "速度", "品質", "最適な用途"],
        "live": ["モデル", "状態", "サイズ", "Thinking", "総合", "速度", "品質", "CPUピーク", "GPUピーク", "VRAMピーク", "最適な用途"],
        "rankings": ["用途", "1位", "2位", "3位"],
        "recommend": ["ページ", "モデル", "Pull名", "量子化", "サイズ", "Thinking", "適合", "スコア", "用途", "DL数", "理由"],
        "usage_active": ["モデル", "サイズ", "プロセッサ", "保持期限", "確認時刻"],
        "usage_overview": ["モデル", "ソース詳細", "元ソース", "操作", "クライアント", "呼び出し", "成功", "失敗", "失敗率", "プロンプト", "応答", "確認済みトークン", "トークンカバー", "未確認トークン呼び出し", "平均秒", "最終使用", "根拠", "直近失敗"],
        "results_base": ["モデル", "インストール", "サイズ", "Thinking", "量子化", "状態", "総合", "速度", "品質", "安定性", "リソース", "用途", "平均 tok/s", "初回トークン"],
        "results_tail": ["最終テスト", "エラー"],
    },
    "zh": {
        "installed": ["模型", "ID", "大小", "参数", "Thinking", "量化", "最新状态", "总分", "速度", "1日Token", "7日Token", "最后测试", "修改时间"],
        "history": ["测试时间", "状态", "总分", "速度", "质量", "最佳用途"],
        "live": ["模型", "状态", "大小", "Thinking", "总分", "速度", "质量", "CPU峰值", "GPU峰值", "VRAM峰值", "最佳用途"],
        "rankings": ["用途", "第1", "第2", "第3"],
        "recommend": ["页", "模型", "Pull 名称", "量化", "大小", "Thinking", "适配", "分数", "用途", "下载", "原因"],
        "usage_active": ["模型", "大小", "处理器", "保留到", "检查时间"],
        "usage_overview": ["模型", "来源详情", "原始来源", "操作", "客户端", "调用", "成功", "失败", "失败率", "提示", "输出", "已知 Token", "Token 覆盖", "未知 Token 调用", "平均秒", "最后使用", "证据", "最近失败"],
        "results_base": ["模型", "安装", "大小", "Thinking", "量化", "状态", "总分", "速度", "质量", "稳定性", "资源", "用途", "平均 tok/s", "首 token"],
        "results_tail": ["最后测试", "错误"],
    },
}


SORTABLE_TABLES = {"installed", "live", "recommend", "results", "usage_active", "usage_model", "usage_source", "usage_failure", "usage_recent", "usage_overview"}
SORT_ROLE = Qt.ItemDataRole.UserRole
SOURCE_INDEX_ROLE = Qt.ItemDataRole.UserRole + 1
BAD_RECOMMENDATION_SCORE = 65.0
PARAMETER_OPTIONS = [
    (0.0, "<1B"),
    (6.0, "6B"),
    (12.0, "12B"),
    (24.0, "24B"),
    (32.0, "32B"),
    (128.0, "128B"),
    (501.0, ">500B"),
]
HF_TASK_OPTIONS = [
    ("Any", ""),
    ("Text Generation", "text-generation"),
    ("Any-to-Any", "any-to-any"),
    ("Image-Text-to-Text", "image-text-to-text"),
    ("Image-to-Text", "image-to-text"),
    ("Image-to-Image", "image-to-image"),
    ("Text-to-Image", "text-to-image"),
    ("Text-to-Video", "text-to-video"),
    ("Text-to-Speech", "text-to-speech"),
]
HF_LIBRARY_OPTIONS = [
    ("Any", ""),
    ("GGUF", "gguf"),
    ("Transformers", "transformers"),
    ("PyTorch", "pytorch"),
    ("TensorFlow", "tensorflow"),
    ("JAX", "jax"),
    ("Diffusers", "diffusers"),
    ("MLX", "mlx"),
    ("Transformers.js", "transformers.js"),
    ("Safetensors", "safetensors"),
    ("sentence-transformers", "sentence-transformers"),
    ("ONNX", "onnx"),
]
HF_APP_OPTIONS = [
    ("Any", ""),
    ("Ollama", "ollama"),
    ("llama.cpp", "llama.cpp"),
    ("vLLM", "vllm"),
    ("MLX LM", "mlx-lm"),
    ("LM Studio", "lmstudio"),
    ("Jan", "jan"),
    ("Draw Things", "draw-things"),
]
HF_LANGUAGE_OPTIONS = [
    ("Any", ""),
    ("English", "en"),
    ("Korean", "ko"),
    ("Japanese", "ja"),
    ("Chinese", "zh"),
    ("Multilingual", "multilingual"),
]
HF_LICENSE_OPTIONS = [
    ("Any", ""),
    ("Apache-2.0", "apache-2.0"),
    ("MIT", "mit"),
    ("CC-BY-4.0", "cc-by-4.0"),
    ("CC-BY-SA-4.0", "cc-by-sa-4.0"),
    ("CC-BY-NC-4.0", "cc-by-nc-4.0"),
    ("Other", "other"),
]
HF_PROVIDER_OPTIONS = [
    ("Any", ""),
    ("Groq", "groq"),
    ("Novita", "novita"),
    ("Cerebras", "cerebras"),
    ("SambaNova", "sambanova"),
    ("Nscale", "nscale"),
    ("fal", "fal"),
    ("Hyperbolic", "hyperbolic"),
    ("Together AI", "together"),
]
HF_OTHER_OPTIONS = [
    ("Any", ""),
    ("Ungated Only", "ungated"),
]
HF_OPTION_LABEL_TRANSLATIONS = {
    "ko": {
        "": "전체",
        "text-generation": "텍스트 생성",
        "any-to-any": "범용 변환",
        "image-text-to-text": "이미지+텍스트→텍스트",
        "image-to-text": "이미지→텍스트",
        "image-to-image": "이미지→이미지",
        "text-to-image": "텍스트→이미지",
        "text-to-video": "텍스트→비디오",
        "text-to-speech": "텍스트→음성",
        "transformers": "Transformers",
        "gguf": "GGUF",
        "pytorch": "PyTorch",
        "tensorflow": "TensorFlow",
        "jax": "JAX",
        "diffusers": "Diffusers",
        "mlx": "MLX",
        "transformers.js": "Transformers.js",
        "safetensors": "Safetensors",
        "sentence-transformers": "sentence-transformers",
        "onnx": "ONNX",
        "ollama": "Ollama",
        "llama.cpp": "llama.cpp",
        "vllm": "vLLM",
        "mlx-lm": "MLX LM",
        "lmstudio": "LM Studio",
        "jan": "Jan",
        "draw-things": "Draw Things",
        "en": "영어",
        "ko": "한국어",
        "ja": "일본어",
        "zh": "중국어",
        "multilingual": "다국어",
        "apache-2.0": "Apache-2.0",
        "mit": "MIT",
        "cc-by-4.0": "CC-BY-4.0",
        "cc-by-sa-4.0": "CC-BY-SA-4.0",
        "cc-by-nc-4.0": "CC-BY-NC-4.0",
        "other": "기타",
        "groq": "Groq",
        "novita": "Novita",
        "cerebras": "Cerebras",
        "sambanova": "SambaNova",
        "nscale": "Nscale",
        "fal": "fal",
        "hyperbolic": "Hyperbolic",
        "together": "Together AI",
        "ungated": "게이트 없음",
    },
    "ja": {
        "": "すべて",
        "text-generation": "テキスト生成",
        "any-to-any": "Any-to-Any",
        "image-text-to-text": "画像+テキスト→テキスト",
        "image-to-text": "画像→テキスト",
        "image-to-image": "画像→画像",
        "text-to-image": "テキスト→画像",
        "text-to-video": "テキスト→動画",
        "text-to-speech": "テキスト→音声",
        "en": "英語",
        "ko": "韓国語",
        "ja": "日本語",
        "zh": "中国語",
        "multilingual": "多言語",
        "other": "その他",
        "ungated": "ゲートなしのみ",
    },
    "zh": {
        "": "全部",
        "text-generation": "文本生成",
        "any-to-any": "任意到任意",
        "image-text-to-text": "图像+文本到文本",
        "image-to-text": "图像到文本",
        "image-to-image": "图像到图像",
        "text-to-image": "文本到图像",
        "text-to-video": "文本到视频",
        "text-to-speech": "文本到语音",
        "en": "英语",
        "ko": "韩语",
        "ja": "日语",
        "zh": "中文",
        "multilingual": "多语言",
        "other": "其他",
        "ungated": "仅无门控",
    },
}
APP_CREATOR = "Vinsentk"
APP_REPOSITORY = "https://github.com/Vinsentk/local-model-bench"


def resource_path(*parts: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
    return base.joinpath(*parts)


class SortTableWidgetItem(QTableWidgetItem):
    def __lt__(self, other: QTableWidgetItem) -> bool:
        left = self.data(SORT_ROLE)
        right = other.data(SORT_ROLE)
        if left is not None and right is not None:
            try:
                return left < right
            except TypeError:
                return str(left).casefold() < str(right).casefold()
        return super().__lt__(other)


class BenchmarkChart(QWidget):
    """Small interactive chart backed by the current benchmark reports."""

    reportSelected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("benchmarkChart")
        self.setMinimumHeight(220)
        self.setMouseTracking(True)
        self.reports: list[BenchmarkReport] = []
        self.chart_metric_key = "score"
        self.selected_index: int | None = None
        self._bars: list[tuple[QRectF, int]] = []

    def set_reports(self, reports: list[BenchmarkReport]) -> None:
        self.reports = reports[:8]
        self.update()

    def set_metric(self, metric: str) -> None:
        self.chart_metric_key = metric
        self.update()

    def set_selected(self, index: int | None) -> None:
        self.selected_index = index
        self.update()

    def _value(self, report: BenchmarkReport) -> float:
        if self.chart_metric_key == "tps":
            return max(0.0, report.avg_tokens_per_second) if report.run_settings.get("tps_source") == "ollama_eval" else 0.0
        if self.chart_metric_key == "vram":
            return max(0.0, float(report.resource_summary.get("vram_used_gb_peak", 0) or 0))
        return max(0.0, report.score.total)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        outer = QRectF(1, 1, self.width() - 2, self.height() - 2)
        painter.setPen(QPen(QColor("#2a4963"), 1))
        painter.setBrush(QColor("#13263c"))
        painter.drawRoundedRect(outer, 12, 12)
        self._bars = []
        if not self.reports:
            painter.setPen(QColor("#9cb5ce"))
            painter.drawText(outer, Qt.AlignmentFlag.AlignCenter, "Run a benchmark to see model comparisons")
            return
        left, right, top, bottom = 44.0, 20.0, 30.0, 48.0
        chart_height = max(1.0, self.height() - top - bottom)
        chart_width = max(1.0, self.width() - left - right)
        values = [self._value(report) for report in self.reports]
        scale = 100.0 if self.chart_metric_key == "score" else max(1.0, max(values) * 1.15)
        for fraction in (0.0, .5, 1.0):
            y = top + chart_height * (1 - fraction)
            painter.setPen(QPen(QColor("#2d4560"), 1, Qt.PenStyle.DashLine))
            painter.drawLine(int(left), int(y), int(self.width() - right), int(y))
            painter.setPen(QColor("#91aac4"))
            painter.drawText(7, int(y) + 4, f"{scale * fraction:g}")
        slot = chart_width / len(self.reports)
        bar_width = min(86.0, slot * .58)
        accent = {"score": ("#2bd6bb", "#1b8e8d"), "tps": ("#5eb9ff", "#2369a8"),
                  "vram": ("#f6c26b", "#a86c3d")}[self.chart_metric_key]
        for index, (report, value) in enumerate(zip(self.reports, values)):
            height = max(3.0, chart_height * min(1.0, value / scale)) if value else 3.0
            x = left + index * slot + (slot - bar_width) / 2
            rect = QRectF(x, top + chart_height - height, bar_width, height)
            gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            colors = accent if report.status == "OK" else (("#eeb36b", "#a55b46") if report.status == "PARTIAL" else ("#e77a87", "#8c3a58"))
            gradient.setColorAt(0, QColor(colors[0]))
            gradient.setColorAt(1, QColor(colors[1]))
            painter.setBrush(QBrush(gradient))
            painter.setPen(QPen(QColor("#e4fcff" if self.selected_index == index else colors[1]),
                                2 if self.selected_index == index else 1))
            painter.drawRoundedRect(rect, 6, 6)
            painter.setPen(QColor("#ecf7ff"))
            suffix = "" if self.chart_metric_key == "score" else " TPS" if self.chart_metric_key == "tps" else " GB"
            value_label = "—" if self.chart_metric_key == "tps" and value <= 0 else f"{value:.1f}{suffix}"
            painter.drawText(QRectF(x - 10, rect.top() - 24, bar_width + 20, 20),
                             Qt.AlignmentFlag.AlignCenter, value_label)
            label = painter.fontMetrics().elidedText(report.model_name, Qt.TextElideMode.ElideMiddle,
                                                      max(48, int(slot - 8)))
            painter.setPen(QColor("#b6cce3"))
            painter.drawText(QRectF(left + index * slot, self.height() - 38, slot, 28),
                             Qt.AlignmentFlag.AlignCenter, label)
            self._bars.append((QRectF(left + index * slot, top, slot, chart_height + 36), index))

    def mouseMoveEvent(self, event) -> None:
        index = next((index for rect, index in self._bars if rect.contains(event.position())), None)
        if index is None:
            self.setToolTip("")
            return
        report = self.reports[index]
        measured_tps = (report.avg_tokens_per_second if report.run_settings.get("tps_source") == "ollama_eval" else 0)
        self.setToolTip(f"{report.model_name}\nStatus: {report.status}\nScore: {report.score.total}"
                        f"\nGeneration TPS: {measured_tps or 'unmeasured'}"
                        f"\nVRAM peak: {report.resource_summary.get('vram_used_gb_peak', 'unmeasured')} GB")

    def mousePressEvent(self, event) -> None:
        index = next((index for rect, index in self._bars if rect.contains(event.position())), None)
        if index is not None:
            self.set_selected(index)
            self.reportSelected.emit(index)


class RangeSlider(QWidget):
    rangeChanged = Signal(int, int)

    def __init__(self, maximum: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._minimum = 0
        self._maximum = maximum
        self._low = 0
        self._high = maximum
        self._active_handle = ""
        self.setMinimumHeight(36)
        self.setMouseTracking(True)

    def lowValue(self) -> int:
        return self._low

    def highValue(self) -> int:
        return self._high

    def setRangeValues(self, low: int, high: int) -> None:
        low = max(self._minimum, min(self._maximum, low))
        high = max(self._minimum, min(self._maximum, high))
        if low > high:
            low, high = high, low
        changed = low != self._low or high != self._high
        self._low = low
        self._high = high
        self.update()
        if changed:
            self.rangeChanged.emit(self._low, self._high)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        y = self.height() / 2
        left = self._x_for_value(self._minimum)
        right = self._x_for_value(self._maximum)
        low_x = self._x_for_value(self._low)
        high_x = self._x_for_value(self._high)

        painter.setPen(QPen(QColor("#d8dee6"), 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(int(left), int(y), int(right), int(y))
        painter.setPen(QPen(QColor("#55c7f7"), 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(int(low_x), int(y), int(high_x), int(y))

        painter.setPen(QPen(QColor("#6b7280"), 1))
        for index in range(self._minimum, self._maximum + 1):
            x = self._x_for_value(index)
            painter.drawLine(int(x), int(y + 9), int(x), int(y + 14))

        self._draw_handle(painter, low_x, y)
        self._draw_handle(painter, high_x, y)

    def mousePressEvent(self, event) -> None:
        value = self._value_for_x(event.position().x())
        self._active_handle = "low" if abs(value - self._low) <= abs(value - self._high) else "high"
        self._move_active_handle(value)

    def mouseMoveEvent(self, event) -> None:
        if self._active_handle:
            self._move_active_handle(self._value_for_x(event.position().x()))

    def mouseReleaseEvent(self, _event) -> None:
        self._active_handle = ""

    def _draw_handle(self, painter: QPainter, x: float, y: float) -> None:
        painter.setPen(QPen(QColor("#314155"), 2))
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(QRectF(x - 6, y - 6, 12, 12))
        painter.setBrush(QColor("#314155"))
        painter.drawEllipse(QRectF(x - 2, y - 2, 4, 4))

    def _move_active_handle(self, value: int) -> None:
        if self._active_handle == "low":
            self.setRangeValues(min(value, self._high), self._high)
        elif self._active_handle == "high":
            self.setRangeValues(self._low, max(value, self._low))

    def _x_for_value(self, value: int) -> float:
        margin = 12
        span = max(1, self._maximum - self._minimum)
        return margin + ((value - self._minimum) / span) * max(1, self.width() - margin * 2)

    def _value_for_x(self, x: float) -> int:
        margin = 12
        width = max(1, self.width() - margin * 2)
        ratio = max(0.0, min(1.0, (x - margin) / width))
        return round(self._minimum + ratio * (self._maximum - self._minimum))


class WorkerSignals(QObject):
    result = Signal(object)
    report = Signal(object)
    error = Signal(str)
    progress = Signal(str)
    finished = Signal()


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
            self._emit(self.signals.result, result)
        except Exception:
            self._emit(self.signals.error, traceback.format_exc())
        finally:
            self._emit(self.signals.finished)

    def _emit(self, signal, *args) -> None:
        try:
            signal.emit(*args)
        except RuntimeError:
            pass


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings_path = Path.home() / ".local-model-bench" / "settings.json"
        self.settings = self._load_settings()
        self.language = str(self.settings.get("language", "en"))
        self.setWindowTitle("Local Model Bench")
        self.resize(1440, 920)

        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(max(2, int(self.settings.get("parallel", 1))))
        self.hardware = HardwareProbe().probe()
        self.client = OllamaClient(str(self.settings.get("endpoint", "http://127.0.0.1:11434")))
        self.store = ResultStore(Path(str(self.settings.get("storage_path", Path.home() / ".local-model-bench" / "results.sqlite"))))
        self.installed_models: list[OllamaModel] = []
        self.installed_hf_metadata: dict[str, dict[str, str]] = dict(self.settings.get("hf_model_metadata", {}))
        self.recommendations: list[ModelRecommendation] = self._load_cached_recommendations()
        self.excluded_recommendations: list[str] = list(self.settings.get("excluded_recommendations", []))
        self.live_reports: list[BenchmarkReport] = []
        self.results_rows: list[dict] = []
        self.active_workers: set[Worker] = set()
        self.pending_download: ModelRecommendation | None = None
        self.download_cancel_requested = False
        self.cancel_requested = False
        self.run_all_active = False
        self.run_all_total = 0
        self.run_all_completed = 0
        self.benchmark_active = False
        self.benchmark_failed = False
        self.benchmark_total_steps = 0
        self.benchmark_seen_steps: set[str] = set()

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        icon_path = resource_path("assets", "app.ico")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self._apply_style()
        self._build_installed_tab()
        self._build_benchmark_tab()
        self._build_recommendations_tab()
        self._build_results_tab()
        self._build_usage_tab()
        self._build_settings_tab()
        self._load_recent_benchmarks()
        self._apply_language()
        self.usage_timer = QTimer(self)
        self.usage_timer.setInterval(30_000)
        self.usage_timer.timeout.connect(self.refresh_usage)
        self.usage_timer.start()
        self.refresh_installed()
        self.refresh_results()
        self.refresh_usage()

    def _load_recent_benchmarks(self) -> None:
        """Show saved model comparisons immediately, before a new run is started."""
        reports: list[BenchmarkReport] = []
        for row in self.store.latest_report_per_model(limit=30, include_deleted=False):
            try:
                status = str(row.get("status", ""))
                settings = row.get("run_settings") or {}
                mode = settings.get("mode")
                measured_tps = settings.get("tps_source") == "ollama_eval"
                score = ScoreBreakdown(
                    total=float(row.get("total_score") or 0), quality=float(row.get("quality") or 0),
                    speed=float(row.get("speed") or 0), stability=float(row.get("stability") or 0),
                    resource_fit=float(row.get("resource_fit") or 0), usability=float(row.get("usability") or 0),
                    use_case_scores=(row.get("use_case_scores") or {}) if status == "OK" and mode != "smoke" else {},
                    recommended_uses=(row.get("recommended_uses") or []) if status == "OK" and mode != "smoke" else ["needs_more_tests"],
                )
                cases = [BenchmarkCaseResult(**(case if measured_tps else {**case, "tokens_per_second": 0,
                                                                           "prompt_tokens_per_second": 0}))
                         for case in row.get("cases", [])]
                reports.append(BenchmarkReport(
                    model_name=row["model_name"], source=row.get("source", "Installed"),
                    size_label=row.get("size_label", ""), quant=row.get("quant", ""),
                    status=status, cases=cases, score=score,
                    avg_tokens_per_second=float(row.get("avg_tokens_per_second") or 0) if measured_tps else 0.0,
                    started_at=row.get("started_at", ""), finished_at=row.get("finished_at", ""),
                    error=row.get("error", ""), resource_summary=row.get("resource_summary") or {},
                    run_settings=settings,
                ))
            except (KeyError, TypeError, ValueError):
                continue
        self.live_reports = reports
        self._render_live_reports()
        self._render_rankings()
        if reports:
            self.live_table.setCurrentCell(0, 0)

    def _build_installed_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        self.hardware_label = QLabel(self._hardware_text())
        self.hardware_label.setObjectName("heroLabel")
        self.installed_table = QTableWidget(0, 13)
        self._setup_table(self.installed_table, "installed")
        self.installed_table.itemSelectionChanged.connect(self.show_selected_model_history)

        controls = QHBoxLayout()
        self.refresh_installed_button = QPushButton()
        self.refresh_installed_button.clicked.connect(self.refresh_installed)
        self.delete_model_button = QPushButton()
        self.delete_model_button.setObjectName("dangerAction")
        self.delete_model_button.clicked.connect(self.delete_selected_model)
        controls.addWidget(self.refresh_installed_button)
        controls.addWidget(self.delete_model_button)
        controls.addStretch()

        connections = QHBoxLayout()
        self.register_ollama_button = QPushButton()
        self.register_ollama_button.setObjectName("secondaryAction")
        self.register_ollama_button.clicked.connect(self.register_gguf_with_ollama)
        self.register_opencodex_button = QPushButton()
        self.register_opencodex_button.setObjectName("secondaryAction")
        self.register_opencodex_button.clicked.connect(self.register_selected_opencodex)
        self.connect_codex_button = QPushButton()
        self.connect_codex_button.setObjectName("secondaryAction")
        self.connect_codex_button.clicked.connect(self.connect_selected_codex)
        self.smoke_model_button = QPushButton()
        self.smoke_model_button.setObjectName("secondaryAction")
        self.smoke_model_button.clicked.connect(self.smoke_selected_model)
        for button in (self.register_ollama_button, self.register_opencodex_button,
                       self.connect_codex_button, self.smoke_model_button):
            connections.addWidget(button)
        connections.addStretch()
        self.connection_status = QLabel()
        self.connection_status.setObjectName("connectionStatus")
        self.connection_status.setWordWrap(True)
        self.connection_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.history_title = QLabel()
        self.history_title.setObjectName("sectionTitle")
        self.history_table = QTableWidget(0, 6)
        self._setup_table(self.history_table, "history")

        layout.addWidget(self.hardware_label)
        layout.addLayout(controls)
        layout.addLayout(connections)
        layout.addWidget(self.connection_status)
        layout.addWidget(self.installed_table, stretch=3)
        layout.addWidget(self.history_title)
        layout.addWidget(self.history_table, stretch=2)
        self.tabs.addTab(page, "")

    def _build_benchmark_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        controls = QHBoxLayout()
        self.model_label = QLabel()
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(520)
        self.mode_label = QLabel()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["smoke", "standard", "thinking"])
        self.run_selected_button = QPushButton()
        self.run_all_button = QPushButton()
        self.retry_needed_button = QPushButton()
        self.stop_button = QPushButton()
        self.stop_button.setEnabled(False)
        self.run_selected_button.clicked.connect(self.run_selected_benchmark)
        self.run_all_button.clicked.connect(self.run_all_benchmarks)
        self.retry_needed_button.clicked.connect(self.retry_needed_benchmarks)
        self.stop_button.clicked.connect(self.stop_after_current)
        controls.addWidget(self.model_label)
        controls.addWidget(self.model_combo, stretch=1)
        controls.addWidget(self.mode_label)
        controls.addWidget(self.mode_combo)
        controls.addWidget(self.run_selected_button)
        controls.addWidget(self.run_all_button)
        controls.addWidget(self.retry_needed_button)
        controls.addWidget(self.stop_button)

        process_frame = QFrame()
        process_frame.setObjectName("panel")
        process_layout = QVBoxLayout(process_frame)
        process_layout.setContentsMargins(16, 12, 16, 12)
        process_layout.setSpacing(8)
        process_header = QHBoxLayout()
        self.process_title = QLabel()
        self.process_title.setObjectName("sectionTitle")
        self.process_state_label = QLabel()
        self.process_state_label.setObjectName("processState")
        self.process_state_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        process_header.addWidget(self.process_title, stretch=1)
        process_header.addWidget(self.process_state_label, stretch=2)
        self.process_progress = QProgressBar()
        self.process_progress.setObjectName("processProgress")
        self.process_progress.setRange(0, 100)
        self.process_progress.setValue(0)
        self.process_steps_label = QLabel()
        self.process_steps_label.setObjectName("processSteps")
        self.process_steps_label.setWordWrap(True)
        self.process_plan_label = QLabel()
        self.process_plan_label.setObjectName("processPlan")
        self.process_plan_label.setWordWrap(True)
        process_layout.addLayout(process_header)
        process_layout.addWidget(self.process_progress)
        process_layout.addWidget(self.process_steps_label)
        process_layout.addWidget(self.process_plan_label)

        score_panel = QGridLayout()
        self.current_case_title, self.current_case_value = self._score_card(score_panel, 0, 0)
        self.last_score_title, self.last_score_value = self._score_card(score_panel, 0, 1)
        self.speed_title, self.speed_value = self._score_card(score_panel, 0, 2)
        self.quality_title, self.quality_value = self._score_card(score_panel, 0, 3)
        self.best_uses_title, self.best_uses_value = self._score_card(score_panel, 0, 4)
        self.progress_title, self.progress_value = self._score_card(score_panel, 0, 5)

        resource_panel = QGridLayout()
        self.cpu_title, self.cpu_value = self._score_card(resource_panel, 0, 0)
        self.gpu_title, self.gpu_value = self._score_card(resource_panel, 0, 1)
        self.ram_title, self.ram_value = self._score_card(resource_panel, 0, 2)
        self.vram_title, self.vram_value = self._score_card(resource_panel, 0, 3)

        chart_header = QHBoxLayout()
        self.chart_title = QLabel()
        self.chart_title.setObjectName("sectionTitle")
        self.chart_metric = QComboBox()
        self.chart_metric.addItem("Score", "score")
        self.chart_metric.addItem("Generation TPS", "tps")
        self.chart_metric.addItem("VRAM peak", "vram")
        self.chart_metric.currentIndexChanged.connect(
            lambda _: self.benchmark_chart.set_metric(self.chart_metric.currentData()))
        chart_header.addWidget(self.chart_title)
        chart_header.addStretch()
        chart_header.addWidget(self.chart_metric)
        self.benchmark_chart = BenchmarkChart()
        self.benchmark_chart.reportSelected.connect(self._select_chart_report)

        self.live_table = QTableWidget(0, 11)
        self._setup_table(self.live_table, "live")
        self.live_table.itemSelectionChanged.connect(self.show_benchmark_detail)

        lower = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()
        self.rankings_title = QLabel()
        self.rankings_title.setObjectName("sectionTitle")
        self.rankings_table = QTableWidget(0, 4)
        self._setup_table(self.rankings_table, "rankings")
        self.benchmark_detail_title = QLabel()
        self.benchmark_detail_title.setObjectName("sectionTitle")
        self.benchmark_detail = QTextEdit()
        self.benchmark_detail.setReadOnly(True)
        self.benchmark_detail.setMinimumHeight(180)
        left.addWidget(self.rankings_title)
        left.addWidget(self.rankings_table)
        right.addWidget(self.benchmark_detail_title)
        right.addWidget(self.benchmark_detail)
        lower.addLayout(left, stretch=1)
        lower.addLayout(right, stretch=2)

        self.progress_log = QTextEdit()
        self.progress_log.setReadOnly(True)
        self.progress_log.setMinimumHeight(130)

        layout.addLayout(controls)
        layout.addWidget(process_frame)
        layout.addLayout(score_panel)
        layout.addLayout(resource_panel)
        layout.addLayout(chart_header)
        layout.addWidget(self.benchmark_chart)
        layout.addWidget(self.live_table, stretch=3)
        layout.addLayout(lower, stretch=2)
        layout.addWidget(self.progress_log, stretch=1)
        self.tabs.addTab(page, "")

    def _build_recommendations_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        controls = QHBoxLayout()
        self.search_button = QPushButton()
        self.download_button = QPushButton()
        self.cancel_download_button = QPushButton()
        self.delete_downloaded_button = QPushButton()
        self.open_hf_button = QPushButton()
        self.search_button.clicked.connect(self.find_recommendations)
        self.download_button.clicked.connect(self.download_selected_recommendation)
        self.cancel_download_button.clicked.connect(self.cancel_download)
        self.delete_downloaded_button.clicked.connect(self.delete_selected_recommendation_model)
        self.open_hf_button.clicked.connect(self.open_selected_hf_model)
        self.cancel_download_button.setEnabled(False)
        controls.addWidget(self.search_button)
        controls.addWidget(self.download_button)
        controls.addWidget(self.cancel_download_button)
        controls.addWidget(self.delete_downloaded_button)
        controls.addWidget(self.open_hf_button)
        controls.addStretch()

        filter_panel = QFrame()
        filter_panel.setObjectName("panel")
        filter_layout = QGridLayout(filter_panel)
        filter_layout.setContentsMargins(14, 12, 14, 12)
        filter_layout.setHorizontalSpacing(10)
        filter_layout.setVerticalSpacing(8)
        self.hf_filter_title = QLabel()
        self.hf_filter_title.setObjectName("sectionTitle")
        self.hf_task_label = QLabel()
        self.hf_library_label = QLabel()
        self.hf_app_label = QLabel()
        self.hf_language_label = QLabel()
        self.hf_license_label = QLabel()
        self.hf_provider_label = QLabel()
        self.hf_other_label = QLabel()
        self.hf_page_range_label = QLabel()
        self.hf_keyword_label = QLabel()
        self.hf_task_combo = self._combo_from_options(HF_TASK_OPTIONS, self.settings.get("hf_task", "text-generation"))
        self.hf_library_combo = self._combo_from_options(HF_LIBRARY_OPTIONS, self.settings.get("hf_library", "gguf"))
        self.hf_app_combo = self._combo_from_options(HF_APP_OPTIONS, self.settings.get("hf_app", ""))
        self.hf_language_combo = self._combo_from_options(HF_LANGUAGE_OPTIONS, self.settings.get("hf_language", ""))
        self.hf_license_combo = self._combo_from_options(HF_LICENSE_OPTIONS, self.settings.get("hf_license", ""))
        self.hf_provider_combo = self._combo_from_options(HF_PROVIDER_OPTIONS, self.settings.get("hf_provider", ""))
        self.hf_other_combo = self._combo_from_options(HF_OTHER_OPTIONS, self.settings.get("hf_other", ""))
        self.hf_page_start_spin = QSpinBox()
        self.hf_page_start_spin.setRange(1, 20)
        self.hf_page_start_spin.setValue(self._settings_int("hf_page_start", 1))
        self.hf_page_start_spin.setMaximumWidth(72)
        self.hf_page_start_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.hf_page_end_spin = QSpinBox()
        self.hf_page_end_spin.setRange(1, 20)
        self.hf_page_end_spin.setValue(self._settings_int("hf_page_end", 3))
        self.hf_page_end_spin.setMaximumWidth(72)
        self.hf_page_end_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.hf_page_separator = QLabel("~")
        self.hf_page_range_widget = QWidget()
        page_range_layout = QHBoxLayout(self.hf_page_range_widget)
        page_range_layout.setContentsMargins(0, 0, 0, 0)
        page_range_layout.setSpacing(6)
        page_range_layout.addWidget(self.hf_page_start_spin)
        page_range_layout.addWidget(self.hf_page_separator)
        page_range_layout.addWidget(self.hf_page_end_spin)
        page_range_layout.addStretch()
        self.hf_page_start_spin.valueChanged.connect(lambda _value: self._save_settings() if hasattr(self, "endpoint_input") else None)
        self.hf_page_end_spin.valueChanged.connect(lambda _value: self._save_settings() if hasattr(self, "endpoint_input") else None)
        self.hf_keyword_input = QLineEdit(str(self.settings.get("hf_keyword", "") or ""))
        self.hf_keyword_input.textChanged.connect(lambda _text: self._save_settings() if hasattr(self, "endpoint_input") else None)
        filter_layout.addWidget(self.hf_filter_title, 0, 0, 1, 10)
        first_row_filters = [
            (self.hf_task_label, self.hf_task_combo),
            (self.hf_library_label, self.hf_library_combo),
            (self.hf_app_label, self.hf_app_combo),
            (self.hf_provider_label, self.hf_provider_combo),
            (self.hf_language_label, self.hf_language_combo),
        ]
        second_row_filters = [
            (self.hf_license_label, self.hf_license_combo, 1),
            (self.hf_other_label, self.hf_other_combo, 1),
            (self.hf_page_range_label, self.hf_page_range_widget, 1),
            (self.hf_keyword_label, self.hf_keyword_input, 2),
        ]
        for index, (label, widget) in enumerate(first_row_filters):
            col = index * 2
            filter_layout.addWidget(label, 1, col)
            filter_layout.addWidget(widget, 1, col + 1)
        next_col = 0
        for label, widget, control_span in second_row_filters:
            filter_layout.addWidget(label, 2, next_col)
            filter_layout.addWidget(widget, 2, next_col + 1, 1, control_span * 2 - 1)
            next_col += control_span * 2
        for col in [1, 3, 5, 7, 9]:
            filter_layout.setColumnStretch(col, 1)
        filter_layout.setColumnStretch(5, 2)
        filter_layout.setColumnStretch(7, 2)
        filter_layout.setColumnStretch(9, 2)

        parameter_panel = QFrame()
        parameter_panel.setObjectName("panel")
        parameter_layout = QVBoxLayout(parameter_panel)
        parameter_layout.setContentsMargins(14, 12, 14, 12)
        parameter_layout.setSpacing(8)
        parameter_header = QHBoxLayout()
        self.parameter_title = QLabel()
        self.parameter_title.setObjectName("sectionTitle")
        self.parameter_value_label = QLabel()
        self.reset_parameters_button = QPushButton()
        self.reset_parameters_button.clicked.connect(self.reset_parameter_range)
        parameter_header.addWidget(self.parameter_title)
        parameter_header.addStretch()
        parameter_header.addWidget(self.parameter_value_label)
        parameter_header.addWidget(self.reset_parameters_button)
        label_row = QHBoxLayout()
        for _, label in PARAMETER_OPTIONS:
            mark = QLabel(label)
            mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label_row.addWidget(mark)
        self.parameter_slider = RangeSlider(len(PARAMETER_OPTIONS) - 1)
        self.parameter_slider.rangeChanged.connect(self._on_parameter_range_changed)
        self.parameter_slider.setRangeValues(
            int(self.settings.get("param_min_index", 0)),
            int(self.settings.get("param_max_index", 4)),
        )
        parameter_layout.addLayout(parameter_header)
        parameter_layout.addLayout(label_row)
        parameter_layout.addWidget(self.parameter_slider)

        progress_row = QHBoxLayout()
        self.recommend_progress_label = QLabel()
        self.recommend_progress = QProgressBar()
        self.recommend_progress.setObjectName("progressBar")
        self.recommend_progress.setRange(0, 100)
        self.recommend_progress.setValue(0)
        progress_row.addWidget(self.recommend_progress_label, stretch=1)
        progress_row.addWidget(self.recommend_progress, stretch=2)

        download_progress_row = QHBoxLayout()
        self.download_progress_label = QLabel()
        self.download_progress = QProgressBar()
        self.download_progress.setObjectName("downloadProgress")
        self.download_progress.setTextVisible(True)
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        download_progress_row.addWidget(self.download_progress_label, stretch=1)
        download_progress_row.addWidget(self.download_progress, stretch=2)

        self.recommend_table = QTableWidget(0, 11)
        self._setup_table(self.recommend_table, "recommend")
        self.recommend_table.itemSelectionChanged.connect(self.show_recommendation_detail)
        self.recommend_detail_title = QLabel()
        self.recommend_detail_title.setObjectName("sectionTitle")
        self.recommend_detail = QTextEdit()
        self.recommend_detail.setReadOnly(True)
        self.recommend_detail.setMinimumHeight(130)
        self.research_report_title = QLabel()
        self.research_report_title.setObjectName("sectionTitle")
        self.recommendation_report = QTextEdit()
        self.recommendation_report.setReadOnly(True)
        self.recommendation_report.setMinimumHeight(130)
        detail_panel = QFrame()
        detail_panel.setObjectName("panel")
        detail_layout = QVBoxLayout(detail_panel)
        detail_layout.setContentsMargins(12, 10, 12, 12)
        detail_layout.addWidget(self.recommend_detail_title)
        detail_layout.addWidget(self.recommend_detail)
        report_panel = QFrame()
        report_panel.setObjectName("panel")
        report_layout = QVBoxLayout(report_panel)
        report_layout.setContentsMargins(12, 10, 12, 12)
        report_layout.addWidget(self.research_report_title)
        report_layout.addWidget(self.recommendation_report)
        lower_splitter = QSplitter(Qt.Orientation.Horizontal)
        lower_splitter.addWidget(detail_panel)
        lower_splitter.addWidget(report_panel)
        lower_splitter.setSizes([1, 1])
        layout.addLayout(controls)
        layout.addWidget(filter_panel)
        layout.addWidget(parameter_panel)
        layout.addLayout(progress_row)
        layout.addLayout(download_progress_row)
        layout.addWidget(self.recommend_table, stretch=7)
        layout.addWidget(lower_splitter, stretch=2)
        self.tabs.addTab(page, "")

    def _build_results_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        controls = QHBoxLayout()
        self.refresh_results_button = QPushButton()
        self.delete_result_button = QPushButton()
        self.delete_model_results_button = QPushButton()
        self.export_csv_button = QPushButton()
        self.export_json_button = QPushButton()
        self.backup_db_button = QPushButton()
        self.show_deleted_checkbox = QCheckBox()
        self.refresh_results_button.clicked.connect(self.refresh_results)
        self.delete_result_button.clicked.connect(self.delete_selected_result)
        self.delete_model_results_button.clicked.connect(self.delete_selected_model_results)
        self.export_csv_button.clicked.connect(self.export_results_csv)
        self.export_json_button.clicked.connect(self.export_results_json)
        self.backup_db_button.clicked.connect(self.backup_result_db)
        self.show_deleted_checkbox.setChecked(bool(self.settings.get("show_deleted_results", True)))
        self.show_deleted_checkbox.stateChanged.connect(lambda _state: self.refresh_results())
        controls.addWidget(self.refresh_results_button)
        controls.addWidget(self.delete_result_button)
        controls.addWidget(self.delete_model_results_button)
        controls.addWidget(self.export_csv_button)
        controls.addWidget(self.export_json_button)
        controls.addWidget(self.backup_db_button)
        controls.addWidget(self.show_deleted_checkbox)
        controls.addStretch()
        self.results_table = QTableWidget(0, 27)
        self._setup_table(self.results_table, "results")
        self.results_table.itemSelectionChanged.connect(self.show_result_detail)
        self.results_detail = QTextEdit()
        self.results_detail.setReadOnly(True)
        self.results_detail.setMinimumHeight(190)
        self.results_leaders = QTextEdit()
        self.results_leaders.setReadOnly(True)
        self.results_leaders.setMinimumHeight(190)
        lower_splitter = QSplitter(Qt.Orientation.Horizontal)
        lower_splitter.addWidget(self.results_detail)
        lower_splitter.addWidget(self.results_leaders)
        lower_splitter.setSizes([2, 1])
        layout.addLayout(controls)
        layout.addWidget(self.results_table, stretch=5)
        layout.addWidget(lower_splitter, stretch=2)
        self.tabs.addTab(page, "")

    def _build_usage_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        controls = QHBoxLayout()
        self.usage_period_label = QLabel()
        self.usage_period_combo = QComboBox()
        self.usage_period_combo.addItem("Today", "today")
        self.usage_period_combo.addItem("7 days", "7d")
        self.usage_period_combo.addItem("30 days", "30d")
        self.usage_period_combo.addItem("All", "all")
        self.usage_period_combo.setCurrentIndex(1)
        self.refresh_usage_button = QPushButton()
        self.refresh_usage_button.clicked.connect(self.refresh_usage)
        self.usage_ollama_checkbox = QCheckBox("Ollama")
        self.usage_ollama_checkbox.setChecked(True)
        self.usage_opencode_checkbox = QCheckBox("OpenCode")
        self.usage_opencode_checkbox.setChecked(True)
        self.import_usage_logs_button = QPushButton()
        self.import_usage_logs_button.clicked.connect(self.import_usage_logs)
        self.usage_import_label = QLabel()
        self.usage_period_combo.currentIndexChanged.connect(lambda _index: self.refresh_usage())
        controls.addWidget(self.usage_period_label)
        controls.addWidget(self.usage_period_combo)
        controls.addWidget(self.usage_ollama_checkbox)
        controls.addWidget(self.usage_opencode_checkbox)
        controls.addWidget(self.refresh_usage_button)
        controls.addWidget(self.import_usage_logs_button)
        controls.addWidget(self.usage_import_label)
        controls.addStretch()

        self.active_models_title = QLabel()
        self.active_models_title.setObjectName("sectionTitle")
        self.usage_active_table = QTableWidget(0, 5)
        self._setup_table(self.usage_active_table, "usage_active")

        summary_grid = QGridLayout()
        self.usage_active_count_title, self.usage_active_count_value = self._score_card(summary_grid, 0, 0)
        self.usage_calls_title, self.usage_calls_value = self._score_card(summary_grid, 0, 1)
        self.usage_success_title, self.usage_success_value = self._score_card(summary_grid, 0, 2)
        self.usage_tokens_title, self.usage_tokens_value = self._score_card(summary_grid, 1, 0)
        self.usage_token_coverage_title, self.usage_token_coverage_value = self._score_card(summary_grid, 1, 1)
        self.usage_top_source_title, self.usage_top_source_value = self._score_card(summary_grid, 1, 2)

        splitter = QSplitter(Qt.Orientation.Vertical)
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)
        top_layout.addLayout(summary_grid)
        top_layout.addWidget(self.active_models_title)
        top_layout.addWidget(self.usage_active_table)

        lower_panel = QWidget()
        lower_layout = QVBoxLayout(lower_panel)
        lower_layout.setContentsMargins(0, 0, 0, 0)
        lower_layout.setSpacing(8)
        self.usage_overview_title = QLabel()
        self.usage_overview_title.setObjectName("sectionTitle")
        self.usage_overview_table = QTableWidget(0, 18)
        self._setup_table(self.usage_overview_table, "usage_overview")
        lower_layout.addWidget(self.usage_overview_title)
        lower_layout.addWidget(self.usage_overview_table)

        splitter.addWidget(top_panel)
        splitter.addWidget(lower_panel)
        splitter.setSizes([260, 520])

        layout.addLayout(controls)
        layout.addWidget(splitter, stretch=1)
        self.tabs.addTab(page, "")

    def _build_settings_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        form_frame = QFrame()
        form_frame.setObjectName("panel")
        form = QFormLayout(form_frame)
        form.setContentsMargins(18, 18, 18, 18)
        form.setSpacing(12)
        self.endpoint_input = QLineEdit(str(self.settings.get("endpoint", "http://127.0.0.1:11434")))
        self.endpoint_auto_button = QPushButton()
        self.endpoint_auto_button.clicked.connect(self.detect_ollama_endpoint)
        endpoint_row = QHBoxLayout()
        endpoint_row.addWidget(self.endpoint_input, stretch=1)
        endpoint_row.addWidget(self.endpoint_auto_button)
        self.storage_path_input = QLineEdit(str(self.store.db_path))
        self.storage_browse_button = QPushButton()
        self.storage_browse_button.clicked.connect(self.browse_result_db_path)
        self.open_data_folder_button = QPushButton()
        self.open_data_folder_button.clicked.connect(self.open_data_folder)
        self.settings_backup_db_button = QPushButton()
        self.settings_backup_db_button.clicked.connect(self.backup_result_db)
        storage_row = QHBoxLayout()
        storage_row.addWidget(self.storage_path_input, stretch=1)
        storage_row.addWidget(self.storage_browse_button)
        storage_row.addWidget(self.open_data_folder_button)
        storage_row.addWidget(self.settings_backup_db_button)
        self.ollama_models_path_input = QLineEdit(str(self.settings.get("ollama_models_path", os.environ.get("OLLAMA_MODELS", ""))))
        self.ollama_models_browse_button = QPushButton()
        self.ollama_models_browse_button.clicked.connect(self.browse_ollama_models_path)
        self.ollama_models_set_button = QPushButton()
        self.ollama_models_set_button.clicked.connect(self.set_ollama_models_env)
        self.ollama_models_open_button = QPushButton()
        self.ollama_models_open_button.clicked.connect(self.open_ollama_models_folder)
        self.ollama_models_hint = QLabel()
        self.ollama_models_hint.setWordWrap(True)
        ollama_models_row = QHBoxLayout()
        ollama_models_row.addWidget(self.ollama_models_path_input, stretch=1)
        ollama_models_row.addWidget(self.ollama_models_browse_button)
        ollama_models_row.addWidget(self.ollama_models_set_button)
        ollama_models_row.addWidget(self.ollama_models_open_button)
        self.default_mode_combo = QComboBox()
        self.default_mode_combo.addItems(["smoke", "standard", "thinking"])
        self.default_mode_combo.setCurrentText(str(self.settings.get("default_mode", "standard")))
        self.parallel_spin = QSpinBox()
        self.parallel_spin.setMinimum(1)
        self.parallel_spin.setMaximum(4)
        self.parallel_spin.setValue(int(self.settings.get("parallel", 1)))
        self.warmup_spin = QSpinBox()
        self.warmup_spin.setMinimum(0)
        self.warmup_spin.setMaximum(5)
        self.warmup_spin.setValue(int(self.settings.get("benchmark_warmup_runs", 0)))
        self.repeat_spin = QSpinBox()
        self.repeat_spin.setMinimum(1)
        self.repeat_spin.setMaximum(5)
        self.repeat_spin.setValue(int(self.settings.get("benchmark_repeat_count", 1)))
        self.language_combo = QComboBox()
        for code, label in LANGUAGES.items():
            self.language_combo.addItem(label, code)
        self._set_language_combo(self.language)
        self.apply_button = QPushButton()
        self.apply_button.clicked.connect(self.apply_settings)
        self.endpoint_label = QLabel()
        self.storage_label = QLabel()
        self.ollama_models_label = QLabel()
        self.default_mode_label = QLabel()
        self.parallel_label = QLabel()
        self.warmup_label = QLabel()
        self.repeat_label = QLabel()
        self.language_label = QLabel()
        form.addRow(self.endpoint_label, endpoint_row)
        form.addRow(self.storage_label, storage_row)
        form.addRow(self.ollama_models_label, ollama_models_row)
        form.addRow("", self.ollama_models_hint)
        form.addRow(self.default_mode_label, self.default_mode_combo)
        form.addRow(self.parallel_label, self.parallel_spin)
        form.addRow(self.warmup_label, self.warmup_spin)
        form.addRow(self.repeat_label, self.repeat_spin)
        form.addRow(self.language_label, self.language_combo)
        about_frame = QFrame()
        about_frame.setObjectName("panel")
        about_layout = QVBoxLayout(about_frame)
        about_layout.setContentsMargins(18, 14, 18, 14)
        about_layout.setSpacing(8)
        self.about_title = QLabel()
        self.about_title.setObjectName("sectionTitle")
        self.about_body = QLabel()
        self.about_body.setWordWrap(True)
        self.about_body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        about_layout.addWidget(self.about_title)
        about_layout.addWidget(self.about_body)
        layout.addWidget(form_frame)
        layout.addWidget(self.apply_button)
        layout.addWidget(about_frame)
        layout.addStretch()
        self.tabs.addTab(page, "")

    def refresh_installed(self) -> None:
        self._run_background(self.client.list_models, self._on_installed)

    def _on_installed(self, models: list[OllamaModel]) -> None:
        self.installed_models = models
        self._render_installed_models(reset_combo=True)
        self.show_selected_model_history()
        self._enrich_installed_hf_metadata(models)

    def _render_installed_models(self, *, reset_combo: bool = False) -> None:
        latest_reports = self.store.latest_report_map(include_deleted=False)
        usage_1d = self._usage_token_totals_for_installed("1d")
        usage_7d = self._usage_token_totals_for_installed("7d")
        self._begin_table_update(self.installed_table)
        self.installed_table.setRowCount(len(self.installed_models))
        if reset_combo:
            self.model_combo.clear()
        for row, model in enumerate(self.installed_models):
            if reset_combo:
                self.model_combo.addItem(model.name)
            thinking = self._thinking_label_for_model(model.name)
            latest = latest_reports.get(model.name, {})
            last_status = latest.get("status", "-")
            total = latest.get("total_score", "-")
            speed = self._row_tps(latest)
            last_tested = self._format_local_time(latest.get("finished_at", ""))
            usage_key = self._usage_model_key(model.name)
            tokens_1d = usage_1d.get(usage_key, 0)
            tokens_7d = usage_7d.get(usage_key, 0)
            values = [
                model.name,
                model.model_id,
                model.size_label,
                self._param_label_for_model(model.name),
                thinking,
                model.quant,
                last_status,
                total,
                speed,
                self._format_token_count(tokens_1d),
                self._format_token_count(tokens_7d),
                last_tested,
                self._format_local_time(model.modified_at),
            ]
            sort_keys = [
                model.name.casefold(),
                model.model_id.casefold(),
                model.size_bytes,
                self._param_size_b(model.name),
                self._thinking_sort_for_model(model.name),
                model.quant.casefold(),
                str(last_status).casefold(),
                self._number_or_zero(total),
                self._number_or_zero(speed),
                tokens_1d,
                tokens_7d,
                self._time_sort_value(latest.get("finished_at", "")),
                self._time_sort_value(model.modified_at),
            ]
            self._set_row(self.installed_table, row, values, sort_keys=sort_keys, source_index=row)
        self._end_table_update(self.installed_table, "installed")

    def _usage_token_totals_for_installed(self, period: str) -> dict[str, int]:
        totals: dict[str, int] = {}
        for model_name, tokens in self.store.usage_token_totals_by_model(period=period).items():
            key = self._usage_model_key(model_name)
            totals[key] = totals.get(key, 0) + int(tokens or 0)
        return totals

    def _usage_model_key(self, model_name: str) -> str:
        text = str(model_name or "").strip()
        for prefix in ["ollama/", "registry.ollama.ai/library/", "library/"]:
            if text.lower().startswith(prefix):
                text = text[len(prefix) :]
                break
        if text.startswith("hf.co/"):
            return text.casefold()
        return text.casefold()

    def _enrich_installed_hf_metadata(self, models: list[OllamaModel]) -> None:
        candidates = [
            model.name
            for model in models
            if self._thinking_sort_value(model.name) == 0
            and model.name not in self.installed_hf_metadata
        ][:30]
        if not candidates:
            return

        def work():
            search = HuggingFaceSearch(self.hardware)
            return {name: search.lookup_thinking_metadata(name) for name in candidates}

        self._run_background(work, self._on_installed_hf_metadata)

    def _on_installed_hf_metadata(self, metadata: dict[str, dict[str, str]]) -> None:
        self.installed_hf_metadata.update(metadata)
        enriched = sum(1 for item in metadata.values() if item.get("thinking_support", "unknown") != "unknown")
        if enriched:
            self._append_log(f"HF Thinking metadata updated: {enriched}")
        self._save_settings()
        self._render_installed_models(reset_combo=False)

    def delete_selected_model(self) -> None:
        model = self._selected_installed_model()
        if model is None:
            self.connection_status.setText(tr(self.language, "select_model_first"))
            return
        ok = QMessageBox.question(self, tr(self.language, "delete_title"),
                                  f"{tr(self.language, 'delete_connections_body')}\n\n{model.name}")
        if ok != QMessageBox.StandardButton.Yes:
            return
        self._run_model_action(lambda: delete_connected_model(model.name, self.client), self._after_delete)

    def _after_delete(self, result: str) -> None:
        self.connection_status.setText(result)
        self._append_log(result)
        self.refresh_installed()

    def register_gguf_with_ollama(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr(self.language, "register_ollama"), "", "GGUF (*.gguf)")
        if not path:
            return
        default_name = re.sub(r"[^a-z0-9_.-]+", "-", Path(path).stem.lower()).strip("-.")[:80]
        name, accepted = QInputDialog.getText(self, tr(self.language, "register_ollama"),
                                              tr(self.language, "ollama_name_prompt"), text=default_name)
        if accepted and name.strip():
            self._run_model_action(lambda: self.client.create_from_gguf(path, name.strip()),
                                   lambda _: self._after_ollama_registration(name.strip()))

    def _after_ollama_registration(self, name: str) -> None:
        self.connection_status.setText(f"Ollama registered: {name}. Run the basic test before connecting it.")
        self.refresh_installed()

    def register_selected_opencodex(self) -> None:
        model = self._selected_installed_model()
        if model is None:
            self.connection_status.setText(tr(self.language, "select_model_first"))
            return
        self._run_model_action(lambda: register_opencodex(model.name, self.client), self.connection_status.setText)

    def connect_selected_codex(self) -> None:
        model = self._selected_installed_model()
        if model is None:
            self.connection_status.setText(tr(self.language, "select_model_first"))
            return
        self._run_model_action(lambda: connect_codex(model.name, self.client), self.connection_status.setText)

    def smoke_selected_model(self) -> None:
        model = self._selected_installed_model()
        if model is None:
            self.connection_status.setText(tr(self.language, "select_model_first"))
            return
        if self.benchmark_active:
            self.connection_status.setText(tr(self.language, "benchmark_running"))
            return
        self.mode_combo.setCurrentText("smoke")
        self.connection_status.setText(f"{model.name}: basic test running...")
        self._run_benchmark(model, basic=True)
        self.tabs.setCurrentIndex(1)

    def _run_model_action(self, fn, on_result) -> None:
        buttons = (self.register_ollama_button, self.register_opencodex_button,
                   self.connect_codex_button, self.delete_model_button)
        for button in buttons:
            button.setEnabled(False)
        self.connection_status.setText(tr(self.language, "connection_working"))
        worker = Worker(fn)
        worker.signals.result.connect(on_result)
        worker.signals.error.connect(lambda error: self.connection_status.setText(error.strip().splitlines()[-1]))
        worker.signals.finished.connect(lambda: [button.setEnabled(True) for button in buttons])
        self._start_worker(worker)

    def show_selected_model_history(self) -> None:
        model = self._selected_installed_model()
        rows = self.store.reports_for_model(model.name) if model else []
        self.history_table.setRowCount(len(rows))
        for row, item in enumerate(rows):
            values = [
                self._format_local_time(item.get("finished_at", "")),
                item.get("status", ""),
                item.get("total_score", ""),
                item.get("speed", ""),
                item.get("quality", ""),
                self._format_use_list(item.get("recommended_uses", [])),
            ]
            self._set_row(self.history_table, row, values, muted=bool(item.get("deleted_at")))
        self._apply_column_layout(self.history_table, "history")

    def _selected_installed_model(self) -> OllamaModel | None:
        index = self._selected_source_index(self.installed_table)
        if index is not None and 0 <= index < len(self.installed_models):
            return self.installed_models[index]
        return None

    def run_selected_benchmark(self) -> None:
        name = self.model_combo.currentText()
        model = next((item for item in self.installed_models if item.name == name), None)
        if model:
            self._run_benchmark(model)

    def run_all_benchmarks(self) -> None:
        mode = self.mode_combo.currentText()
        models = list(self.installed_models)
        self._run_benchmark_queue(models, mode, tr(self.language, "queued_all"))

    def retry_needed_benchmarks(self) -> None:
        mode = self.mode_combo.currentText()
        models = self._models_needing_retry()
        if not models:
            self._append_log(tr(self.language, "retry_none"))
            return
        self._run_benchmark_queue(models, mode, tr(self.language, "queued_retry"))

    def _run_benchmark_queue(self, models: list[OllamaModel], mode: str, label: str) -> None:
        self.cancel_requested = False
        self.run_all_active = True
        self.run_all_total = len(models)
        self.run_all_completed = 0
        self._start_benchmark_process(models, mode, label)
        self._update_run_progress()
        worker = None

        def work():
            runner = BenchmarkRunner(self.client, self.hardware)
            warmup_runs = self.warmup_spin.value()
            repeat_count = self.repeat_spin.value()
            for model in models:
                if self.cancel_requested:
                    break
                report = runner.run(
                    model,
                    mode=mode,
                    warmup_runs=warmup_runs,
                    repeat_count=repeat_count,
                    on_progress=lambda m, c: worker.signals.progress.emit(f"{m}: {c}"),  # type: ignore[union-attr]
                    on_usage=self._record_usage_event,
                )
                worker.signals.report.emit(report)  # type: ignore[union-attr]
            return None

        self._append_log(f"{label}: {len(models)} ({mode})")
        worker = Worker(work)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.report.connect(self._on_report)
        worker.signals.error.connect(self._on_benchmark_error)
        worker.signals.finished.connect(self._finish_run_all)
        self._start_worker(worker)

    def _models_needing_retry(self) -> list[OllamaModel]:
        installed = {model.name: model for model in self.installed_models}
        latest_by_model: dict[str, dict] = {}
        for report in self.store.latest_reports(limit=1000):
            name = str(report.get("model_name", ""))
            if name and name not in latest_by_model:
                latest_by_model[name] = report
        models: list[OllamaModel] = []
        for name, report in latest_by_model.items():
            model = installed.get(name)
            if model is None:
                continue
            status = str(report.get("status", "")).upper()
            score = self._number_or_zero(report.get("total_score", 0))
            uses = report.get("recommended_uses", [])
            if status in {"FAIL", "PARTIAL", "RETRY"} or score < BAD_RECOMMENDATION_SCORE or "needs_more_tests" in uses:
                models.append(model)
        return models

    def stop_after_current(self) -> None:
        self.cancel_requested = True
        self._append_log(tr(self.language, "stop_requested"))

    def _run_benchmark(self, model: OllamaModel, *, basic: bool = False) -> None:
        mode = self.mode_combo.currentText()
        self.run_all_active = False
        self.run_all_total = 1
        self.run_all_completed = 0
        self._start_benchmark_process([model], mode, tr(self.language, "queued_benchmark"))
        self._update_run_progress()
        worker = None

        def work():
            runner = BenchmarkRunner(self.client, self.hardware)
            return runner.run(
                model,
                mode=mode,
                warmup_runs=0 if basic else self.warmup_spin.value(),
                repeat_count=1 if basic else self.repeat_spin.value(),
                on_progress=lambda m, c: worker.signals.progress.emit(f"{m}: {c}"),  # type: ignore[union-attr]
                on_usage=self._record_usage_event,
            )

        self._append_log(f"{tr(self.language, 'queued_benchmark')}: {model.name} ({mode})")
        worker = Worker(work)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.result.connect(self._on_report)
        worker.signals.error.connect(self._on_benchmark_error)
        worker.signals.finished.connect(self._finish_single_benchmark)
        self._start_worker(worker)

    def _on_progress(self, text: str) -> None:
        self.current_case_value.setText(text)
        self.current_case_value.setToolTip(text)
        self._mark_benchmark_step(text)
        self._append_log(text)

    def _on_report(self, report: BenchmarkReport) -> None:
        if self.benchmark_active:
            self._set_process_stage("score", tr(self.language, "process_score"), 88)
        self.store.save_report(report)
        if self.benchmark_active:
            self._set_process_stage("save", tr(self.language, "process_save"), 94)
        self.live_reports.insert(0, report)
        self.live_reports = self.live_reports[:100]
        if self.run_all_total > 0:
            self.run_all_completed = min(self.run_all_completed + 1, self.run_all_total)
            self._update_run_progress()
        self._set_score_cards(report)
        self._set_resource_cards(report)
        self._render_live_reports()
        self._render_rankings()
        self.show_benchmark_detail()
        progress = f" [{self._run_progress_text()}]" if self.run_all_total > 0 else ""
        self._append_log(f"{report.model_name}: {report.status}, score {report.score.total}, {self._report_tps(report)}{progress}")
        if (report.run_settings or {}).get("mode") == "smoke":
            passed = report.status == "OK" and bool(report.cases) and all(case.passed for case in report.cases)
            verdict = tr(self.language, "basic_test_pass") if passed else tr(self.language, "basic_test_fail")
            self.connection_status.setText(f"{report.model_name}: {verdict} | {self._report_tps(report)}")
        self.refresh_results()
        self._render_installed_models(reset_combo=False)

    def find_recommendations(self) -> None:
        self._append_log(tr(self.language, "researching"))
        self.search_button.setEnabled(False)
        self._set_recommend_progress(5, tr(self.language, "recommend_progress_start"))
        worker = None

        def work():
            worker.signals.progress.emit(f"25|{tr(self.language, 'recommend_progress_fetch')}")  # type: ignore[union-attr]
            installed = list(self.installed_models)
            min_params, max_params = self._parameter_range_values()
            filters = self._hf_filter_values()
            recs = HuggingFaceSearch(self.hardware).recommendations(
                min_params_b=min_params,
                max_params_b=max_params,
                task=filters["task"],
                library=filters["library"],
                app=filters["app"],
                language=filters["language"],
                license=filters["license"],
                provider=filters["provider"],
                other=filters["other"],
                keyword=filters["keyword"],
                page_start=int(filters["page_start"]),
                page_end=int(filters["page_end"]),
            )
            worker.signals.progress.emit(f"65|{tr(self.language, 'recommend_progress_filter')}")  # type: ignore[union-attr]
            recs, excluded_bad = self._filter_new_recommendations(recs, installed)
            worker.signals.progress.emit(f"85|{tr(self.language, 'recommend_progress_report')}")  # type: ignore[union-attr]
            report = self._build_recommendation_report(recs, installed)
            return {"recommendations": recs, "report": report, "excluded_bad": excluded_bad}

        worker = Worker(work)
        worker.signals.progress.connect(self._on_recommendation_progress)
        worker.signals.result.connect(self._on_recommendation_result)
        worker.signals.error.connect(self._on_recommendation_error)
        worker.signals.finished.connect(lambda: self.search_button.setEnabled(True))
        self._start_worker(worker)

    def _on_recommendation_result(self, result: dict) -> None:
        self._on_recommendations(result.get("recommendations", []))
        report = str(result.get("report", ""))
        excluded = list(result.get("excluded_bad", []) or [])
        self.excluded_recommendations = excluded
        if excluded:
            report = (
                f"{tr(self.language, 'excluded_models')}: {len(excluded)}\n"
                + "\n".join(f"- {item}" for item in excluded[:40])
                + "\n\n"
                + report
            )
            self._append_log(f"{tr(self.language, 'excluded_models')}: {len(excluded)}")
        self.recommendation_report.setPlainText(report)
        self._set_recommend_progress(100, tr(self.language, "recommend_progress_done"))
        self._save_settings()

    def _on_recommendation_progress(self, payload: str) -> None:
        percent_text, _, message = payload.partition("|")
        try:
            percent = int(percent_text)
        except ValueError:
            percent = self.recommend_progress.value()
            message = payload
        self._set_recommend_progress(percent, message)

    def _on_recommendation_error(self, error: str) -> None:
        self._set_recommend_progress(0, tr(self.language, "recommend_progress_error"))
        self.recommendation_report.setPlainText(f"{tr(self.language, 'recommend_retry_hint')}\n\n{error}")
        self._append_log(error)

    def _on_recommendations(self, recs: list[ModelRecommendation]) -> None:
        self.recommendations = recs
        self._render_recommendations()
        self._append_log(f"{tr(self.language, 'recommendations_loaded')}: {len(recs)}")
        self._save_settings()

    def download_selected_recommendation(self) -> None:
        rec = self._selected_recommendation()
        if rec is None:
            QMessageBox.information(self, tr(self.language, "no_selection_title"), tr(self.language, "no_selection_body"))
            return
        ok = QMessageBox.question(self, tr(self.language, "download_title"), f"{tr(self.language, 'download_body')}\n\n{rec.pull_name}")
        if ok != QMessageBox.StandardButton.Yes:
            return
        self.pending_download = rec
        self.download_cancel_requested = False
        self.download_button.setEnabled(False)
        self.cancel_download_button.setEnabled(True)
        self._set_download_progress(0, f"{tr(self.language, 'downloading')}: {rec.pull_name}", busy=True)
        self._append_log(f"{tr(self.language, 'downloading')}: {rec.pull_name}")
        worker = None

        def work():
            errors: list[str] = []
            for candidate in self._download_pull_candidates(rec):
                self._raise_if_download_canceled()
                worker.signals.progress.emit(f"{tr(self.language, 'download_trying')}: {candidate}")  # type: ignore[union-attr]
                try:
                    self.client.pull(candidate, on_line=lambda line: self._emit_download_line(worker, line))  # type: ignore[arg-type]
                    if candidate != rec.pull_name:
                        worker.signals.progress.emit(f"{tr(self.language, 'download_fallback_used')}: {candidate}")  # type: ignore[union-attr]
                    return self.client.list_models()
                except Exception as exc:
                    errors.append(f"{candidate}: {exc}")
                    if not self._should_retry_pull_with_latest(str(exc)):
                        break
            raise RuntimeError("\n".join(errors))

        worker = Worker(work)
        worker.signals.progress.connect(self._on_download_progress)
        worker.signals.result.connect(self._after_download)
        worker.signals.error.connect(self._on_download_error)
        worker.signals.finished.connect(self._finish_download_worker)
        self._start_worker(worker)

    def _emit_download_line(self, worker: Worker, line: str) -> None:
        self._raise_if_download_canceled()
        worker.signals.progress.emit(line)

    def _raise_if_download_canceled(self) -> None:
        if self.download_cancel_requested:
            raise RuntimeError(tr(self.language, "download_canceled"))

    def _finish_download_worker(self) -> None:
        self.download_button.setEnabled(True)
        self.cancel_download_button.setEnabled(False)

    def cancel_download(self) -> None:
        if self.pending_download is None:
            return
        self.download_cancel_requested = True
        self._set_download_progress(0, tr(self.language, "download_cancel_requested"), busy=True)
        self._append_log(tr(self.language, "download_cancel_requested"))

    def _download_pull_candidates(self, rec: ModelRecommendation) -> list[str]:
        candidates = []
        if self._pull_name_has_safe_tag(rec.pull_name):
            candidates.append(rec.pull_name)
        base = f"hf.co/{rec.repo_id}"
        if base not in candidates:
            candidates.append(base)
        latest = f"hf.co/{rec.repo_id}:latest"
        if latest not in candidates:
            candidates.append(latest)
        return candidates

    def _pull_name_has_safe_tag(self, pull_name: str) -> bool:
        if not pull_name.startswith("hf.co/"):
            return True
        tail = pull_name.removeprefix("hf.co/")
        if ":" not in tail:
            return True
        tag = tail.rsplit(":", 1)[-1].upper()
        if tag in {"LATEST", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"}:
            return True
        if re.fullmatch(r"Q[2-6]_K(?:_[SML])?", tag):
            return True
        if tag.endswith(".GGUF"):
            return True
        return False

    def _should_retry_pull_with_latest(self, error: str) -> bool:
        lowered = error.lower()
        return "valid quantization scheme" in lowered or "specified tag" in lowered or "manifest" in lowered

    def _after_download(self, models: list[OllamaModel]) -> None:
        self._on_installed(models)
        self._set_download_progress(100, tr(self.language, "download_finished"))
        self._append_log(tr(self.language, "download_finished"))
        if self.pending_download:
            model = next((item for item in models if item.name == self.pending_download.pull_name), None)
            if model is None:
                model = OllamaModel(
                    name=self.pending_download.pull_name,
                    size_label=self.pending_download.size_label,
                    quant=self.pending_download.quant,
                    source="Downloaded",
                )
            self._append_log(f"{tr(self.language, 'auto_benchmark')}: {model.name}")
            self._run_benchmark(model)
            self.pending_download = None

    def _on_download_progress(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        percent = self._download_percent_from_line(text)
        if percent is None:
            self._set_download_progress(0, text, busy=True)
        else:
            self._set_download_progress(percent, text, busy=False)
        self._append_log(text)

    def _on_download_error(self, error: str) -> None:
        label = tr(self.language, "download_canceled") if tr(self.language, "download_canceled") in error else f"{tr(self.language, 'download_failed')}: {error}"
        self._set_download_progress(0, label, busy=False)
        self.pending_download = None
        self._append_log(error)

    def delete_selected_recommendation_model(self) -> None:
        rec = self._selected_recommendation()
        if rec is None:
            QMessageBox.information(self, tr(self.language, "no_selection_title"), tr(self.language, "no_selection_body"))
            return
        candidates = set(self._download_pull_candidates(rec))
        installed = [model.name for model in self.installed_models if model.name in candidates or model.name.startswith(f"hf.co/{rec.repo_id}:")]
        if not installed:
            self._append_log(f"{tr(self.language, 'delete_downloaded_none')}: {rec.repo_id}")
            return
        ok = QMessageBox.question(
            self,
            tr(self.language, "delete_downloaded_title"),
            f"{tr(self.language, 'delete_downloaded_body')}\n\n" + "\n".join(installed),
        )
        if ok != QMessageBox.StandardButton.Yes:
            return

        def work():
            for name in installed:
                delete_connected_model(name, self.client)
            return installed

        self._run_background(work, self._after_delete_downloaded)

    def _after_delete_downloaded(self, names: list[str]) -> None:
        self._append_log(f"{tr(self.language, 'delete_finished')}: {', '.join(names)}")
        self.refresh_installed()

    def _set_download_progress(self, percent: int, message: str, *, busy: bool = False) -> None:
        if busy:
            self.download_progress.setRange(0, 0)
            self.download_progress.setFormat("...")
        else:
            self.download_progress.setRange(0, 100)
            self.download_progress.setValue(max(0, min(100, percent)))
            self.download_progress.setFormat("%p%")
        self.download_progress_label.setText(message)

    def _download_percent_from_line(self, line: str) -> int | None:
        matches = re.findall(r"(\d{1,3})%", line)
        if not matches:
            return None
        return max(0, min(100, int(matches[-1])))

    def open_selected_hf_model(self) -> None:
        rec = self._selected_recommendation()
        if rec is None:
            QMessageBox.information(self, tr(self.language, "no_selection_title"), tr(self.language, "no_selection_body"))
            return
        webbrowser.open(f"https://huggingface.co/{rec.repo_id}")

    def refresh_results(self) -> None:
        include_deleted = self.show_deleted_checkbox.isChecked()
        rows = self.store.latest_report_per_model(include_deleted=include_deleted)
        self.results_rows = rows
        installed_keys = self._installed_model_keys()
        self._begin_table_update(self.results_table)
        self.results_table.setRowCount(len(rows))
        for row, item in enumerate(rows):
            use_scores = item.get("use_case_scores", {})
            resources = item.get("resource_summary", {})
            model_name = str(item.get("model_name", ""))
            deleted = bool(item.get("deleted_at"))
            installed = bool(self._model_match_keys(model_name) & installed_keys)
            install_text = tr(self.language, "installed_status_installed") if installed else tr(self.language, "installed_status_missing")
            status_text = str(item.get("status", ""))
            if deleted:
                status_text = f"{tr(self.language, 'deleted')} ({status_text})"
            values = [
                model_name,
                install_text,
                item.get("size_label", ""),
                self._thinking_label_for_model(model_name),
                item.get("quant", ""),
                status_text,
                item.get("total_score", ""),
                item.get("speed", ""),
                item.get("quality", ""),
                item.get("stability", ""),
                item.get("resource_fit", ""),
                self._format_use_list(item.get("recommended_uses", [])),
                self._row_tps(item),
                item.get("avg_first_token_seconds", ""),
            ]
            values.extend([use_scores.get(key, "") for key in USE_CASES])
            values.extend(
                [
                    self._percent(resources, "cpu_percent_peak"),
                    self._percent(resources, "gpu_percent_peak"),
                    self._gb_delta(resources, "ram_used_gb"),
                    self._gb_delta(resources, "vram_used_gb"),
                    self._format_local_time(item.get("finished_at", "")),
                    item.get("error", ""),
                ]
            )
            sort_keys = [
                model_name.casefold(),
                1 if installed else 0,
                self._size_sort_value(str(item.get("size_label", ""))),
                self._thinking_sort_for_model(model_name),
                str(item.get("quant", "")).casefold(),
                status_text.casefold(),
                self._number_or_zero(item.get("total_score", 0)),
                self._number_or_zero(item.get("speed", 0)),
                self._number_or_zero(item.get("quality", 0)),
                self._number_or_zero(item.get("stability", 0)),
                self._number_or_zero(item.get("resource_fit", 0)),
                self._format_use_list(item.get("recommended_uses", [])).casefold(),
                self._number_or_zero(item.get("avg_tokens_per_second", 0)) if self._has_measured_tps(item) else 0,
                self._number_or_zero(item.get("avg_first_token_seconds", 0)),
            ]
            sort_keys.extend(self._number_or_zero(use_scores.get(key, 0)) for key in USE_CASES)
            sort_keys.extend(
                [
                    self._percent_sort_value(resources, "cpu_percent_peak"),
                    self._percent_sort_value(resources, "gpu_percent_peak"),
                    self._gb_sort_value(resources, "ram_used_gb"),
                    self._gb_sort_value(resources, "vram_used_gb"),
                    self._time_sort_value(item.get("finished_at", "")),
                    str(item.get("error", "")).casefold(),
                ]
            )
            self._set_row(
                self.results_table,
                row,
                values,
                sort_keys=sort_keys,
                source_index=row,
                muted=deleted,
                soft_muted=(not installed and not deleted),
            )
        self._end_table_update(self.results_table, "results")
        self.show_result_detail()
        self._save_settings()

    def refresh_usage(self) -> None:
        if not hasattr(self, "usage_period_combo"):
            return
        period = self.usage_period_combo.currentData() or "7d"
        active_count = self._refresh_active_models()
        self._render_usage_overview_table(period, active_count)

    def import_usage_logs(self) -> None:
        if not hasattr(self, "import_usage_logs_button"):
            return
        self.import_usage_logs_button.setEnabled(False)
        self.usage_import_label.setText(tr(self.language, "usage_import_running"))
        sources = self._selected_usage_import_sources()
        worker = Worker(lambda: self._import_usage_logs(sources))
        worker.signals.result.connect(self._on_usage_logs_imported)
        worker.signals.error.connect(self._on_usage_logs_import_failed)
        worker.signals.finished.connect(lambda: self.import_usage_logs_button.setEnabled(True))
        self._start_worker(worker)

    def _selected_usage_import_sources(self) -> list[str]:
        sources = []
        if self.usage_ollama_checkbox.isChecked():
            sources.append("ollama")
        if self.usage_opencode_checkbox.isChecked():
            sources.append("opencode")
        return sources

    def _import_usage_logs(self, sources: list[str]) -> dict:
        if not sources:
            return {"sources": [], "parsed": 0, "inserted": 0, "skipped": 0}
        inserted = 0
        parsed = 0
        by_source: dict[str, dict[str, int]] = {}
        if "ollama" in sources:
            events = OllamaLogImporter().parse_paths(max_events=3000)
            stats = self._save_imported_usage_events(events)
            by_source["ollama"] = stats
            inserted += stats["inserted"]
            parsed += stats["parsed"]
        if "opencode" in sources:
            events = OpenCodeLogImporter().parse_paths(max_events=3000)
            stats = self._save_imported_usage_events(events)
            by_source["opencode"] = stats
            inserted += stats["inserted"]
            parsed += stats["parsed"]
        return {"sources": sources, "parsed": parsed, "inserted": inserted, "skipped": parsed - inserted, "by_source": by_source}

    def _save_imported_usage_events(self, events: list[Any]) -> dict[str, int]:
        inserted = 0
        for event in events:
            if self.store.save_usage_event(
                model_name=event.model_name,
                source=event.source,
                operation=event.operation,
                status=event.status,
                failure_type=event.failure_type,
                prompt_tokens=getattr(event, "prompt_tokens", None),
                eval_tokens=getattr(event, "eval_tokens", None),
                duration_seconds=event.duration_seconds,
                details=event.details,
                created_at=event.created_at,
                external_id=event.external_id,
            ):
                inserted += 1
        return {"parsed": len(events), "inserted": inserted, "skipped": len(events) - inserted}

    def _on_usage_logs_imported(self, result: dict) -> None:
        selected = ", ".join(result.get("sources", [])) or "none"
        text = tr(self.language, "usage_import_done").format(
            parsed=int(result.get("parsed", 0) or 0),
            inserted=int(result.get("inserted", 0) or 0),
            skipped=int(result.get("skipped", 0) or 0),
            sources=selected,
        )
        self.usage_import_label.setText(text)
        self._append_log(text)
        self.refresh_usage()

    def _on_usage_logs_import_failed(self, error: str) -> None:
        text = f"{tr(self.language, 'usage_import_failed')}: {error}"
        self.usage_import_label.setText(text)
        self._append_log(text)

    def _refresh_active_models(self) -> int:
        checked = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        unavailable = False
        try:
            rows = self.client.ps()
        except Exception as exc:
            rows = [{"model": "unavailable", "details": str(exc), "size": "", "processor": "", "expires_at": ""}]
            unavailable = True
        self._begin_table_update(self.usage_active_table)
        if not rows:
            self.usage_active_table.setRowCount(1)
            values = [tr(self.language, "no_active_models"), "-", "-", "-", checked]
            self._set_row(self.usage_active_table, 0, values, sort_keys=["", 0, "", 0, checked], muted=True)
            self._end_table_update(self.usage_active_table, "usage_active")
            return 0
        self.usage_active_table.setRowCount(len(rows))
        for row, item in enumerate(rows):
            model = str(item.get("model") or item.get("name") or item.get("details") or "unknown")
            size = str(item.get("size_label") or self._format_size_value(item.get("size", item.get("size_vram", ""))))
            details = item.get("details")
            processor = str(item.get("processor") or (details.get("processor", "") if isinstance(details, dict) else ""))
            until = str(item.get("expires_at") or item.get("until") or "")
            values = [model, size, processor or "unknown", self._format_local_time(until), checked]
            self._set_row(self.usage_active_table, row, values, sort_keys=[model.casefold(), self._size_sort_value(size), processor, self._time_sort_value(until), checked])
        self._end_table_update(self.usage_active_table, "usage_active")
        return 0 if unavailable else len(rows)

    def _render_usage_overview_table(self, period: str, active_count: int) -> None:
        rows = self.store.usage_overview(period=period)
        total_calls = sum(int(item.get("calls", 0) or 0) for item in rows)
        total_successes = sum(int(item.get("successes", 0) or 0) for item in rows)
        total_failures = sum(int(item.get("failures", 0) or 0) for item in rows)
        total_tokens = sum(int(item.get("total_tokens", 0) or 0) for item in rows)
        known_token_calls = sum(int(item.get("known_token_calls", 0) or 0) for item in rows)
        unknown_token_calls = sum(int(item.get("unknown_token_calls", 0) or 0) for item in rows)
        success_rate = round((total_successes / total_calls) * 100, 1) if total_calls else 0.0
        token_coverage = round((known_token_calls / total_calls) * 100, 1) if total_calls else 0.0
        top_source = str(rows[0].get("source_detail", rows[0].get("source", "-"))) if rows else "-"
        self.usage_active_count_value.setText(str(active_count))
        self.usage_calls_value.setText(f"{total_calls} / {total_failures}")
        self.usage_success_value.setText(f"{success_rate}%")
        self.usage_tokens_value.setText(str(total_tokens))
        self.usage_token_coverage_value.setText(f"{token_coverage}% / {unknown_token_calls} unknown")
        self.usage_top_source_value.setText(top_source)

        self._begin_table_update(self.usage_overview_table)
        self.usage_overview_table.setRowCount(len(rows))
        for row, item in enumerate(rows):
            calls = int(item.get("calls", 0) or 0)
            failures = int(item.get("failures", 0) or 0)
            failure_rate = round((failures / calls) * 100, 1) if calls else 0.0
            model = str(item.get("model_name", ""))
            source = str(item.get("source", "unknown"))
            source_detail = str(item.get("source_detail", source))
            operation = str(item.get("operation", ""))
            client_ip = str(item.get("client_ip", ""))
            token_coverage = float(item.get("token_coverage", 0) or 0)
            avg_duration = item.get("avg_duration_seconds")
            values = [
                model,
                source_detail,
                source,
                operation,
                client_ip or "-",
                calls,
                int(item.get("successes", 0) or 0),
                failures,
                f"{failure_rate}%",
                self._token_value(item.get("prompt_tokens")),
                self._token_value(item.get("eval_tokens")),
                self._token_value(item.get("total_tokens")),
                f"{token_coverage}%",
                int(item.get("unknown_token_calls", 0) or 0),
                "-" if avg_duration is None else avg_duration,
                self._format_local_time(item.get("last_used", "")),
                item.get("source_evidence", ""),
                item.get("recent_failure") or "-",
            ]
            sort_keys = [
                model.casefold(),
                source_detail.casefold(),
                source.casefold(),
                operation.casefold(),
                client_ip,
                calls,
                int(item.get("successes", 0) or 0),
                failures,
                failure_rate,
                self._number_or_zero(values[9]),
                self._number_or_zero(values[10]),
                self._number_or_zero(values[11]),
                token_coverage,
                int(item.get("unknown_token_calls", 0) or 0),
                self._number_or_zero(avg_duration),
                self._time_sort_value(item.get("last_used", "")),
                str(item.get("source_evidence", "")).casefold(),
                str(item.get("recent_failure", "")).casefold(),
            ]
            self._set_row(self.usage_overview_table, row, values, sort_keys=sort_keys, soft_muted=failures > 0)
        self._end_table_update(self.usage_overview_table, "usage_overview")

    def _render_usage_model_table(self, period: str) -> None:
        rows = self.store.usage_by_model(period=period)
        self._begin_table_update(self.usage_model_table)
        self.usage_model_table.setRowCount(len(rows))
        for row, item in enumerate(rows):
            calls = int(item.get("calls", 0) or 0)
            failures = int(item.get("failures", 0) or 0)
            failure_rate = round((failures / calls) * 100, 1) if calls else 0
            values = [
                item.get("model_name", ""),
                item.get("source", "unknown"),
                calls,
                int(item.get("successes", 0) or 0),
                failures,
                f"{failure_rate}%",
                self._token_value(item.get("prompt_tokens")),
                self._token_value(item.get("eval_tokens")),
                self._token_value(item.get("total_tokens")),
                self._format_local_time(item.get("last_used", "")),
            ]
            sort_keys = [str(values[0]).casefold(), str(values[1]).casefold(), calls, int(values[3]), failures, failure_rate, self._number_or_zero(values[6]), self._number_or_zero(values[7]), self._number_or_zero(values[8]), self._time_sort_value(item.get("last_used", ""))]
            self._set_row(self.usage_model_table, row, values, sort_keys=sort_keys)
        self._end_table_update(self.usage_model_table, "usage_model")

    def _render_usage_source_table(self, period: str) -> None:
        rows = self.store.usage_by_source(period=period)
        self._begin_table_update(self.usage_source_table)
        self.usage_source_table.setRowCount(len(rows))
        for row, item in enumerate(rows):
            values = [
                item.get("source", "unknown"),
                int(item.get("calls", 0) or 0),
                int(item.get("successes", 0) or 0),
                int(item.get("failures", 0) or 0),
                self._token_value(item.get("total_tokens")),
            ]
            self._set_row(self.usage_source_table, row, values, sort_keys=[str(values[0]).casefold(), values[1], values[2], values[3], self._number_or_zero(values[4])])
        self._end_table_update(self.usage_source_table, "usage_source")

    def _render_usage_failure_table(self, period: str) -> None:
        rows = self.store.failures_by_type(period=period)
        self._begin_table_update(self.usage_failure_table)
        self.usage_failure_table.setRowCount(len(rows))
        for row, item in enumerate(rows):
            values = [item.get("failure_type", "unknown"), int(item.get("failures", 0) or 0)]
            self._set_row(self.usage_failure_table, row, values, sort_keys=[str(values[0]).casefold(), values[1]])
        self._end_table_update(self.usage_failure_table, "usage_failure")

    def _render_usage_recent_table(self, period: str) -> None:
        rows = self.store.usage_events(period=period, limit=80)
        self._begin_table_update(self.usage_recent_table)
        self.usage_recent_table.setRowCount(len(rows))
        for row, item in enumerate(rows):
            values = [
                self._format_local_time(item.get("created_at", "")),
                item.get("model_name", ""),
                item.get("source", "unknown"),
                item.get("operation", ""),
                item.get("status", ""),
                item.get("failure_type", ""),
                self._token_value(item.get("prompt_tokens")),
                self._token_value(item.get("eval_tokens")),
            ]
            sort_keys = [self._time_sort_value(item.get("created_at", "")), str(values[1]).casefold(), str(values[2]).casefold(), str(values[3]).casefold(), str(values[4]).casefold(), str(values[5]).casefold(), self._number_or_zero(values[6]), self._number_or_zero(values[7])]
            self._set_row(self.usage_recent_table, row, values, sort_keys=sort_keys)
        self._end_table_update(self.usage_recent_table, "usage_recent")

    def delete_selected_result(self) -> None:
        rows = self._selected_result_rows()
        if not rows:
            return
        result_ids = [int(item.get("result_id", 0) or 0) for item in rows]
        result_ids = [result_id for result_id in result_ids if result_id > 0]
        if not result_ids:
            return
        names = [str(item.get("model_name", "")) for item in rows[:5] if item.get("model_name")]
        if len(rows) > 1:
            body = f"{tr(self.language, 'delete_result_body')}\n\n{tr(self.language, 'selected_count')}: {len(result_ids)}"
            if names:
                body += "\n" + "\n".join(names)
            if len(rows) > len(names):
                body += f"\n... +{len(rows) - len(names)}"
        else:
            body = f"{tr(self.language, 'delete_result_body')}\n\n{rows[0].get('model_name', '')}"
        ok = QMessageBox.question(
            self,
            tr(self.language, "delete_result_title"),
            body,
        )
        if ok != QMessageBox.StandardButton.Yes:
            return
        self.store.delete_reports(result_ids)
        self.refresh_results()
        self._render_installed_models(reset_combo=False)
        self._append_log(tr(self.language, "result_deleted"))

    def delete_selected_model_results(self) -> None:
        item = self._selected_result_row()
        if item is None:
            return
        model_name = str(item.get("model_name", ""))
        if not model_name:
            return
        ok = QMessageBox.question(
            self,
            tr(self.language, "delete_model_results_title"),
            f"{tr(self.language, 'delete_model_results_body')}\n\n{model_name}",
        )
        if ok != QMessageBox.StandardButton.Yes:
            return
        self.store.delete_reports_for_model(model_name)
        self.refresh_results()
        self._render_installed_models(reset_combo=False)
        self.show_selected_model_history()
        self._append_log(tr(self.language, "model_results_deleted"))

    def _selected_result_row(self) -> dict | None:
        rows = self._selected_result_rows()
        return rows[0] if rows else None

    def _selected_result_rows(self) -> list[dict]:
        rows: list[dict] = []
        for index in self._selected_source_indexes(self.results_table):
            if 0 <= index < len(self.results_rows):
                rows.append(self.results_rows[index])
        return rows

    def export_results_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            tr(self.language, "export_csv"),
            str(self.store.db_path.with_suffix(".csv")),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        rows = self.results_rows or self.store.latest_report_per_model(include_deleted=self.show_deleted_checkbox.isChecked())
        fields = self._export_fields()
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: self._export_value(row, field) for field in fields})
        self._append_log(f"{tr(self.language, 'export_finished')}: {path}")

    def export_results_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            tr(self.language, "export_json"),
            str(self.store.db_path.with_suffix(".json")),
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        rows = self.results_rows or self.store.latest_report_per_model(include_deleted=self.show_deleted_checkbox.isChecked())
        Path(path).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        self._append_log(f"{tr(self.language, 'export_finished')}: {path}")

    def backup_result_db(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        default = self.store.db_path.with_name(f"{self.store.db_path.stem}-{stamp}.backup{self.store.db_path.suffix}")
        path, _ = QFileDialog.getSaveFileName(
            self,
            tr(self.language, "backup_db"),
            str(default),
            "SQLite Database (*.sqlite *.db);;All Files (*)",
        )
        if not path:
            return
        shutil.copy2(self.store.db_path, path)
        self._append_log(f"{tr(self.language, 'backup_finished')}: {path}")

    def open_data_folder(self) -> None:
        folder = Path(self.storage_path_input.text().strip() or str(self.store.db_path)).expanduser().parent
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder))  # type: ignore[attr-defined]

    def show_result_detail(self) -> None:
        item = self._selected_result_row() or (self.results_rows[0] if self.results_rows else None)
        if item is None:
            self.results_detail.setPlainText(tr(self.language, "no_results"))
            self.results_leaders.setPlainText("")
            return
        model_name = str(item.get("model_name", ""))
        history = self.store.reports_for_model(model_name, limit=12, include_deleted=self.show_deleted_checkbox.isChecked())
        active_rows = self.store.latest_report_per_model(include_deleted=False)
        lines = [
            f"{tr(self.language, 'model')}: {model_name}",
            f"{tr(self.language, 'last_score')}: {item.get('total_score', '-')}",
            f"{tr(self.language, 'speed')}: {self._row_tps(item)}",
            f"{tr(self.language, 'quality')}: {item.get('quality', '-')}",
            f"{tr(self.language, 'best_uses')}: {self._format_use_list(item.get('recommended_uses', []))}",
            "",
            tr(self.language, "score_trend"),
        ]
        for report in reversed(history):
            score = self._number_or_zero(report.get("total_score", 0))
            bar = "#" * int(max(0, min(20, round(score / 5))))
            deleted = f" {tr(self.language, 'deleted')}" if report.get("deleted_at") else ""
            lines.append(f"{self._format_local_time(report.get('finished_at', ''))} | {score:5.1f} | {bar}{deleted}")
        quant_lines = self._quant_comparison_lines(model_name)
        if quant_lines:
            lines.extend(["", tr(self.language, "quant_compare"), *quant_lines])
        previous = history[1] if len(history) > 1 else None
        if previous:
            delta = self._number_or_zero(item.get("total_score", 0)) - self._number_or_zero(previous.get("total_score", 0))
            sign = "+" if delta >= 0 else ""
            lines.extend(["", f"{tr(self.language, 'score_delta')}: {sign}{round(delta, 2)}"])
        resources = item.get("resource_summary", {}) or {}
        if resources:
            lines.extend(
                [
                    "",
                    tr(self.language, "resource_summary"),
                    f"- CPU avg/peak: {resources.get('cpu_percent_avg', 0)}% / {resources.get('cpu_percent_peak', 0)}%",
                    f"- GPU avg/peak: {resources.get('gpu_percent_avg', 0)}% / {resources.get('gpu_percent_peak', 0)}%",
                    f"- RAM avg/peak: {resources.get('ram_used_gb_avg', 0)}GB / {resources.get('ram_used_gb_peak', 0)}GB",
                    f"- VRAM avg/peak: {resources.get('vram_used_gb_avg', 0)}GB / {resources.get('vram_used_gb_peak', 0)}GB",
                    f"- Samples: {int(resources.get('sample_count', 0) or 0)}",
                ]
            )
        self.results_detail.setPlainText("\n".join(lines))
        self.results_leaders.setPlainText(self._use_case_leaders_text(active_rows))

    def apply_settings(self) -> None:
        self.language = self.language_combo.currentData() or "en"
        self.client = OllamaClient(self.endpoint_input.text().strip())
        self.store = ResultStore(Path(self.storage_path_input.text().strip()))
        self.mode_combo.setCurrentText(self.default_mode_combo.currentText())
        self.thread_pool.setMaxThreadCount(max(2, self.parallel_spin.value()))
        self._save_settings()
        self._apply_language()
        self.refresh_installed()
        self.refresh_results()
        self._append_log(tr(self.language, "settings_applied"))

    def browse_result_db_path(self) -> None:
        current = Path(self.storage_path_input.text().strip() or str(self.store.db_path))
        selected, _ = QFileDialog.getSaveFileName(
            self,
            tr(self.language, "select_result_db"),
            str(current if current.name else current / "results.sqlite"),
            "SQLite Database (*.sqlite *.db);;All Files (*)",
        )
        if selected:
            self.storage_path_input.setText(selected)

    def browse_ollama_models_path(self) -> None:
        current = self.ollama_models_path_input.text().strip() or str(Path.home() / ".ollama" / "models")
        selected = QFileDialog.getExistingDirectory(self, tr(self.language, "select_ollama_models_path"), current)
        if selected:
            self.ollama_models_path_input.setText(selected)

    def open_ollama_models_folder(self) -> None:
        text = self.ollama_models_path_input.text().strip() or os.environ.get("OLLAMA_MODELS", "")
        folder = Path(text).expanduser() if text else Path.home() / ".ollama" / "models"
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder))  # type: ignore[attr-defined]

    def set_ollama_models_env(self) -> None:
        text = self.ollama_models_path_input.text().strip()
        if not text:
            return
        folder = Path(text).expanduser()
        ok = QMessageBox.question(
            self,
            tr(self.language, "ollama_models_set_title"),
            f"{tr(self.language, 'ollama_models_set_body')}\n\nOLLAMA_MODELS={folder}",
        )
        if ok != QMessageBox.StandardButton.Yes:
            return
        folder.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["setx", "OLLAMA_MODELS", str(folder)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        if result.returncode != 0:
            self._append_log(f"{tr(self.language, 'ollama_models_set_failed')}: {(result.stderr or result.stdout).strip()}")
            return
        os.environ["OLLAMA_MODELS"] = str(folder)
        self._save_settings()
        self._append_log(f"{tr(self.language, 'ollama_models_set_done')}: {folder}")

    def detect_ollama_endpoint(self) -> None:
        self.endpoint_auto_button.setEnabled(False)
        self._append_log(tr(self.language, "endpoint_detecting"))
        worker = Worker(self._detect_ollama_endpoint)
        worker.signals.result.connect(self._on_endpoint_detected)
        worker.signals.error.connect(lambda err: self._append_log(err))
        worker.signals.finished.connect(lambda: self.endpoint_auto_button.setEnabled(True))
        self._start_worker(worker)

    def _detect_ollama_endpoint(self) -> str:
        for endpoint in self._endpoint_candidates():
            try:
                OllamaClient(endpoint, timeout_seconds=2).list_models()
                return endpoint
            except Exception:
                continue
        return ""

    def _endpoint_candidates(self) -> list[str]:
        raw_values = [
            self.endpoint_input.text().strip(),
            os.environ.get("OLLAMA_HOST", "").strip(),
            "http://127.0.0.1:11434",
            "http://localhost:11434",
        ]
        candidates: list[str] = []
        seen: set[str] = set()
        for value in raw_values:
            endpoint = self._normalize_endpoint(value)
            if endpoint and endpoint not in seen:
                candidates.append(endpoint)
                seen.add(endpoint)
        return candidates

    def _normalize_endpoint(self, value: str) -> str:
        text = value.strip().rstrip("/")
        if not text:
            return ""
        if text.startswith("0.0.0.0:"):
            text = "127.0.0.1:" + text.split(":", 1)[1]
        if text.startswith(":"):
            text = "127.0.0.1" + text
        if not text.startswith(("http://", "https://")):
            text = "http://" + text
        return text.rstrip("/")

    def _on_endpoint_detected(self, endpoint: str) -> None:
        if endpoint:
            self.endpoint_input.setText(endpoint)
            self._append_log(f"{tr(self.language, 'endpoint_detected')}: {endpoint}")
        else:
            self._append_log(tr(self.language, "endpoint_not_found"))

    def _render_live_reports(self) -> None:
        self.benchmark_chart.set_reports(self.live_reports)
        self._begin_table_update(self.live_table)
        self.live_table.setRowCount(len(self.live_reports))
        for row, report in enumerate(self.live_reports):
            resources = report.resource_summary
            values = [
                report.model_name,
                report.status,
                report.size_label,
                self._thinking_label_for_model(report.model_name),
                report.score.total,
                report.score.speed,
                report.score.quality,
                self._percent(resources, "cpu_percent_peak"),
                self._percent(resources, "gpu_percent_peak"),
                self._gb_delta(resources, "vram_used_gb"),
                self._format_use_list(report.score.recommended_uses),
            ]
            sort_keys = [
                report.model_name.casefold(),
                report.status.casefold(),
                self._size_sort_value(report.size_label),
                self._thinking_sort_for_model(report.model_name),
                report.score.total,
                report.score.speed,
                report.score.quality,
                self._percent_sort_value(resources, "cpu_percent_peak"),
                self._percent_sort_value(resources, "gpu_percent_peak"),
                self._gb_sort_value(resources, "vram_used_gb"),
                self._format_use_list(report.score.recommended_uses).casefold(),
            ]
            self._set_row(self.live_table, row, values, sort_keys=sort_keys, source_index=row)
        self._end_table_update(self.live_table, "live")

    def _render_rankings(self) -> None:
        self.rankings_table.setRowCount(len(USE_CASES))
        for row, use_case in enumerate(USE_CASES):
            ranked = sorted(
                [report for report in self.live_reports
                 if report.status == "OK" and (report.run_settings or {}).get("mode") in {"standard", "thinking"}],
                key=lambda report: report.score.use_case_scores.get(use_case, 0),
                reverse=True,
            )
            values = [use_case_label(use_case, self.language)]
            for report in ranked[:3]:
                score = report.score.use_case_scores.get(use_case, 0)
                values.append(f"{report.model_name} ({score})")
            while len(values) < 4:
                values.append("-")
            self._set_row(self.rankings_table, row, values)
        self._apply_column_layout(self.rankings_table, "rankings")

    def show_benchmark_detail(self) -> None:
        report = self._selected_live_report()
        if report is None:
            self.benchmark_detail.clear()
            return
        index = self._selected_source_index(self.live_table)
        self.benchmark_chart.set_selected(index)
        resources = report.resource_summary
        run_settings = report.run_settings or {}
        lines = [
            f"Model: {report.model_name}",
            f"Status: {report.status}",
            f"Size: {report.size_label or '-'}",
            f"Thinking: {self._thinking_label_for_model(report.model_name)}",
            f"Thinking evidence: {self._thinking_evidence_for_model(report.model_name) or '-'}",
            f"Total: {report.score.total} | Speed: {report.score.speed} | Quality: {report.score.quality}",
            f"Best uses: {self._format_use_list(report.score.recommended_uses)}",
            "Practical fit (basic benchmarks only):",
            *[f"- {use_case_label(key, self.language)}: {FIT_EXPLANATIONS.get(self.language, FIT_EXPLANATIONS['en']).get(key, FIT_EXPLANATIONS['en'].get(key, ''))}"
              for key in report.score.recommended_uses],
            "",
            "Benchmark settings:",
            f"- Mode: {run_settings.get('mode', '-')}",
            f"- Warmup: {run_settings.get('warmup_runs', 0)}",
            f"- Repeats: {run_settings.get('repeat_count', 1)}",
            f"- Timeout: {run_settings.get('timeout_seconds', '-')}s",
            "",
            "Timing summary:",
            f"- Output generation: {self._report_tps(report)}",
            f"- Prompt processing: {self._avg_case_value(report.cases, 'prompt_tokens_per_second')} tok/s",
            f"- Model load: {self._avg_case_value(report.cases, 'load_seconds')}s",
            "",
            "Resource usage:",
            f"- CPU peak: {self._percent(resources, 'cpu_percent_peak')} / delta {self._percent(resources, 'cpu_percent_delta')}",
            f"- GPU peak: {self._percent(resources, 'gpu_percent_peak')} / delta {self._percent(resources, 'gpu_percent_delta')}",
            f"- RAM peak: {resources.get('ram_used_gb_peak', 0)}GB / delta {resources.get('ram_used_gb_delta', 0)}GB",
            f"- VRAM peak: {resources.get('vram_used_gb_peak', 0)}GB / delta {resources.get('vram_used_gb_delta', 0)}GB",
            "",
            "Cases:",
        ]
        for case in report.cases:
            lines.append(
                f"- {case.name}: {'OK' if case.passed else 'FAIL'} | score {case.score} | "
                f"{f'{case.tokens_per_second:.2f} tok/s' if case.tokens_per_second > 0 else tr(self.language, 'tps_unmeasured')} | prompt {round(case.prompt_tokens_per_second, 2)} tok/s | "
                f"load {round(case.load_seconds, 2)}s | first token {round(case.first_token_seconds, 2)}s"
            )
            if case.error:
                lines.append(f"  error: {case.error}")
        self.benchmark_detail.setPlainText("\n".join(lines))

    def _report_tps(self, report: BenchmarkReport) -> str:
        return (f"{report.avg_tokens_per_second:.2f} tok/s"
                if report.run_settings.get("tps_source") == "ollama_eval" and
                any(case.tokens_per_second > 0 for case in report.cases)
                else tr(self.language, "tps_unmeasured"))

    @staticmethod
    def _has_measured_tps(row: dict) -> bool:
        return (row.get("run_settings") or {}).get("tps_source") == "ollama_eval" and float(row.get("avg_tokens_per_second") or 0) > 0

    def _row_tps(self, row: dict) -> str:
        return (f"{float(row['avg_tokens_per_second']):.2f} tok/s"
                if self._has_measured_tps(row) else tr(self.language, "tps_unmeasured"))

    def _select_chart_report(self, index: int) -> None:
        for row in range(self.live_table.rowCount()):
            item = self.live_table.item(row, 0)
            if item is not None and item.data(SOURCE_INDEX_ROLE) == index:
                self.live_table.setCurrentCell(row, 0)
                self.show_benchmark_detail()
                break

    def _selected_live_report(self) -> BenchmarkReport | None:
        index = self._selected_source_index(self.live_table)
        if index is not None and 0 <= index < len(self.live_reports):
            return self.live_reports[index]
        return self.live_reports[0] if self.live_reports else None

    def _render_recommendations(self) -> None:
        self._begin_table_update(self.recommend_table)
        self.recommend_table.setRowCount(len(self.recommendations))
        for row, rec in enumerate(self.recommendations):
            values = [
                rec.trending_page or "",
                rec.model_name,
                rec.pull_name,
                rec.quant,
                rec.size_label,
                self._thinking_label_from_support(rec.thinking_support, rec.model_name, rec.pull_name, rec.repo_id),
                rec.fit,
                rec.score,
                self._format_use_list(rec.recommended_uses),
                rec.downloads,
                rec.reason,
            ]
            sort_keys = [
                rec.trending_page or 0,
                rec.model_name.casefold(),
                rec.pull_name.casefold(),
                rec.quant.casefold(),
                self._size_sort_value(rec.size_label),
                self._thinking_sort_from_support(rec.thinking_support, rec.model_name, rec.pull_name, rec.repo_id),
                rec.fit.casefold(),
                rec.score,
                self._format_use_list(rec.recommended_uses).casefold(),
                rec.downloads,
                rec.reason.casefold(),
            ]
            self._set_row(self.recommend_table, row, values, sort_keys=sort_keys, source_index=row)
        self._end_table_update(self.recommend_table, "recommend")
        self.show_recommendation_detail()

    def show_recommendation_detail(self) -> None:
        rec = self._selected_recommendation()
        if rec is None:
            self.recommend_detail.clear()
            return
        detail = [
            f"{tr(self.language, 'model')}: {rec.model_name}",
            f"{tr(self.language, 'trending_page')}: {rec.trending_page or '-'}",
            f"Repository: {rec.repo_id}",
            f"Pull: {rec.pull_name}",
            f"Quant: {rec.quant or '-'}",
            f"Size: {rec.size_label}",
            f"Thinking: {self._thinking_label_from_support(rec.thinking_support, rec.model_name, rec.pull_name, rec.repo_id)}",
            f"Thinking evidence: {rec.thinking_evidence or '-'}",
            f"Fit: {rec.fit}",
            f"Score: {rec.score}",
            f"Uses: {self._format_use_list(rec.recommended_uses)}",
            f"Downloads: {rec.downloads}",
            f"Likes: {rec.likes}",
            f"Updated: {rec.updated_at}",
            "",
            f"Reason: {rec.reason}",
        ]
        self.recommend_detail.setPlainText("\n".join(detail))

    def _selected_recommendation(self) -> ModelRecommendation | None:
        index = self._selected_source_index(self.recommend_table)
        if index is not None and 0 <= index < len(self.recommendations):
            return self.recommendations[index]
        return self.recommendations[0] if self.recommendations else None

    def _set_score_cards(self, report: BenchmarkReport) -> None:
        self.current_case_value.setText(self._short(report.model_name, 48))
        self.current_case_value.setToolTip(report.model_name)
        self.last_score_value.setText(f"{report.score.total}/100")
        self.speed_value.setText(f"{report.score.speed} | {self._report_tps(report)}")
        self.quality_value.setText(str(report.score.quality))
        self.best_uses_value.setText(self._format_use_list(report.score.recommended_uses))

    def _set_resource_cards(self, report: BenchmarkReport) -> None:
        resources = report.resource_summary
        self.cpu_value.setText(f"{self._percent(resources, 'cpu_percent_peak')}  Δ {self._percent(resources, 'cpu_percent_delta')}")
        self.gpu_value.setText(f"{self._percent(resources, 'gpu_percent_peak')}  Δ {self._percent(resources, 'gpu_percent_delta')}")
        self.ram_value.setText(f"{resources.get('ram_used_gb_peak', 0)}GB  Δ {resources.get('ram_used_gb_delta', 0)}GB")
        self.vram_value.setText(f"{resources.get('vram_used_gb_peak', 0)}GB  Δ {resources.get('vram_used_gb_delta', 0)}GB")

    def _update_run_progress(self) -> None:
        self.progress_value.setText(self._run_progress_text())

    def _run_progress_text(self) -> str:
        if self.run_all_total <= 0:
            return "-"
        remaining = max(0, self.run_all_total - self.run_all_completed)
        percent = round((self.run_all_completed / self.run_all_total) * 100, 1)
        return f"{self.run_all_completed}/{self.run_all_total} ({percent}%) · {remaining} {tr(self.language, 'remaining')}"

    def _finish_run_all(self) -> None:
        if self.run_all_active:
            self._update_run_progress()
        self.run_all_active = False
        self._finish_benchmark_process()

    def _finish_single_benchmark(self) -> None:
        self._finish_benchmark_process()

    def _on_benchmark_error(self, error: str) -> None:
        self._append_log(error)
        self.benchmark_failed = True
        if self.benchmark_active:
            self._set_process_stage("canceled", error.splitlines()[-1][:180], self.process_progress.value())

    def _start_benchmark_process(self, models: list[OllamaModel], mode: str, label: str) -> None:
        warmup_runs = self.warmup_spin.value()
        repeat_count = self.repeat_spin.value()
        case_count = self._benchmark_case_count(mode)
        planned_steps = len(models) * (warmup_runs + case_count * repeat_count)
        self.benchmark_total_steps = max(1, planned_steps)
        self.benchmark_seen_steps = set()
        self.benchmark_active = True
        self.benchmark_failed = False
        self._set_benchmark_buttons_running(True)
        self.process_plan_label.setText(
            tr(self.language, "process_plan_template").format(
                label=label,
                mode=mode,
                models=len(models),
                warmup=warmup_runs,
                repeats=repeat_count,
                steps=planned_steps,
            )
        )
        self._set_process_stage("queue", tr(self.language, "process_queue"), 5)

    def _mark_benchmark_step(self, text: str) -> None:
        if not self.benchmark_active:
            return
        if text not in self.benchmark_seen_steps:
            self.benchmark_seen_steps.add(text)
        stage_key = "warmup" if "warmup" in text.lower() else "cases"
        completed_ratio = min(1.0, len(self.benchmark_seen_steps) / max(1, self.benchmark_total_steps))
        percent = 10 + int(completed_ratio * 72)
        self._set_process_stage(stage_key, text, percent)

    def _finish_benchmark_process(self) -> None:
        if not self.benchmark_active:
            return
        canceled = self.benchmark_failed or (self.cancel_requested and self.run_all_completed < self.run_all_total)
        if canceled:
            self._set_process_stage("canceled", tr(self.language, "process_canceled"), self.process_progress.value())
        else:
            self._set_process_stage("done", tr(self.language, "process_done"), 100)
        self.benchmark_active = False
        self._set_benchmark_buttons_running(False)

    def _set_benchmark_buttons_running(self, running: bool) -> None:
        self.run_selected_button.setEnabled(not running)
        self.run_all_button.setEnabled(not running)
        self.retry_needed_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def _set_process_stage(self, stage: str, detail: str, percent: int) -> None:
        self.process_progress.setValue(max(0, min(100, percent)))
        self.process_state_label.setText(detail)
        self.process_steps_label.setText(self._process_steps_text(stage))

    def _process_steps_text(self, active_stage: str) -> str:
        stages = [
            ("queue", tr(self.language, "process_queue")),
            ("warmup", tr(self.language, "process_warmup")),
            ("cases", tr(self.language, "process_cases")),
            ("score", tr(self.language, "process_score")),
            ("save", tr(self.language, "process_save")),
        ]
        order = {key: index for index, (key, _label) in enumerate(stages)}
        active_index = order.get(active_stage, len(stages) if active_stage == "done" else 0)
        parts = []
        for index, (key, label) in enumerate(stages):
            if active_stage == "canceled" and index > active_index:
                prefix = "-"
            elif index < active_index or active_stage == "done":
                prefix = "[OK]"
            elif key == active_stage:
                prefix = ">"
            else:
                prefix = "-"
            parts.append(f"{prefix} {label}")
        return "   ".join(parts)

    def _benchmark_case_count(self, mode: str) -> int:
        if mode == "smoke":
            return 1
        if mode == "thinking":
            return 7
        return 6

    def _filter_new_recommendations(
        self,
        recs: list[ModelRecommendation],
        installed: list[OllamaModel],
    ) -> tuple[list[ModelRecommendation], list[str]]:
        installed_names = {model.name.lower() for model in installed}
        installed_roots = {model.name.split(":", 1)[0].lower() for model in installed}
        bad_keys = self._bad_benchmark_model_keys()
        filtered = []
        excluded: list[str] = []
        for rec in recs:
            names = {
                rec.model_name.lower(),
                rec.pull_name.lower(),
                rec.pull_name.replace("hf.co/", "", 1).lower(),
                rec.repo_id.lower(),
                rec.repo_id.split("/", 1)[-1].lower(),
            }
            root_names = {name.split(":", 1)[0] for name in names}
            if names & installed_names or root_names & installed_roots:
                excluded.append(f"{rec.model_name} - {tr(self.language, 'exclude_installed')}")
                continue
            if self._model_match_keys(rec.model_name, rec.pull_name, rec.repo_id) & bad_keys:
                excluded.append(f"{rec.model_name} - {tr(self.language, 'exclude_poor_result')}")
                continue
            filtered.append(rec)
        return filtered, excluded

    def _bad_benchmark_model_keys(self) -> set[str]:
        bad_keys: set[str] = set()
        seen: set[str] = set()
        for row in self.store.latest_reports(limit=1000):
            model_name = str(row.get("model_name", ""))
            if not model_name:
                continue
            primary_key = model_name.casefold()
            if primary_key in seen:
                continue
            seen.add(primary_key)
            status = str(row.get("status", "")).upper()
            score = self._number_or_zero(row.get("total_score", 0))
            if status != "OK" or score < BAD_RECOMMENDATION_SCORE:
                bad_keys.update(self._model_match_keys(model_name))
        return bad_keys

    def _model_match_keys(self, *values: str) -> set[str]:
        keys: set[str] = set()
        for value in values:
            text = str(value).strip().casefold()
            if not text:
                continue
            queue = [text, text.replace("hf.co/", "", 1)]
            variants = set(queue)
            for variant in queue:
                no_tag = variant.rsplit(":", 1)[0]
                variants.add(no_tag)
                if "/" in variant:
                    variants.add(variant.rsplit("/", 1)[-1])
                if "/" in no_tag:
                    variants.add(no_tag.rsplit("/", 1)[-1])
            keys.update(item for item in variants if item)
        return keys

    def _installed_model_keys(self) -> set[str]:
        keys: set[str] = set()
        for model in self.installed_models:
            keys.update(self._model_match_keys(model.name))
        return keys

    def _build_recommendation_report(self, recs: list[ModelRecommendation], installed: list[OllamaModel]) -> str:
        if not installed or not recs:
            return self._fallback_recommendation_report(recs)
        model_name = installed[0].name
        candidate_lines = "\n".join(
            f"- {rec.model_name} | page={rec.trending_page} | pull={rec.pull_name} | "
            f"thinking={rec.thinking_support} | fit={rec.fit} | score={rec.score}"
            for rec in recs[:8]
        )
        usage_context = self._usage_context_for_recommendations()
        report_language = LANGUAGES.get(self.language, "English")
        prompt = (
            "You are helping choose local Ollama models for this Windows PC. "
            f"Write a concise recommendation report in {report_language}. "
            "Pick the best 3 candidates and explain why. Consider benchmark score, best use case, usage volume, failure rate, installed status, quant, and size. "
            "Do not recommend deleting low-usage models only because they are low usage.\n\n"
            f"Hardware: CPU={self.hardware.cpu_name}, RAM={self.hardware.ram_gb}GB, "
            f"GPU={self.hardware.gpu_name}, VRAM={self.hardware.vram_gb}GB.\n\n"
            f"Local usage and failure context:\n{usage_context}\n\n"
            f"Candidate models from Hugging Face GGUF Trending pages 1-3:\n{candidate_lines}"
        )
        try:
            result = self.client.generate(model_name, prompt, num_predict=320, num_ctx=2048, temperature=0.2, timeout_seconds=180)
            self._record_usage_event(
                {
                    "model_name": model_name,
                    "operation": "recommendation:local_report",
                    "status": "success" if result.text.strip() else "failure",
                    "failure_type": "" if result.text.strip() else "empty response",
                    "prompt_tokens": result.prompt_eval_count,
                    "eval_tokens": result.eval_count,
                    "duration_seconds": round(result.elapsed_seconds, 3),
                    "details": {"candidates": len(recs)},
                }
            )
            if result.text.strip():
                return result.text.strip()
        except Exception as exc:
            self._record_usage_event(
                {
                    "model_name": model_name,
                    "operation": "recommendation:local_report",
                    "status": "failure",
                    "failure_type": self._usage_failure_type(str(exc)),
                    "details": {"error": str(exc)[:500]},
                }
            )
            return f"{tr(self.language, 'research_fallback')}\n\n{exc}\n\n{self._fallback_recommendation_report(recs)}"
        return self._fallback_recommendation_report(recs)

    def _fallback_recommendation_report(self, recs: list[ModelRecommendation]) -> str:
        if not recs:
            return tr(self.language, "research_fallback")
        lines = [tr(self.language, "research_fallback"), "", self._usage_context_for_recommendations(), ""]
        for index, rec in enumerate(recs[:5], start=1):
            thinking = self._thinking_label_from_support(rec.thinking_support, rec.model_name, rec.pull_name, rec.repo_id)
            lines.append(
                f"{index}. P{rec.trending_page} {rec.model_name} ({rec.score}) "
                f"[Thinking: {thinking}] - {self._format_use_list(rec.recommended_uses)}"
            )
            lines.append(f"   {rec.pull_name}")
        return "\n".join(lines)

    def _usage_context_for_recommendations(self) -> str:
        rows = self.store.usage_by_model(period="30d")[:8]
        if not rows:
            return "Usage context: unavailable. No local-model-bench usage events are recorded yet."
        lines = ["Usage context from the last 30 days:"]
        for row in rows:
            calls = int(row.get("calls", 0) or 0)
            failures = int(row.get("failures", 0) or 0)
            failure_rate = round((failures / calls) * 100, 1) if calls else 0
            lines.append(
                f"- {row.get('model_name', '')} | source={row.get('source', 'unknown')} | "
                f"calls={calls} | failure_rate={failure_rate}% | tokens={self._token_value(row.get('total_tokens'))}"
            )
        return "\n".join(lines)

    def _export_fields(self) -> list[str]:
        return [
            "result_id",
            "model_name",
            "size_label",
            "quant",
            "status",
            "total_score",
            "quality",
            "speed",
            "stability",
            "resource_fit",
            "avg_tokens_per_second",
            "avg_first_token_seconds",
            "avg_prompt_tokens_per_second",
            "avg_load_seconds",
            "benchmark_mode",
            "warmup_runs",
            "repeat_count",
            "timeout_seconds",
            "recommended_uses",
            "finished_at",
            "deleted_at",
            "error",
        ]

    def _export_value(self, row: dict, field: str) -> Any:
        value = row.get(field, "")
        if field in {"benchmark_mode", "warmup_runs", "repeat_count", "timeout_seconds"}:
            settings = row.get("run_settings", {}) or {}
            return settings.get(
                {
                    "benchmark_mode": "mode",
                    "warmup_runs": "warmup_runs",
                    "repeat_count": "repeat_count",
                    "timeout_seconds": "timeout_seconds",
                }[field],
                "",
            )
        if field == "recommended_uses":
            return self._format_use_list(value or [])
        return value

    def _quant_comparison_lines(self, model_name: str) -> list[str]:
        root = model_name.split(":", 1)[0].casefold()
        rows = [
            row
            for row in self.store.latest_report_per_model(include_deleted=False)
            if str(row.get("model_name", "")).split(":", 1)[0].casefold() == root
        ]
        if len(rows) <= 1:
            return []
        ranked = sorted(rows, key=lambda row: self._number_or_zero(row.get("total_score", 0)), reverse=True)
        return [
            f"- {row.get('model_name', '')} | {row.get('quant') or '-'} | {row.get('total_score', '-')} | {self._row_tps(row)}"
            for row in ranked[:8]
        ]

    def _use_case_leaders_text(self, rows: list[dict]) -> str:
        lines = [tr(self.language, "best_use_leaders")]
        for use_case in USE_CASES:
            ranked = sorted(
                rows,
                key=lambda row: self._number_or_zero((row.get("use_case_scores", {}) or {}).get(use_case, 0)),
                reverse=True,
            )
            leader = ranked[0] if ranked else None
            if leader is None:
                lines.append(f"- {use_case_label(use_case, self.language)}: -")
                continue
            score = (leader.get("use_case_scores", {}) or {}).get(use_case, 0)
            lines.append(f"- {use_case_label(use_case, self.language)}: {leader.get('model_name', '')} ({score})")
        return "\n".join(lines)

    def _load_cached_recommendations(self) -> list[ModelRecommendation]:
        recs: list[ModelRecommendation] = []
        for item in self.settings.get("recommendation_cache", []) or []:
            if isinstance(item, dict):
                try:
                    payload = {
                        "model_name": str(item.get("model_name", "")),
                        "repo_id": str(item.get("repo_id", "")),
                        "pull_name": str(item.get("pull_name", "")),
                        "quant": str(item.get("quant", "")),
                        "size_label": str(item.get("size_label", "")),
                        "fit": str(item.get("fit", "")),
                        "score": float(item.get("score", 0) or 0),
                        "recommended_uses": list(item.get("recommended_uses", []) or []),
                        "reason": str(item.get("reason", "")),
                        "downloads": int(item.get("downloads", 0) or 0),
                        "likes": int(item.get("likes", 0) or 0),
                        "updated_at": str(item.get("updated_at", "")),
                        "trending_page": int(item.get("trending_page", 0) or 0),
                        "thinking_support": str(item.get("thinking_support", "unknown")),
                        "thinking_evidence": str(item.get("thinking_evidence", "")),
                    }
                    if payload["model_name"] and payload["repo_id"]:
                        recs.append(ModelRecommendation(**payload))
                except TypeError:
                    continue
        return recs

    def _recommendation_cache_payload(self) -> list[dict]:
        return [rec.__dict__ for rec in self.recommendations[:50]]

    def _run_background(self, fn, on_result) -> None:
        worker = Worker(fn)
        worker.signals.result.connect(on_result)
        worker.signals.error.connect(lambda err: self._append_log(err))
        self._start_worker(worker)

    def _start_worker(self, worker: Worker) -> None:
        self.active_workers.add(worker)
        worker.signals.finished.connect(lambda w=worker: self.active_workers.discard(w))
        self.thread_pool.start(worker)

    def _append_log(self, text: str) -> None:
        self.progress_log.append(text)

    def _record_usage_event(self, event: dict) -> None:
        self.store.save_usage_event(
            model_name=str(event.get("model_name", "")),
            source="local-model-bench",
            operation=str(event.get("operation", "")),
            status=str(event.get("status", "failure")),
            failure_type=str(event.get("failure_type", "")),
            prompt_tokens=event.get("prompt_tokens"),
            eval_tokens=event.get("eval_tokens"),
            duration_seconds=event.get("duration_seconds"),
            details=event.get("details", {}),
        )

    def _usage_failure_type(self, message: str) -> str:
        lowered = message.lower()
        if "timeout" in lowered or "timed out" in lowered:
            return "timeout"
        if "length" in lowered:
            return "length"
        if "empty" in lowered:
            return "empty response"
        if "connection" in lowered or "refused" in lowered:
            return "connection error"
        if "ollama" in lowered or "runtime" in lowered:
            return "runtime error"
        return "unknown"

    def _format_use_list(self, values: list[str]) -> str:
        return format_use_cases(values, self.language)

    def _on_parameter_range_changed(self, _low: int | None = None, _high: int | None = None) -> None:
        self._update_parameter_range_label()
        if hasattr(self, "endpoint_input"):
            self._save_settings()

    def reset_parameter_range(self) -> None:
        self.parameter_slider.setRangeValues(0, 4)
        self._update_parameter_range_label()
        self._save_settings()

    def _parameter_range_values(self) -> tuple[float, float]:
        return (
            PARAMETER_OPTIONS[self.parameter_slider.lowValue()][0],
            PARAMETER_OPTIONS[self.parameter_slider.highValue()][0],
        )

    def _update_parameter_range_label(self) -> None:
        min_label = PARAMETER_OPTIONS[self.parameter_slider.lowValue()][1]
        max_label = PARAMETER_OPTIONS[self.parameter_slider.highValue()][1]
        self.parameter_value_label.setText(f"{min_label} - {max_label}")

    def _thinking_label(self, *values: str) -> str:
        score = self._thinking_sort_value(*values)
        if score >= 2:
            return tr(self.language, "thinking_yes")
        if score >= 1:
            return tr(self.language, "thinking_likely")
        return tr(self.language, "thinking_unknown")

    def _thinking_label_for_model(self, model_name: str) -> str:
        metadata = self.installed_hf_metadata.get(model_name, {})
        return self._thinking_label_from_support(metadata.get("thinking_support", "unknown"), model_name)

    def _thinking_sort_for_model(self, model_name: str) -> int:
        metadata = self.installed_hf_metadata.get(model_name, {})
        return self._thinking_sort_from_support(metadata.get("thinking_support", "unknown"), model_name)

    def _thinking_evidence_for_model(self, model_name: str) -> str:
        metadata = self.installed_hf_metadata.get(model_name, {})
        return str(metadata.get("thinking_evidence", ""))

    def _thinking_label_from_support(self, support: str, *fallback_values: str) -> str:
        score = self._thinking_sort_from_support(support, *fallback_values)
        if score >= 2:
            return tr(self.language, "thinking_yes")
        if score >= 1:
            return tr(self.language, "thinking_likely")
        return tr(self.language, "thinking_unknown")

    def _thinking_sort_from_support(self, support: str, *fallback_values: str) -> int:
        normalized = str(support).casefold()
        if normalized == "yes":
            return 2
        if normalized == "likely":
            return 1
        return self._thinking_sort_value(*fallback_values)

    def _thinking_sort_value(self, *values: str) -> int:
        text = " ".join(str(value).lower() for value in values)
        if any(marker in text for marker in ["thinking", "reasoning", "reasoner", "deepseek-r1", "qwq", "qwen3"]):
            return 2
        if self._param_size_b(text) >= 27:
            return 1
        return 0

    def _param_size_b(self, text: str) -> float:
        match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", text, flags=re.IGNORECASE)
        return float(match.group(1)) if match else 0.0

    def _param_label_for_model(self, text: str) -> str:
        value = self._param_size_b(text)
        if value <= 0:
            return "-"
        if value.is_integer():
            return f"{int(value)}B"
        return f"{value:g}B"

    def _set_recommend_progress(self, percent: int, message: str) -> None:
        self.recommend_progress.setValue(max(0, min(100, percent)))
        self.recommend_progress_label.setText(message)

    def _apply_language(self) -> None:
        self.tabs.setTabText(0, tr(self.language, "installed"))
        self.tabs.setTabText(1, tr(self.language, "benchmark"))
        self.tabs.setTabText(2, tr(self.language, "recommendations"))
        self.tabs.setTabText(3, tr(self.language, "results"))
        self.tabs.setTabText(4, tr(self.language, "usage"))
        self.tabs.setTabText(5, tr(self.language, "settings"))
        self.refresh_installed_button.setText(tr(self.language, "refresh_installed"))
        self.delete_model_button.setText(tr(self.language, "delete_model"))
        self.register_ollama_button.setText(tr(self.language, "register_ollama"))
        self.register_opencodex_button.setText(tr(self.language, "register_opencodex"))
        self.connect_codex_button.setText(tr(self.language, "connect_codex"))
        self.smoke_model_button.setText(tr(self.language, "basic_test"))
        self.chart_title.setText(tr(self.language, "chart_comparison"))
        if not self.connection_status.text():
            self.connection_status.setText(tr(self.language, "connection_hint"))
        self.history_title.setText(tr(self.language, "model_history"))
        self.model_label.setText(tr(self.language, "model"))
        self.mode_label.setText(tr(self.language, "mode"))
        self.run_selected_button.setText(tr(self.language, "run_selected"))
        self.run_all_button.setText(tr(self.language, "run_all"))
        self.retry_needed_button.setText(tr(self.language, "retry_needed"))
        self.stop_button.setText(tr(self.language, "stop_after_current"))
        self.search_button.setText(tr(self.language, "find_models"))
        self.download_button.setText(tr(self.language, "download_selected"))
        self.cancel_download_button.setText(tr(self.language, "cancel_download"))
        self.delete_downloaded_button.setText(tr(self.language, "delete_downloaded"))
        self.open_hf_button.setText(tr(self.language, "open_hf"))
        self.download_progress_label.setText(tr(self.language, "download_progress_idle"))
        self.hf_filter_title.setText(tr(self.language, "hf_filters"))
        self.hf_task_label.setText(tr(self.language, "hf_tasks"))
        self.hf_library_label.setText(tr(self.language, "hf_libraries"))
        self.hf_app_label.setText(tr(self.language, "hf_apps"))
        self.hf_language_label.setText(tr(self.language, "hf_languages"))
        self.hf_license_label.setText(tr(self.language, "hf_licenses"))
        self.hf_provider_label.setText(tr(self.language, "hf_providers"))
        self.hf_other_label.setText(tr(self.language, "hf_other"))
        self.hf_page_range_label.setText(tr(self.language, "hf_page_range"))
        self.hf_keyword_label.setText(tr(self.language, "hf_keyword"))
        self.hf_keyword_input.setPlaceholderText(tr(self.language, "hf_keyword_placeholder"))
        self._translate_hf_filter_options()
        self.parameter_title.setText(tr(self.language, "parameters"))
        self.reset_parameters_button.setText(tr(self.language, "reset_parameters"))
        self._update_parameter_range_label()
        self.recommend_progress_label.setText(tr(self.language, "recommend_progress_idle"))
        self.recommend_detail_title.setText(tr(self.language, "recommendation_detail"))
        self.research_report_title.setText(tr(self.language, "research_report"))
        self.refresh_results_button.setText(tr(self.language, "refresh_results"))
        self.delete_result_button.setText(tr(self.language, "delete_result"))
        self.delete_model_results_button.setText(tr(self.language, "delete_model_results"))
        self.export_csv_button.setText(tr(self.language, "export_csv"))
        self.export_json_button.setText(tr(self.language, "export_json"))
        self.backup_db_button.setText(tr(self.language, "backup_db"))
        self.show_deleted_checkbox.setText(tr(self.language, "show_deleted"))
        self.usage_period_label.setText(tr(self.language, "period"))
        self.refresh_usage_button.setText(tr(self.language, "refresh_usage"))
        self.usage_ollama_checkbox.setText(tr(self.language, "usage_source_ollama"))
        self.usage_opencode_checkbox.setText(tr(self.language, "usage_source_opencode"))
        self.import_usage_logs_button.setText(tr(self.language, "import_usage_logs"))
        self.usage_import_label.setText(tr(self.language, "usage_import_idle"))
        self.active_models_title.setText(tr(self.language, "active_ollama_models"))
        self.usage_overview_title.setText(tr(self.language, "usage_overview"))
        self.usage_active_count_title.setText(tr(self.language, "usage_active_count"))
        self.usage_calls_title.setText(tr(self.language, "usage_calls"))
        self.usage_success_title.setText(tr(self.language, "usage_success_rate"))
        self.usage_tokens_title.setText(tr(self.language, "usage_total_tokens"))
        self.usage_token_coverage_title.setText(tr(self.language, "usage_token_coverage"))
        self.usage_top_source_title.setText(tr(self.language, "usage_top_source"))
        self.endpoint_label.setText(tr(self.language, "ollama_endpoint"))
        self.endpoint_auto_button.setText(tr(self.language, "auto_endpoint"))
        self.storage_label.setText(tr(self.language, "result_db_path"))
        self.storage_browse_button.setText(tr(self.language, "browse"))
        self.open_data_folder_button.setText(tr(self.language, "open_data_folder"))
        self.settings_backup_db_button.setText(tr(self.language, "backup_db"))
        self.ollama_models_label.setText(tr(self.language, "ollama_models_path"))
        self.ollama_models_browse_button.setText(tr(self.language, "browse"))
        self.ollama_models_set_button.setText(tr(self.language, "set_ollama_models"))
        self.ollama_models_open_button.setText(tr(self.language, "open_ollama_models"))
        self.ollama_models_hint.setText(tr(self.language, "ollama_models_hint"))
        self.default_mode_label.setText(tr(self.language, "default_benchmark"))
        self.parallel_label.setText(tr(self.language, "concurrency"))
        self.warmup_label.setText(tr(self.language, "benchmark_warmup_runs"))
        self.repeat_label.setText(tr(self.language, "benchmark_repeat_count"))
        self.language_label.setText(tr(self.language, "language"))
        self.apply_button.setText(tr(self.language, "apply_settings"))
        self.about_title.setText(tr(self.language, "about_app"))
        self.about_body.setText(
            "\n".join(
                [
                    f"{tr(self.language, 'version')}: {__version__}",
                    f"{tr(self.language, 'creator')}: {APP_CREATOR}",
                    f"{tr(self.language, 'repository')}: {APP_REPOSITORY}",
                    f"{tr(self.language, 'local_data_path')}: {self.settings_path.parent}",
                ]
            )
        )
        self.current_case_title.setText(tr(self.language, "current_case"))
        self.last_score_title.setText(tr(self.language, "last_score"))
        self.speed_title.setText(tr(self.language, "speed"))
        self.quality_title.setText(tr(self.language, "quality"))
        self.best_uses_title.setText(tr(self.language, "best_uses"))
        self.progress_title.setText(tr(self.language, "progress"))
        self.process_title.setText(tr(self.language, "benchmark_process"))
        if not self.benchmark_active:
            self.process_state_label.setText(tr(self.language, "process_idle"))
            self.process_progress.setValue(0)
            self.process_steps_label.setText(self._process_steps_text("queue"))
            self.process_plan_label.setText(tr(self.language, "process_idle_plan"))
        self._update_run_progress()
        self.cpu_title.setText(tr(self.language, "cpu_peak"))
        self.gpu_title.setText(tr(self.language, "gpu_peak"))
        self.ram_title.setText(tr(self.language, "ram_peak"))
        self.vram_title.setText(tr(self.language, "vram_peak"))
        self.rankings_title.setText(tr(self.language, "rankings"))
        self.benchmark_detail_title.setText(tr(self.language, "benchmark_detail"))
        self.hardware_label.setText(self._hardware_text())
        for table, name in [
            (self.installed_table, "installed"),
            (self.history_table, "history"),
            (self.live_table, "live"),
            (self.rankings_table, "rankings"),
            (self.recommend_table, "recommend"),
            (self.usage_active_table, "usage_active"),
            (self.usage_overview_table, "usage_overview"),
        ]:
            table.setHorizontalHeaderLabels(self._table_headers(name))
            self._apply_column_layout(table, name)
        self.results_table.setHorizontalHeaderLabels(self._results_headers())
        self._apply_column_layout(self.results_table, "results")
        self._render_recommendations()
        self._render_live_reports()
        self._render_rankings()
        self.refresh_results()
        self.show_selected_model_history()

    def _hardware_text(self) -> str:
        return (
            f"{tr(self.language, 'hardware_prefix')}: CPU {self.hardware.cpu_name} | "
            f"RAM {self.hardware.ram_gb}GB | GPU {self.hardware.gpu_name} {self.hardware.vram_gb}GB | "
            f"Drive {self.hardware.preferred_drive} ({self.hardware.preferred_drive_free_gb}GB free)"
        )

    def _results_headers(self) -> list[str]:
        return (
            self._table_headers("results_base")
            + [use_case_label(key, self.language) for key in USE_CASES]
            + ["CPU Peak", "GPU Peak", "RAM Δ", "VRAM Δ"]
            + self._table_headers("results_tail")
        )

    def _score_card(self, layout: QGridLayout, row: int, col: int) -> tuple[QLabel, QLabel]:
        frame = QFrame()
        frame.setObjectName("scoreCard")
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        box = QVBoxLayout(frame)
        box.setContentsMargins(14, 12, 14, 12)
        title = QLabel()
        title.setObjectName("cardTitle")
        value = QLabel("-")
        value.setObjectName("cardValue")
        value.setWordWrap(True)
        box.addWidget(title)
        box.addWidget(value)
        layout.addWidget(frame, row, col)
        return title, value

    def _setup_table(self, table: QTableWidget, name: str) -> None:
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setTextElideMode(Qt.TextElideMode.ElideRight)
        table.setWordWrap(False)
        table.setSortingEnabled(False)
        table.setHorizontalHeaderLabels(self._table_headers(name))
        table.horizontalHeader().setMinimumSectionSize(70)
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setDefaultSectionSize(26)
        self._apply_column_layout(table, name)
        table.setSortingEnabled(name in SORTABLE_TABLES)

    def _apply_column_layout(self, table: QTableWidget, name: str) -> None:
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        widths = {
            "installed": [260, 100, 80, 85, 95, 90, 90, 70, 75, 95, 95, 140, 140],
            "history": [170, 80, 75, 75, 75, 300],
            "live": [260, 80, 85, 95, 70, 70, 70, 85, 85, 95, 300],
            "rankings": [150, 270, 270, 270],
            "recommend": [60, 230, 350, 90, 80, 95, 95, 70, 230, 95, 360],
            "usage_active": [280, 90, 110, 160, 150],
            "usage_model": [260, 140, 70, 80, 80, 85, 95, 95, 95, 150],
            "usage_source": [150, 80, 80, 80, 110],
            "usage_failure": [170, 80],
            "usage_recent": [145, 240, 130, 180, 85, 130, 90, 90],
            "usage_overview": [300, 150, 105, 135, 105, 70, 70, 70, 80, 90, 90, 105, 100, 115, 80, 145, 170, 150],
        }.get(name)
        if widths:
            for index, width in enumerate(widths[: table.columnCount()]):
                table.setColumnWidth(index, width)
        if name == "results":
            widths = [280, 80, 85, 95, 90, 80, 70, 70, 70, 75, 80, 260, 85, 85]
            widths += [95 for _ in USE_CASES]
            widths += [80, 80, 80, 85, 145, 260]
            for index, width in enumerate(widths[: table.columnCount()]):
                table.setColumnWidth(index, width)

    def _begin_table_update(self, table: QTableWidget) -> None:
        table.setSortingEnabled(False)

    def _end_table_update(self, table: QTableWidget, name: str) -> None:
        self._apply_column_layout(table, name)
        table.setSortingEnabled(name in SORTABLE_TABLES)

    def _set_row(
        self,
        table: QTableWidget,
        row: int,
        values: list[Any],
        sort_keys: list[Any] | None = None,
        source_index: int | None = None,
        muted: bool = False,
        soft_muted: bool = False,
    ) -> None:
        for col, value in enumerate(values):
            text = str(value)
            item = SortTableWidgetItem(text)
            item.setToolTip(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            if muted:
                item.setForeground(QColor("#8a96a3"))
                item.setBackground(QColor("#f7f9fb"))
            elif soft_muted:
                item.setForeground(QColor("#697586"))
            if sort_keys and col < len(sort_keys):
                item.setData(SORT_ROLE, sort_keys[col])
            if source_index is not None:
                item.setData(SOURCE_INDEX_ROLE, source_index)
            table.setItem(row, col, item)

    def _selected_source_index(self, table: QTableWidget) -> int | None:
        row = table.currentRow()
        if row < 0:
            return None
        item = table.item(row, 0)
        if item is None:
            return None
        value = item.data(SOURCE_INDEX_ROLE)
        return value if isinstance(value, int) else None

    def _selected_source_indexes(self, table: QTableWidget) -> list[int]:
        selected_rows = {index.row() for index in table.selectedIndexes()}
        if not selected_rows and table.currentRow() >= 0:
            selected_rows.add(table.currentRow())
        source_indexes: list[int] = []
        seen: set[int] = set()
        for row in sorted(selected_rows):
            item = table.item(row, 0)
            if item is None:
                continue
            value = item.data(SOURCE_INDEX_ROLE)
            if isinstance(value, int) and value not in seen:
                seen.add(value)
                source_indexes.append(value)
        return source_indexes

    def _size_sort_value(self, value: str) -> float:
        text = str(value).strip().lower()
        number = self._leading_number(text)
        if number is None:
            return -1.0
        if "gb" in text:
            return number * 1024
        if "mb" in text:
            return number
        if text.endswith("b"):
            return number
        return number

    def _format_size_value(self, value: Any) -> str:
        try:
            number = float(value or 0)
        except (TypeError, ValueError):
            return str(value or "")
        if number <= 0:
            return ""
        if number > 1024**3:
            return f"{number / (1024**3):.1f} GB"
        if number > 1024**2:
            return f"{number / (1024**2):.0f} MB"
        return str(value)

    def _token_value(self, value: Any) -> str:
        if value is None:
            return "unavailable"
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return "unavailable"

    def _format_token_count(self, value: Any) -> str:
        try:
            return f"{int(value or 0):,}"
        except (TypeError, ValueError):
            return "0"

    def _percent_sort_value(self, resources: dict, key: str) -> float:
        return self._number_or_zero(resources.get(key, 0))

    def _gb_sort_value(self, resources: dict, key: str) -> float:
        return self._number_or_zero(resources.get(f"{key}_delta", 0))

    def _number_or_zero(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return self._leading_number(str(value)) or 0.0

    def _avg_case_value(self, cases: list[Any], field: str) -> float:
        values = [self._number_or_zero(getattr(case, field, 0)) for case in cases]
        values = [value for value in values if value > 0]
        return round(sum(values) / len(values), 2) if values else 0.0

    def _format_local_time(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return "-"
        dt = self._parse_time(text)
        if dt is None:
            return text
        return dt.astimezone().strftime("%Y-%m-%d %H:%M")

    def _time_sort_value(self, value: Any) -> float:
        text = str(value or "").strip()
        if not text:
            return 0.0
        dt = self._parse_time(text)
        return dt.astimezone().timestamp() if dt else 0.0

    def _parse_time(self, value: str) -> datetime | None:
        normalized = value.replace("Z", "+00:00")
        normalized = re.sub(r"(\.\d{6})\d+([+-]\d\d:\d\d)?$", r"\1\2", normalized)
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return None

    def _leading_number(self, value: str) -> float | None:
        chars = []
        for char in value.strip():
            if char.isdigit() or char in ".-":
                chars.append(char)
                continue
            break
        if not chars:
            return None
        try:
            return float("".join(chars))
        except ValueError:
            return None

    def _table_headers(self, name: str) -> list[str]:
        return TABLE_HEADER_TRANSLATIONS.get(self.language, {}).get(name, TABLE_HEADERS.get(name, []))

    def _set_language_combo(self, language: str) -> None:
        for index in range(self.language_combo.count()):
            if self.language_combo.itemData(index) == language:
                self.language_combo.setCurrentIndex(index)
                return
        self.language_combo.setCurrentIndex(0)

    def _combo_from_options(self, options: list[tuple[str, str]], selected_value: Any) -> QComboBox:
        combo = QComboBox()
        for label, value in options:
            combo.addItem(label, value)
        selected = str(selected_value or "")
        for index in range(combo.count()):
            if combo.itemData(index) == selected:
                combo.setCurrentIndex(index)
                break
        combo.currentIndexChanged.connect(lambda _index: self._save_settings() if hasattr(self, "endpoint_input") else None)
        return combo

    def _settings_int(self, key: str, default: int) -> int:
        try:
            return int(self.settings.get(key, default))
        except (TypeError, ValueError):
            return default

    def _translate_hf_filter_options(self) -> None:
        for combo, options in [
            (self.hf_task_combo, HF_TASK_OPTIONS),
            (self.hf_library_combo, HF_LIBRARY_OPTIONS),
            (self.hf_app_combo, HF_APP_OPTIONS),
            (self.hf_language_combo, HF_LANGUAGE_OPTIONS),
            (self.hf_license_combo, HF_LICENSE_OPTIONS),
            (self.hf_provider_combo, HF_PROVIDER_OPTIONS),
            (self.hf_other_combo, HF_OTHER_OPTIONS),
        ]:
            self._translate_combo_options(combo, options)

    def _translate_combo_options(self, combo: QComboBox, options: list[tuple[str, str]]) -> None:
        translations = HF_OPTION_LABEL_TRANSLATIONS.get(str(self.language), {})
        for index, (label, value) in enumerate(options):
            combo.setItemText(index, translations.get(value, label))

    def _hf_filter_values(self) -> dict[str, str]:
        page_start, page_end = self._hf_page_range_values()
        return {
            "task": str(self.hf_task_combo.currentData() or ""),
            "library": str(self.hf_library_combo.currentData() or ""),
            "app": str(self.hf_app_combo.currentData() or ""),
            "language": str(self.hf_language_combo.currentData() or ""),
            "license": str(self.hf_license_combo.currentData() or ""),
            "provider": str(self.hf_provider_combo.currentData() or ""),
            "other": str(self.hf_other_combo.currentData() or ""),
            "page_start": str(page_start),
            "page_end": str(page_end),
            "keyword": self.hf_keyword_input.text().strip(),
        }

    def _hf_page_range_values(self) -> tuple[int, int]:
        start = int(self.hf_page_start_spin.value())
        end = int(self.hf_page_end_spin.value())
        if start > end:
            start, end = end, start
        return start, end

    def _load_settings(self) -> dict:
        try:
            if self.settings_path.exists():
                return json.loads(self.settings_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return {}

    def _save_settings(self) -> None:
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        hf_page_start, hf_page_end = self._hf_page_range_values()
        payload = {
            "language": self.language,
            "endpoint": self.endpoint_input.text().strip(),
            "storage_path": self.storage_path_input.text().strip(),
            "ollama_models_path": self.ollama_models_path_input.text().strip(),
            "default_mode": self.default_mode_combo.currentText(),
            "parallel": self.parallel_spin.value(),
            "benchmark_warmup_runs": self.warmup_spin.value(),
            "benchmark_repeat_count": self.repeat_spin.value(),
            "hf_model_metadata": self.installed_hf_metadata,
            "param_min_index": self.parameter_slider.lowValue(),
            "param_max_index": self.parameter_slider.highValue(),
            "hf_task": self.hf_task_combo.currentData(),
            "hf_library": self.hf_library_combo.currentData(),
            "hf_app": self.hf_app_combo.currentData(),
            "hf_language": self.hf_language_combo.currentData(),
            "hf_license": self.hf_license_combo.currentData(),
            "hf_provider": self.hf_provider_combo.currentData(),
            "hf_other": self.hf_other_combo.currentData(),
            "hf_page_start": hf_page_start,
            "hf_page_end": hf_page_end,
            "hf_keyword": self.hf_keyword_input.text().strip(),
            "show_deleted_results": self.show_deleted_checkbox.isChecked(),
            "recommendation_cache": self._recommendation_cache_payload(),
            "excluded_recommendations": self.excluded_recommendations[:100],
        }
        self.settings_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _percent(self, resources: dict, key: str) -> str:
        return f"{resources.get(key, 0)}%"

    def _gb_delta(self, resources: dict, key: str) -> str:
        return f"{resources.get(f'{key}_delta', 0)}GB"

    def _short(self, value: str, limit: int) -> str:
        return value if len(value) <= limit else value[: max(0, limit - 3)] + "..."

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background: #0b1220; color: #e8f1ff; font-size: 13px; font-family: 'Malgun Gothic', 'Segoe UI'; }
            QTabWidget::pane { border: 1px solid #253650; background: #0b1220; border-radius: 10px; }
            QTabBar::tab {
                background: #152137; color: #9fb3ce; border: 1px solid #253650; padding: 11px 18px;
                margin-right: 4px; border-top-left-radius: 9px; border-top-right-radius: 9px;
            }
            QTabBar::tab:hover { background: #1d304d; color: #e8f1ff; }
            QTabBar::tab:selected { background: #203a59; color: #74e2d2; font-weight: 700; border-bottom: 2px solid #29c4b3; }
            QPushButton {
                background: #1769b7; color: #ffffff; border: 1px solid #3386cf; border-radius: 8px;
                padding: 9px 14px; font-weight: 650;
            }
            QPushButton:hover { background: #2387db; }
            QPushButton:pressed { background: #0f4d89; }
            QPushButton:disabled { background: #27354a; color: #798aa3; border-color: #33455d; }
            QPushButton#secondaryAction { background: #183344; color: #90f0e2; border-color: #2c6870; }
            QPushButton#secondaryAction:hover { background: #245461; }
            QPushButton#dangerAction { background: #553042; color: #ffd8df; border-color: #a45b70; }
            QPushButton#dangerAction:hover { background: #783d53; }
            QLineEdit, QComboBox, QSpinBox, QTextEdit {
                background: #111e31; color: #e8f1ff; border: 1px solid #344961; border-radius: 8px; padding: 8px;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QTextEdit:focus { border-color: #29c4b3; }
            QComboBox QAbstractItemView { background: #17273d; color: #e8f1ff; selection-background-color: #21678d; }
            QTableWidget {
                background: #111d2f; alternate-background-color: #17263a; color: #e5efff;
                gridline-color: #2a3b52; border: 1px solid #2b405b; border-radius: 9px;
            }
            QTableWidget::item:selected { background: #1d6687; color: #ffffff; }
            QTableWidget::item:hover { background: #244363; }
            QHeaderView::section {
                background: #1c344e; color: #bfeadd; padding: 8px; border: 0; border-right: 1px solid #2b4961; font-weight: 700;
            }
            QProgressBar {
                background: #15253a; color: #e8f1ff; border: 1px solid #36506a;
                border-radius: 7px; min-height: 16px; max-height: 18px; text-align: center;
                font-size: 11px; font-weight: 600;
            }
            QProgressBar::chunk {
                background: #29c4b3; border-radius: 6px;
            }
            QProgressBar#downloadProgress::chunk {
                background: #39b77a;
            }
            QProgressBar#processProgress::chunk {
                background: #32a4d6;
            }
            QFrame#scoreCard, QFrame#panel {
                background: #14243a; border: 1px solid #2b4861; border-radius: 11px;
            }
            QLabel#cardTitle {
                background: transparent; color: #a7bed6; font-size: 12px; font-weight: 600;
            }
            QLabel#cardValue {
                background: transparent; color: #75ecdb; font-size: 17px; font-weight: 700;
            }
            QLabel#heroLabel {
                background: #142941; border-left: 4px solid #2bd5be; border-radius: 9px;
                padding: 14px; font-weight: 700; color: #ddf8f2;
            }
            QLabel#sectionTitle {
                background: transparent; color: #e9f4ff; font-size: 15px; font-weight: 700; padding-top: 7px;
            }
            QLabel#processState {
                background: transparent; color: #70d8ee; font-weight: 700;
            }
            QLabel#processSteps {
                background: #182e45; border: 1px solid #35516b; border-radius: 8px;
                color: #d2e8fa; padding: 9px; font-weight: 600;
            }
            QLabel#processPlan {
                background: transparent; color: #a7bed6; font-size: 12px;
            }
            QLabel#connectionStatus {
                background: #102637; border: 1px solid #2c5f66; border-radius: 8px;
                color: #aee9e0; padding: 10px; min-height: 18px;
            }
            """
        )


def run_app() -> int:
    app = QApplication([])
    window = MainWindow()
    window.show()
    return app.exec()
