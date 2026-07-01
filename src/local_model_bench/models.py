from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class HardwareInfo:
    cpu_name: str
    cpu_cores: int
    cpu_threads: int
    ram_gb: float
    gpu_name: str
    vram_gb: float
    preferred_drive: str
    preferred_drive_free_gb: float


@dataclass(frozen=True)
class OllamaModel:
    name: str
    model_id: str = ""
    size_bytes: int = 0
    size_label: str = ""
    modified_at: str = ""
    quant: str = ""
    source: str = "Installed"
    thinking_support: str = "unknown"
    thinking_evidence: str = ""


@dataclass(frozen=True)
class GenerationResult:
    text: str
    elapsed_seconds: float
    first_token_seconds: float = 0.0
    eval_count: int = 0
    eval_duration_ns: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration_ns: int = 0
    load_duration_ns: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        if self.eval_count and self.eval_duration_ns:
            seconds = self.eval_duration_ns / 1_000_000_000
            if seconds > 0:
                return self.eval_count / seconds
        if self.elapsed_seconds > 0:
            return max(0.0, len(self.text.split()) / self.elapsed_seconds)
        return 0.0

    @property
    def prompt_tokens_per_second(self) -> float:
        if self.prompt_eval_count and self.prompt_eval_duration_ns:
            seconds = self.prompt_eval_duration_ns / 1_000_000_000
            if seconds > 0:
                return self.prompt_eval_count / seconds
        return 0.0

    @property
    def load_seconds(self) -> float:
        return self.load_duration_ns / 1_000_000_000 if self.load_duration_ns else 0.0


@dataclass(frozen=True)
class BenchmarkCaseResult:
    name: str
    passed: bool
    score: float
    elapsed_seconds: float
    first_token_seconds: float
    tokens_per_second: float
    prompt_tokens_per_second: float
    load_seconds: float
    output: str
    error: str = ""


@dataclass(frozen=True)
class ScoreBreakdown:
    total: float
    quality: float
    speed: float
    stability: float
    resource_fit: float
    usability: float
    use_case_scores: dict[str, float]
    recommended_uses: list[str]


@dataclass(frozen=True)
class BenchmarkReport:
    model_name: str
    source: str
    size_label: str
    quant: str
    status: str
    cases: list[BenchmarkCaseResult]
    score: ScoreBreakdown
    avg_tokens_per_second: float
    started_at: str
    finished_at: str
    error: str = ""
    resource_summary: dict[str, Any] = field(default_factory=dict)
    run_settings: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRecommendation:
    model_name: str
    repo_id: str
    pull_name: str
    quant: str
    size_label: str
    fit: str
    score: float
    recommended_uses: list[str]
    reason: str
    downloads: int = 0
    likes: int = 0
    updated_at: str = ""
    trending_page: int = 0
    thinking_support: str = "unknown"
    thinking_evidence: str = ""


def utc_now_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()
