from __future__ import annotations

from .i18n import USE_CASES
from .models import BenchmarkCaseResult, HardwareInfo, OllamaModel, ScoreBreakdown


class ScoreEngine:
    def score(
        self,
        cases: list[BenchmarkCaseResult],
        model: OllamaModel,
        hardware: HardwareInfo,
        error: str = "",
    ) -> ScoreBreakdown:
        if not cases:
            return ScoreBreakdown(0, 0, 0, 0, 0, 0, {key: 0 for key in USE_CASES}, [])

        pass_rate = sum(1 for case in cases if case.passed) / len(cases)
        quality = min(35.0, sum(case.score for case in cases) / len(cases) * 35.0)

        avg_tps = sum(case.tokens_per_second for case in cases) / len(cases)
        first_token_values = [case.first_token_seconds for case in cases if case.first_token_seconds > 0]
        avg_first_token = sum(first_token_values) / len(first_token_values) if first_token_values else 10.0
        throughput_points = min(17.0, (avg_tps / 35.0) * 17.0)
        latency_points = max(0.0, min(8.0, 8.0 - avg_first_token))
        speed = min(25.0, throughput_points + latency_points)

        failures = sum(1 for case in cases if not case.passed or case.error)
        stability = max(0.0, 15.0 - failures * 3.0)
        if error:
            stability = min(stability, 5.0)

        resource_fit = self._resource_fit(model, hardware)
        usability = 10.0 if pass_rate >= 0.8 else 6.0 if pass_rate >= 0.5 else 2.0
        total = min(100.0, quality + speed + stability + resource_fit + usability)

        use_case_scores = self._use_case_scores(cases, avg_tps, quality, resource_fit, stability)
        recommended_uses = [
            name for name, value in sorted(use_case_scores.items(), key=lambda item: item[1], reverse=True) if value >= 65
        ][:3]
        if not recommended_uses:
            recommended_uses = ["needs_more_tests"]

        return ScoreBreakdown(
            total=round(total, 1),
            quality=round(quality, 1),
            speed=round(speed, 1),
            stability=round(stability, 1),
            resource_fit=round(resource_fit, 1),
            usability=round(usability, 1),
            use_case_scores={key: round(value, 1) for key, value in use_case_scores.items()},
            recommended_uses=recommended_uses,
        )

    def _resource_fit(self, model: OllamaModel, hardware: HardwareInfo) -> float:
        if model.size_bytes <= 0 or hardware.vram_gb <= 0:
            return 9.0
        model_gb = model.size_bytes / (1024**3)
        if model_gb <= hardware.vram_gb * 0.45:
            return 15.0
        if model_gb <= hardware.vram_gb * 0.75:
            return 12.0
        if model_gb <= hardware.vram_gb * 0.95:
            return 8.0
        if model_gb <= hardware.ram_gb * 0.45:
            return 5.0
        return 2.0

    def _use_case_scores(
        self,
        cases: list[BenchmarkCaseResult],
        avg_tps: float,
        quality: float,
        resource_fit: float,
        stability: float,
    ) -> dict[str, float]:
        by_name = {case.name: case for case in cases}
        code = by_name.get("code")
        review = by_name.get("review")
        korean = by_name.get("korean")
        reasoning = by_name.get("reasoning")
        json_case = by_name.get("json")

        return {
            "fast_draft": min(100.0, avg_tps * 2.5 + stability * 3),
            "coding_assistant": self._case_value(code) * 55 + self._case_value(json_case) * 20 + stability * 1.5,
            "code_review": self._case_value(review) * 65 + stability * 2,
            "long_analysis": self._case_value(reasoning) * 50 + quality * 1.2 + resource_fit,
            "korean_summary": self._case_value(korean) * 70 + stability * 2,
            "low_resource_fast": min(100.0, resource_fit * 4 + avg_tps * 2),
            "high_quality_slow": min(100.0, quality * 2 + self._case_value(reasoning) * 25),
        }

    def _case_value(self, case: BenchmarkCaseResult | None) -> float:
        if case is None:
            return 0.0
        return max(0.0, min(1.0, case.score))
