from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import replace
from statistics import mean

from .models import BenchmarkCaseResult, BenchmarkReport, HardwareInfo, OllamaModel, utc_now_iso
from .ollama_client import OllamaClient, OllamaError
from .resources import ResourceMonitor
from .scoring import ScoreEngine


ProgressCallback = Callable[[str, str], None]
UsageCallback = Callable[[dict], None]


class BenchmarkRunner:
    def __init__(self, client: OllamaClient, hardware: HardwareInfo) -> None:
        self.client = client
        self.hardware = hardware
        self.scoring = ScoreEngine()

    def run(
        self,
        model: OllamaModel,
        *,
        mode: str = "standard",
        warmup_runs: int = 0,
        repeat_count: int = 1,
        on_progress: ProgressCallback | None = None,
        on_usage: UsageCallback | None = None,
    ) -> BenchmarkReport:
        started = utc_now_iso()
        cases: list[BenchmarkCaseResult] = []
        error = ""
        status = "OK"
        monitor = ResourceMonitor()
        timeout_seconds = self._timeout_seconds(model, mode)
        warmup_runs = max(0, min(5, int(warmup_runs or 0)))
        repeat_count = max(1, min(5, int(repeat_count or 1)))
        keep_alive = "2m" if warmup_runs or repeat_count > 1 else "0s"
        if warmup_runs:
            self._warmup_model(model, warmup_runs, timeout_seconds, keep_alive, on_progress, on_usage)
        monitor.start()
        try:
            for name, prompt, checker, tokens in self._cases(mode):
                for repeat_index in range(repeat_count):
                    repeat_label = f" r{repeat_index + 1}/{repeat_count}" if repeat_count > 1 else ""
                    if on_progress:
                        on_progress(model.name, f"{name}{repeat_label} ({timeout_seconds}s)")
                    try:
                        result = self.client.generate(
                            model.name,
                            prompt,
                            num_predict=tokens,
                            num_ctx=4096 if mode == "thinking" else 2048,
                            temperature=0,
                            keep_alive=keep_alive,
                            timeout_seconds=timeout_seconds,
                        )
                        case_error = self._response_issue(result.text, result.raw)
                        passed, score = checker(result.text)
                        if case_error:
                            passed = False
                            score = 0
                            status = self._status_for_issue(case_error)
                            error = case_error
                        if on_usage:
                            on_usage(
                                {
                                    "model_name": model.name,
                                    "operation": f"benchmark:{name}",
                                    "status": "success" if passed else "failure",
                                    "failure_type": "" if passed else self._failure_type(case_error or "unknown"),
                                    "prompt_tokens": result.prompt_eval_count,
                                    "eval_tokens": result.eval_count,
                                    "duration_seconds": round(result.elapsed_seconds, 3),
                                    "details": {
                                        "mode": mode,
                                        "case": name,
                                        "score": score,
                                        "repeat": repeat_index + 1,
                                        "repeat_count": repeat_count,
                                    },
                                }
                            )
                        cases.append(
                            BenchmarkCaseResult(
                                name=name,
                                passed=passed,
                                score=score,
                                elapsed_seconds=result.elapsed_seconds,
                                first_token_seconds=result.first_token_seconds,
                                tokens_per_second=result.tokens_per_second,
                                prompt_tokens_per_second=result.prompt_tokens_per_second,
                                load_seconds=result.load_seconds,
                                output=result.text.strip()[:2000],
                                error=case_error,
                            )
                        )
                        if case_error:
                            break
                    except OllamaError as exc:
                        error = str(exc)
                        status = self._status_for_issue(error)
                        if on_usage:
                            on_usage(
                                {
                                    "model_name": model.name,
                                    "operation": f"benchmark:{name}",
                                    "status": "failure",
                                    "failure_type": self._failure_type(error),
                                    "prompt_tokens": None,
                                    "eval_tokens": None,
                                    "duration_seconds": None,
                                    "details": {
                                        "mode": mode,
                                        "case": name,
                                        "repeat": repeat_index + 1,
                                        "repeat_count": repeat_count,
                                        "error": error[:500],
                                    },
                                }
                            )
                        cases.append(
                            BenchmarkCaseResult(
                                name=name,
                                passed=False,
                                score=0,
                                elapsed_seconds=0,
                                first_token_seconds=0,
                                tokens_per_second=0,
                                prompt_tokens_per_second=0,
                                load_seconds=0,
                                output="",
                                error=str(exc),
                            )
                        )
                        break
                if error:
                    break
        finally:
            resource_summary = monitor.stop()

        if cases and any(not case.passed for case in cases) and status == "OK":
            status = "PARTIAL"
        measured = [case.tokens_per_second for case in cases if case.tokens_per_second > 0]
        avg_tps = mean(measured) if measured else 0.0
        score = self.scoring.score(cases, model, self.hardware, error)
        if mode == "smoke" or status != "OK":
            score = replace(score, recommended_uses=["needs_more_tests"])
        return BenchmarkReport(
            model_name=model.name,
            source=model.source,
            size_label=model.size_label,
            quant=model.quant,
            status=status,
            cases=cases,
            score=score,
            avg_tokens_per_second=round(avg_tps, 2),
            started_at=started,
            finished_at=utc_now_iso(),
            error=error,
            resource_summary=resource_summary,
            run_settings={
                "mode": mode,
                "tps_source": "ollama_eval",
                "warmup_runs": warmup_runs,
                "repeat_count": repeat_count,
                "timeout_seconds": timeout_seconds,
                "keep_alive": keep_alive,
            },
        )

    def _warmup_model(
        self,
        model: OllamaModel,
        warmup_runs: int,
        timeout_seconds: int,
        keep_alive: str,
        on_progress: ProgressCallback | None,
        on_usage: UsageCallback | None,
    ) -> None:
        for index in range(warmup_runs):
            if on_progress:
                on_progress(model.name, f"warmup {index + 1}/{warmup_runs} ({timeout_seconds}s)")
            try:
                result = self.client.generate(
                    model.name,
                    "Reply with exactly: OK",
                    num_predict=8,
                    num_ctx=1024,
                    temperature=0,
                    keep_alive=keep_alive,
                    timeout_seconds=timeout_seconds,
                )
                issue = self._response_issue(result.text, result.raw)
                if on_usage:
                    on_usage(
                        {
                            "model_name": model.name,
                            "operation": "benchmark:warmup",
                            "status": "success" if not issue else "failure",
                            "failure_type": "" if not issue else self._failure_type(issue),
                            "prompt_tokens": result.prompt_eval_count,
                            "eval_tokens": result.eval_count,
                            "duration_seconds": round(result.elapsed_seconds, 3),
                            "details": {"warmup": index + 1, "warmup_runs": warmup_runs, "error": issue},
                        }
                    )
            except OllamaError as exc:
                if on_usage:
                    on_usage(
                        {
                            "model_name": model.name,
                            "operation": "benchmark:warmup",
                            "status": "failure",
                            "failure_type": self._failure_type(str(exc)),
                            "prompt_tokens": None,
                            "eval_tokens": None,
                            "duration_seconds": None,
                            "details": {"warmup": index + 1, "warmup_runs": warmup_runs, "error": str(exc)[:500]},
                        }
                    )

    def _timeout_seconds(self, model: OllamaModel, mode: str) -> int:
        if mode == "smoke":
            base = 90
            long_timeout = 240
        elif mode == "thinking":
            base = 360
            long_timeout = 900
        else:
            base = 180
            long_timeout = 600
        return long_timeout if self._needs_long_timeout(model) else base

    def _needs_long_timeout(self, model: OllamaModel) -> bool:
        name = model.name.lower()
        if any(marker in name for marker in ["thinking", "reasoning", "reasoner", "deepseek-r1", "qwq", "qwen3"]):
            return True
        size_b = self._param_size_b(model.name)
        return size_b >= 27

    def _param_size_b(self, text: str) -> float:
        match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", text, flags=re.IGNORECASE)
        return float(match.group(1)) if match else 0.0

    def _is_timeout_error(self, message: str) -> bool:
        lowered = message.lower()
        return "timed out" in lowered or "timeout" in lowered

    def _cases(self, mode: str):
        smoke = [
            ("smoke", "Reply with exactly: OK", self._check_ok, 8),
        ]
        if mode == "smoke":
            return smoke
        standard = smoke + [
            (
                "json",
                'Return only valid JSON with keys "status" and "count"; status must be "ok" and count must be 3.',
                self._check_json,
                64,
            ),
            (
                "code",
                "Write a Python function add_even_numbers(values) that returns the sum of even integers in values. Return code only.",
                self._check_code,
                160,
            ),
            (
                "review",
                "Review this code and name one bug: def div(a,b): return a / b. Keep answer under 40 words.",
                self._check_review,
                96,
            ),
            (
                "korean",
                "다음 문장을 한국어 한 문장으로 요약해라: Local models are useful for private, offline draft generation and fast iteration.",
                self._check_korean,
                96,
            ),
            (
                "reasoning",
                "A box has 3 red balls and 2 blue balls. If one red ball is added, how many balls are in the box? Answer with the number only.",
                self._check_reasoning,
                32,
            ),
        ]
        if mode == "thinking":
            return standard + [
                (
                    "thinking",
                    "Use brief reasoning internally, then answer only the final number: A laptop costs 1200. It is discounted 15%, then tax adds 8%. What is the final price rounded to the nearest dollar?",
                    self._check_thinking_math,
                    192,
                )
            ]
        return standard

    def _check_ok(self, text: str) -> tuple[bool, float]:
        text = self._strip_thinking(text)
        normalized = text.strip().upper()
        passed = "OK" in normalized and len(normalized) <= 40
        return passed, 1.0 if passed else 0.3 if "OK" in normalized else 0.0

    def _check_json(self, text: str) -> tuple[bool, float]:
        text = self._strip_thinking(text)
        candidate = text.strip()
        try:
            if "```" in candidate:
                candidate = candidate.split("```")[1].replace("json", "", 1).strip()
            data = json.loads(candidate)
            passed = data.get("status") == "ok" and int(data.get("count", -1)) == 3
            return passed, 1.0 if passed else 0.5
        except Exception:
            return False, 0.0

    def _check_code(self, text: str) -> tuple[bool, float]:
        text = self._strip_thinking(text)
        lowered = text.lower()
        score = 0.0
        score += 0.25 if "def add_even_numbers" in lowered else 0
        score += 0.25 if "return" in lowered else 0
        score += 0.25 if "% 2" in lowered or "mod" in lowered or "even" in lowered else 0
        score += 0.25 if "sum(" in lowered or "+=" in lowered else 0
        return score >= 0.75, score

    def _check_review(self, text: str) -> tuple[bool, float]:
        text = self._strip_thinking(text)
        lowered = text.lower()
        score = 0.0
        score += 0.5 if "zero" in lowered or "0" in lowered else 0
        score += 0.3 if "division" in lowered or "divide" in lowered else 0
        score += 0.2 if len(text.split()) <= 60 else 0
        return score >= 0.7, score

    def _check_korean(self, text: str) -> tuple[bool, float]:
        text = self._strip_thinking(text)
        has_hangul = any("\uac00" <= char <= "\ud7a3" for char in text)
        concise = len(text.strip()) <= 180
        keyword = any(term in text for term in ["로컬", "모델", "비공개", "초안", "오프라인", "빠른"])
        score = (0.5 if has_hangul else 0) + (0.25 if concise else 0) + (0.25 if keyword else 0)
        return score >= 0.75, score

    def _check_reasoning(self, text: str) -> tuple[bool, float]:
        text = self._strip_thinking(text)
        cleaned = "".join(ch for ch in text if ch.isdigit())
        passed = cleaned.startswith("6")
        return passed, 1.0 if passed else 0.0

    def _check_thinking_math(self, text: str) -> tuple[bool, float]:
        text = self._strip_thinking(text)
        digits = "".join(ch for ch in text if ch.isdigit())
        passed = digits.startswith("1102")
        return passed, 1.0 if passed else 0.0

    def _response_issue(self, text: str, raw: dict) -> str:
        if not text.strip():
            return "empty response"
        reason = str(raw.get("done_reason", "")).lower()
        if reason == "length":
            return "done_reason=length"
        return ""

    def _status_for_issue(self, message: str) -> str:
        lowered = message.lower()
        if self._is_timeout_error(lowered):
            return "TIMEOUT"
        if "connection" in lowered or "refused" in lowered:
            return "RUNTIME"
        if "done_reason=length" in lowered or "length" in lowered:
            return "LENGTH"
        if "empty response" in lowered:
            return "EMPTY"
        return "RUNTIME"

    def _failure_type(self, message: str) -> str:
        lowered = message.lower()
        if self._is_timeout_error(lowered):
            return "timeout"
        if "done_reason=length" in lowered or "length" in lowered:
            return "length"
        if "empty response" in lowered:
            return "empty response"
        if "connection" in lowered or "refused" in lowered:
            return "connection error"
        if "runtime" in lowered or "ollama" in lowered:
            return "runtime error"
        return "unknown"

    def _strip_thinking(self, text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
