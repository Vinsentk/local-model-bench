from local_model_bench.benchmark import BenchmarkRunner
from local_model_bench.models import GenerationResult, HardwareInfo, OllamaModel


def runner() -> BenchmarkRunner:
    return BenchmarkRunner(client=None, hardware=HardwareInfo("cpu", 1, 1, 16, "gpu", 8, "C:\\", 10))  # type: ignore[arg-type]


def test_json_checker_accepts_valid_json() -> None:
    passed, score = runner()._check_json('{"status":"ok","count":3}')
    assert passed
    assert score == 1.0


def test_korean_checker_requires_hangul() -> None:
    passed, score = runner()._check_korean("로컬 모델은 비공개 오프라인 초안 작성에 유용하다.")
    assert passed
    assert score >= 0.75


def test_reasoning_checker() -> None:
    passed, score = runner()._check_reasoning("6")
    assert passed
    assert score == 1.0


def test_reasoning_checker_ignores_thinking_block() -> None:
    passed, score = runner()._check_reasoning("<think>3 plus 2 plus 1</think>6")
    assert passed
    assert score == 1.0


def test_large_or_thinking_models_get_longer_timeout() -> None:
    bench = runner()
    small = OllamaModel("qwen2.5:9b")
    large = OllamaModel("qwen3:27b")
    thinking = OllamaModel("reasoning-model:14b")

    assert bench._timeout_seconds(small, "standard") == 180
    assert bench._timeout_seconds(large, "standard") == 600
    assert bench._timeout_seconds(thinking, "smoke") == 240
    assert bench._timeout_seconds(thinking, "thinking") == 900


def test_benchmark_status_classifies_generation_issues() -> None:
    bench = runner()
    assert bench._status_for_issue("timed out") == "TIMEOUT"
    assert bench._status_for_issue("done_reason=length") == "LENGTH"
    assert bench._status_for_issue("empty response") == "EMPTY"
    assert bench._status_for_issue("connection refused") == "RUNTIME"


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, model: str, prompt: str, **kwargs) -> GenerationResult:
        self.calls.append({"model": model, "prompt": prompt, **kwargs})
        return GenerationResult(
            text="OK",
            elapsed_seconds=0.5,
            first_token_seconds=0.1,
            eval_count=2,
            eval_duration_ns=100_000_000,
            prompt_eval_count=4,
            prompt_eval_duration_ns=200_000_000,
            load_duration_ns=300_000_000,
            raw={},
        )


def test_benchmark_warmup_and_repeats_are_recorded() -> None:
    client = FakeClient()
    bench = BenchmarkRunner(client=client, hardware=HardwareInfo("cpu", 1, 1, 16, "gpu", 8, "C:\\", 10))

    report = bench.run(OllamaModel("tiny:1b"), mode="smoke", warmup_runs=1, repeat_count=2)

    assert len(client.calls) == 3
    assert len(report.cases) == 2
    assert report.run_settings["warmup_runs"] == 1
    assert report.run_settings["repeat_count"] == 2
    assert {call["keep_alive"] for call in client.calls} == {"2m"}
    assert report.cases[0].prompt_tokens_per_second == 20.0
    assert report.cases[0].load_seconds == 0.3
