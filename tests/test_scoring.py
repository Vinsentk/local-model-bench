from local_model_bench.models import BenchmarkCaseResult, HardwareInfo, OllamaModel
from local_model_bench.scoring import ScoreEngine


def test_score_maps_recommended_uses() -> None:
    hardware = HardwareInfo("cpu", 16, 32, 61.6, "gpu", 16.0, "E:\\", 1000)
    model = OllamaModel("qwen:test", size_bytes=6 * 1024**3)
    cases = [
        case("smoke", 40.0, "OK"),
        case("json", 30.0, "{}"),
        case("code", 25.0, "def"),
        case("review", 25.0, "zero"),
        case("korean", 25.0, "로컬 모델"),
        case("reasoning", 25.0, "6"),
    ]
    score = ScoreEngine().score(cases, model, hardware)
    assert score.total > 80
    assert "coding_assistant" in score.recommended_uses or "fast_draft" in score.recommended_uses


def case(name: str, tps: float, output: str) -> BenchmarkCaseResult:
    return BenchmarkCaseResult(
        name=name,
        passed=True,
        score=1.0,
        elapsed_seconds=1.0,
        first_token_seconds=0.2,
        tokens_per_second=tps,
        prompt_tokens_per_second=100.0,
        load_seconds=0.1,
        output=output,
    )
