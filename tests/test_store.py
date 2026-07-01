from local_model_bench.models import BenchmarkCaseResult, BenchmarkReport, ScoreBreakdown
from local_model_bench.store import ResultStore
from datetime import datetime, timedelta


def make_report(model_name: str, total: float, finished_at: str) -> BenchmarkReport:
    return BenchmarkReport(
        model_name=model_name,
        source="Installed",
        size_label="9B",
        quant="Q4",
        status="OK",
        cases=[
            BenchmarkCaseResult(
                name="smoke",
                passed=True,
                score=1.0,
                elapsed_seconds=1.0,
                first_token_seconds=0.1,
                tokens_per_second=10.0,
                prompt_tokens_per_second=20.0,
                load_seconds=0.2,
                output="OK",
            )
        ],
        score=ScoreBreakdown(total, 10, 10, 10, 10, 10, {}, []),
        avg_tokens_per_second=10.0,
        started_at=finished_at,
        finished_at=finished_at,
    )


def test_latest_report_per_model_returns_one_latest_row(tmp_path) -> None:
    store = ResultStore(tmp_path / "results.sqlite")
    store.save_report(make_report("model-a", 10, "2026-06-20T01:00:00+09:00"))
    store.save_report(make_report("model-a", 90, "2026-06-20T02:00:00+09:00"))
    store.save_report(make_report("model-b", 70, "2026-06-20T01:30:00+09:00"))

    rows = store.latest_report_per_model()

    assert [row["model_name"] for row in rows] == ["model-b", "model-a"]
    assert {row["model_name"]: row["total_score"] for row in rows} == {"model-a": 90, "model-b": 70}


def test_deleted_report_is_kept_but_hidden_from_active_latest(tmp_path) -> None:
    store = ResultStore(tmp_path / "results.sqlite")
    store.save_report(make_report("model-a", 10, "2026-06-20T01:00:00+09:00"))
    store.save_report(make_report("model-a", 90, "2026-06-20T02:00:00+09:00"))
    latest = store.latest_report_per_model()

    store.delete_report(int(latest[0]["result_id"]))

    visible_latest = store.latest_report_per_model()
    active_latest = store.latest_report_per_model(include_deleted=False)

    assert visible_latest[0]["total_score"] == 90
    assert visible_latest[0]["deleted_at"]
    assert active_latest[0]["total_score"] == 10


def test_delete_reports_marks_multiple_results(tmp_path) -> None:
    store = ResultStore(tmp_path / "results.sqlite")
    store.save_report(make_report("model-a", 10, "2026-06-20T01:00:00+09:00"))
    store.save_report(make_report("model-b", 11, "2026-06-20T01:00:00+09:00"))
    rows = store.latest_report_per_model(include_deleted=True)

    store.delete_reports([int(row["result_id"]) for row in rows])

    deleted_rows = store.latest_report_per_model(include_deleted=True)
    assert len(deleted_rows) == 2
    assert all(row["deleted_at"] for row in deleted_rows)
    assert store.latest_report_per_model(include_deleted=False) == []


def test_report_payload_includes_raw_benchmark_metrics(tmp_path) -> None:
    store = ResultStore(tmp_path / "results.sqlite")
    store.save_report(make_report("model-a", 10, "2026-06-20T01:00:00+09:00"))

    row = store.latest_report_per_model()[0]

    assert row["avg_prompt_tokens_per_second"] == 20.0
    assert row["avg_load_seconds"] == 0.2
    assert row["cases"][0]["prompt_tokens_per_second"] == 20.0
    assert row["cases"][0]["load_seconds"] == 0.2


def test_usage_events_are_aggregated_by_model_source_and_failure(tmp_path) -> None:
    store = ResultStore(tmp_path / "results.sqlite")
    store.save_usage_event(
        model_name="model-a",
        source="local-model-bench",
        operation="benchmark:smoke",
        status="success",
        prompt_tokens=5,
        eval_tokens=7,
        duration_seconds=1.2,
    )
    store.save_usage_event(
        model_name="model-a",
        source="local-model-bench",
        operation="benchmark:json",
        status="failure",
        failure_type="timeout",
    )

    by_model = store.usage_by_model(period="all")
    by_source = store.usage_by_source(period="all")
    failures = store.failures_by_type(period="all")
    events = store.usage_events(period="all")

    assert by_model[0]["calls"] == 2
    assert by_model[0]["successes"] == 1
    assert by_model[0]["failures"] == 1
    assert by_model[0]["total_tokens"] == 12
    assert by_source[0]["source"] == "local-model-bench"
    assert failures[0]["failure_type"] == "timeout"
    assert events[0]["operation"] == "benchmark:json"


def test_usage_external_id_prevents_duplicate_imports(tmp_path) -> None:
    store = ResultStore(tmp_path / "results.sqlite")
    first = store.save_usage_event(
        model_name="model-a",
        source="unknown",
        operation="ollama:generate",
        status="success",
        created_at="2026-06-20T05:00:00+09:00",
        external_id="ollama-log:test",
    )
    second = store.save_usage_event(
        model_name="model-a",
        source="unknown",
        operation="ollama:generate",
        status="success",
        created_at="2026-06-20T05:00:00+09:00",
        external_id="ollama-log:test",
    )

    events = store.usage_events(period="all")

    assert first is True
    assert second is False
    assert len(events) == 1
    assert events[0]["external_id"] == "ollama-log:test"


def test_usage_overview_separates_known_and_unknown_tokens(tmp_path) -> None:
    store = ResultStore(tmp_path / "results.sqlite")
    store.save_usage_event(
        model_name="model-a",
        source="local-model-bench",
        operation="benchmark:smoke",
        status="success",
        prompt_tokens=10,
        eval_tokens=20,
        duration_seconds=3.0,
    )
    store.save_usage_event(
        model_name="model-a",
        source="unknown",
        operation="ollama:generate",
        status="success",
        duration_seconds=5.0,
        details={
            "client_ip": "127.0.0.1",
            "source_evidence": "ollama_server_log_only",
            "token_counts": "unavailable_in_ollama_server_log",
        },
    )

    rows = store.usage_overview(period="all")
    direct = next(row for row in rows if row["source"] == "local-model-bench")
    imported = next(row for row in rows if row["source"] == "unknown")

    assert direct["total_tokens"] == 30
    assert direct["known_token_calls"] == 1
    assert direct["unknown_token_calls"] == 0
    assert direct["token_coverage"] == 100.0
    assert imported["source_detail"] == "Ollama localhost"
    assert imported["known_token_calls"] == 0
    assert imported["unknown_token_calls"] == 1
    assert imported["token_coverage"] == 0.0


def test_usage_token_totals_by_model_respects_recent_periods(tmp_path) -> None:
    store = ResultStore(tmp_path / "results.sqlite")
    now = datetime.now().astimezone().replace(microsecond=0)
    store.save_usage_event(
        model_name="model-a",
        source="local-model-bench",
        operation="benchmark:smoke",
        status="success",
        prompt_tokens=10,
        eval_tokens=20,
        created_at=(now - timedelta(hours=2)).isoformat(),
    )
    store.save_usage_event(
        model_name="model-a",
        source="opencode",
        operation="chat",
        status="success",
        prompt_tokens=30,
        eval_tokens=40,
        created_at=(now - timedelta(days=3)).isoformat(),
    )
    store.save_usage_event(
        model_name="model-a",
        source="opencode",
        operation="chat",
        status="success",
        prompt_tokens=100,
        eval_tokens=200,
        created_at=(now - timedelta(days=8)).isoformat(),
    )

    assert store.usage_token_totals_by_model(period="1d")["model-a"] == 30
    assert store.usage_token_totals_by_model(period="7d")["model-a"] == 100
