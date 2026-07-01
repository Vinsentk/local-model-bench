from local_model_bench.opencode_logs import OpenCodeLogImporter


def test_opencode_log_importer_extracts_llm_failure(tmp_path) -> None:
    log_path = tmp_path / "opencode.log"
    log_path.write_text(
        '2026-05-01T13:22:04.041483Z  INFO sidecar: opencode_lib::cli: ERROR 2026-05-01T13:22:04 +46874ms service=llm providerID=ollama modelID=qwen3.5:9b session.id=ses_123 small=false agent=thinker mode=primary error={"error":{"message":"model does not support tools"}}',
        encoding="utf-8",
    )

    events = OpenCodeLogImporter([tmp_path]).parse_paths([log_path])

    assert len(events) == 1
    assert events[0].source == "opencode"
    assert events[0].model_name == "ollama/qwen3.5:9b"
    assert events[0].operation == "opencode:thinker"
    assert events[0].status == "failure"
    assert events[0].failure_type == "tools unsupported"
    assert events[0].duration_seconds == 46.874
    assert events[0].details["session_id"] == "ses_123"


def test_opencode_log_importer_extracts_usage_tokens(tmp_path) -> None:
    log_path = tmp_path / "opencode.log"
    log_path.write_text(
        '2026-05-01T13:22:04.041483Z  INFO sidecar: opencode_lib::cli: service=llm providerID=ollama modelID=model-a session.id=ses_123 agent=build mode=primary usage={"prompt_tokens":11,"completion_tokens":22,"total_tokens":33}',
        encoding="utf-8",
    )

    events = OpenCodeLogImporter([tmp_path]).parse_paths([log_path])

    assert len(events) == 1
    assert events[0].status == "success"
    assert events[0].prompt_tokens == 11
    assert events[0].eval_tokens == 22
    assert events[0].details["token_counts"] == "available"
