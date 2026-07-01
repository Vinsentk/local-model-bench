from local_model_bench.ollama_logs import OllamaLogImporter


def test_ollama_log_importer_extracts_generate_events(tmp_path) -> None:
    log_path = tmp_path / "server-1.log"
    log_path.write_text(
        "\n".join(
            [
                'time=2026-06-20T04:47:28.961+09:00 level=INFO source=images.go:354 msg="template selection" model=hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:latest selected=qwen3',
                '[GIN] 2026/06/20 - 04:47:30 | 200 |   53.7865125s |       127.0.0.1 | POST     "/api/generate"',
                '[GIN] 2026/06/20 - 04:48:10 | 499 |    1m10.25s |       127.0.0.1 | POST     "/api/chat"',
            ]
        ),
        encoding="utf-8",
    )

    events = OllamaLogImporter(tmp_path).parse_paths([log_path])

    assert len(events) == 2
    assert events[0].model_name == "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:latest"
    assert events[0].operation == "ollama:generate"
    assert events[0].status == "success"
    assert events[0].duration_seconds == 53.7865
    assert events[0].details["source_evidence"] == "ollama_server_log_only"
    assert events[1].operation == "ollama:chat"
    assert events[1].status == "failure"
    assert events[1].failure_type == "connection error"
    assert events[1].duration_seconds == 70.25


def test_ollama_log_importer_limits_to_latest_events(tmp_path) -> None:
    log_path = tmp_path / "server-1.log"
    log_path.write_text(
        "\n".join(
            [
                'time=2026-06-20T04:47:28.961+09:00 level=INFO source=images.go:354 msg="template selection" model=model-a selected=qwen3',
                '[GIN] 2026/06/20 - 04:47:30 | 200 | 1s | 127.0.0.1 | POST     "/api/generate"',
                '[GIN] 2026/06/20 - 04:47:31 | 200 | 2s | 127.0.0.1 | POST     "/api/generate"',
            ]
        ),
        encoding="utf-8",
    )

    events = OllamaLogImporter(tmp_path).parse_paths([log_path], max_events=1)

    assert len(events) == 1
    assert events[0].duration_seconds == 2.0
