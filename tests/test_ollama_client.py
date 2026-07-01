from local_model_bench.ollama_client import OllamaClient


def test_pull_status_formats_percentage_and_size() -> None:
    line = OllamaClient._format_pull_status(
        {
            "status": "pulling manifest layer",
            "digest": "sha256:abcdef1234567890",
            "completed": 512 * 1024 * 1024,
            "total": 1024 * 1024 * 1024,
        }
    )

    assert "50%" in line
    assert "512.0 MB" in line
    assert "1.0 GB" in line
    assert "sha256:abcdef12345" in line


def test_pull_status_handles_non_progress_status() -> None:
    assert OllamaClient._format_pull_status({"status": "verifying sha256 digest"}) == "verifying sha256 digest"


def test_ps_falls_back_to_ollama_cli_text(monkeypatch) -> None:
    class Result:
        returncode = 0
        stdout = (
            "NAME                                      ID              SIZE      PROCESSOR    CONTEXT    UNTIL\n"
            "hf.co/example/model:latest               abcdef123456    7.4 GB    100% GPU     4096       4 minutes from now\n"
        )

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: Result())

    client = OllamaClient()
    client._get_json = lambda path: {"models": []}  # type: ignore[method-assign]

    rows = client.ps()

    assert rows == [
        {
            "model": "hf.co/example/model:latest",
            "name": "hf.co/example/model:latest",
            "digest": "abcdef123456",
            "size_label": "7.4 GB",
            "processor": "100% GPU",
            "context": "4096",
            "expires_at": "4 minutes from now",
            "source": "ollama ps",
        }
    ]
