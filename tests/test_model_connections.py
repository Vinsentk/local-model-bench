from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_model_bench import model_connections as links
from local_model_bench.models import GenerationResult, OllamaModel
from local_model_bench.ollama_client import OllamaClient


class FakeOllama:
    def __init__(self) -> None:
        self.names = {"existing:latest", "new:latest"}
        self.fail_delete = False
        self.loaded = set()

    def list_models(self):
        return [OllamaModel(name) for name in sorted(self.names)]

    def show(self, _name):
        return {"capabilities": ["completion", "tools"], "parameters": "num_ctx 16384"}

    def delete(self, name):
        if self.fail_delete:
            raise RuntimeError("Ollama refused deletion")
        self.names.remove(name)

    def ps(self):
        return [{"name": name} for name in self.loaded]


@pytest.fixture
def configured_bridge(tmp_path, monkeypatch):
    root = tmp_path / ".opencodex"
    desktop = root / "desktop"
    live = root / "live/home"
    desktop.mkdir(parents=True)
    live.mkdir(parents=True)
    catalog = desktop / "opencodex-catalog.json"
    template = {"slug": "ollama/existing:latest", "display_name": "existing", "context_window": 8192,
                "base_instructions": "You are a coding agent powered by the ollama/existing:latest. If asked which model you are, identify as ollama/existing:latest. Generic rules.",
                "opencodex_capability_provenance": {"provider": "ollama", "model_id": "existing:latest"}}
    (desktop / "opencodex-catalog.json").write_text(json.dumps({"models": [template]}), encoding="utf-8")
    (desktop / "allowed-local-models.json").write_text(json.dumps(["existing:latest"]), encoding="utf-8")
    (desktop / "ollama-capabilities.json").write_text(json.dumps([{"name": "existing:latest", "capabilities": ["completion"]}]), encoding="utf-8")
    (live / "config.json").write_text(json.dumps({"providers": {"ollama": {
        "models": ["existing:latest"], "selectedModels": ["existing:latest"],
        "modelContextWindows": {"existing:latest": 8192}}}}), encoding="utf-8")
    codex = tmp_path / "codex-home"
    codex.mkdir()
    (codex / "config.toml").write_text(
        f'model_catalog_json = {json.dumps(str(catalog))}\nopenai_base_url = "http://127.0.0.1:19108/v1"\n',
        encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(codex))
    monkeypatch.setenv("LOCAL_MODEL_BENCH_OPENCODEX_ROOT", str(root))
    monkeypatch.setenv("LOCAL_MODEL_BENCH_DATA_DIR", str(tmp_path / "app-data"))
    return root


def test_register_connect_and_delete_only_selected_model(configured_bridge, monkeypatch):
    client = FakeOllama()
    root = configured_bridge
    message = links.register_opencodex("new:latest", client)
    assert "registered" in message
    catalog_path = root / "desktop/opencodex-catalog.json"
    rows = json.loads(catalog_path.read_text(encoding="utf-8"))["models"]
    assert {row["slug"] for row in rows} == {"ollama/existing:latest", "ollama/new:latest"}
    assert "new:latest" in json.loads((root / "live/home/config.json").read_text(encoding="utf-8"))["providers"]["ollama"]["models"]
    monkeypatch.setattr(links, "urlopen", lambda *_args, **_kwargs:
                        BytesIO(json.dumps({"data": [{"id": "ollama/new:latest"}]}).encode()))
    assert "proxy lists" in links.connect_codex("new:latest", client)
    assert "references removed" in links.delete_connected_model("new:latest", client)
    assert client.names == {"existing:latest"}
    rows = json.loads(catalog_path.read_text(encoding="utf-8"))["models"]
    assert [row["slug"] for row in rows] == ["ollama/existing:latest"]
    assert json.loads((root / "desktop/allowed-local-models.json").read_text(encoding="utf-8")) == ["existing:latest"]
    assert json.loads((root / "live/home/config.json").read_text(encoding="utf-8"))["providers"]["ollama"]["models"] == ["existing:latest"]
    assert list((Path(os.environ["LOCAL_MODEL_BENCH_DATA_DIR"]) / "connection-backups").glob("*/manifest.json"))


def test_ollama_delete_failure_keeps_connections(configured_bridge):
    client = FakeOllama()
    links.register_opencodex("new:latest", client)
    paths = links._paths(configured_bridge, codex=True)
    before = {path: path.read_bytes() for path in paths}
    client.fail_delete = True
    with pytest.raises(RuntimeError, match="refused"):
        links.delete_connected_model("new:latest", client)
    assert {path: path.read_bytes() for path in paths} == before


def test_loaded_model_cannot_be_deleted_or_disconnected(configured_bridge):
    client = FakeOllama()
    links.register_opencodex("new:latest", client)
    paths = links._paths(configured_bridge, codex=True)
    before = {path: path.read_bytes() for path in paths}
    client.loaded.add("new:latest")
    with pytest.raises(links.ConnectionError, match="loaded"):
        links.delete_connected_model("new:latest", client)
    assert "new:latest" in client.names
    assert {path: path.read_bytes() for path in paths} == before


def test_tps_never_uses_word_count_as_token_count():
    assert GenerationResult(text="two words", elapsed_seconds=1.0).tokens_per_second == 0
    assert GenerationResult(text="", elapsed_seconds=10, eval_count=20,
                            eval_duration_ns=1_000_000_000).tokens_per_second == 20


def test_gguf_import_preserves_source_and_verifies_ollama_inventory(tmp_path, monkeypatch):
    client = OllamaClient()
    source = tmp_path / "tiny.gguf"
    source.write_bytes(b"GGUF")
    created = {"done": False}

    def run(argv, **_kwargs):
        assert argv[:3] == ["ollama", "create", "tiny:latest"]
        assert str(source.as_posix()) in Path(argv[-1]).read_text(encoding="utf-8")
        created["done"] = True
        return SimpleNamespace(returncode=0, stdout="success", stderr="")

    monkeypatch.setattr("local_model_bench.ollama_client.subprocess.run", run)
    monkeypatch.setattr(client, "list_models", lambda: [OllamaModel("tiny:latest")] if created["done"] else [])
    client.create_from_gguf(str(source), "tiny:latest")
    assert source.read_bytes() == b"GGUF"
