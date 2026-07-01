from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from .models import GenerationResult, OllamaModel


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, endpoint: str = "http://127.0.0.1:11434", timeout_seconds: int = 120) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def list_models(self) -> list[OllamaModel]:
        payload = self._get_json("/api/tags")
        models = []
        for item in payload.get("models", []):
            name = str(item.get("name", ""))
            if not name:
                continue
            size = int(item.get("size", 0) or 0)
            models.append(
                OllamaModel(
                    name=name,
                    model_id=str(item.get("digest", ""))[:12],
                    size_bytes=size,
                    size_label=self._format_size(size),
                    modified_at=str(item.get("modified_at", "")),
                    quant=self.extract_quant(name),
                )
            )
        return models

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        num_predict: int = 128,
        num_ctx: int = 2048,
        temperature: float = 0.0,
        keep_alive: str = "0s",
        timeout_seconds: int | None = None,
    ) -> GenerationResult:
        start = time.perf_counter()
        body = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": keep_alive,
            "options": {
                "num_predict": num_predict,
                "num_ctx": num_ctx,
                "temperature": temperature,
            },
        }
        payload, text, first_token_seconds = self._post_stream_json(
            "/api/generate", body, start=start, timeout_seconds=timeout_seconds
        )
        elapsed = time.perf_counter() - start
        return GenerationResult(
            text=text,
            elapsed_seconds=elapsed,
            first_token_seconds=first_token_seconds,
            eval_count=int(payload.get("eval_count", 0) or 0),
            eval_duration_ns=int(payload.get("eval_duration", 0) or 0),
            prompt_eval_count=int(payload.get("prompt_eval_count", 0) or 0),
            prompt_eval_duration_ns=int(payload.get("prompt_eval_duration", 0) or 0),
            load_duration_ns=int(payload.get("load_duration", 0) or 0),
            raw=payload,
        )

    def pull(self, model: str, on_line: Callable[[str], None] | None = None) -> None:
        self._pull_api(model, on_line=on_line)

    def _pull_api(self, model: str, on_line: Callable[[str], None] | None = None) -> None:
        body = {"name": model, "stream": True}
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint + "/api/pull",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    if payload.get("error"):
                        raise OllamaError(f"ollama pull failed for {model}: {payload.get('error')}")
                    if on_line:
                        on_line(self._format_pull_status(payload))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Ollama pull failed for {model}: {exc}") from exc

    def pull_with_cli(self, model: str, on_line: Callable[[str], None] | None = None) -> None:
        cmd = ["ollama", "pull", model]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            **_hidden_subprocess_kwargs(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            if on_line:
                on_line(line.rstrip())
        code = process.wait()
        if code != 0:
            raise OllamaError(f"ollama pull failed with exit code {code}: {model}")

    def delete(self, model: str) -> None:
        cmd = ["ollama", "rm", model]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            **_hidden_subprocess_kwargs(),
        )
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            raise OllamaError(f"ollama rm failed for {model}: {details}")

    def ps(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            rows = self._get_json("/api/ps").get("models", [])
        except OllamaError:
            rows = []
        return rows or self._ps_cli()

    def _ps_cli(self) -> list[dict[str, Any]]:
        result = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            **_hidden_subprocess_kwargs(),
        )
        if result.returncode != 0:
            return []
        lines = [line.rstrip("\n") for line in result.stdout.splitlines() if line.strip()]
        if len(lines) <= 1:
            return []
        return self._ps_cli_split(lines[1:])

    def _ps_cli_split(self, lines: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line in lines:
            parts = re.split(r"\s{2,}", line.strip(), maxsplit=5)
            if not parts:
                continue
            rows.append(
                {
                    "model": parts[0],
                    "name": parts[0],
                    "digest": parts[1] if len(parts) > 1 else "",
                    "size_label": parts[2] if len(parts) > 2 else "",
                    "processor": parts[3] if len(parts) > 3 else "",
                    "context": parts[4] if len(parts) > 4 else "",
                    "expires_at": parts[5] if len(parts) > 5 else "",
                    "source": "ollama ps",
                }
            )
        return rows

    def show(self, model: str) -> dict[str, Any]:
        return self._post_json("/api/show", {"model": model})

    def _get_json(self, path: str) -> dict[str, Any]:
        url = self.endpoint + path
        try:
            with urllib.request.urlopen(url, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Ollama request failed: {url}: {exc}") from exc

    def _post_json(
        self,
        path: str,
        body: dict[str, Any],
        *,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds or self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Ollama generate failed for {body.get('model')}: {exc}") from exc

    def _post_stream_json(
        self,
        path: str,
        body: dict[str, Any],
        *,
        start: float,
        timeout_seconds: int | None = None,
    ) -> tuple[dict[str, Any], str, float]:
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        chunks: list[str] = []
        first_token_seconds = 0.0
        final_payload: dict[str, Any] = {}
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds or self.timeout_seconds) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    piece = str(payload.get("response", ""))
                    if piece:
                        if first_token_seconds <= 0:
                            first_token_seconds = time.perf_counter() - start
                        chunks.append(piece)
                    if payload.get("done"):
                        final_payload = payload
                        break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Ollama streaming generate failed for {body.get('model')}: {exc}") from exc
        return final_payload, "".join(chunks), first_token_seconds

    @staticmethod
    def extract_quant(name: str) -> str:
        patterns = [
            r"(UD-Q\d_[A-Z_]+)",
            r"(Q\d_[A-Z_]+)",
            r"(IQ\d_[A-Za-z0-9_]+)",
            r"(q\d(?:_[a-z]+)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, name, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        if size_bytes <= 0:
            return ""
        gb = size_bytes / (1024**3)
        if gb >= 1:
            return f"{gb:.1f} GB"
        return f"{size_bytes / (1024**2):.0f} MB"

    @classmethod
    def _format_pull_status(cls, payload: dict[str, Any]) -> str:
        status = str(payload.get("status", "") or "pulling")
        completed = int(payload.get("completed", 0) or 0)
        total = int(payload.get("total", 0) or 0)
        digest = str(payload.get("digest", "") or "")
        if total > 0 and completed >= 0:
            percent = max(0, min(100, round((completed / total) * 100)))
            label = f"{status} {percent}% ({cls._format_bytes(completed)} / {cls._format_bytes(total)})"
        else:
            label = status
        if digest:
            return f"{label} {digest[:18]}"
        return label

    @staticmethod
    def _format_bytes(value: int) -> str:
        if value >= 1024**3:
            return f"{value / (1024**3):.1f} GB"
        if value >= 1024**2:
            return f"{value / (1024**2):.1f} MB"
        if value >= 1024:
            return f"{value / 1024:.1f} KB"
        return f"{value} B"


def _hidden_subprocess_kwargs() -> dict:
    kwargs: dict = {}
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    if hasattr(subprocess, "STARTUPINFO"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        kwargs["startupinfo"] = startupinfo
    return kwargs
