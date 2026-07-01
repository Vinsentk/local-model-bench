from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re


@dataclass(frozen=True)
class OpenCodeLogUsageEvent:
    created_at: str
    model_name: str
    source: str
    operation: str
    status: str
    failure_type: str
    duration_seconds: float | None
    prompt_tokens: int | None
    eval_tokens: int | None
    external_id: str
    details: dict


class OpenCodeLogImporter:
    LINE_TIME_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}T\S+)\s+")
    FIELD_RE = re.compile(r"(?P<key>[A-Za-z_.]+)=(?P<value>\"[^\"]*\"|\S+)")
    TOKEN_PATTERNS = {
        "prompt": [
            re.compile(r'"prompt_tokens"\s*:\s*(\d+)'),
            re.compile(r'"input_tokens"\s*:\s*(\d+)'),
            re.compile(r"\bprompt_tokens=(\d+)"),
            re.compile(r"\binput_tokens=(\d+)"),
        ],
        "eval": [
            re.compile(r'"completion_tokens"\s*:\s*(\d+)'),
            re.compile(r'"output_tokens"\s*:\s*(\d+)'),
            re.compile(r"\bcompletion_tokens=(\d+)"),
            re.compile(r"\boutput_tokens=(\d+)"),
        ],
        "total": [
            re.compile(r'"total_tokens"\s*:\s*(\d+)'),
            re.compile(r"\btotal_tokens=(\d+)"),
        ],
    }

    def __init__(self, log_dirs: list[Path] | None = None) -> None:
        self.log_dirs = log_dirs or self.default_log_dirs()

    @staticmethod
    def default_log_dirs() -> list[Path]:
        paths: list[Path] = []
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            paths.extend(
                [
                    Path(local_app_data) / "ai.opencode.desktop" / "logs",
                    Path(local_app_data) / "OpenCode" / "logs",
                    Path(local_app_data) / "opencode" / "logs",
                ]
            )
        home = Path.home()
        paths.extend(
            [
                home / ".config" / "opencode" / "logs",
                home / ".local" / "share" / "opencode" / "logs",
                home / "Library" / "Logs" / "opencode",
            ]
        )
        seen: set[str] = set()
        unique = []
        for path in paths:
            key = str(path).casefold()
            if key not in seen:
                seen.add(key)
                unique.append(path)
        return unique

    def default_log_paths(self) -> list[Path]:
        paths: list[Path] = []
        for log_dir in self.log_dirs:
            if not log_dir.exists():
                continue
            paths.extend(log_dir.glob("*.log"))
            paths.extend(log_dir.glob("*.jsonl"))
        return sorted(set(paths), key=lambda path: path.stat().st_mtime)

    def parse_paths(self, paths: list[Path] | None = None, *, max_events: int = 3000) -> list[OpenCodeLogUsageEvent]:
        selected_paths = paths if paths is not None else self.default_log_paths()
        events: list[OpenCodeLogUsageEvent] = []
        for path in selected_paths:
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    for line_number, line in enumerate(handle, 1):
                        event = self._event_from_line(path, line_number, line)
                        if event:
                            events.append(event)
            except OSError:
                continue
        if max_events > 0:
            return events[-max_events:]
        return events

    def _event_from_line(self, path: Path, line_number: int, line: str) -> OpenCodeLogUsageEvent | None:
        if "service=llm" not in line or "modelID=" not in line:
            return None
        fields = self._fields(line)
        provider = fields.get("providerID", "")
        model_id = fields.get("modelID", "")
        if not model_id:
            return None
        prompt_tokens, eval_tokens = self._tokens(line)
        status = "failure" if " ERROR " in line or " error=" in line else "success"
        failure_type = self._failure_type(line) if status == "failure" else ""
        agent = fields.get("agent", "")
        operation = f"opencode:{agent}" if agent else "opencode:llm"
        details = {
            "provider_id": provider,
            "model_id": model_id,
            "agent": agent,
            "mode": fields.get("mode", ""),
            "session_id": fields.get("session.id", ""),
            "small": fields.get("small", ""),
            "log_file": str(path),
            "log_line": line_number,
            "source_evidence": "opencode_log",
            "token_counts": "available" if prompt_tokens is not None or eval_tokens is not None else "unavailable_in_opencode_log",
        }
        return OpenCodeLogUsageEvent(
            created_at=self._timestamp(line),
            model_name=f"{provider}/{model_id}" if provider else model_id,
            source="opencode",
            operation=operation,
            status=status,
            failure_type=failure_type,
            duration_seconds=self._duration_seconds(line),
            prompt_tokens=prompt_tokens,
            eval_tokens=eval_tokens,
            external_id=self._external_id(path, line_number, line),
            details=details,
        )

    def _fields(self, line: str) -> dict[str, str]:
        values = {}
        for match in self.FIELD_RE.finditer(line):
            values[match.group("key")] = match.group("value").strip('"')
        return values

    def _tokens(self, line: str) -> tuple[int | None, int | None]:
        prompt = self._first_int(line, self.TOKEN_PATTERNS["prompt"])
        eval_count = self._first_int(line, self.TOKEN_PATTERNS["eval"])
        total = self._first_int(line, self.TOKEN_PATTERNS["total"])
        if total is not None and prompt is not None and eval_count is None:
            eval_count = max(0, total - prompt)
        if total is not None and eval_count is not None and prompt is None:
            prompt = max(0, total - eval_count)
        return prompt, eval_count

    @staticmethod
    def _first_int(line: str, patterns: list[re.Pattern[str]]) -> int | None:
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                return int(match.group(1))
        return None

    @classmethod
    def _timestamp(cls, line: str) -> str:
        match = cls.LINE_TIME_RE.search(line)
        if not match:
            return datetime.now().astimezone().isoformat()
        text = match.group("ts").replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text).astimezone().isoformat()
        except ValueError:
            return datetime.now().astimezone().isoformat()

    @staticmethod
    def _duration_seconds(line: str) -> float | None:
        match = re.search(r"\+(?P<ms>\d+)ms\b", line)
        if match:
            return round(int(match.group("ms")) / 1000, 3)
        return None

    @staticmethod
    def _failure_type(line: str) -> str:
        lowered = line.lower()
        if "does not support tools" in lowered or "unsupported" in lowered:
            return "tools unsupported"
        if "timeout" in lowered:
            return "timeout"
        if "rate" in lowered and "limit" in lowered:
            return "rate limit"
        if "connection" in lowered or "connect" in lowered:
            return "connection error"
        return "runtime error"

    @staticmethod
    def _external_id(path: Path, line_number: int, line: str) -> str:
        payload = f"{path.resolve()}:{line_number}:{line.strip()}".encode("utf-8", errors="replace")
        return "opencode-log:" + hashlib.sha256(payload).hexdigest()
