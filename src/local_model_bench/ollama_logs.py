from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import os
from pathlib import Path
import re


@dataclass(frozen=True)
class OllamaLogUsageEvent:
    created_at: str
    model_name: str
    source: str
    operation: str
    status: str
    failure_type: str
    duration_seconds: float | None
    external_id: str
    details: dict


class OllamaLogImporter:
    TEMPLATE_RE = re.compile(r'time=(?P<ts>\S+).*msg="template selection"\s+model=(?P<model>\S+)')
    GIN_RE = re.compile(
        r'\[GIN\]\s+'
        r'(?P<date>\d{4}/\d{2}/\d{2})\s+-\s+'
        r'(?P<time>\d{2}:\d{2}:\d{2})\s+\|\s+'
        r'(?P<status>\d{3})\s+\|\s+'
        r'(?P<duration>[^|]+)\|\s+'
        r'(?P<ip>[^|]+)\|\s+'
        r'(?P<method>\S+)\s+"(?P<path>[^"]+)"'
    )
    API_PATHS = {"/api/generate", "/api/chat"}

    def __init__(self, log_dir: Path | None = None) -> None:
        self.log_dir = log_dir or self.default_log_dir()

    @staticmethod
    def default_log_dir() -> Path:
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "Ollama"
        return Path.home() / "AppData" / "Local" / "Ollama"

    def default_log_paths(self) -> list[Path]:
        if not self.log_dir.exists():
            return []
        return sorted(self.log_dir.glob("server*.log"), key=lambda path: path.stat().st_mtime)

    def parse_paths(self, paths: list[Path] | None = None, *, max_events: int = 3000) -> list[OllamaLogUsageEvent]:
        events: list[OllamaLogUsageEvent] = []
        current_model = "unknown"
        current_model_time = ""
        selected_paths = paths if paths is not None else self.default_log_paths()
        for path in selected_paths:
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    for line_number, line in enumerate(handle, 1):
                        model_match = self.TEMPLATE_RE.search(line)
                        if model_match:
                            current_model = model_match.group("model")
                            current_model_time = model_match.group("ts")
                            continue

                        request_match = self.GIN_RE.search(line)
                        if not request_match or request_match.group("path") not in self.API_PATHS:
                            continue

                        event = self._event_from_request(
                            path=path,
                            line_number=line_number,
                            line=line,
                            match=request_match,
                            current_model=current_model,
                            current_model_time=current_model_time,
                        )
                        events.append(event)
            except OSError:
                continue
        if max_events > 0:
            return events[-max_events:]
        return events

    def _event_from_request(
        self,
        *,
        path: Path,
        line_number: int,
        line: str,
        match: re.Match[str],
        current_model: str,
        current_model_time: str,
    ) -> OllamaLogUsageEvent:
        status_code = int(match.group("status"))
        success = 200 <= status_code < 300
        request_path = match.group("path")
        failure_type = "" if success else ("connection error" if status_code == 499 else "runtime error")
        operation = "ollama:chat" if request_path == "/api/chat" else "ollama:generate"
        created_at = self._request_timestamp(match.group("date"), match.group("time"))
        details = {
            "http_status": status_code,
            "http_method": match.group("method"),
            "http_path": request_path,
            "client_ip": match.group("ip").strip(),
            "log_file": str(path),
            "log_line": line_number,
            "source_evidence": "ollama_server_log_only",
            "model_evidence_time": current_model_time,
            "token_counts": "unavailable_in_ollama_server_log",
        }
        return OllamaLogUsageEvent(
            created_at=created_at,
            model_name=current_model or "unknown",
            source="unknown",
            operation=operation,
            status="success" if success else "failure",
            failure_type=failure_type,
            duration_seconds=self._duration_seconds(match.group("duration")),
            external_id=self._external_id(path, line_number, line),
            details=details,
        )

    @staticmethod
    def _request_timestamp(date_text: str, time_text: str) -> str:
        parsed = datetime.strptime(f"{date_text} {time_text}", "%Y/%m/%d %H:%M:%S")
        return parsed.astimezone().isoformat()

    @staticmethod
    def _duration_seconds(value: str) -> float | None:
        text = value.strip()
        try:
            if text.endswith("ms"):
                return round(float(text[:-2].strip()) / 1000, 4)
            if text.endswith("us") or text.endswith("µs"):
                return round(float(text[:-2].strip()) / 1_000_000, 6)
            if text.endswith("s") and "m" not in text and "h" not in text:
                return round(float(text[:-1].strip()), 4)
            match = re.fullmatch(
                r"(?:(?P<hours>\d+(?:\.\d+)?)h)?"
                r"(?:(?P<minutes>\d+(?:\.\d+)?)m)?"
                r"(?:(?P<seconds>\d+(?:\.\d+)?)s)?",
                text,
            )
            if match:
                hours = float(match.group("hours") or 0)
                minutes = float(match.group("minutes") or 0)
                seconds = float(match.group("seconds") or 0)
                return round((hours * 3600) + (minutes * 60) + seconds, 4)
        except ValueError:
            return None
        return None

    @staticmethod
    def _external_id(path: Path, line_number: int, line: str) -> str:
        payload = f"{path.resolve()}:{line_number}:{line.strip()}".encode("utf-8", errors="replace")
        return "ollama-log:" + hashlib.sha256(payload).hexdigest()
