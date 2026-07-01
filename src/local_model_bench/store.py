from __future__ import annotations

from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path

from .models import BenchmarkReport


class ResultStore:
    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".local-model-bench" / "results.sqlite"
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _init(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_score REAL NOT NULL,
                    avg_tokens_per_second REAL NOT NULL,
                    finished_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_results_model ON benchmark_results(model_name, finished_at)")
            columns = {row[1] for row in con.execute("PRAGMA table_info(benchmark_results)").fetchall()}
            if "deleted_at" not in columns:
                con.execute("ALTER TABLE benchmark_results ADD COLUMN deleted_at TEXT")
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    status TEXT NOT NULL,
                    failure_type TEXT NOT NULL,
                    prompt_tokens INTEGER,
                    eval_tokens INTEGER,
                    total_tokens INTEGER,
                    duration_seconds REAL,
                    external_id TEXT,
                    details_json TEXT NOT NULL
                )
                """
            )
            usage_columns = {row[1] for row in con.execute("PRAGMA table_info(usage_events)").fetchall()}
            if "external_id" not in usage_columns:
                con.execute("ALTER TABLE usage_events ADD COLUMN external_id TEXT")
            con.execute("CREATE INDEX IF NOT EXISTS idx_usage_created ON usage_events(created_at)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_usage_model ON usage_events(model_name, created_at)")
            con.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_usage_external_id
                ON usage_events(external_id)
                WHERE external_id IS NOT NULL
                """
            )

    def save_report(self, report: BenchmarkReport) -> None:
        payload = self._report_to_dict(report)
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                INSERT INTO benchmark_results
                  (model_name, source, status, total_score, avg_tokens_per_second, finished_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.model_name,
                    report.source,
                    report.status,
                    report.score.total,
                    report.avg_tokens_per_second,
                    report.finished_at,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )

    def latest_reports(self, limit: int = 200, *, include_deleted: bool = True) -> list[dict]:
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            where = "" if include_deleted else "WHERE deleted_at IS NULL"
            rows = con.execute(
                f"""
                SELECT id, payload_json, deleted_at FROM benchmark_results
                {where}
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        results = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            payload["result_id"] = int(row["id"])
            payload["deleted_at"] = row["deleted_at"]
            results.append(payload)
        return results

    def all_reports(self, *, include_deleted: bool = True) -> list[dict]:
        return self.latest_reports(limit=100_000, include_deleted=include_deleted)

    def latest_report_per_model(self, limit: int = 500, *, include_deleted: bool = True) -> list[dict]:
        rows = self.latest_reports(limit=limit, include_deleted=include_deleted)
        latest: dict[str, dict] = {}
        for row in rows:
            model_name = str(row.get("model_name", ""))
            if model_name and model_name not in latest:
                latest[model_name] = row
        return list(latest.values())

    def latest_report_map(self, limit: int = 1000, *, include_deleted: bool = True) -> dict[str, dict]:
        return {
            row.get("model_name", ""): row
            for row in self.latest_report_per_model(limit=limit, include_deleted=include_deleted)
        }

    def reports_for_model(self, model_name: str, limit: int = 25, *, include_deleted: bool = True) -> list[dict]:
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            deleted_filter = "" if include_deleted else "AND deleted_at IS NULL"
            rows = con.execute(
                f"""
                SELECT id, payload_json, deleted_at FROM benchmark_results
                WHERE model_name = ?
                {deleted_filter}
                ORDER BY id DESC
                LIMIT ?
                """,
                (model_name, limit),
            ).fetchall()
        results = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            payload["result_id"] = int(row["id"])
            payload["deleted_at"] = row["deleted_at"]
            results.append(payload)
        return results

    def delete_report(self, result_id: int) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "UPDATE benchmark_results SET deleted_at = COALESCE(deleted_at, ?) WHERE id = ?",
                (self._now_local_iso(), result_id),
            )

    def delete_reports(self, result_ids: list[int]) -> None:
        ids = [int(result_id) for result_id in result_ids if int(result_id) > 0]
        if not ids:
            return
        now = self._now_local_iso()
        placeholders = ",".join("?" for _ in ids)
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                f"UPDATE benchmark_results SET deleted_at = COALESCE(deleted_at, ?) WHERE id IN ({placeholders})",
                [now, *ids],
            )

    def delete_reports_for_model(self, model_name: str) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "UPDATE benchmark_results SET deleted_at = COALESCE(deleted_at, ?) WHERE model_name = ?",
                (self._now_local_iso(), model_name),
            )

    def _now_local_iso(self) -> str:
        return datetime.now().astimezone().isoformat()

    def save_usage_event(
        self,
        *,
        model_name: str,
        source: str,
        operation: str,
        status: str,
        failure_type: str = "",
        prompt_tokens: int | None = None,
        eval_tokens: int | None = None,
        duration_seconds: float | None = None,
        details: dict | None = None,
        created_at: str | None = None,
        external_id: str | None = None,
    ) -> bool:
        prompt = int(prompt_tokens or 0)
        eval_count = int(eval_tokens or 0)
        total = prompt + eval_count if prompt or eval_count else None
        with sqlite3.connect(self.db_path) as con:
            cursor = con.execute(
                """
                INSERT OR IGNORE INTO usage_events
                  (created_at, model_name, source, operation, status, failure_type,
                   prompt_tokens, eval_tokens, total_tokens, duration_seconds, external_id, details_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at or self._now_local_iso(),
                    model_name,
                    source,
                    operation,
                    status,
                    failure_type or "",
                    prompt_tokens,
                    eval_tokens,
                    total,
                    duration_seconds,
                    external_id,
                    json.dumps(details or {}, ensure_ascii=False),
                ),
            )
        return cursor.rowcount > 0

    def usage_events(self, *, period: str = "all", limit: int = 500) -> list[dict]:
        where, params = self._usage_period_filter(period)
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                f"""
                SELECT * FROM usage_events
                {where}
                ORDER BY id DESC
                LIMIT ?
                """,
                [*params, limit],
            ).fetchall()
        return [self._usage_row_to_dict(row) for row in rows]

    def usage_by_model(self, *, period: str = "all") -> list[dict]:
        where, params = self._usage_period_filter(period)
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                f"""
                SELECT
                  model_name,
                  source,
                  COUNT(*) AS calls,
                  SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS successes,
                  SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) AS failures,
                  SUM(prompt_tokens) AS prompt_tokens,
                  SUM(eval_tokens) AS eval_tokens,
                  SUM(total_tokens) AS total_tokens,
                  MAX(created_at) AS last_used
                FROM usage_events
                {where}
                GROUP BY model_name, source
                ORDER BY calls DESC, failures DESC, model_name ASC
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def usage_token_totals_by_model(self, *, period: str = "all") -> dict[str, int]:
        where, params = self._usage_period_filter(period)
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                f"""
                SELECT model_name, SUM(COALESCE(total_tokens, 0)) AS total_tokens
                FROM usage_events
                {where}
                GROUP BY model_name
                """,
                params,
            ).fetchall()
        return {str(row["model_name"]): int(row["total_tokens"] or 0) for row in rows}

    def usage_by_source(self, *, period: str = "all") -> list[dict]:
        where, params = self._usage_period_filter(period)
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                f"""
                SELECT source, COUNT(*) AS calls,
                       SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS successes,
                       SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) AS failures,
                       SUM(total_tokens) AS total_tokens
                FROM usage_events
                {where}
                GROUP BY source
                ORDER BY calls DESC
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def usage_overview(self, *, period: str = "all", limit: int = 100_000) -> list[dict]:
        events = self.usage_events(period=period, limit=limit)
        groups: dict[tuple[str, str, str, str, str], dict] = {}
        for event in events:
            details = event.get("details") or {}
            source = str(event.get("source") or "unknown")
            operation = str(event.get("operation") or "unknown")
            client_ip = str(details.get("client_ip") or "")
            evidence = str(details.get("source_evidence") or ("direct_usage_event" if source != "unknown" else "unknown"))
            source_detail = self._usage_source_detail(source, details)
            key = (str(event.get("model_name") or ""), source, operation, client_ip, evidence)
            group = groups.setdefault(
                key,
                {
                    "model_name": key[0],
                    "source": source,
                    "source_detail": source_detail,
                    "operation": operation,
                    "client_ip": client_ip,
                    "source_evidence": evidence,
                    "calls": 0,
                    "successes": 0,
                    "failures": 0,
                    "prompt_tokens": 0,
                    "eval_tokens": 0,
                    "total_tokens": 0,
                    "known_token_calls": 0,
                    "unknown_token_calls": 0,
                    "duration_total": 0.0,
                    "duration_count": 0,
                    "last_used": "",
                    "recent_failure": "",
                },
            )
            group["calls"] += 1
            if str(event.get("status") or "").lower() == "success":
                group["successes"] += 1
            else:
                group["failures"] += 1
                if not group["recent_failure"]:
                    group["recent_failure"] = str(event.get("failure_type") or event.get("status") or "failed")

            total_tokens = event.get("total_tokens")
            if total_tokens is None:
                group["unknown_token_calls"] += 1
            else:
                group["known_token_calls"] += 1
                group["total_tokens"] += int(total_tokens or 0)
                group["prompt_tokens"] += int(event.get("prompt_tokens") or 0)
                group["eval_tokens"] += int(event.get("eval_tokens") or 0)

            duration = event.get("duration_seconds")
            if duration is not None:
                group["duration_total"] += float(duration or 0)
                group["duration_count"] += 1

            created_at = str(event.get("created_at") or "")
            if created_at > str(group["last_used"] or ""):
                group["last_used"] = created_at

        rows = []
        for group in groups.values():
            calls = int(group["calls"] or 0)
            duration_count = int(group.pop("duration_count") or 0)
            duration_total = float(group.pop("duration_total") or 0)
            group["avg_duration_seconds"] = round(duration_total / duration_count, 3) if duration_count else None
            group["token_coverage"] = round((int(group["known_token_calls"] or 0) / calls) * 100, 1) if calls else 0.0
            rows.append(group)
        return sorted(
            rows,
            key=lambda row: (
                int(row.get("calls", 0) or 0),
                int(row.get("unknown_token_calls", 0) or 0),
                int(row.get("total_tokens", 0) or 0),
                str(row.get("last_used", "")),
            ),
            reverse=True,
        )

    def failures_by_type(self, *, period: str = "all") -> list[dict]:
        where, params = self._usage_period_filter(period)
        clause = f"{where} AND status != 'success'" if where else "WHERE status != 'success'"
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                f"""
                SELECT COALESCE(NULLIF(failure_type, ''), 'unknown') AS failure_type,
                       COUNT(*) AS failures
                FROM usage_events
                {clause}
                GROUP BY COALESCE(NULLIF(failure_type, ''), 'unknown')
                ORDER BY failures DESC
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def _usage_period_filter(self, period: str) -> tuple[str, list[str]]:
        now = datetime.now().astimezone()
        if period == "today":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "1d":
            start = now.replace(microsecond=0) - timedelta(days=1)
        elif period == "7d":
            start = now.replace(microsecond=0) - timedelta(days=7)
        elif period == "30d":
            start = now.replace(microsecond=0) - timedelta(days=30)
        else:
            return "", []
        return "WHERE created_at >= ?", [start.isoformat()]

    def _usage_row_to_dict(self, row: sqlite3.Row) -> dict:
        item = dict(row)
        try:
            item["details"] = json.loads(item.pop("details_json") or "{}")
        except json.JSONDecodeError:
            item["details"] = {}
        return item

    @staticmethod
    def _usage_source_detail(source: str, details: dict) -> str:
        evidence = str(details.get("source_evidence") or "")
        client_ip = str(details.get("client_ip") or "")
        if source == "opencode":
            agent = str(details.get("agent") or "")
            provider = str(details.get("provider_id") or "")
            return f"OpenCode {agent}" if agent else f"OpenCode {provider}".strip()
        if source != "unknown":
            return source
        if evidence == "ollama_server_log_only":
            if client_ip in {"127.0.0.1", "::1"}:
                return "Ollama localhost"
            if client_ip.startswith(("192.168.", "172.", "10.")):
                return f"Ollama LAN/WSL {client_ip}"
            if client_ip:
                return f"Ollama remote {client_ip}"
            return "Ollama server log"
        return "unknown"

    def _report_to_dict(self, report: BenchmarkReport) -> dict:
        first_token_values = [case.first_token_seconds for case in report.cases if case.first_token_seconds > 0]
        avg_first_token = sum(first_token_values) / len(first_token_values) if first_token_values else 0.0
        prompt_tps_values = [case.prompt_tokens_per_second for case in report.cases if case.prompt_tokens_per_second > 0]
        avg_prompt_tps = sum(prompt_tps_values) / len(prompt_tps_values) if prompt_tps_values else 0.0
        load_values = [case.load_seconds for case in report.cases if case.load_seconds > 0]
        avg_load_seconds = sum(load_values) / len(load_values) if load_values else 0.0
        return {
            "model_name": report.model_name,
            "source": report.source,
            "size_label": report.size_label,
            "quant": report.quant,
            "status": report.status,
            "total_score": report.score.total,
            "quality": report.score.quality,
            "speed": report.score.speed,
            "stability": report.score.stability,
            "resource_fit": report.score.resource_fit,
            "usability": report.score.usability,
            "recommended_uses": report.score.recommended_uses,
            "use_case_scores": report.score.use_case_scores,
            "avg_tokens_per_second": report.avg_tokens_per_second,
            "avg_first_token_seconds": round(avg_first_token, 2),
            "avg_prompt_tokens_per_second": round(avg_prompt_tps, 2),
            "avg_load_seconds": round(avg_load_seconds, 2),
            "started_at": report.started_at,
            "finished_at": report.finished_at,
            "error": report.error,
            "resource_summary": report.resource_summary,
            "run_settings": report.run_settings,
            "cases": [
                {
                    "name": case.name,
                    "passed": case.passed,
                    "score": case.score,
                    "elapsed_seconds": case.elapsed_seconds,
                    "first_token_seconds": case.first_token_seconds,
                    "tokens_per_second": case.tokens_per_second,
                    "prompt_tokens_per_second": case.prompt_tokens_per_second,
                    "load_seconds": case.load_seconds,
                    "output": case.output,
                    "error": case.error,
                }
                for case in report.cases
            ],
        }
