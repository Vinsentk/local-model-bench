# Architecture

## Components

- `ui.py` / `main.py`: PySide6 Windows UI, settings, tabs, workers, progress, exports, and user actions.
- `ollama_client.py`: Ollama HTTP API and hidden CLI subprocess wrappers for listing, generating, pulling, and removing models.
- `benchmark.py`: benchmark modes/cases, timeouts, resource-monitor integration, result classification, and report assembly.
- `scoring.py` / `ranker.py`: practical score calculation and hardware-aware recommendation ranking.
- `hardware.py` / `resources.py`: Windows CPU/GPU/RAM/VRAM/disk probing and live benchmark resource observation.
- `store.py`: SQLite schema and queries for reports, soft deletion, usage events, summaries, exports, and backups.
- `ollama_logs.py` / `opencode_logs.py`: read-only parsing and deduplication of local usage evidence.
- `hf_search.py`: Hugging Face Trending/API/README retrieval, filtering, metadata extraction, and candidate preparation.
- `models.py` / `i18n.py`: typed data records and localized labels/use-case text.

## Data flow

```text
PySide6 UI
  -> settings + OllamaClient
  -> BenchmarkRunner -> ResourceMonitor + ScoreEngine -> ResultStore(SQLite)
  -> Ollama/OpenCode log importers -> usage events -> ResultStore
  -> HuggingFaceSearch -> ModelRanker + HardwareProbe -> recommendations
  -> OllamaClient pull/remove -> installed-model refresh
```

Benchmark requests use the configured endpoint (default `http://127.0.0.1:11434`). Reports retain raw case metrics, score breakdowns, status, and failure type. Usage imports retain source evidence and mark unavailable token counts instead of estimating them.

## Integration boundaries

- **Ollama:** HTTP requests and CLI fallback are isolated in `OllamaClient`; connection, timeout, malformed JSON, and pull/remove errors become `OllamaError` or explicit failure states.
- **Hugging Face:** `HuggingFaceSearch` is the optional network boundary for Trending pages and model metadata. Search failures can leave the UI with rule-based local recommendations.
- **SQLite:** `ResultStore` is the persistence boundary. Writes are local; latest/soft-deleted/report/usage queries provide the UI read models.
- **PySide6:** UI work is dispatched through worker paths so long benchmark, search, import, and download operations can report progress and errors without blocking the main window.
- **Windows environment:** `OLLAMA_HOST`, `OLLAMA_MODELS`, `LOCALAPPDATA`, and user-selected paths configure local resources. Hardware probing falls back across fixed drive roots (`E:\`, `D:\`, `C:\`) only to find available free space; these roots are environment-dependent and do not expose a user's private path in the repository.

## Failure boundaries and fallbacks

Benchmark cases distinguish `OK`, `PARTIAL`, `TIMEOUT`, `LENGTH`, `EMPTY`, and `RUNTIME`; usage aggregates distinguish timeout, length, empty response, runtime, connection, and unknown failures. Missing Ollama/Hugging Face evidence is shown as unavailable or failed rather than guessed. Recommendation research uses a rule-based fallback when its optional report path is unavailable. Log importers cap events and hash source lines to avoid duplicate ingestion. No component silently converts an external or missing failure into a successful benchmark result.
