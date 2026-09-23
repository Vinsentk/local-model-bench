# Changelog

## 0.2.2 - 2026-09-23

- Clarify that Standard and Thinking benchmarks begin with a basic response check; progress now shows the selected run mode separately from the current case.
- Keep the mode selector fixed during a running benchmark so its visible value matches the active run.

## 0.2.1 - 2026-09-23

- Show every installed Ollama model in the benchmark comparison, including models without a benchmark result.
- Remove the comparison graph and show scores, measured generation TPS, and recommended uses in a prominent table on a vertically scrollable benchmark page.

## 0.2.0 - 2026-09-23

- Added selected-model GGUF import, OpenCodex registration, Codex catalog connection, and connection cleanup on Ollama deletion.
- Added a one-run basic model check, practical-use guidance, and measured generation TPS from Ollama evaluation metrics.
- Added an interactive benchmark chart and refreshed dark desktop UI.
- Fixed the Windows package startup failure caused by an incompatible ICU DLL collected from another tool on PATH.

## 0.1.0 - 2026-06-20

- Added PySide6 desktop UI for installed Ollama models, benchmarks, recommendations, results, and settings.
- Added smoke, standard, and thinking benchmark modes with score, speed, quality, resource, and use-case ranking output.
- Added CPU/GPU/RAM/VRAM sampling during benchmarks.
- Added Hugging Face GGUF Trending recommendation search with user-selected page range, HF-style filters, keyword refinement, parameter-range filtering, and local research notes.
- Added multilingual UI support for English, Korean, Japanese, and Chinese.
- Added SQLite result storage, soft deletion, CSV/JSON export, DB backup, and data-folder access.
- Added usage analytics for local-model-bench Ollama calls, read-only Ollama/OpenCode log import, active Ollama model display, source/failure aggregation, token counters when available, recent usage events, and installed-model 1-day/7-day token totals.
- Added Windows build script, app icon, version metadata, GitHub Actions workflow, and MIT license.
