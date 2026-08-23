# Product Requirements

## Problem

Local-model evaluation can leave benchmark output, failures, resource observations, and usage logs scattered across a Windows machine. Users need a repeatable way to compare installed Ollama models and decide what to keep, retry, remove, or test next.

## Users and scope

The primary user is a Windows desktop user who runs local Ollama models and wants practical, local-use comparisons. The MVP covers:

- smoke, standard, and thinking benchmark modes;
- quality/speed/stability/resource-fit/usability scoring out of 100;
- CPU, GPU, RAM, VRAM, and free-storage observations;
- SQLite result history, soft deletion, CSV/JSON export, and database backup;
- Ollama/OpenCode usage-log import with source, status, failure type, and available token evidence;
- Hugging Face GGUF Trending discovery, filtering, hardware-aware ranking, and Ollama download/delete actions;
- a PySide6 UI with English, Korean, Japanese, and Chinese text.

## Non-goals

This is not a scientific benchmark, a hosted model service, a proxy, packet sniffer, background service, or Ollama-settings replacement. Scores are practical signals for one PC, not universal model quality claims. The app does not infer caller identity when logs do not provide evidence, and it does not estimate missing historical token counts.

## Main flows

1. Start the Windows UI, detect an Ollama endpoint, and list installed models.
2. Select a benchmark mode; generate benchmark cases through Ollama, monitor resources, classify outcomes, score the report, and persist it to SQLite.
3. Review latest results, trends, use-case leaders, retry-needed models, soft-deleted history, exports, and backups.
4. Open Recommendations, search selected Hugging Face Trending pages, apply filters and local keywords, rank candidates against detected hardware, and optionally pull or remove a model through Ollama.
5. Open Usage, import local Ollama/OpenCode evidence, deduplicate imported log lines, and review model/source/failure/token summaries.

## Acceptance criteria

- A reachable Ollama server can be listed and benchmarked in all three modes; each case records pass/failure and a bounded failure type.
- A completed report contains the documented score categories and raw benchmark metrics and can be reopened from SQLite.
- Deleted reports remain traceable but are excluded from the active latest view; CSV/JSON export and DB backup are available.
- Recommendation filtering applies the selected page range and filters before the final local keyword refinement, then exposes the Hugging Face model link.
- Usage imports do not duplicate the same external log line and distinguish known from unavailable token counts.
- Ollama/network, subprocess, malformed-response, timeout, and empty-output failures surface as explicit UI/report states rather than being treated as successful results.
- Automated tests and the CI compile/test/build workflow remain green.

## Safety, privacy, and operational limits

The app communicates directly with the configured Ollama endpoint and optional Hugging Face pages. It does not proxy traffic or send data to a project backend. Local result/settings files use the Windows user data area by default; users should choose paths appropriate for their machine and avoid importing sensitive logs. Hardware probing checks available Windows drive roots (`E:\`, `D:\`, and `C:\`) as an environment-dependent fallback; these are not user-specific paths or model-storage guarantees. `OLLAMA_MODELS` remains controlled by Ollama and changing it requires an Ollama restart.
