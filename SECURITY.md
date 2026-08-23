# Security

## Local and offline boundaries

Local Model Bench is a Windows desktop application. Benchmark results,
settings, and the SQLite history are written to user-selected local paths;
there is no project backend. Ollama is contacted directly at the configured
local endpoint (normally `127.0.0.1:11434`). Optional Hugging Face search and
model pulls are explicit network actions. The app does not proxy traffic,
sniff packets, or run a background service.

Avoid importing confidential prompts, personal data, credentials, tokens,
cookies, or private logs. Do not commit secrets or real logs to this
repository. The files under `samples/` are synthetic parser fixtures only and
contain no measured model performance or real user data.

## Responsible reporting

Report a suspected vulnerability privately through the repository's GitHub
Security Advisories page:
<https://github.com/Vinsentk/local-model-bench/security/advisories/new>.
Include a minimal reproduction and affected version, but never include live
credentials, access tokens, private logs, or other sensitive data. Do not
open a public issue for an undisclosed vulnerability.
