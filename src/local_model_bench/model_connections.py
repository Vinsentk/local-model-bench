"""Selected-model registration for this host's OpenCodex/Codex bridge.

No live process is restarted here. A file edit is reported separately from live
model discovery, which depends on the proxy and Codex reloading their catalog.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tomllib
from urllib.request import urlopen


class ConnectionError(RuntimeError):
    pass


def _codex_config() -> Path:
    return Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "config.toml"


def _config_values() -> dict:
    path = _codex_config()
    if not path.is_file():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def discover_root() -> Path:
    explicit = os.environ.get("LOCAL_MODEL_BENCH_OPENCODEX_ROOT", "").strip()
    if explicit:
        root = Path(explicit)
    else:
        catalog = _config_values().get("model_catalog_json")
        if not isinstance(catalog, str) or not catalog:
            raise ConnectionError("Codex model catalog is not configured. Set LOCAL_MODEL_BENCH_OPENCODEX_ROOT.")
        path = Path(catalog).resolve()
        root = next((folder for folder in path.parents if
                     (folder / "live/home/config.json").is_file() and
                     (folder / "desktop/opencodex-catalog.json").is_file()), Path())
    if not root.is_dir() or not (root / "live/home/config.json").is_file() or not (root / "desktop/opencodex-catalog.json").is_file():
        raise ConnectionError("OpenCodex root not found. Set LOCAL_MODEL_BENCH_OPENCODEX_ROOT to its .opencodex folder.")
    return root.resolve()


def _paths(root: Path, *, codex: bool = False) -> list[Path]:
    paths = [root / "desktop/opencodex-catalog.json", root / "desktop/allowed-local-models.json",
             root / "desktop/ollama-capabilities.json", root / "live/home/config.json"]
    if codex:
        config = _config_values()
        catalog = config.get("model_catalog_json")
        if isinstance(catalog, str) and catalog:
            path = Path(catalog)
            if path.resolve() not in {item.resolve() for item in paths}:
                paths.append(path)
    if any(not path.is_file() for path in paths):
        raise ConnectionError("OpenCodex registration files are missing; no changes made.")
    return paths


def _read(paths: list[Path]) -> tuple[dict[Path, bytes], dict[Path, object]]:
    raw = {path: path.read_bytes() for path in paths}
    try:
        values = {path: json.loads(content) for path, content in raw.items()}
    except (ValueError, UnicodeDecodeError) as exc:
        raise ConnectionError("OpenCodex registration JSON is invalid; no changes made.") from exc
    return raw, values


def _commit(raw: dict[Path, bytes], values: dict[Path, object], name: str) -> Path | None:
    changed = {path: json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
               for path, value in values.items() if path in raw}
    changed = {path: content for path, content in changed.items() if content != raw[path]}
    if not changed:
        return None
    if any(path.read_bytes() != before for path, before in raw.items()):
        raise ConnectionError("Registration files changed during operation; no changes made.")
    backup = Path(os.environ.get("LOCAL_MODEL_BENCH_DATA_DIR", str(Path.home() / ".local-model-bench"))) / "connection-backups" / (name + "-" + os.urandom(6).hex())
    backup.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for index, (path, before) in enumerate(raw.items()):
        filename = f"{index}-{path.name}"
        (backup / filename).write_bytes(before)
        manifest[str(path)] = {"backup": filename, "sha256": hashlib.sha256(before).hexdigest()}
    (backup / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    applied: list[Path] = []
    try:
        for path, content in changed.items():
            if path.read_bytes() != raw[path]:
                raise ConnectionError("Registration files changed during operation.")
            temp = path.with_name(path.name + ".local-model-bench.tmp")
            with temp.open("xb") as stream:
                stream.write(content)
            try:
                os.replace(temp, path)
            finally:
                temp.unlink(missing_ok=True)
            applied.append(path)
    except Exception:
        for path in reversed(applied):
            if path.read_bytes() == changed[path]:
                path.write_bytes(raw[path])
        raise
    return backup


def _slug(name: str) -> str:
    return "ollama/" + name.replace("/", "-")


def _matching(row: dict, name: str) -> bool:
    provenance = row.get("opencodex_capability_provenance") or {}
    return row.get("slug", "").startswith("ollama/") and (
        provenance.get("model_id") == name or row.get("slug") in {_slug(name), "ollama/" + name})


def _context(detail: dict) -> int:
    matches = re.findall(r"(?m)^num_ctx\s+(\d+)\s*$", str(detail.get("parameters", "")))
    return min(int(matches[0]), 65536) if len(matches) == 1 and int(matches[0]) > 0 else 8192


def register_opencodex(name: str, client) -> str:
    if name not in {model.name for model in client.list_models()}:
        raise ConnectionError("Selected model is no longer installed in Ollama.")
    detail = client.show(name)
    caps = detail.get("capabilities") or []
    if "completion" not in caps or detail.get("remote_model") or detail.get("remote_host"):
        raise ConnectionError("Only local completion models can be registered in OpenCodex.")
    root = discover_root()
    paths = _paths(root)
    raw, values = _read(paths)
    catalog, allowed, capabilities, proxy = (values[path] for path in paths)
    if not isinstance(catalog, dict) or not isinstance(catalog.get("models"), list) or not isinstance(allowed, list) or not isinstance(capabilities, list):
        raise ConnectionError("OpenCodex registration schema is unsupported.")
    provider = proxy.get("providers", {}).get("ollama") if isinstance(proxy, dict) else None
    if not isinstance(provider, dict) or not isinstance(provider.get("models"), list):
        raise ConnectionError("OpenCodex Ollama provider is missing.")
    rows = catalog["models"]
    matching = next((row for row in rows if isinstance(row, dict) and _matching(row, name)), None)
    context = _context(detail)
    if matching is None:
        slug = _slug(name)
        if any(row.get("slug") == slug for row in rows if isinstance(row, dict)):
            raise ConnectionError("Codex model slug collision; no changes made.")
        template = next((row for row in rows if isinstance(row, dict) and row.get("slug", "").startswith("ollama/")), None)
        if template is None:
            raise ConnectionError("No existing OpenCodex Ollama catalog template; no changes made.")
        matching = copy.deepcopy(template)
        instruction = str(matching.get("base_instructions", ""))
        instruction = re.sub(r"\AYou are a coding agent powered by the .*?\. If asked which model you are, identify as .*?\. ",
                             f"You are a coding agent powered by the {slug}. If asked which model you are, identify as {slug}. ",
                             instruction, count=1)
        matching.update(slug=slug, display_name="Local · " + name,
                        description="Ollama local model. Native Codex tools are not yet qualified.",
                        base_instructions=instruction, priority=20 + len(rows),
                        default_reasoning_level="none",
                        supported_reasoning_levels=[{"effort": "none", "description": "No reasoning effort"}],
                        context_window=context, max_context_window=context,
                        auto_compact_token_limit=min(54000, int(context * .9)),
                        input_modalities=["text", "image"] if "vision" in caps else ["text"],
                        supports_image_detail_original="vision" in caps,
                        supports_reasoning_summaries=False,
                        opencodex_capability_provenance={"provider": "ollama", "model_id": name})
        if "tools" not in caps:
            matching["supports_parallel_tool_calls"] = False
        rows.append(matching)
    if name not in allowed:
        allowed.append(name)
    capabilities[:] = [row for row in capabilities if row.get("name") != name]
    capabilities.append({"name": name, "capabilities": caps})
    if name not in provider["models"]:
        provider["models"].append(name)
    selected = provider.get("selectedModels")
    if isinstance(selected, list) and name not in selected:
        selected.append(name)
    provider.setdefault("modelContextWindows", {})[name] = context
    provider.setdefault("modelAutoCompactTokenLimits", {})[name] = min(54000, int(context * .9))
    backup = _commit(raw, values, "register")
    return f"OpenCodex registered: {_slug(name)}. Proxy reload may be needed. Backup: {backup or 'unchanged'}"


def connect_codex(name: str, client) -> str:
    root = discover_root()
    config = _config_values()
    catalog_path = config.get("model_catalog_json")
    proxy_url = config.get("openai_base_url")
    if not isinstance(catalog_path, str) or not catalog_path or not isinstance(proxy_url, str) or not proxy_url:
        raise ConnectionError("Codex has no OpenCodex catalog/proxy route. Configure model_catalog_json and openai_base_url first.")
    selected_catalog = Path(catalog_path).resolve()
    if selected_catalog not in {(root / "desktop/opencodex-catalog.json").resolve(),
                                (root / "live/codex-home/opencodex-catalog.json").resolve()}:
        raise ConnectionError("Codex points to a different catalog; no changes made.")
    register_opencodex(name, client)
    if selected_catalog != (root / "desktop/opencodex-catalog.json").resolve():
        raw, values = _read([selected_catalog])
        current = values[selected_catalog]
        source = json.loads((root / "desktop/opencodex-catalog.json").read_text(encoding="utf-8"))
        row = next(row for row in source["models"] if isinstance(row, dict) and _matching(row, name))
        if not isinstance(current, dict) or not isinstance(current.get("models"), list):
            raise ConnectionError("Codex catalog schema is unsupported.")
        current["models"] = [item for item in current["models"]
                             if not (isinstance(item, dict) and _matching(item, name))] + [copy.deepcopy(row)]
        _commit(raw, values, "codex")
    catalog = json.loads(selected_catalog.read_text(encoding="utf-8"))
    if not any(_matching(row, name) for row in catalog.get("models", []) if isinstance(row, dict)):
        raise ConnectionError("Codex catalog readback failed.")
    slug = next(row["slug"] for row in catalog["models"] if isinstance(row, dict) and _matching(row, name))
    try:
        with urlopen(proxy_url.rstrip("/").removesuffix("/v1") + "/v1/models", timeout=4) as response:
            live = json.load(response)
        visible = any(row.get("id") == slug for row in live.get("data", []))
    except (OSError, ValueError):
        visible = False
    return (f"Codex proxy lists {slug}; restart Codex and run a native tool test to confirm use." if visible else
            f"Codex catalog prepared: {slug}. Proxy/Codex reload and live test required.")


def disconnect_model(name: str) -> str:
    root = discover_root()
    paths = _paths(root, codex=True)
    raw, values = _read(paths)
    for path, value in values.items():
        if isinstance(value, dict) and isinstance(value.get("models"), list):
            value["models"] = [row for row in value["models"] if not (isinstance(row, dict) and _matching(row, name))]
        elif isinstance(value, list):
            value[:] = [row for row in value if row != name and not (isinstance(row, dict) and row.get("name") == name)]
        elif isinstance(value, dict) and isinstance(value.get("providers", {}).get("ollama"), dict):
            provider = value["providers"]["ollama"]
            for key in ("models", "selectedModels"):
                if isinstance(provider.get(key), list):
                    provider[key] = [item for item in provider[key] if item != name]
            for key in ("modelReasoningEfforts", "modelDefaultReasoningEfforts", "modelContextWindows",
                        "modelAutoCompactTokenLimits", "modelSupportsReasoningSummaries", "modelReasoningEffortMap"):
                if isinstance(provider.get(key), dict):
                    provider[key].pop(name, None)
    backup = _commit(raw, values, "disconnect")
    return f"OpenCodex/Codex references removed for {name}. Backup: {backup or 'unchanged'}"


def delete_connected_model(name: str, client) -> str:
    # Preflight all registration files before the irreversible Ollama removal.
    root = discover_root()
    _read(_paths(root, codex=True))
    if name not in {model.name for model in client.list_models()}:
        raise ConnectionError("Selected model is no longer installed in Ollama; connections were left unchanged.")
    if hasattr(client, "ps") and any(row.get("name") == name or row.get("model") == name
                                      for row in client.ps()):
        raise ConnectionError("Selected model is loaded in Ollama. Unload it after active work before deleting.")
    client.delete(name)
    if name in {model.name for model in client.list_models()}:
        raise ConnectionError("Ollama still lists the model; connection files were left unchanged.")
    return disconnect_model(name)
