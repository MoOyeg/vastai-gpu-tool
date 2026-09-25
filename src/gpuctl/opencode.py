"""Wire a live Vast instance into opencode as an OpenAI-compatible provider.

opencode's ProviderConfig (schema: https://opencode.ai/config.json) takes an
`npm` adapter, `options.baseURL` / `options.apiKey`, and a `models` map. That
is exactly the shape already used for the local llama.cpp / MLX providers in
~/.config/opencode/opencode.json, so a rented box drops in beside them.

Every write backs the file up first and never touches providers we did not
create.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import OPENCODE_GLOBAL_CONFIG

NPM_ADAPTER = "@ai-sdk/openai-compatible"

# A rented box pulls 39-63 GB of weights and warms CUDA graphs; a first request
# can sit a long time before the first token. Keep opencode from giving up.
HEADER_TIMEOUT_MS = 900_000
CHUNK_TIMEOUT_MS = 300_000


class OpencodeError(RuntimeError):
    """The opencode config could not be read or safely updated."""


@dataclass
class LinkResult:
    path: Path
    provider_id: str
    backup: Path | None
    set_default: bool
    previous_default: str | None


def target_path(explicit: str | None = None, project: bool = False) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if project:
        return Path.cwd() / "opencode.json"
    return OPENCODE_GLOBAL_CONFIG


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"$schema": "https://opencode.ai/config.json"}
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return {"$schema": "https://opencode.ai/config.json"}
    try:
        return json.loads(raw)
    except ValueError as exc:
        # Refuse rather than clobber a file we cannot faithfully round-trip
        # (e.g. a .jsonc with comments).
        raise OpencodeError(
            f"{path} is not plain JSON we can safely rewrite ({exc}).\n"
            "Point --config at a plain .json file, or fix the file first."
        ) from exc


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    # Matches the opencode.<epoch>.bak convention already in that directory.
    backup = path.with_name(f"{path.stem}.{int(time.time())}.bak")
    backup.write_bytes(path.read_bytes())
    return backup


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def provider_block(
    *,
    display_name: str,
    endpoint: str,
    serve_key: str,
    model_id: str,
    context_len: int,
    max_output: int = 8192,
) -> dict[str, Any]:
    return {
        "npm": NPM_ADAPTER,
        "name": display_name,
        "options": {
            "baseURL": endpoint.rstrip("/") + "/v1",
            "apiKey": serve_key,
            "headerTimeout": HEADER_TIMEOUT_MS,
            "chunkTimeout": CHUNK_TIMEOUT_MS,
        },
        "models": {
            model_id: {
                "name": f"{model_id} @ {display_name}",
                "tool_call": True,
                "limit": {"context": context_len, "output": max_output},
            }
        },
    }


def link(
    *,
    path: Path,
    provider_id: str,
    block: dict[str, Any],
    model_id: str,
    set_default: bool,
) -> LinkResult:
    data = _load(path)
    backup = _backup(path)

    providers = data.setdefault("provider", {})
    if not isinstance(providers, dict):
        raise OpencodeError(f"{path}: 'provider' is not an object; refusing to edit.")

    previous_default = data.get("model") if isinstance(data.get("model"), str) else None
    providers[provider_id] = block

    if set_default:
        data["model"] = f"{provider_id}/{model_id}"

    _write(path, data)
    return LinkResult(path, provider_id, backup, set_default, previous_default)


def unlink(*, path: Path, provider_id: str, restore_model: str | None = None) -> bool:
    """Remove a provider we added. Returns True if something was removed."""
    if not path.exists():
        return False
    data = _load(path)
    providers = data.get("provider")
    if not isinstance(providers, dict) or provider_id not in providers:
        return False

    _backup(path)
    providers.pop(provider_id, None)

    # If we had made this provider the default, don't leave a dangling pointer.
    current = data.get("model")
    if isinstance(current, str) and current.startswith(f"{provider_id}/"):
        if restore_model:
            data["model"] = restore_model
        else:
            data.pop("model", None)

    if not providers:
        data.pop("provider", None)

    _write(path, data)
    return True


def linked_providers(path: Path, prefix: str = "vast-") -> list[str]:
    try:
        data = _load(path)
    except OpencodeError:
        return []
    providers = data.get("provider")
    if not isinstance(providers, dict):
        return []
    return sorted(p for p in providers if p.startswith(prefix))
