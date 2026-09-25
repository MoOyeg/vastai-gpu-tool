"""Readiness probing for the remote vLLM server."""
from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class Probe:
    reachable: bool
    serving: bool
    model_ids: list[str]
    detail: str = ""

    @property
    def best_model(self) -> str | None:
        return self.model_ids[0] if self.model_ids else None


def probe(endpoint: str, serve_key: str, timeout: float = 6.0) -> Probe:
    """Ask the OpenAI-compatible server what it is serving.

    `/v1/models` answering with at least one model is the real readiness
    signal: the container can be 'running' on Vast for many minutes while
    vLLM is still pulling weights and warming CUDA graphs.
    """
    url = endpoint.rstrip("/") + "/v1/models"
    headers = {"Authorization": f"Bearer {serve_key}"}
    try:
        r = httpx.get(url, headers=headers, timeout=timeout)
    except httpx.HTTPError as exc:
        return Probe(False, False, [], f"{type(exc).__name__}: {exc}")
    if r.status_code == 401:
        return Probe(True, False, [], "401 - serving key mismatch")
    if r.status_code != 200:
        return Probe(True, False, [], f"HTTP {r.status_code}")
    try:
        payload = r.json()
    except ValueError:
        return Probe(True, False, [], "non-JSON from /v1/models")
    ids = [m.get("id") for m in payload.get("data", []) if m.get("id")]
    return Probe(True, bool(ids), ids, "ok" if ids else "no models listed yet")


def tool_call_smoke(
    endpoint: str, serve_key: str, model: str, timeout: float = 120.0
) -> tuple[bool, str]:
    """Verify the server can actually emit an OpenAI tool call.

    `/v1/models` answering is not enough for an agent client: vLLM only serves
    function calling when launched with --enable-auto-tool-choice and a
    --tool-call-parser. Without them this returns the server's own complaint,
    which is what opencode would otherwise hit on its first request.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
        "max_tokens": 128,
        "temperature": 0.0,
        "tool_choice": "auto",
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }],
    }
    try:
        r = httpx.post(endpoint.rstrip("/") + "/v1/chat/completions",
                       json=body, timeout=timeout,
                       headers={"Authorization": f"Bearer {serve_key}"})
    except httpx.HTTPError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if r.status_code != 200:
        return False, r.text[:220]
    try:
        msg = r.json()["choices"][0]["message"]
    except (ValueError, KeyError, IndexError) as exc:
        return False, f"unexpected response shape: {exc}"
    calls = msg.get("tool_calls") or []
    if not calls:
        return False, "server accepted tools but returned no tool_calls"
    fn = (calls[0].get("function") or {}).get("name", "?")
    return True, f"tool_calls ok (called {fn})"


def completion_smoke(endpoint: str, serve_key: str, model: str, timeout: float = 120.0) -> tuple[bool, str]:
    """One tiny chat completion — proves the model actually generates."""
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    try:
        r = httpx.post(
            url, json=body, timeout=timeout,
            headers={"Authorization": f"Bearer {serve_key}"},
        )
    except httpx.HTTPError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    try:
        text = r.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError) as exc:
        return False, f"unexpected response shape: {exc}"
    return True, (text or "").strip()
