"""Paths, API-key resolution and user-tunable defaults."""
from __future__ import annotations

import os
from pathlib import Path

APP = "gpuctl"

CONFIG_DIR = Path(os.environ.get("GPUCTL_CONFIG_DIR", Path.home() / ".config" / APP))
STATE_DIR = Path(os.environ.get("GPUCTL_STATE_DIR", Path.home() / ".local" / "state" / APP))
STATE_FILE = STATE_DIR / "deployments.json"

OPENCODE_GLOBAL_CONFIG = Path.home() / ".config" / "opencode" / "opencode.json"

# Where we look for the Vast API key, in order.
_KEY_FILES = (
    CONFIG_DIR / "api_key",
    Path.home() / ".config" / "vastai" / "vast_api_key",
    Path.home() / ".vast_api_key",
)

VAST_API_BASE = "https://console.vast.ai/api/v0"
# Instance *listing* moved to v1; everything else we use is still on v0.
INSTANCES_V1_URL = "https://console.vast.ai/api/v1/instances/"

# Safety rails. Renting GPUs spends real money every hour, so these are
# deliberately conservative and every one of them is overridable per-launch.
DEFAULT_MAX_DPH = 2.00        # refuse any offer above this $/hr
DEFAULT_TTL_HOURS = 3.0       # auto-destroy deadline written at launch time
DEFAULT_DISK_GB = 150         # a 70B Q4 download plus HF cache needs room
DEFAULT_PORT = 8000           # vLLM's OpenAI-compatible server


class ConfigError(RuntimeError):
    """Raised when the tool is not set up well enough to run."""


def ensure_dirs() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def api_key(required: bool = True) -> str | None:
    """Resolve the Vast.ai API key from env, then from known key files."""
    key = os.environ.get("VAST_API_KEY", "").strip()
    if key:
        return key
    for path in _KEY_FILES:
        try:
            key = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if key:
            return key
    if required:
        raise ConfigError(
            "No Vast.ai API key found.\n"
            "  Set VAST_API_KEY=... in your environment, or write the key to\n"
            f"  {_KEY_FILES[0]}\n"
            "  Get one at https://cloud.vast.ai/manage-keys/"
        )
    return None


def save_api_key(key: str) -> Path:
    ensure_dirs()
    path = _KEY_FILES[0]
    path.write_text(key.strip() + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path
