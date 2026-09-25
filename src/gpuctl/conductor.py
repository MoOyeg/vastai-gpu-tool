"""Point Conductor's default model at a gpuctl-served instance.

Conductor (conductor.build) keeps personal settings in ~/.conductor/settings.toml
and, for the OpenCode harness, "asks OpenCode for the models available to your
provider configuration". So a deployment that gpuctl has already linked into
opencode.json is visible to Conductor with no further work; what remains is
telling Conductor to *use* it, via the provider-qualified id in `[models]`:

    [models]
    default = "vast-50546033/llama-3.3-70b-instruct-awq"

Editing strategy: `tomllib` is read-only, and re-serialising with a writer would
discard the comments, key order and quoting style of a file the user maintains
by hand. So this makes a surgical single-line edit to the text and then verifies
it by re-parsing and diffing against the expected document -- if anything other
than the intended key moved, the write is rolled back.
"""
from __future__ import annotations

import re
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONDUCTOR_DIR = Path.home() / ".conductor"
SETTINGS_PATH = CONDUCTOR_DIR / "settings.toml"

# Keys under [models] this feature is allowed to touch.
MANAGED_KEYS = ("default", "review")


class ConductorError(RuntimeError):
    """Conductor's settings file is missing, unreadable, or unsafe to edit."""


@dataclass
class ConductorView:
    path: Path
    exists: bool
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def models(self) -> dict[str, Any]:
        m = self.data.get("models")
        return m if isinstance(m, dict) else {}

    def get(self, key: str) -> str | None:
        v = self.models.get(key)
        return v if isinstance(v, str) else None

    @property
    def opencode_executable(self) -> str | None:
        v = self.data.get("opencode_executable_path")
        return v if isinstance(v, str) else None


@dataclass
class ConductorEdit:
    path: Path
    backup: Path | None
    changed: dict[str, str]           # key -> new value
    previous: dict[str, str | None]   # key -> value before the edit (None = absent)


def read(path: Path | None = None) -> ConductorView:
    path = path or SETTINGS_PATH
    if not path.exists():
        return ConductorView(path, False)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConductorError(f"{path} is not readable TOML: {exc}") from None
    return ConductorView(path, True, data)


def _toml_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


_HEADER = re.compile(r"^\s*\[")


def _table_span(lines: list[str], table: str) -> tuple[int, int] | None:
    """Line range holding `table`'s own keys, excluding any sub-tables.

    [models] ends where the next header begins -- including [models.codex],
    whose keys belong to the sub-table, not to [models].
    """
    want = re.compile(rf"^\s*\[\s*{re.escape(table)}\s*\]")
    start = next((i for i, ln in enumerate(lines) if want.match(ln)), None)
    if start is None:
        return None
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if _HEADER.match(lines[i]):
            end = i
            break
    return start, end


def _set_key(text: str, table: str, key: str, value: str) -> str:
    """Replace or insert `key = value` inside `table`, touching nothing else."""
    lines = text.splitlines()
    rendered = f"{key} = {_toml_str(value)}"
    span = _table_span(lines, table)

    if span is None:
        body = "" if not lines or lines[-1].strip() == "" else "\n"
        return text.rstrip("\n") + f"{body}\n[{table}]\n{rendered}\n"

    start, end = span
    # Capture indentation and any trailing comment so a user's annotation on the
    # line survives the value change.
    assign = re.compile(
        rf"^(\s*)(?:{re.escape(key)}|\"{re.escape(key)}\")\s*="
        r"\s*(?:\"(?:[^\"\\]|\\.)*\"|'[^']*'|[^#\n]*?)\s*(#.*)?$"
    )
    for i in range(start + 1, end):
        m = assign.match(lines[i])
        if m:
            trailing = f"  {m.group(2)}" if m.group(2) else ""
            lines[i] = f"{m.group(1)}{rendered}{trailing}"
            break
    else:
        # Append after the table's existing keys, skipping back over the blank
        # lines that separate it from the next section.
        at = end
        while at > start + 1 and not lines[at - 1].strip():
            at -= 1
        lines.insert(at, rendered)
    return "\n".join(lines) + "\n"


def _expected(data: dict[str, Any], updates: dict[str, str]) -> dict[str, Any]:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in data.items()}
    models = dict(out.get("models") or {})
    models.update(updates)
    out["models"] = models
    return out


def set_models(
    updates: dict[str, str],
    *,
    path: Path | None = None,
    create: bool = True,
) -> ConductorEdit:
    """Set keys under [models], preserving the rest of the file verbatim."""
    path = path or SETTINGS_PATH
    bad = set(updates) - set(MANAGED_KEYS)
    if bad:
        raise ConductorError(f"refusing to write unmanaged key(s) {sorted(bad)}")
    if not updates:
        raise ConductorError("nothing to set")

    view = read(path)
    if not view.exists:
        if not create:
            raise ConductorError(
                f"{path} does not exist. Is Conductor installed? "
                f"Pass --create to write a new settings file."
            )
        original = '"$schema" = "https://conductor.build/schemas/settings.schema.json"\n'
        path.parent.mkdir(parents=True, exist_ok=True)
        base: dict[str, Any] = {"$schema": "https://conductor.build/schemas/settings.schema.json"}
        backup = None
    else:
        original = path.read_text(encoding="utf-8")
        base = view.data
        backup = path.with_name(f"{path.stem}.{int(time.time())}.bak")
        backup.write_text(original, encoding="utf-8")

    updated = original
    for key, value in updates.items():
        updated = _set_key(updated, "models", key, value)

    # Verify the surgical edit did exactly what was intended and nothing more.
    try:
        reparsed = tomllib.loads(updated)
    except tomllib.TOMLDecodeError as exc:
        raise ConductorError(f"edit produced invalid TOML, not written: {exc}") from None
    want = _expected(base, updates)
    if reparsed != want:
        raise ConductorError(
            "edit would have changed more than the requested keys, not written. "
            f"Expected {want}, got {reparsed}."
        )

    path.write_text(updated, encoding="utf-8")
    return ConductorEdit(
        path=path,
        backup=backup,
        changed=dict(updates),
        previous={k: (base.get("models") or {}).get(k) for k in updates},
    )


def revert(previous: dict[str, str | None], *, path: Path | None = None) -> list[str]:
    """Restore keys to what they were before `set_models`.

    A key that did not exist before is removed rather than blanked, so the file
    returns to its original shape.
    """
    path = path or SETTINGS_PATH
    view = read(path)
    if not view.exists:
        return []

    restored: list[str] = []
    text = path.read_text(encoding="utf-8")
    to_set = {k: v for k, v in previous.items() if isinstance(v, str)}
    to_drop = [k for k, v in previous.items() if v is None]

    for key, value in to_set.items():
        text = _set_key(text, "models", key, value)
        restored.append(key)
    if to_drop:
        lines = text.splitlines()
        span = _table_span(lines, "models")
        if span:
            start, end = span
            keep = []
            for i, ln in enumerate(lines):
                if start < i < end and any(
                    re.match(rf"^\s*(?:{re.escape(k)}|\"{re.escape(k)}\")\s*=", ln)
                    for k in to_drop
                ):
                    restored.append(next(k for k in to_drop if re.match(
                        rf"^\s*(?:{re.escape(k)}|\"{re.escape(k)}\")\s*=", ln)))
                    continue
                keep.append(ln)
            text = "\n".join(keep) + "\n"

    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConductorError(f"revert produced invalid TOML, not written: {exc}") from None
    path.write_text(text, encoding="utf-8")
    return restored


def model_ref(provider_id: str, model_id: str) -> str:
    """Conductor and opencode both address models as provider/model."""
    return f"{provider_id}/{model_id}"
