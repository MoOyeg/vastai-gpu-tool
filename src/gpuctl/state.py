"""Local record of what we have rented.

Vast is the source of truth for instance *state*; this file is the source of
truth for our *intent* — which model we asked for, the serving key we minted,
the auto-destroy deadline, and which opencode config we edited so we can undo
it cleanly on teardown.
"""
from __future__ import annotations

import json
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import STATE_FILE, ensure_dirs


@dataclass
class Deployment:
    instance_id: int
    recipe: str
    model: str
    served_name: str
    offer_id: int
    port: int
    serve_key: str
    created_at: float
    ttl_hours: float
    dph_at_launch: float
    gpu_label: str = ""
    label: str = ""
    opencode_provider: str = ""
    opencode_target: str = ""
    conductor_target: str = ""
    # [models] keys we overwrote, and their prior values (None = key was absent).
    conductor_prev: dict[str, Any] = field(default_factory=dict)
    endpoint: str = ""
    linked_at: float | None = None
    destroyed_at: float | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def provider_id(self) -> str:
        return self.opencode_provider or f"vast-{self.instance_id}"

    @property
    def deadline(self) -> float:
        return self.created_at + self.ttl_hours * 3600.0

    def expired(self, now: float | None = None) -> bool:
        if self.ttl_hours <= 0:
            return False
        return (now or time.time()) >= self.deadline

    def age_hours(self, now: float | None = None) -> float:
        return max(0.0, ((now or time.time()) - self.created_at) / 3600.0)

    def accrued_cost(self, now: float | None = None) -> float:
        """Best-effort spend estimate from our own launch clock.

        Vast bills from when the instance is created, including the image
        download, so this intentionally counts provisioning time too.
        """
        return self.age_hours(now) * self.dph_at_launch


def new_serve_key() -> str:
    return "sk-vast-" + secrets.token_urlsafe(24)


def _read_raw() -> dict[str, Any]:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"deployments": []}


def load_all(include_destroyed: bool = False) -> list[Deployment]:
    out: list[Deployment] = []
    known = set(Deployment.__dataclass_fields__)
    for row in _read_raw().get("deployments", []):
        try:
            dep = Deployment(**{k: v for k, v in row.items() if k in known})
        except TypeError:
            continue
        if dep.destroyed_at and not include_destroyed:
            continue
        out.append(dep)
    return sorted(out, key=lambda d: d.created_at)


def _write_all(deps: list[Deployment]) -> None:
    ensure_dirs()
    payload = {"version": 1, "deployments": [asdict(d) for d in deps]}
    tmp = Path(str(STATE_FILE) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(STATE_FILE)
    # The file holds serving keys.
    STATE_FILE.chmod(0o600)


def save(dep: Deployment) -> None:
    deps = load_all(include_destroyed=True)
    for i, existing in enumerate(deps):
        if existing.instance_id == dep.instance_id:
            deps[i] = dep
            break
    else:
        deps.append(dep)
    _write_all(deps)


def find(instance_id: int, include_destroyed: bool = False) -> Deployment | None:
    for dep in load_all(include_destroyed=include_destroyed):
        if dep.instance_id == instance_id:
            return dep
    return None


def resolve(ref: str | None) -> Deployment | None:
    """Resolve a user-supplied reference: an instance id, or a recipe name.

    With no reference at all, return the only live deployment if there is
    exactly one — the common case when you rent one box at a time.
    """
    live = load_all()
    if ref is None:
        return live[-1] if len(live) == 1 else None
    ref = ref.strip()
    if ref.isdigit():
        return find(int(ref))
    matches = [d for d in live if d.recipe == ref]
    return matches[-1] if len(matches) == 1 else None
