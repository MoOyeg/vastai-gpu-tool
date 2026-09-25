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
    # Last observed sign of forward progress, and when we saw it. Persisted so
    # any command (ps, watch, reap) can spot a stall — the failure mode that
    # matters most is a box that hangs while nobody is watching.
    progress_marker: str = ""
    progress_at: float = 0.0
    log_size: int = 0          # last observed container-log length
    log_checked_at: float = 0.0  # when we last paid for a log fetch
    conductor_target: str = ""
    # [models] keys we overwrote, and their prior values (None = key was absent).
    conductor_prev: dict[str, Any] = field(default_factory=dict)
    endpoint: str = ""
    linked_at: float | None = None
    served_at: float | None = None   # first time /v1/models answered — the success marker
    destroyed_at: float | None = None
    final_cost: float | None = None  # spend frozen at teardown
    end_reason: str = ""             # manual | ttl | stalled | error
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

    def ran_seconds(self, now: float | None = None) -> float:
        """Billable lifetime: creation until teardown, or until now if still up.

        The clamp matters. Without it a destroyed instance keeps accruing
        forever — a box that ran 34 minutes was reporting $353 of spend two
        weeks later.
        """
        end = self.destroyed_at or (now or time.time())
        return max(0.0, end - self.created_at)

    def age_hours(self, now: float | None = None) -> float:
        return self.ran_seconds(now) / 3600.0

    def accrued_cost(self, now: float | None = None) -> float:
        """Spend estimate: billable hours x the rate agreed at launch.

        Vast bills from creation, including the image download, so provisioning
        time counts. Once torn down the figure is frozen in `final_cost` so the
        ledger stays stable.
        """
        if self.final_cost is not None:
            return self.final_cost
        return self.age_hours(now) * self.dph_at_launch

    @property
    def first_served_at(self) -> float | None:
        """When the model first answered.

        `linked_at` is a sound fallback for records written before `served_at`
        existed: a deployment is only ever linked into opencode *after* its
        /v1/models probe succeeded, so a link timestamp proves it served.
        """
        return self.served_at or self.linked_at

    @property
    def succeeded(self) -> bool:
        """Did this instance ever actually serve a model?

        `notes["linked_model_id"]` is the durable witness. It is written only by
        link_opencode, which runs only after a /v1/models probe succeeded — and
        unlike `linked_at`, teardown does not clear it. (Teardown *does* clear
        linked_at, since that means "currently linked", which is why it cannot
        be the sole record.)
        """
        return (self.first_served_at is not None
                or bool(self.notes.get("linked_model_id")))

    @property
    def time_to_serve(self) -> float | None:
        """Seconds from renting to the model answering — provisioning latency."""
        served = self.first_served_at
        if served is None:
            return None
        return max(0.0, served - self.created_at)

    @property
    def outcome(self) -> str:
        if not self.destroyed_at:
            return "serving" if self.succeeded else "starting"
        if self.succeeded:
            return "served"
        return f"never served ({self.end_reason})" if self.end_reason else "never served"

    def close(self, reason: str, now: float | None = None) -> None:
        """Freeze the record at teardown."""
        now = now or time.time()
        self.destroyed_at = now
        self.final_cost = self.ran_seconds(now) / 3600.0 * self.dph_at_launch
        self.end_reason = reason


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
