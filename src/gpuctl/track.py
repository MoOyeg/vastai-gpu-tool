"""Track a rented instance from 'paid for' to 'usable by opencode'."""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from . import opencode
from .health import Probe, probe
from .state import Deployment, save
from .vast import VastClient, endpoint_for


class Phase(str, Enum):
    PENDING = "pending"       # rented, not yet visible in the instance list
    LOADING = "loading"       # Vast is pulling the image / starting the container
    RUNNING = "running"       # container up, vLLM still loading weights
    SERVING = "serving"       # /v1/models answers - the real readiness signal
    LINKED = "linked"         # written into opencode's config
    STOPPED = "stopped"
    ERROR = "error"
    GONE = "gone"             # destroyed, or no longer on the account

    @property
    def terminal(self) -> bool:
        return self in (Phase.LINKED, Phase.STOPPED, Phase.ERROR, Phase.GONE)


PHASE_STYLE = {
    Phase.PENDING: "dim",
    Phase.LOADING: "yellow",
    Phase.RUNNING: "cyan",
    Phase.SERVING: "green",
    Phase.LINKED: "bold green",
    Phase.STOPPED: "dim red",
    Phase.ERROR: "bold red",
    Phase.GONE: "dim",
}


@dataclass
class Snapshot:
    dep: Deployment
    phase: Phase
    instance: dict[str, Any] | None
    endpoint: str | None
    probe: Probe | None
    detail: str
    cost: float

    @property
    def actual_status(self) -> str:
        return (self.instance or {}).get("actual_status") or "-"

    @property
    def status_msg(self) -> str:
        return ((self.instance or {}).get("status_msg") or "").strip()


def snapshot(client: VastClient, dep: Deployment, *, deep: bool = True) -> Snapshot:
    """One observation of where a deployment actually is."""
    cost = dep.accrued_cost()

    if dep.destroyed_at:
        return Snapshot(dep, Phase.GONE, None, None, None, "destroyed", cost)

    try:
        inst = client.get_instance(dep.instance_id)
    except Exception as exc:  # network hiccup - report, don't crash the watch
        return Snapshot(dep, Phase.PENDING, None, None, None, f"lookup failed: {exc}", cost)

    if inst is None:
        # Right after renting it can take a few seconds to appear; after a
        # destroy it never comes back. The caller's clock disambiguates.
        phase = Phase.PENDING if dep.age_hours() * 3600 < 120 else Phase.GONE
        return Snapshot(dep, phase, None, None, None, "not visible on the account yet", cost)

    # Vast bills from its own start_date; prefer it over our launch clock.
    start = inst.get("start_date")
    dph = float(inst.get("dph_total") or dep.dph_at_launch or 0.0)
    if isinstance(start, (int, float)) and start > 0:
        cost = max(0.0, (time.time() - float(start)) / 3600.0) * dph

    status = (inst.get("actual_status") or "").lower()
    status_msg = (inst.get("status_msg") or "").strip()

    if status in ("exited", "stopped"):
        return Snapshot(dep, Phase.STOPPED, inst, None, None, status_msg or status, cost)
    if status in ("error", "failed"):
        return Snapshot(dep, Phase.ERROR, inst, None, None, status_msg or status, cost)
    # NB: never infer failure from status_msg. During provisioning Vast streams
    # the image build log through that field, and apt package names such as
    # "liberror-perl" contain the substring "error". actual_status is the only
    # trustworthy signal; a container that really dies reports "exited".
    if status != "running":
        # created / loading / scheduling - image is coming down the wire.
        return Snapshot(dep, Phase.LOADING, inst, None, None, status_msg or status or "starting", cost)

    endpoint = endpoint_for(inst, dep.port)
    if not endpoint:
        return Snapshot(dep, Phase.RUNNING, inst, None, None, "waiting for port mapping", cost)

    if not deep:
        return Snapshot(dep, Phase.RUNNING, inst, endpoint, None, "container up", cost)

    p = probe(endpoint, dep.serve_key)
    if p.serving:
        phase = Phase.LINKED if dep.linked_at else Phase.SERVING
        return Snapshot(dep, phase, inst, endpoint, p, ", ".join(p.model_ids), cost)

    detail = "vLLM loading weights" if not p.reachable else p.detail
    return Snapshot(dep, Phase.RUNNING, inst, endpoint, p, detail, cost)


def link_opencode(
    snap: Snapshot,
    *,
    config_path,
    set_default: bool,
    context_len: int | None = None,
) -> opencode.LinkResult:
    """Write the serving instance into opencode's provider map."""
    dep = snap.dep
    if not snap.endpoint:
        raise RuntimeError("cannot link: no public endpoint yet")

    model_id = (snap.probe.best_model if snap.probe else None) or dep.served_name
    ctx = context_len or _context_from_args(dep) or 32768

    block = opencode.provider_block(
        display_name=f"Vast {dep.gpu_label or dep.recipe} ({dep.instance_id})",
        endpoint=snap.endpoint,
        serve_key=dep.serve_key,
        model_id=model_id,
        context_len=ctx,
    )
    result = opencode.link(
        path=config_path,
        provider_id=dep.provider_id,
        block=block,
        model_id=model_id,
        set_default=set_default,
    )

    dep.opencode_provider = result.provider_id
    dep.opencode_target = str(result.path)
    dep.endpoint = snap.endpoint
    dep.linked_at = time.time()
    dep.notes["linked_model_id"] = model_id
    if result.previous_default and set_default:
        dep.notes["previous_default_model"] = result.previous_default
    save(dep)
    return result


def _context_from_args(dep: Deployment) -> int | None:
    from .recipes import RECIPES

    recipe = RECIPES.get(dep.recipe)
    if not recipe:
        return None
    args = recipe.vllm_args
    if "--max-model-len" in args:
        try:
            return int(args[args.index("--max-model-len") + 1])
        except (IndexError, ValueError):
            return None
    return None


def unlink_opencode(dep: Deployment) -> bool:
    from pathlib import Path

    if not dep.opencode_target:
        return False
    removed = opencode.unlink(
        path=Path(dep.opencode_target),
        provider_id=dep.provider_id,
        restore_model=dep.notes.get("previous_default_model"),
    )
    if removed:
        dep.linked_at = None
        save(dep)
    return removed
