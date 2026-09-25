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
    STALLED = "stalled"       # container up but showing no sign of progress
    LINKED = "linked"         # written into opencode's config
    STOPPED = "stopped"
    ERROR = "error"
    GONE = "gone"             # destroyed, or no longer on the account

    @property
    def terminal(self) -> bool:
        return self in (Phase.LINKED, Phase.STOPPED, Phase.ERROR, Phase.GONE)

    @property
    def wasting_money(self) -> bool:
        """Billing, but will not become useful without intervention."""
        return self in (Phase.STALLED, Phase.ERROR)


PHASE_STYLE = {
    Phase.PENDING: "dim",
    Phase.LOADING: "yellow",
    Phase.RUNNING: "cyan",
    Phase.SERVING: "green",
    Phase.STALLED: "bold red",
    Phase.LINKED: "bold green",
    Phase.STOPPED: "dim red",
    Phase.ERROR: "bold red",
    Phase.GONE: "dim",
}


# How long a phase may show no observable progress before we call it stalled.
# Image pulls are quiet for a while but Vast streams layer progress through
# status_msg, so LOADING still moves; once the container is up, a healthy vLLM
# is always doing something visible (downloading, loading, or capturing graphs).
# Provisioning trouble worth reacting to, keyed on an unambiguous phrase Vast
# surfaces through status_msg. Each is specific enough that it cannot match
# benign setup output -- note the deliberate contrast with the earlier bug where
# a bare "error" substring matched the apt package name "liberror-perl".
#
# These are NOT treated as terminal. Vast retries its provisioning steps, and a
# host observed emitting "curl: (6) Could not resolve host: cloud.vast.ai"
# recovered on its own and went on to pull the image — condemning it would have
# destroyed a working box. Instead a match shortens the stall threshold, so a
# blip that resolves costs nothing while one that does not is caught in minutes
# rather than half an hour.
#
# curl's wording is "Could not resolve host: <h>"; apt's is "Could not resolve
# '<h>'", so requiring the literal word "host" keeps this to Vast's own step.
SUSPICIOUS_STATUS_PATTERNS: tuple[tuple[str, str], ...] = (
    ("could not resolve host",
     "container DNS failed to resolve; if it persists the box cannot reach HuggingFace either"),
    ("no space left on device", "the machine ran out of disk"),
    ("manifest unknown", "that image tag does not exist"),
    ("pull access denied", "the image is private, or the tag is wrong"),
    ("repository does not exist", "the image name is wrong"),
    ("toomanyrequests", "the registry rate-limited the image pull"),
)


def suspicious_status(status_msg: str | None) -> tuple[str, str] | None:
    """Recognise provisioning trouble. Returns (phrase, why) or None.

    Matching is narrow on purpose: status_msg also carries Vast's streamed
    image-build log, which is full of innocent words.
    """
    text = (status_msg or "").lower()
    for phrase, why in SUSPICIOUS_STATUS_PATTERNS:
        if phrase in text:
            return phrase, why
    return None


# Threshold used instead of the phase default once status_msg shows trouble.
SUSPICIOUS_STALL_AFTER = 480.0   # 8 min

# Minimum gap between log fetches while a stall is suspected.
LOG_CHECK_INTERVAL = 120.0

STALL_AFTER = {
    Phase.LOADING: 1800.0,   # 30 min
    Phase.RUNNING: 600.0,    # 10 min
    Phase.PENDING: 900.0,    # 15 min
}


def progress_marker(instance: dict[str, Any], phase: "Phase | None" = None) -> str:
    """Fingerprint of the signals that reflect the *workload* progressing.

    Deliberately excludes gpu_util, cpu_util and mem_usage. Vast caches that
    telemetry and refreshes it on its own schedule, so those values change
    merely because the host is alive — on a box observed genuinely hung, they
    sat frozen for a minute and then jumped, which would read as progress and
    reset the stall clock forever.

    What is left is honest: actual_status and status_msg move while Vast is
    pulling the image, and disk_usage grows while weights download. Once the
    container is up and the model is cached, neither moves — so the container
    log becomes the deciding signal (see _log_grew).
    """
    def num(key: str, places: int = 1) -> str:
        try:
            return f"{round(float(instance.get(key) or 0), places):.{places}f}"
        except (TypeError, ValueError):
            return "?"

    parts = [str(instance.get("actual_status") or ""), num("disk_usage")]
    # status_msg legitimately streams docker layer progress while the image is
    # being pulled, so it is real progress during LOADING. Once the container is
    # up it is a static banner that Vast nonetheless rewrites occasionally
    # (observed flapping between ".../ssh" and "..."), which would reset the
    # stall clock forever. Past LOADING, the log is the signal instead.
    if phase in (None, Phase.PENDING, Phase.LOADING):
        parts.append((str(instance.get("status_msg") or ""))[:200])
    return "|".join(parts)


@dataclass
class Snapshot:
    dep: Deployment
    phase: Phase
    instance: dict[str, Any] | None
    endpoint: str | None
    probe: Probe | None
    detail: str
    cost: float
    stalled_for: float = 0.0   # seconds without observable progress

    @property
    def actual_status(self) -> str:
        return (self.instance or {}).get("actual_status") or "-"

    @property
    def status_msg(self) -> str:
        return ((self.instance or {}).get("status_msg") or "").strip()


def snapshot(
    client: VastClient,
    dep: Deployment,
    *,
    deep: bool = True,
    detect_stall: bool = True,
    confirm_with_logs: bool = True,
) -> Snapshot:
    """Where a deployment is, with a stall check layered on top.

    The raw observation cannot tell a slow box from a wedged one — that needs
    two points in time. The progress marker is persisted on the deployment, so
    consecutive runs of any command supply those points even when nothing is
    actively watching.
    """
    snap = _observe(client, dep, deep=deep)
    if not detect_stall:
        return snap

    limit = STALL_AFTER.get(snap.phase)
    if limit is None or snap.instance is None:
        return snap

    # Provisioning trouble does not condemn a box -- Vast retries and hosts do
    # recover -- but it does mean we should stop waiting sooner.
    trouble = suspicious_status(snap.instance.get("status_msg"))
    if trouble:
        limit = min(limit, SUSPICIOUS_STALL_AFTER)

    marker = progress_marker(snap.instance, snap.phase)
    now = time.time()
    if marker != dep.progress_marker or not dep.progress_at:
        # Something moved (or this is the first look): reset the clock.
        dep.progress_marker = marker
        dep.progress_at = now
        save(dep)
        return snap

    stalled_for = now - dep.progress_at
    snap.stalled_for = stalled_for

    # Past the halfway mark, start checking the container log — the one signal a
    # working vLLM always produces. Rate-limited so a tight `watch` loop does not
    # pay for a log fetch on every poll.
    if confirm_with_logs and stalled_for >= limit / 2:
        if now - dep.log_checked_at >= LOG_CHECK_INTERVAL:
            grew, size = _log_grew(client, dep)
            dep.log_checked_at = now
            if size:
                dep.log_size = size
            if grew:
                dep.progress_at = now
                snap.stalled_for = 0.0
                save(dep)
                return snap
            save(dep)

    if stalled_for < limit:
        return snap

    snap.phase = Phase.STALLED
    mins = stalled_for / 60
    detail = f"no progress for {mins:.0f}m"
    if trouble:
        detail += f"; {trouble[1]}"
    elif confirm_with_logs and dep.log_size:
        detail += "; container log not growing either"
    snap.detail = f"{detail} (was: {snap.detail})" if snap.detail else detail
    return snap


def _log_grew(client: VastClient, dep: Deployment) -> tuple[bool, int]:
    """Has the container log grown since we last looked? (grew, current_size)."""
    try:
        text = client.logs(dep.instance_id, tail=2000)
    except Exception:
        # A log fetch failure is not evidence of progress.
        return False, 0
    size = len(text or "")
    if not size:
        return False, 0
    if not dep.log_size:
        # First measurement is a baseline, not evidence either way. Report no
        # growth but let the caller store the size; the stall threshold has not
        # necessarily elapsed yet, so this does not by itself condemn the box.
        return False, size
    return size > dep.log_size, size


def _observe(client: VastClient, dep: Deployment, *, deep: bool = True) -> Snapshot:
    """One raw observation of where a deployment actually is."""
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
        if dep.served_at is None:
            # First confirmed serve: the success marker for the ledger.
            dep.served_at = time.time()
            save(dep)
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


def link_conductor(
    dep: Deployment,
    *,
    path=None,
    also_review: bool = False,
):
    """Point Conductor's default model at this deployment.

    Conductor discovers OpenCode models from opencode's own provider config, so
    this only makes sense once the deployment is linked there.
    """
    from . import conductor

    if not (dep.linked_at and dep.opencode_provider):
        raise RuntimeError(
            f"instance {dep.instance_id} is not linked into opencode yet; "
            f"run `gpuctl link {dep.instance_id}` first."
        )
    model_id = dep.notes.get("linked_model_id") or dep.served_name
    ref = conductor.model_ref(dep.opencode_provider, model_id)
    updates = {"default": ref}
    if also_review:
        updates["review"] = ref

    edit = conductor.set_models(updates, path=path)
    dep.conductor_target = str(edit.path)
    # Keep the first-seen previous values: re-pointing twice must not record
    # our own value as the thing to restore.
    for key, was in edit.previous.items():
        dep.conductor_prev.setdefault(key, was)
    save(dep)
    return edit


def unlink_conductor(dep: Deployment) -> list[str]:
    """Restore whatever Conductor's [models] keys were before we changed them."""
    from pathlib import Path as _Path

    from . import conductor

    if not dep.conductor_target or not dep.conductor_prev:
        return []
    restored = conductor.revert(dict(dep.conductor_prev), path=_Path(dep.conductor_target))
    dep.conductor_target = ""
    dep.conductor_prev = {}
    save(dep)
    return restored


def _context_from_args(dep: Deployment) -> int | None:
    from .recipes import all_recipes

    recipe = all_recipes().get(dep.recipe)
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
