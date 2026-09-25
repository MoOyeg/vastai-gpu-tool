"""Given a model, find the cheapest hardware on Vast that can actually host it.

Fetches the market once and fits locally, rather than one search per
model/config pair - Vast rate-limits bursts (HTTP 429) and the whole matrix is
answerable from a single broad query.
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import GIB, Model
from .vast import VastClient

# vLLM's default headroom; the rest of the card is weights + KV.
GPU_MEM_UTIL = 0.92

# Decode efficiency factors -- see docs/METHOD.md §2. Applied uniformly across
# every configuration: mixing an optimistic figure for new hardware with a
# measured one for old hardware silently biases a comparison.
EFFICIENCY = {1: 0.55, 2: 0.40, 4: 0.33, 8: 0.30}

# vllm/vllm-openai:latest ships CUDA 13 / torch cu130. Its forward-compat libs
# only work on datacenter GPUs, so a GeForce host with an older driver dies with
# CUDA error 804. Require a driver new enough to run CUDA 13 natively.
# See docs/METHOD.md §6.
MIN_CUDA_FOR_LATEST = 13.0


@dataclass
class Fit:
    model: Model
    offer: dict
    num_gpus: int
    gpu_name: str
    vram_gib: float
    need_gib: float
    context: int
    fp8_kv: bool
    dph: float
    parallel: str
    est_tokps: float | None

    @property
    def headroom_gib(self) -> float:
        return self.vram_gib * GPU_MEM_UTIL - self.need_gib

    @property
    def max_context(self) -> int:
        """Largest context this offer could hold, given the weights."""
        spare = self.vram_gib * GPU_MEM_UTIL - self.model.weights_gib - (
            2.0 + 0.6 * (self.num_gpus - 1))
        per_token = self.model.kv_bytes_per_token / (2 if self.fp8_kv else 1)
        return max(0, int(spare * GIB / per_token))


def fetch_market(
    client: VastClient,
    *,
    min_disk_gb: float = 120,
    min_reliability: float = 0.95,
    min_inet_down: float = 300,
    min_cuda: float = MIN_CUDA_FOR_LATEST,
    limit: int = 800,
    verified_only: bool = True,
) -> list[dict]:
    q: dict = {
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "disk_space": {"gte": min_disk_gb},
        "reliability": {"gte": min_reliability},
        "inet_down": {"gte": min_inet_down},
        "cuda_max_good": {"gte": min_cuda},
    }
    if not verified_only:
        q["verified"] = {"eq": True} if verified_only else {"in": [True, False]}
    return client.search_offers(
        q, limit=limit, order=[["dph_total", "asc"]], storage_gb=float(min_disk_gb)
    )


def _vram_gib(offer: dict) -> float:
    """Total VRAM across the offer's GPUs. Vast reports gpu_ram in MiB."""
    per_gpu_mib = float(offer.get("gpu_ram") or 0)
    return per_gpu_mib * int(offer.get("num_gpus") or 1) / 1024.0


def estimate_tokps(model: Model, offer: dict, num_gpus: int) -> float | None:
    """Decode tok/s from the doc's bandwidth model. Treat as +/-25%."""
    bw = float(offer.get("gpu_mem_bw") or 0)      # GB/s per GPU
    if bw <= 0:
        return None
    eff = EFFICIENCY.get(num_gpus, 0.30)
    read = model.bytes_read_per_token
    if read <= 0:
        return None
    return (bw * num_gpus * eff) / read


def fit_model(
    model: Model,
    offers: list[dict],
    *,
    context: int,
    fp8_kv: bool = False,
    max_dph: float | None = None,
    allow_pipeline: bool = True,
) -> list[Fit]:
    """Every offer that can host `model` at `context`, cheapest first."""
    out: list[Fit] = []
    need_cc = model.required_compute_cap
    for o in offers:
        n = int(o.get("num_gpus") or 0)
        if n < 1:
            continue
        # A card that cannot run the quantisation's kernels is not a candidate,
        # however cheap it is.
        if int(o.get("compute_cap") or 0) < need_cc:
            continue
        if model.valid_tp(n):
            parallel = f"TP={n}" if n > 1 else "single"
        elif allow_pipeline and n > 1:
            # vLLM falls back to pipeline parallelism; works, but no
            # single-stream speedup (docs/METHOD.md §5).
            parallel = f"PP={n}"
        else:
            continue
        vram = _vram_gib(o)
        need = model.vram_gib(context, fp8_kv=fp8_kv, num_gpus=n)
        if vram * GPU_MEM_UTIL < need:
            continue
        dph = float(o.get("dph_total") or 0)
        if max_dph is not None and dph > max_dph:
            continue
        tokps = estimate_tokps(model, o, n) if parallel.startswith(("TP", "single")) else None
        out.append(Fit(model, o, n, str(o.get("gpu_name") or "?"), vram, need,
                       context, fp8_kv, dph, parallel, tokps))
    out.sort(key=lambda f: f.dph)
    return out


def cheapest(
    model: Model, offers: list[dict], *, context: int, fp8_kv: bool = False
) -> Fit | None:
    fits = fit_model(model, offers, context=context, fp8_kv=fp8_kv)
    return fits[0] if fits else None
