"""Launch recipes.

Each recipe is one row of the rent-test matrix in docs/METHOD.md §8 — the
configurations worth measuring before committing to buying any of them.
`est_tokps` carries the *predicted* decode rate so `gpuctl bench` can be compared
against the estimate a hardware decision would otherwise rest on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Recipe:
    key: str
    title: str
    gpu_name: str
    num_gpus: int
    model: str
    vllm_args: list[str]
    disk_gb: int = 150
    max_dph: float = 2.00
    min_inet_down: int = 400          # Mbps; a 39 GB pull dominates cold start
    min_gpu_ram_mb: int | None = None
    min_cuda: float = 13.0   # vllm/vllm-openai:latest is CUDA 13; see planner.MIN_CUDA_FOR_LATEST
    est_tokps: str = "?"
    doc_ref: str = ""
    notes: str = ""
    query_extra: dict[str, Any] = field(default_factory=dict)

    tool_parser: str = ""   # see models.Model.tool_parser — required for agent clients

    @property
    def tool_args(self) -> list[str]:
        return (["--enable-auto-tool-choice", "--tool-call-parser", self.tool_parser]
                if self.tool_parser else [])

    @property
    def served_name(self) -> str:
        """Short, stable id used by vLLM and by the opencode provider block."""
        return self.model.split("/")[-1]

    def search_query(self, *, max_dph: float | None = None) -> dict[str, Any]:
        from .vast import normalize_gpu_name

        q: dict[str, Any] = {
            "gpu_name": {"eq": normalize_gpu_name(self.gpu_name)},
            "num_gpus": {"eq": self.num_gpus},
            "dph_total": {"lte": float(max_dph if max_dph is not None else self.max_dph)},
            "disk_space": {"gte": float(self.disk_gb)},
            "inet_down": {"gte": float(self.min_inet_down)},
            "reliability": {"gte": 0.97},
            # A GeForce host on an older driver dies with CUDA error 804:
            # the image's forward-compat libs only work on datacenter GPUs.
            "cuda_max_good": {"gte": float(self.min_cuda)},
        }
        if self.min_gpu_ram_mb:
            q["gpu_ram"] = {"gte": float(self.min_gpu_ram_mb)}
        q.update(self.query_extra)
        return q


# The 70B Q4 weights the doc sizes at ~39 GB.
_L70 = "casperhansen/llama-3.3-70b-instruct-awq"

RECIPES: dict[str, Recipe] = {
    "smoke": Recipe(
        key="smoke", tool_parser="hermes",
        title="Smoke test — 1x cheap 24 GB, 7B AWQ",
        gpu_name="RTX 3090",
        num_gpus=1,
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        vllm_args=["--max-model-len", "16384", "--gpu-memory-utilization", "0.90"],
        disk_gb=40,
        max_dph=0.30,
        min_inet_down=200,
        est_tokps="~80-120",
        doc_ref="n/a",
        notes="Exercises the whole pipeline for pennies. Run this first.",
    ),
    "build-a": Recipe(
        key="build-a", tool_parser="llama3_json",
        title="Build A — 2x RTX 3090, TP=2, 70B Q4",
        gpu_name="RTX 3090",
        num_gpus=2,
        model=_L70,
        vllm_args=[
            "--tensor-parallel-size", "2",
            # At fp16 KV a 48 GiB rig leaves ~5 GiB for cache (~16k ctx);
            # fp8 KV halves KV to 160 KiB/token and roughly doubles that.
            "--kv-cache-dtype", "fp8",
            "--max-model-len", "14336",
            "--gpu-memory-utilization", "0.92",
        ],
        max_dph=0.70,
        est_tokps="~18-22",
        doc_ref="METHOD §8",
        notes="Is 48 GB usable at fp8 KV, and does awq_marlin engage?",
    ),
    "build-b": Recipe(
        key="build-b", tool_parser="llama3_json",
        title="Build B — 2x RTX 4090, TP=2, 70B Q4",
        gpu_name="RTX 4090",
        num_gpus=2,
        model=_L70,
        vllm_args=[
            "--tensor-parallel-size", "2",
            "--kv-cache-dtype", "fp8",
            "--max-model-len", "14336",
            "--gpu-memory-utilization", "0.92",
        ],
        max_dph=1.00,
        est_tokps="~20-25",
        doc_ref="METHOD §8",
        notes="Is 2x 4090 really ~1.7x a 3090 pair, or ~1.1x? Bandwidth says the latter.",
    ),
    "build-f": Recipe(
        key="build-f", tool_parser="llama3_json",
        title="Build F+ — 4x RTX 3090, TP=4, 70B Q4",
        gpu_name="RTX 3090",
        num_gpus=4,
        model=_L70,
        vllm_args=[
            "--tensor-parallel-size", "4",
            "--max-model-len", "32768",
            "--gpu-memory-utilization", "0.92",
        ],
        max_dph=1.30,
        est_tokps="~28-35",
        doc_ref="METHOD §8",
        notes="96 GB, fp16 KV, real 32k context. Fastest of the 3090 configurations.",
    ),
    "a6000": Recipe(
        key="a6000", tool_parser="llama3_json",
        title="1x RTX A6000 48 GB — long-context single card",
        gpu_name="RTX A6000",
        num_gpus=1,
        model=_L70,
        vllm_args=[
            "--max-model-len", "28672",
            "--gpu-memory-utilization", "0.92",
        ],
        max_dph=0.60,
        est_tokps="~11",
        doc_ref="METHOD §3",
        notes="Slow but ~28k ctx at fp16 KV on one card. Tests context-vs-speed directly.",
    ),
    "pro6000": Recipe(
        key="pro6000", tool_parser="openai",
        title="1x RTX PRO 6000 Blackwell 96 GB — the 120B MoE question",
        gpu_name="RTX PRO 6000 WS",
        num_gpus=1,
        model="openai/gpt-oss-120b",
        vllm_args=[
            "--max-model-len", "32768",
            "--gpu-memory-utilization", "0.92",
        ],
        disk_gb=250,
        max_dph=2.50,
        min_inet_down=800,
        min_gpu_ram_mb=90000,
        est_tokps="~28 dense / MoE ≫",
        doc_ref="METHOD §6",
        notes=(
            "Does the 120B MoE class actually run, and how fast? "
            "Three Blackwell variants exist (WS / S / Max-Q); none were rentable on 2026-09-10."
        ),
    ),
}


def get(key: str) -> Recipe:
    try:
        return RECIPES[key]
    except KeyError:
        raise KeyError(
            f"Unknown recipe {key!r}. Known: {', '.join(sorted(RECIPES))}"
        ) from None
