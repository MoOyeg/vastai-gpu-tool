"""Turn a recipe into a running, vLLM-serving Vast instance."""
from __future__ import annotations

import time
from typing import Any

from .config import DEFAULT_TTL_HOURS
from .recipes import Recipe
from .state import Deployment, new_serve_key
from .vast import VastClient, VastError

ONSTART_LIMIT = 4048  # Vast rejects longer onstart payloads.


def search(
    client: VastClient,
    recipe: Recipe,
    *,
    max_dph: float | None = None,
    limit: int = 20,
    order: list[list[str]] | None = None,
) -> list[dict[str, Any]]:
    query = recipe.search_query(max_dph=max_dph)
    # Validate against the live catalogue: a bad name matches nothing silently.
    query["gpu_name"] = {"eq": client.resolve_gpu_name(recipe.gpu_name)}
    return client.search_offers(
        query,
        limit=limit,
        order=order or [["dph_total", "asc"]],
        storage_gb=float(recipe.disk_gb),
    )


def offer_summary(offer: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": offer.get("id"),
        "gpu": f"{offer.get('num_gpus', '?')}x {offer.get('gpu_name', '?')}",
        "dph": float(offer.get("dph_total") or 0.0),
        "reliability": float(offer.get("reliability2") or offer.get("reliability") or 0.0),
        "inet_down": float(offer.get("inet_down") or 0.0),
        "disk": float(offer.get("disk_space") or 0.0),
        "cuda": offer.get("cuda_max_good"),
        "geo": offer.get("geolocation") or "?",
        "machine": offer.get("machine_id"),
    }


def build_env(
    *, port: int, serve_key: str, hf_token: str | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Vast's env map doubles as the Docker run-options map.

    A key of "-p 8000:8000" (value "1") is how a container port is requested;
    Vast then maps it to a random external port on the machine's public IP.
    """
    env: dict[str, str] = {
        f"-p {port}:{port}": "1",
        "VLLM_API_KEY": serve_key,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    if hf_token:
        env["HF_TOKEN"] = hf_token
    if extra:
        env.update(extra)
    return env


def build_onstart(recipe: Recipe, *, port: int, extra_args: list[str] | None = None) -> str:
    """The command Vast runs once the container is up.

    The serving key is referenced through $VLLM_API_KEY rather than baked in,
    so the secret never appears in the onstart text Vast stores.
    """
    args = [
        "vllm", "serve", recipe.model,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--served-model-name", recipe.served_name,
        "--api-key", '"$VLLM_API_KEY"',
        *recipe.vllm_args,
        *recipe.tool_args,
        *(extra_args or []),
    ]
    script = "\n".join(
        [
            "#!/bin/bash",
            "export HF_HOME=/workspace/hf",
            'mkdir -p "$HF_HOME" /var/log',
            "touch /var/log/vllm.log",
            f"echo \"[gpuctl] serving {recipe.model} on :{port}\" | tee -a /var/log/vllm.log",
            "nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a /var/log/vllm.log",
            " ".join(args) + " 2>&1 | tee -a /var/log/vllm.log",
        ]
    )
    if len(script) > ONSTART_LIMIT:
        raise VastError(
            f"onstart script is {len(script)} chars, over Vast's {ONSTART_LIMIT} limit."
        )
    return script


def launch(
    client: VastClient,
    recipe: Recipe,
    offer: dict[str, Any],
    *,
    port: int,
    disk_gb: int,
    ttl_hours: float = DEFAULT_TTL_HOURS,
    hf_token: str | None = None,
    extra_args: list[str] | None = None,
    label: str | None = None,
    image: str | None = None,
) -> Deployment:
    offer_id = int(offer["id"])
    serve_key = new_serve_key()
    # A recipe may pin its own image; only fall back to the default if it did not.
    image = image or recipe.image
    env = build_env(port=port, serve_key=serve_key, hf_token=hf_token,
                    extra=recipe.extra_env)
    onstart = build_onstart(recipe, port=port, extra_args=extra_args)
    label = label or f"gpuctl/{recipe.key}"

    instance_id = client.create_instance(
        offer_id,
        image=image,
        disk_gb=disk_gb,
        env=env,
        onstart=onstart,
        label=label,
        runtype="ssh",  # Vast injects sshd; the image needs no openssh-server.
    )

    summary = offer_summary(offer)
    return Deployment(
        instance_id=instance_id,
        recipe=recipe.key,
        model=recipe.model,
        served_name=recipe.served_name,
        offer_id=offer_id,
        port=port,
        serve_key=serve_key,
        created_at=time.time(),
        ttl_hours=ttl_hours,
        dph_at_launch=summary["dph"],
        gpu_label=summary["gpu"],
        label=label,
        notes={
            "geolocation": summary["geo"],
            "machine_id": summary["machine"],
            "inet_down_mbps": summary["inet_down"],
            "reliability": summary["reliability"],
            "est_tokps": recipe.est_tokps,
            "doc_ref": recipe.doc_ref,
            "disk_gb": disk_gb,
            "image": image,
        },
    )
