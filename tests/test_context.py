"""Context window advertised to opencode must match the GPU and model in use."""
import time

import pytest

from gpuctl import health, state, track
from gpuctl.state import Deployment
from gpuctl.track import ContextChoice, Snapshot, resolve_context

L70 = "casperhansen/llama-3.3-70b-instruct-awq"


def dep(model=L70, recipe="build-a", **kw):
    d = Deployment(instance_id=1, recipe=recipe, model=model, served_name="x",
                   offer_id=1, port=8000, serve_key="k", created_at=time.time(),
                   ttl_hours=3.0, dph_at_launch=1.0, gpu_label="g", **kw)
    state.save(d)
    return d


def snap(d, *, server_ctx=None, gpu_ram=24576, gpus=2):
    probe = None
    if server_ctx is not None:
        probe = health.Probe(True, True, ["m"], "ok", {"m": server_ctx})
    elif server_ctx is None:
        probe = health.Probe(True, True, ["m"], "ok", {})
    inst = {"gpu_ram": gpu_ram, "num_gpus": gpus} if gpu_ram else None
    return Snapshot(d, track.Phase.SERVING, inst, "http://x", probe, "", 0.0)


# ------------------------------------------------------------ the server wins


def test_server_reported_length_wins():
    """vLLM may boot with less than the recipe asked for; it knows what fit."""
    got = resolve_context(dep(), snap(dep(), server_ctx=8192))
    assert got.tokens == 8192
    assert "server" in got.source


def test_server_value_beats_a_larger_recipe_request():
    """build-a asks for 14336; if the server only managed 6144, say 6144."""
    got = resolve_context(dep(), snap(dep(), server_ctx=6144))
    assert got.tokens == 6144


# ------------------------------------------------- fallbacks when it is silent


def test_falls_back_to_the_recipe_value():
    d = dep(recipe="build-a")           # --max-model-len 14336
    got = resolve_context(d, snap(d, gpu_ram=None))
    assert got.tokens == 14336
    assert "recipe" in got.source


def test_non_numeric_max_model_len_is_not_mistaken_for_a_number():
    """mimo-v2.6-pro passes `--max-model-len auto`, which must not become 32768
    by accident — there is simply no recipe figure to use."""
    assert track._recipe_context(dep(recipe="mimo-v2.6-pro")) is None


def test_vram_capacity_caps_an_over_optimistic_recipe():
    """A recipe asking 32768 on a card that can only hold less gets the lower."""
    d = dep(recipe="build-f")           # asks 32768, fp16 KV
    # 2x24 GiB = 48 GiB: 37.1 GiB of weights leaves only a few GiB of KV.
    got = resolve_context(d, snap(d, gpu_ram=24576, gpus=2))
    assert got.tokens < 32768
    assert "GPU" in got.source


def test_roomy_gpu_keeps_the_recipe_value():
    d = dep(recipe="build-f")           # asks 32768
    got = resolve_context(d, snap(d, gpu_ram=98304, gpus=4))   # 384 GiB
    assert got.tokens == 32768
    assert "recipe" in got.source


def test_fp8_kv_doubles_the_computed_capacity():
    """build-a uses --kv-cache-dtype fp8; halving KV per token doubles context."""
    a = dep(recipe="build-a")            # fp8 KV
    f = dep(recipe="build-f")            # fp16 KV
    cap_fp8 = track._vram_context(a, {"gpu_ram": 24576, "num_gpus": 2})
    cap_fp16 = track._vram_context(f, {"gpu_ram": 24576, "num_gpus": 2})
    assert cap_fp8 == pytest.approx(cap_fp16 * 2, rel=0.01)


def test_unknown_model_cannot_be_sized_from_vram():
    d = dep(model="someone/unlisted-model", recipe="build-a")
    assert track._vram_context(d, {"gpu_ram": 24576, "num_gpus": 2}) is None


def test_last_resort_default_is_labelled_as_such():
    d = dep(model="someone/unlisted-model", recipe="smoke")
    # smoke asks 16384, so remove that too by using a recipe-less deployment
    d = dep(model="someone/unlisted-model", recipe="does-not-exist")
    got = resolve_context(d, snap(d, gpu_ram=None))
    assert got.tokens == 32768
    assert "default" in got.source


def test_architectural_maximum_is_never_exceeded():
    """A server reporting more than the model supports is still capped."""
    d = dep(recipe="build-a")            # llama70b max_context 131072
    got = resolve_context(d, snap(d, server_ctx=999_999))
    assert got.tokens == 131072
    assert "architectural" in got.source


# -------------------------------------------------------------- output budget


def test_reasoning_models_get_a_bigger_output_share():
    """Qwen3.8 spent 100% of a 400-token budget thinking before answering."""
    reasoning = dep(recipe="qwen3.8-27b", model="Qwen/Qwen3.8-27B-FP8")
    plain = dep(recipe="build-a")
    r = resolve_context(reasoning, snap(reasoning, server_ctx=65536))
    p = resolve_context(plain, snap(plain, server_ctx=65536))
    assert r.output > p.output
    assert r.output == 32768 and p.output == 16384


def test_output_never_exceeds_context():
    d = dep(recipe="build-a")
    got = resolve_context(d, snap(d, server_ctx=2048))
    assert got.output <= got.tokens


def test_output_has_a_floor():
    d = dep(recipe="build-a")
    got = resolve_context(d, snap(d, server_ctx=8192))
    assert got.output >= 2048
