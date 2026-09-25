"""The sizing arithmetic in docs/METHOD.md, pinned."""
from gpuctl.models import GIB, MODELS, get


def test_kv_per_token_matches_hand_derivation():
    """2 x 80 layers x 8 kv_heads x 128 head_dim x 2 bytes = 320 KiB (METHOD §3)."""
    assert get("llama70b").kv_bytes_per_token == 327_680


def test_tensor_parallel_divisibility_rule():
    """64 attention heads / 8 KV heads => 1,2,4,8 valid; 3 is not (METHOD §5)."""
    m = get("llama70b")
    assert [n for n in range(1, 9) if m.valid_tp(n)] == [1, 2, 4, 8]
    assert not m.valid_tp(3)


def test_fp8_kv_halves_the_cache():
    m = get("llama70b")
    assert m.kv_gib(32768, fp8_kv=True) == m.kv_gib(32768) / 2


def test_gb_vs_gib_conversion():
    """39.8 decimal GB of weights is 37.1 GiB (METHOD §4)."""
    m = get("llama70b")
    assert m.weights_gb == 39.8
    assert round(m.weights_gib, 1) == 37.1


def test_moe_reads_only_active_weights():
    """An MoE's decode speed follows its active parameters, not its total."""
    moe = get("gpt-oss-120b")
    assert moe.bytes_read_per_token < moe.weights_gb / 10
    dense = get("llama70b")
    assert dense.bytes_read_per_token == dense.weights_gb


def test_compute_capability_requirements():
    """VRAM alone is not enough; the kernels need the right silicon (METHOD §6)."""
    assert get("llama70b").required_compute_cap == 800       # AWQ Marlin: Ampere+
    assert get("mistral-24b").required_compute_cap == 800    # bf16: Ampere+
    assert get("gpt-oss-120b").required_compute_cap == 900   # MXFP4: Hopper+


def test_every_model_declares_a_tool_parser():
    for key, m in MODELS.items():
        assert m.tool_parser, f"{key} would be unusable by an agent client"
        assert m.tool_args[0] == "--enable-auto-tool-choice"


def test_vram_need_grows_with_context():
    m = get("llama70b")
    assert m.vram_gib(4096) < m.vram_gib(32768)
    # weights + overhead + KV, in GiB
    expected = m.weights_gib + 2.0 + 32768 * 327_680 / GIB
    assert abs(m.vram_gib(32768) - expected) < 0.01
