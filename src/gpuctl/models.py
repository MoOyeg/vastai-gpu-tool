"""Catalogue of servable open-weight models, with the arithmetic to size them.

Weight sizes are measured from the HuggingFace file listing (counting only the
top-level shards vLLM actually loads — repos like gpt-oss and Mistral also ship
`original/` and `consolidated.safetensors` copies that double the apparent
size). Architecture numbers come from each repo's config.json.

Units: everything internal is GiB. HuggingFace reports decimal GB, and Vast
reports gpu_ram in MiB, so conversions happen once, here, on the way in.
See KV_NOTE below - conflating the two is what makes 48 GB cards look worse
than they are.
"""
from __future__ import annotations

from dataclasses import dataclass, field

GIB = 1024 ** 3

KV_NOTE = (
    "GB vs GiB: HuggingFace reports decimal GB, NVIDIA and Vast report binary "
    "GiB. Llama-70B AWQ is 39.8 GB = 37.1 GiB; 2x3090 is 48 GiB = 51.5 GB. "
    "Sizing '39 GB' of weights against '48 GB' of VRAM suggests ~7k tokens of "
    "context; done consistently it is ~16k. See docs/METHOD.md §4."
)


@dataclass(frozen=True)
class Model:
    key: str
    hf_id: str
    label: str
    weights_gb: float          # decimal GB, measured from the HF file listing
    layers: int
    kv_heads: int
    head_dim: int
    attn_heads: int
    max_context: int
    quant: str                 # awq | mxfp4 | bf16 | fp8
    family: str = ""
    active_b: float | None = None   # MoE: active params/token, drives decode speed
    total_b: float | None = None    # MoE: total params, for the active fraction
    tool_call: bool = True
    # vLLM only exposes OpenAI-style function calling when BOTH
    # --enable-auto-tool-choice and a matching --tool-call-parser are passed.
    # Without them an agent client fails with:
    #   "auto" tool choice requires --enable-auto-tool-choice and
    #   --tool-call-parser to be set
    # Names are from `vllm serve --help=all` on vLLM 0.29.
    tool_parser: str = "hermes"
    min_compute_cap: int | None = None   # sm x100 (860 = sm_86); None = infer from quant
    notes: str = ""
    vllm_args: list[str] = field(default_factory=list)

    @property
    def weights_gib(self) -> float:
        return self.weights_gb * 1e9 / GIB

    @property
    def kv_bytes_per_token(self) -> int:
        """K and V, every layer, per KV head. fp16 KV cache.

        For Llama-70B this yields 2*80*8*128*2 = 327,680 B = 320 KiB/token,
        matching the hand derivation in docs/METHOD.md §3.
        """
        return 2 * self.layers * self.kv_heads * self.head_dim * 2

    def kv_gib(self, context: int, fp8_kv: bool = False) -> float:
        per_token = self.kv_bytes_per_token / (2 if fp8_kv else 1)
        return context * per_token / GIB

    def vram_gib(self, context: int, *, fp8_kv: bool = False, num_gpus: int = 1) -> float:
        """Weights + activations/CUDA graphs + KV cache, in GiB."""
        overhead = 2.0 + 0.6 * (num_gpus - 1)
        return self.weights_gib + overhead + self.kv_gib(context, fp8_kv)

    # What each quantisation actually needs from the silicon. Vast reports
    # compute_cap as sm x 100, so Ampere = 800/860, Ada = 890, Hopper = 900.
    _QUANT_MIN_CC = {
        "awq": 800,     # AWQ Marlin kernels want Ampere; sm_75 is unsupported/slow
        "bf16": 800,    # bf16 datapath is Ampere+
        "fp8": 890,     # native FP8 is Ada+
        "mxfp4": 900,   # Hopper/Blackwell. On Ampere vLLM dequantises to bf16,
                        # which blows past the VRAM budget entirely - this is
                        # exactly the risk docs/METHOD.md §6 describes.
    }

    @property
    def tool_args(self) -> list[str]:
        """Flags that make this model usable by an agent like opencode."""
        if not (self.tool_call and self.tool_parser):
            return []
        return ["--enable-auto-tool-choice", "--tool-call-parser", self.tool_parser]

    @property
    def required_compute_cap(self) -> int:
        if self.min_compute_cap is not None:
            return self.min_compute_cap
        return self._QUANT_MIN_CC.get(self.quant, 800)

    @property
    def bytes_read_per_token(self) -> float:
        """Weight bytes pulled from VRAM per decoded token (decimal GB).

        Decode is memory-bandwidth-bound (docs/METHOD.md §1), so this is the
        denominator of the speed estimate. An MoE only reads its active experts,
        which is the whole reason a 120B MoE can outrun a 70B dense model.
        """
        if self.active_b and self.total_b:
            return self.weights_gb * (self.active_b / self.total_b)
        return self.weights_gb

    def valid_tp(self, n: int) -> bool:
        """vLLM's tensor-parallel divisibility rule (docs/METHOD.md §5).

        TP must divide num_attention_heads, and either divide or be a multiple
        of num_kv_heads (KV heads are replicated when TP > num_kv_heads). This
        is why 3 GPUs cannot run a 64-head model and get forced into pipeline
        parallelism instead.
        """
        if n <= 0 or self.attn_heads % n:
            return False
        return (self.kv_heads % n == 0) or (n % self.kv_heads == 0)


def _m(**kw) -> Model:
    return Model(**kw)


MODELS: dict[str, Model] = {m.key: m for m in [
    _m(key="qwen7b", tool_parser="hermes", hf_id="Qwen/Qwen2.5-7B-Instruct-AWQ", label="Qwen2.5 7B Instruct",
       weights_gb=5.6, layers=28, kv_heads=4, head_dim=128, attn_heads=28,
       max_context=32768, quant="awq", family="qwen2.5",
       notes="Cheapest useful smoke test; fits any 24 GB card.",
       vllm_args=["--max-model-len", "16384"]),

    _m(key="gpt-oss-20b", tool_parser="openai", hf_id="openai/gpt-oss-20b", label="gpt-oss 20B (MoE)",
       weights_gb=13.8, layers=24, kv_heads=8, head_dim=64, attn_heads=64,
       max_context=131072, quant="mxfp4", family="gpt-oss", active_b=3.6, total_b=21.0,
       notes="MXFP4 needs Hopper/Blackwell or a recent vLLM on Ampere; verify before relying on it.",
       vllm_args=["--max-model-len", "32768"]),

    _m(key="coder-30b", tool_parser="qwen3_coder", hf_id="cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
       label="Qwen3-Coder 30B-A3B (MoE, AWQ)",
       weights_gb=18.1, layers=48, kv_heads=4, head_dim=128, attn_heads=32,
       max_context=262144, quant="awq", family="qwen3-coder", active_b=3.3, total_b=30.5,
       notes="Best coding-agent value: 30B quality, ~3B active -> fast decode, fits one 24 GB card.",
       vllm_args=["--max-model-len", "65536"]),

    _m(key="qwen3-32b", tool_parser="hermes", hf_id="Qwen/Qwen3-32B-AWQ", label="Qwen3 32B (dense, AWQ)",
       weights_gb=19.3, layers=64, kv_heads=8, head_dim=128, attn_heads=64,
       max_context=40960, quant="awq", family="qwen3",
       notes="Strong dense mid-size; heavy 256 KiB/token KV for its size.",
       vllm_args=["--max-model-len", "32768"]),

    _m(key="coder-30b-bf16", tool_parser="qwen3_coder", hf_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
       label="Qwen3-Coder 30B-A3B (MoE, bf16)",
       weights_gb=61.1, layers=48, kv_heads=4, head_dim=128, attn_heads=32,
       max_context=262144, quant="bf16", family="qwen3-coder", active_b=3.3, total_b=30.5,
       notes="Unquantized version of coder-30b — only worth it if you have 96 GB.",
       vllm_args=["--max-model-len", "65536"]),

    _m(key="llama70b", tool_parser="llama3_json", hf_id="casperhansen/llama-3.3-70b-instruct-awq",
       label="Llama 3.3 70B Instruct (AWQ)",
       weights_gb=39.8, layers=80, kv_heads=8, head_dim=128, attn_heads=64,
       max_context=131072, quant="awq", family="llama-3.3",
       notes="The classic 70B dense reference point.",
       vllm_args=["--max-model-len", "32768"]),

    _m(key="qwen72b", tool_parser="hermes", hf_id="Qwen/Qwen2.5-72B-Instruct-AWQ", label="Qwen2.5 72B Instruct (AWQ)",
       weights_gb=41.6, layers=80, kv_heads=8, head_dim=128, attn_heads=64,
       max_context=32768, quant="awq", family="qwen2.5",
       notes="Same class as llama70b, slightly larger weights.",
       vllm_args=["--max-model-len", "32768"]),

    _m(key="gpt-oss-120b", tool_parser="openai", hf_id="openai/gpt-oss-120b", label="gpt-oss 120B (MoE)",
       weights_gb=65.2, layers=36, kv_heads=8, head_dim=64, attn_heads=64,
       max_context=131072, quant="mxfp4", family="gpt-oss", active_b=5.1, total_b=117.0,
       notes="Tiny 72 KiB/token KV, ~5B active -> should beat 70B dense on speed. Needs sm_90+.",
       vllm_args=["--max-model-len", "32768"]),

    _m(key="mistral-24b", tool_parser="mistral", hf_id="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
       label="Mistral Small 3.2 24B (bf16)",
       weights_gb=48.0, layers=40, kv_heads=8, head_dim=128, attn_heads=32,
       max_context=131072, quant="bf16", family="mistral",
       notes="Unquantized 24B; needs 48 GB+ purely because it is bf16.",
       vllm_args=["--max-model-len", "32768"]),
]}


def get(key: str) -> Model:
    try:
        return MODELS[key]
    except KeyError:
        raise KeyError(f"Unknown model {key!r}. Known: {', '.join(MODELS)}") from None
