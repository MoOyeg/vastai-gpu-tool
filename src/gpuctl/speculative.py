"""Speculative decoding configuration for vLLM.

vLLM takes this as one JSON blob: `--speculative-config '{"method": ...}'`.
The method names, required fields and defaults below were read from
vllm/config/speculative.py at v0.30.0, not inferred.

The idea is to draft several tokens cheaply and have the target model verify
them in one forward pass. Decode is memory-bandwidth-bound (docs/METHOD.md §1),
so verifying K drafts costs about the same as generating one token — which is
why this can lift tok/s well past the bandwidth ceiling a single-token-at-a-time
read implies. The catch is that it only helps to the extent drafts are
*accepted*; a low acceptance rate buys nothing and can cost a little. That is
why `gpuctl bench` reports the acceptance rate alongside tok/s.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Multi-token-prediction heads shipped inside a target model. No draft
# checkpoint needed: the weights are already there.
MTP_METHODS = (
    "mtp", "deepseek_mtp", "minimax_m3_mtp", "kimi_k3_mtp", "qwen3_next_mtp",
    "qwen3_5_mtp", "glm4_moe_mtp", "glm5_next_mtp", "ernie_mtp", "mimo_mtp",
    "mimo_v2_mtp", "nemotron_h_mtp", "longcat_flash_mtp", "hy_v3_mtp",
    "hy_v4_mtp", "gemma4_mtp", "step3p5_mtp", "exaone4_5_mtp",
)

# Everything vLLM accepts as `method`.
METHODS = (
    "ngram", "ngram_gpu", "suffix", "draft_model", "medusa", "mlp_speculator",
    "eagle", "eagle3", "dflash", "dspark", "extract_hidden_states",
    "custom_class",
) + MTP_METHODS

# Methods that need a separate draft checkpoint in `model`.
NEEDS_DRAFT_MODEL = (
    "draft_model", "eagle", "eagle3", "medusa", "mlp_speculator", "dflash",
    "dspark", "custom_class",
)

# Fields we pass through untouched. Anything else is rejected so a typo does not
# silently reach vLLM and get ignored.
PASSTHROUGH_FIELDS = (
    "prompt_lookup_max", "prompt_lookup_min", "draft_tensor_parallel_size",
    "quantization", "max_model_len", "revision", "enforce_eager",
    "disable_padded_drafter_batch", "parallel_drafting",
    "suffix_decoding_max_tree_depth", "rejection_sample_method",
    "draft_sample_method",
)


class SpeculativeError(ValueError):
    """The speculative-decoding settings are not usable."""


@dataclass(frozen=True)
class Speculative:
    method: str
    num_speculative_tokens: int
    model: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_draft_model(self) -> bool:
        return self.method in NEEDS_DRAFT_MODEL

    @property
    def is_mtp(self) -> bool:
        return self.method in MTP_METHODS

    def payload(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "method": self.method,
            "num_speculative_tokens": int(self.num_speculative_tokens),
        }
        if self.model:
            body["model"] = self.model
        body.update(self.options)
        return body

    def to_json(self) -> str:
        # Compact: the result is passed as one shell word.
        return json.dumps(self.payload(), separators=(",", ":"), sort_keys=False)

    def cli_args(self) -> list[str]:
        """Args for the vllm command line.

        The JSON is wrapped in single quotes so the shell hands it over as one
        argument; the payload itself only ever contains double quotes.
        """
        return ["--speculative-config", f"'{self.to_json()}'"]

    def describe(self) -> str:
        bits = [f"{self.method} ×{self.num_speculative_tokens}"]
        if self.model:
            bits.append(f"draft={self.model}")
        return "  ".join(bits)


def from_mapping(data: dict[str, Any], where: str = "speculative") -> Speculative:
    """Build and validate from a TOML table or CLI flags."""
    if not isinstance(data, dict):
        raise SpeculativeError(f"{where}: must be a table of settings")
    data = dict(data)

    method = str(data.pop("method", "") or "").strip()
    if not method:
        raise SpeculativeError(f"{where}: `method` is required")
    if method not in METHODS:
        raise SpeculativeError(
            f"{where}: unknown method {method!r}. Known: {', '.join(sorted(METHODS))}"
        )

    raw_tokens = data.pop("num_speculative_tokens", None)
    if raw_tokens is None:
        raise SpeculativeError(f"{where}: `num_speculative_tokens` is required")
    try:
        tokens = int(raw_tokens)
    except (TypeError, ValueError):
        raise SpeculativeError(
            f"{where}: num_speculative_tokens must be a whole number, got {raw_tokens!r}"
        ) from None
    if tokens < 1:
        raise SpeculativeError(f"{where}: num_speculative_tokens must be at least 1")

    model = data.pop("model", None)
    if model is not None and not isinstance(model, str):
        raise SpeculativeError(f"{where}: `model` must be text")
    if method in NEEDS_DRAFT_MODEL and not model:
        raise SpeculativeError(
            f"{where}: method {method!r} needs a draft checkpoint — set `model`. "
            f"Methods needing none: ngram, suffix, and the *_mtp heads."
        )
    if method in MTP_METHODS and model:
        raise SpeculativeError(
            f"{where}: method {method!r} uses heads inside the target model, "
            f"so `model` must not be set."
        )

    unknown = set(data) - set(PASSTHROUGH_FIELDS)
    if unknown:
        raise SpeculativeError(
            f"{where}: unknown field(s) {sorted(unknown)}. "
            f"Valid: method, num_speculative_tokens, model, "
            f"{', '.join(PASSTHROUGH_FIELDS)}"
        )
    return Speculative(method=method, num_speculative_tokens=tokens,
                       model=model or None, options=data)


# ------------------------------------------------------------------ metrics

# Counter names from vllm/v1/spec_decode/metrics.py at v0.30.0. Prometheus
# exposition appends _total to counters, so both spellings are accepted.
ACCEPTED_METRIC = "vllm:spec_decode_num_accepted_tokens"
DRAFT_METRIC = "vllm:spec_decode_num_draft_tokens"
DRAFTS_METRIC = "vllm:spec_decode_num_drafts"


@dataclass
class Acceptance:
    accepted: float
    drafted: float
    drafts: float

    @property
    def rate(self) -> float | None:
        """Fraction of drafted tokens the target model kept."""
        return self.accepted / self.drafted if self.drafted else None

    @property
    def tokens_per_draft(self) -> float | None:
        """Mean accepted tokens per draft round — the actual speedup lever."""
        return self.accepted / self.drafts if self.drafts else None


def parse_metrics(text: str) -> Acceptance | None:
    """Pull speculative counters out of a Prometheus /metrics body."""
    wanted = {ACCEPTED_METRIC: 0.0, DRAFT_METRIC: 0.0, DRAFTS_METRIC: 0.0}
    seen = False
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, rest = line.partition(" ")
        base = name.split("{", 1)[0]
        if base.endswith("_total"):
            base = base[: -len("_total")]
        if base not in wanted:
            continue
        try:
            wanted[base] += float(rest.strip().split()[0])
        except (ValueError, IndexError):
            continue
        seen = True
    if not seen:
        return None
    return Acceptance(wanted[ACCEPTED_METRIC], wanted[DRAFT_METRIC],
                      wanted[DRAFTS_METRIC])
