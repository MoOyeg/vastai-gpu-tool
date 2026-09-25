# Method: sizing and predicting a vLLM deployment

The arithmetic `gpuctl` uses to decide whether a GPU offer can host a model, and
how fast it will decode. Everything here is either derived from first principles
or measured on rented hardware; where the two disagree, that is called out.

## 1. Decode is memory-bandwidth-bound

At batch size 1, producing each output token requires reading the model's active
weights out of VRAM once. So:

```
decode tok/s  ≈  (aggregate memory bandwidth × efficiency) / bytes read per token
```

Compute barely enters into it. This single relationship is why VRAM bandwidth,
not FLOPS or price, predicts single-stream speed — and why a card with 8% more
bandwidth is ~8% faster at this job, not 70% faster.

Implemented in `planner.estimate_tokps()`.

### Bytes read per token

For a dense model, that is the whole weight file. For a Mixture-of-Experts model
only the active experts are read:

```
bytes_per_token = weights_bytes × (active_params / total_params)
```

This is the entire reason a 120B MoE with ~5B active parameters can decode
faster than a 70B dense model while being a stronger model.

Implemented in `Model.bytes_read_per_token`.

## 2. Efficiency factors

The fraction of theoretical bandwidth actually achieved. Apply **one** model
uniformly across every row of a comparison — mixing an optimistic figure for new
hardware with a measured one for old hardware silently biases the whole table
toward the new hardware.

| Configuration | Efficiency |
|---|---|
| Single GPU, GDDR | ~55% |
| TP=2 | ~40% |
| TP=4 | ~33% |
| TP=8 | ~30% |

**These are conservative — measurement disagrees.** See §7: a measured TP=2 run
came in at 67–84% rather than 40%. The table is kept as the pessimistic default
because it was derived on Ampere; treat `est tok/s` as a floor and use
`gpuctl bench` for the real number.

## 3. KV cache

```
kv_bytes_per_token = 2 (K and V) × layers × kv_heads × head_dim × dtype_bytes
```

For Llama-3.3-70B (80 layers, 8 KV heads, head_dim 128, fp16):

```
2 × 80 × 8 × 128 × 2 = 327,680 B = 320 KiB per token
```

`--kv-cache-dtype fp8` halves this and roughly doubles the context that fits.

KV is usually the binding constraint, not the weights. A 48 GiB rig holding
37.1 GiB of 4-bit 70B weights has only ~5 GiB left for KV — enough for ~16k
tokens at fp16, and `--max-model-len 32768` will simply refuse to start.

Implemented in `Model.kv_bytes_per_token` and `Model.vram_gib`.

## 4. GB vs GiB — the mistake worth avoiding

HuggingFace reports file sizes in decimal **GB**. NVIDIA and Vast report VRAM in
binary **MiB/GiB**. Mixing them makes 48 GB cards look ~20% worse than they are:

| quantity | decimal | binary |
|---|---|---|
| Llama-70B AWQ shards | 39.8 GB | **37.1 GiB** |
| 2× RTX 3090 | 51.5 GB | **48 GiB** |

Treating "39 GB of weights" as 39 GiB against "48 GB" of VRAM yields ~7k tokens
of context. Done consistently it is **~16k**. `gpuctl` converts once, on ingest,
and works in GiB throughout.

## 5. Tensor-parallel divisibility

vLLM requires that TP divide `num_attention_heads`, **and** that it either divide
or be a multiple of `num_key_value_heads` (KV heads are replicated when
TP > num_kv_heads).

Llama-70B and Qwen-72B have 64 attention heads and 8 KV heads, so valid TP is
**1, 2, 4, 8, 16, 32, 64**. Notably **TP=3 is invalid** — 64 % 3 ≠ 0 — so three
GPUs cannot run tensor parallelism at all and fall back to pipeline parallelism,
which pipelines requests across stages but gives *no* single-stream speedup.

Implemented in `Model.valid_tp()`.

## 6. Compute capability per quantisation

A card that has enough VRAM may still be unable to run the kernels. Vast reports
`compute_cap` as sm × 100.

| Quantisation | Needs | Why |
|---|---|---|
| AWQ (Marlin) | sm_80+ | Marlin kernels are Ampere-and-later |
| bf16 | sm_80+ | bf16 datapath is Ampere+ |
| FP8 | sm_89+ | native FP8 is Ada+ |
| MXFP4 | sm_90+ | on Ampere vLLM dequantises to bf16, blowing the VRAM budget |

Without this filter a planner will cheerfully recommend a Tesla P40 for an AWQ
model. Implemented in `Model.required_compute_cap`.

## 7. Measured results

Verified with `gpuctl bench`, using vLLM's server-reported token counts.

### Llama 3.3 70B AWQ · 2× RTX 5090 · TP=2 · 32k ctx · $0.99/hr

| metric | value |
|---|---|
| decode | **60.6 tok/s** (consistent across 128/256/512-token runs) |
| TTFT | ~500 ms |
| ITL / TPOT | 17 ms |
| §2 model predicted | ~29 tok/s |

Back-solving efficiency from 60.6 tok/s gives **67%** against spec bandwidth
(1792 GB/s/card) or **84%** against Vast's measured figure (~1443 GB/s) — far
above the 40% §2 assumes for TP=2. Whether Ampere reaches the same is a separate
measurement; the conservative table stands until someone runs it.

Also confirmed: `awq_marlin` engages on Blackwell (sm_120) —
`Using MarlinLinearKernel for AutoAWQMarlinLinearMethod`.

### Qwen3.8 27B FP8 · 1× RTX PRO 6000 WS · TP=1 · 131k ctx · $1.49/hr

| metric | value |
|---|---|
| decode | **50.7 tok/s** |
| TTFT | ~175 ms |
| ITL / TPOT | 20 ms |

**Reasoning tokens are decode tokens.** At its default `xhigh` effort this model
spent **100% of a 400-token budget thinking** and never reached an answer. Two
consequences: a throughput measurement must count `delta.reasoning`, not just
`delta.content` (counting only content measures zero and looks like a broken
endpoint); and an agent using it needs a generous `max_tokens`, or a lower
`reasoning_effort`, or it will burn the whole budget before replying.

## 8. The rent-test matrix

The preset recipes in `recipes.py` exist to answer, with measurements rather than
estimates, which hardware configuration is worth buying:

| recipe | configuration | question it settles |
|---|---|---|
| `smoke` | 1× 3090, 7B AWQ | does the pipeline work at all (pennies) |
| `build-a` | 2× 3090, TP=2 | is 48 GB usable at fp8 KV? does awq_marlin engage? |
| `build-b` | 2× 4090, TP=2 | is 2× 4090 really ~1.7× a 3090 pair, or ~1.1×? |
| `build-f` | 4× 3090, TP=4 | does 96 GB + TP=4 beat both at a lower price? |
| `a6000` | 1× A6000 48 GB | long context on one card vs speed |
| `pro6000` | 1× PRO 6000 96 GB | does the 120B MoE class run, and how fast? |

A few hours of rental across these costs about as much as a takeaway meal, and
replaces every estimate above with a number.
