# gpuctl

Provision GPU boxes on [Vast.ai](https://vast.ai), track them from "rented" to
"actually serving tokens", and wire them into [opencode](https://opencode.ai)
the moment they are ready.

Built for a specific question: **before buying GPUs to run a large model
locally, rent the candidate configurations for an hour and measure them.** A few
dollars of rental replaces a pile of estimates.

The sizing and speed arithmetic is written up in [docs/METHOD.md](docs/METHOD.md).

## Why it exists

Renting is a metered activity, and the failure mode is not "it didn't work",
it's "it worked and then billed for nine hours". So the design leans on three
things:

- **Every launch has a deadline.** `--ttl` is written into local state at rent
  time, `gpuctl watch` enforces it, and `gpuctl reap` is cron-safe.
- **Readiness means *serving*, not *running*.** A Vast container reports
  `running` for many minutes while vLLM is still pulling 39 GB of weights.
  `gpuctl` polls `/v1/models` and only calls it ready when a model answers.
- **Teardown is symmetric.** `gpuctl down` destroys the instance *and* removes
  the provider block it added to opencode, so you never point opencode at a
  dead IP.

## Install

```bash
uv venv --python 3.11
uv pip install -e .
export VAST_API_KEY=...        # or: uv run gpuctl set-key <key>
uv run gpuctl doctor
```

Add your SSH public key at <https://cloud.vast.ai/manage-keys/> before renting —
keys only attach to *new* instances.

## Use

```bash
uv run gpuctl models                  # what can I serve, and what VRAM does it need
uv run gpuctl plan                    # cheapest offer that fits each model
uv run gpuctl plan llama70b -n 10     # alternatives for one model
uv run gpuctl launch llama70b         # rent the cheapest fit, boot, wire into opencode
uv run gpuctl ps                      # phase + spend so far
uv run gpuctl bench                   # measured tok/s vs the doc's estimate
uv run gpuctl down                    # stop the meter, unlink from opencode
```

`launch` picks hardware from the model: parallelism from the offer's GPU count
(respecting vLLM's TP divisibility rule), context from what the VRAM allows,
disk from the weight size. `up <recipe>` still runs the fixed presets.

Useful guards: `--min-tokps 25` skips offers that fit but are too slow to use,
`--exclude-geo ", CN"` avoids hosts that cannot reach HuggingFace, and `--ttl`
sets the auto-destroy deadline.

`gpuctl up` prints a spend plan and waits for confirmation before renting
anything. Then it watches the instance through its phases and, on
`serving`, writes the provider into `~/.config/opencode/opencode.json`:

```
pending → loading → running → serving → linked
```

## Recipes

Recipes are **data, not code** — one TOML file each, read at runtime:

```
src/gpuctl/recipes.d/        shipped with the package
~/.config/gpuctl/recipes/    yours; same filename replaces a builtin
$GPUCTL_RECIPES_DIR          one or more extra dirs (os.pathsep-separated)
```

Later directories win, so dropping `build-a.toml` in your own directory
overrides the shipped one, and `my-rig.toml` adds `gpuctl up my-rig`.
`gpuctl recipes --paths` prints the live search path; `--show <key>` prints one
file verbatim.

```toml
# ~/.config/gpuctl/recipes/dual-5090.toml
title      = "2x RTX 5090 — measured 60 tok/s on 70B"
gpu_name   = "RTX 5090"          # must match Vast's catalogue; see `gpuctl gpus`
num_gpus   = 2
model_key  = "llama70b"          # inherits the HF id AND the tool-call parser
disk_gb    = 110
max_dph    = 1.30
est_tokps  = "~60 (measured)"
vllm_args  = ["--tensor-parallel-size", "2", "--max-model-len", "32768"]
```

Frontier-model recipes take their flags from the **official vLLM recipes**
(`recipes.vllm.ai/<org>/<model>`) rather than from guesswork — including the
mandatory ones that are easy to miss, like MiniMax-M3's `--block-size 128` and
DeepSeek V4.1's `--tokenizer-mode deepseek_v41`.

Reasoning effort (`xhigh`, max) is **not a serve flag** — it is a per-request
`chat_template_kwargs` value, pinned server-wide here via
`--default-chat-template-kwargs`. Two models need a non-default container:
Kimi K3 ships as `vllm/vllm-openai:kimi-k3`, and DeepSeek V4.1 Flash needs
vLLM ≥ 0.30 (`DeepseekV41ForCausalLM` is not registered in 0.29), so it pins
`:nightly`. `gpuctl up` highlights a non-default image in the spend plan.

`model_key` points at the model catalogue (`gpuctl models`) so a recipe inherits
the HuggingFace id and — importantly — the right `--tool-call-parser`. Set
`model` and `tool_parser` directly instead for anything not in the catalogue.
Files are validated on load: unknown fields, wrong types and unknown model keys
fail with the filename and the offending field rather than surfacing later as a
malformed Vast query.

Each recipe is one configuration worth measuring before buying it
([docs/METHOD.md §8](docs/METHOD.md)). `est tok/s` is the *prediction*;
`gpuctl bench` produces the measurement to check it against.

| key | hardware | model | doc |
|---|---|---|---|
| `smoke` | 1× RTX 3090 | Qwen2.5-7B-AWQ | pipeline test, pennies |
| `qwen3.8-27b` | 2× RTX 5090 | Qwen3.8-27B-FP8 | ~$1.0/hr — frontier reasoning, cheap |
| `mimo-v2.6-pro` | 8× H200 | MiMo-V2.6-Pro-RL | ~$48/hr |
| `minimax-m3` | 8× H200 | MiniMax-M3 | ~$49/hr |
| `deepseek-v4.1-flash-max` | 8× H200 | DeepSeek-V4.1-Flash | ~$48/hr, needs a nightly image |
| `kimi-k3-max` | 8× B300 | Kimi-K3 | ~$86/hr |
| `build-a` | 2× RTX 3090, TP=2 | Llama-3.3-70B-AWQ | METHOD §8 |
| `build-b` | 2× RTX 4090, TP=2 | Llama-3.3-70B-AWQ | METHOD §8 |
| `build-f` | 4× RTX 3090, TP=4 | Llama-3.3-70B-AWQ | METHOD §8 |
| `a6000` | 1× RTX A6000 48 GB | Llama-3.3-70B-AWQ | METHOD §3 |
| `pro6000` | 1× RTX PRO 6000 96 GB | gpt-oss-120b | METHOD §6 |

Override anything: `gpuctl up build-a --model <hf-id> --ttl 1 --max-dph 0.55`.

## How the opencode wiring works

opencode's provider schema takes an `npm` adapter plus a `baseURL`, which is
the same shape as the local MLX/llama.cpp providers already in your config. A
linked instance shows up as:

```json
"provider": {
  "vast-1234567": {
    "npm": "@ai-sdk/openai-compatible",
    "name": "Vast 2x RTX 3090 (1234567)",
    "options": {
      "baseURL": "http://1.2.3.4:40021/v1",
      "apiKey": "sk-vast-…",
      "headerTimeout": 900000,
      "chunkTimeout": 300000
    },
    "models": { "llama-3.3-70b-instruct-awq": { "tool_call": true, "limit": { … } } }
  }
}
```

The model id is read back from the instance's own `/v1/models` rather than
assumed. The config file is backed up (`opencode.<epoch>.bak`) before every
edit, and only providers prefixed `vast-` are ever touched. A config that
isn't plain JSON is refused rather than rewritten.

## Layout

| file | role |
|---|---|
| `vast.py` | Vast REST client (`/bundles/`, `/asks/{id}/`, `/instances/`) |
| `models.py` | model catalogue + VRAM/KV/TP arithmetic |
| `planner.py` | fits models to live offers, cheapest first |
| `recipes.py` | TOML recipe loader, merge order and validation |
| `recipes.d/*.toml` | the shipped recipes themselves |
| `provision.py` | offer search, port/env mapping, vLLM onstart script |
| `track.py` | phase state machine, cost accounting, link/unlink |
| `health.py` | `/v1/models` readiness probe |
| `opencode.py` | safe merge into opencode's config |
| `conductor.py` | comment-preserving edits to Conductor's settings.toml |
| `state.py` | local record of intent: TTL, serving key, what we edited |

## Measured results (2026-09-10)

**Llama 3.3 70B AWQ, 2x RTX 5090 (TP=2, 32k ctx), $0.99/hr, Taiwan:**

| metric | value |
|---|---|
| decode | **60.6 tok/s** (verified over 128/256/512-token runs, server-reported token counts) |
| TTFT | ~500 ms |
| ITL / TPOT | 17 ms |
| doc §2 model predicted | ~29 tok/s |

Two things this settles:

- **`awq_marlin` engages** on current hardware. The log says
  `Using MarlinLinearKernel for AutoAWQMarlinLinearMethod`, on Blackwell (sm_120).
- **The conventional 40% TP=2 efficiency figure is far too harsh.** Back-solving from 60.6 tok/s
  gives 67-84% depending on whether you use spec or Vast-measured bandwidth.
  Whether Ampere reaches the same is a separate measurement — `gpuctl launch
  llama70b --offer <a 3090 box>` answers it.

## Serving gotchas (found the hard way, verified live 2026-09-10)

These cost real debugging time, so they are written down:

- **A vLLM server is not agent-ready just because `/v1/models` answers.**
  Function calling requires BOTH `--enable-auto-tool-choice` and a matching
  `--tool-call-parser`; without them opencode fails on its first request with
  `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser
  to be set`. Every model and recipe now carries its parser, and
  `health.tool_call_smoke()` asserts a real `tool_calls` response before we call
  anything ready. Parser names move between releases — get them from
  `vllm serve --help=all` (in 0.29 plain `--help` is an 80-line summary that
  omits the flag entirely).
- **`vllm/vllm-openai:latest` is CUDA 13.** Its NVIDIA forward-compatibility
  libraries only work on *datacenter* GPUs, so a GeForce host on an older driver
  dies with `CUDA error 804: forward compatibility was attempted on non
  supported HW` — while Vast happily reports the instance as `running` with
  `status_msg: "success"`. Recipes now require `cuda_max_good >= 13.0`.
- **Cheap hardware that "fits" often cannot run the kernels.** AWQ Marlin and
  bf16 need sm_80+, MXFP4 needs sm_90+. Without a compute-capability filter the
  planner happily recommends a Tesla P40.

- **`gpu_name` must contain spaces, not underscores.** `{"eq": "RTX_3090"}`
  returns *zero offers with HTTP 200* — no error, just silence. The underscore
  form is a CLI shell-quoting convention; Vast's own API docs example
  (`["RTX_4090","RTX_3090"]`) is misleading. `gpuctl` now resolves every name
  against `/gpu_names/unique/` and raises with suggestions rather than
  returning an empty list.
- **`GET /api/v0/instances/` is retired** — HTTP 410 `deprecated_endpoint`.
  Listing moved to `/api/v1/instances/`. Single-instance `GET`, `PUT /asks/{id}/`
  (create), `DELETE /instances/{id}/` and `request_logs` are all still v0.
- **The API rate-limits bursts** (HTTP 429, ~5 requests/window) and returns a
  `retry_after`. The client honours it with exponential-backoff fallback;
  without this, a multi-config search or a tight `watch` trips it.
- **GPU names are more specific than you expect.** There is no `RTX PRO 6000` —
  it is `RTX PRO 6000 WS` / `S` / `Max-Q`. Check with `gpuctl gpus -f 6000`.

## Conductor

[Conductor](https://conductor.build) runs OpenCode as a harness and *"asks
OpenCode for the models available to your provider configuration"* — so a
deployment gpuctl has linked into `opencode.json` already shows up in
Conductor's model picker. What remains is telling Conductor to use it:

```bash
uv run gpuctl launch llama70b --conductor   # set it once the box is serving
uv run gpuctl conductor                    # show current settings + liveness
uv run gpuctl conductor set 50546033        # point it at a linked deployment
uv run gpuctl conductor set --review        # also set the code-review model
uv run gpuctl conductor revert              # put back what was there before
```

`--conductor` also works on `up`, `watch` and `link`. `gpuctl down` reverts
automatically, so Conductor is never left pointing at a destroyed instance —
and `gpuctl conductor` flags it loudly if something else left it that way.

The written value is the provider-qualified id Conductor expects, which is the
same one opencode uses:

```toml
[models]
default = "vast-50546033/llama-3.3-70b-instruct-awq"
```

**On editing someone else's config file.** `tomllib` is read-only, and
re-serialising with a TOML writer would discard the comments, key order and
quoting style of a file you maintain by hand. So gpuctl makes a surgical
single-line edit to the text, preserving standalone *and* inline comments, then
verifies the result by re-parsing it and diffing against the expected document —
if anything but the requested key moved, the write is abandoned. Only
`models.default` and `models.review` are ever touched, a `.bak` is written
first, and a key that did not exist before is *removed* on revert rather than
blanked. Pass `--settings .conductor/settings.toml` to target a project's
committed defaults instead of your personal ones.

## Tests

```bash
uv run --group dev pytest        # 64 tests, no network, ~0.15s
```

The suite stubs out the readiness probe globally, so nothing reaches the network
and no test touches your real opencode config, state or recipe directories.

## Notes

- Vast injects sshd into any image, so `vllm/vllm-openai:latest` works with
  `runtype: ssh` — you get both the server and a shell.
- Container ports are requested through the env map (`"-p 8000:8000": "1"`)
  and land on a *random* external port; `gpuctl` reads the real mapping out of
  the instance's `ports` field.
- The serving key is passed as `$VLLM_API_KEY` and referenced by name in the
  onstart script, so the secret is not stored in Vast's onstart text.
- Gated models need `--hf-token` (or `HF_TOKEN`).
- GPU names differ by host. If a search comes back empty, check the exact
  string with `gpuctl gpus -f 6000`.
