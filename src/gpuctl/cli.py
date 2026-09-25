"""gpuctl command line."""
from __future__ import annotations

import dataclasses
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import conductor as conductor_mod
from . import models as model_mod, opencode, planner, provision, recipes as recipe_mod, state
from .config import (
    DEFAULT_DISK_GB,
    DEFAULT_MAX_DPH,
    DEFAULT_PORT,
    DEFAULT_TTL_HOURS,
    ConfigError,
    api_key,
    save_api_key,
)
from .health import completion_smoke
from .state import Deployment
from .track import (
    PHASE_STYLE, Phase, link_conductor, link_opencode, snapshot,
    unlink_conductor, unlink_opencode,
)
from .recipes import BUILTIN_DIRNAME as BUILTIN_DIR
from .vast import VastClient, VastError, normalize_gpu_name, ssh_target

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Provision, track and auto-wire Vast.ai GPU instances into opencode.",
)
console = Console()
err = Console(stderr=True)


# --------------------------------------------------------------------- helpers


def _client() -> VastClient:
    try:
        return VastClient()
    except ConfigError as exc:
        err.print(f"[bold red]{exc}[/]")
        raise typer.Exit(2) from None


def _fail(msg: str, code: int = 1) -> None:
    err.print(f"[bold red]error:[/] {msg}")
    raise typer.Exit(code)


def _need(ref: Optional[str]) -> Deployment:
    dep = state.resolve(ref)
    if dep is None:
        live = state.load_all()
        if not live:
            _fail("no tracked deployments. Launch one with `gpuctl up <recipe>`.")
        listing = ", ".join(f"{d.instance_id} ({d.recipe})" for d in live)
        _fail(f"ambiguous or unknown deployment. Pick one of: {listing}")
    return dep  # type: ignore[return-value]


def _money(v: float) -> str:
    return f"${v:,.2f}"


def _dur(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


# ---------------------------------------------------------------------- doctor


@app.command()
def doctor() -> None:
    """Check that everything gpuctl depends on is present and wired up."""
    rows: list[tuple[str, bool, str]] = []

    key = api_key(required=False)
    rows.append(("Vast API key", bool(key), f"…{key[-6:]}" if key else "set VAST_API_KEY"))

    if key:
        try:
            with VastClient() as c:
                insts = c.list_instances()
            rows.append(("Vast API reachable", True, f"{len(insts)} instance(s) on the account"))
        except VastError as exc:
            rows.append(("Vast API reachable", False, str(exc)[:80]))

    oc = opencode.target_path()
    rows.append(("opencode config", oc.exists(), str(oc)))
    if oc.exists():
        linked = opencode.linked_providers(oc)
        rows.append(("gpuctl providers in opencode", True, ", ".join(linked) or "none"))
    sibling = oc.with_suffix(".jsonc")
    if sibling.exists():
        rows.append((
            "note: sibling .jsonc present", True,
            f"{sibling.name} also exists; gpuctl only edits {oc.name}",
        ))

    ssh_pub = Path.home() / ".ssh" / "id_ed25519.pub"
    rows.append((
        "SSH public key", ssh_pub.exists(),
        f"{ssh_pub} (add it at cloud.vast.ai/manage-keys/)" if ssh_pub.exists() else "none found",
    ))

    deps = state.load_all()
    rows.append(("tracked deployments", True, str(len(deps))))

    table = Table(box=None, pad_edge=False)
    table.add_column("")
    table.add_column("check")
    table.add_column("detail", style="dim")
    for name, ok, detail in rows:
        table.add_row("[green]✓[/]" if ok else "[red]✗[/]", name, detail)
    console.print(table)


@app.command("set-key")
def set_key(key: str = typer.Argument(..., help="Vast.ai API key")) -> None:
    """Store a Vast.ai API key at ~/.config/gpuctl/api_key (mode 0600)."""
    path = save_api_key(key)
    console.print(f"[green]saved[/] {path}")


@app.command()
def gpus(filter: str = typer.Option("", "--filter", "-f", help="substring match")) -> None:
    """List the canonical GPU names Vast accepts in a search."""
    with _client() as c:
        names = c.gpu_names()
    if filter:
        needle = filter.lower().replace("_", " ")
        names = [n for n in names if needle in n.lower().replace("_", " ")]
    if not names:
        console.print("[yellow]no matching GPU names[/]")
        return
    for n in names:
        console.print(f"  {n}  [dim]({normalize_gpu_name(n)})[/]")


# --------------------------------------------------------------------- recipes


@app.command("recipes")
def list_recipes(
    paths: bool = typer.Option(False, "--paths", help="show the directories recipes are read from"),
    show: Optional[str] = typer.Option(None, "--show", help="print one recipe's TOML verbatim"),
) -> None:
    """Show the launch recipes loaded from disk."""
    if paths:
        console.print("[bold]recipe search path[/] [dim](later overrides earlier)[/]")
        for d in recipe_mod.search_dirs():
            n = len(list(d.glob("*.toml"))) if d.is_dir() else 0
            mark = "[green]✓[/]" if d.is_dir() else "[dim]·[/]"
            console.print(f"  {mark} {d}  [dim]({n} recipe(s))[/]")
        console.print(f"\n[dim]Drop a .toml in {recipe_mod.USER_RECIPES_DIR} to add or override one.[/]")
        console.print(f"[dim]Or point ${recipe_mod.ENV_VAR} at one or more directories.[/]")
        return

    if show:
        try:
            r = recipe_mod.get(show)
        except KeyError as exc:
            _fail(str(exc))
        console.print(f"[dim]{r.source}[/]")
        console.print(Path(r.source).read_text().rstrip())
        return

    try:
        loaded = recipe_mod.all_recipes()
    except recipe_mod.RecipeError as exc:
        _fail(str(exc))
    table = Table(title="gpuctl recipes", header_style="bold")
    table.add_column("key", style="bold cyan")
    table.add_column("hardware")
    table.add_column("model", style="dim")
    table.add_column("max $/hr", justify="right")
    table.add_column("est tok/s", justify="right")
    table.add_column("doc")
    table.add_column("from", style="dim")
    for r in loaded.values():
        table.add_row(
            r.key,
            f"{r.num_gpus}x {r.gpu_name}",
            r.model,
            f"{r.max_dph:.2f}",
            r.est_tokps,
            r.doc_ref,
            "builtin" if BUILTIN_DIR in r.source else "custom",
        )
    console.print(table)
    console.print("[dim]est tok/s are modelled (docs/METHOD.md §2) — `gpuctl bench` measures the truth.[/]")
    console.print("[dim]`gpuctl recipes --paths` shows where these load from; `--show <key>` prints one.[/]")


def _offer_table(offers: list[dict[str, Any]], title: str) -> Table:
    table = Table(title=title, header_style="bold")
    table.add_column("offer", style="bold")
    table.add_column("gpu")
    table.add_column("$/hr", justify="right")
    table.add_column("$/3h", justify="right", style="dim")
    table.add_column("rel.", justify="right")
    table.add_column("net ↓", justify="right")
    table.add_column("disk", justify="right")
    table.add_column("location", style="dim")
    for o in offers:
        s = provision.offer_summary(o)
        table.add_row(
            str(s["id"]),
            s["gpu"],
            f"{s['dph']:.3f}",
            f"{s['dph'] * 3:.2f}",
            f"{s['reliability'] * 100:.1f}%",
            f"{s['inet_down']:.0f}",
            f"{s['disk']:.0f}G",
            str(s["geo"])[:28],
        )
    return table


@app.command()
def search(
    recipe: str = typer.Argument(..., help="recipe key (see `gpuctl recipes`)"),
    max_dph: Optional[float] = typer.Option(None, "--max-dph", help="price ceiling, $/hr"),
    limit: int = typer.Option(15, "--limit", "-n"),
) -> None:
    """Search Vast for offers matching a recipe."""
    try:
        r = recipe_mod.get(recipe)
    except (KeyError, recipe_mod.RecipeError) as exc:
        _fail(str(exc))
    with _client() as c:
        try:
            offers = provision.search(c, r, max_dph=max_dph, limit=limit)
        except VastError as exc:
            _fail(str(exc))
    if not offers:
        console.print(
            f"[yellow]no offers[/] for {r.num_gpus}x {r.gpu_name} under "
            f"${max_dph or r.max_dph:.2f}/hr with ≥{r.disk_gb}G disk.\n"
            "[dim]Try --max-dph higher, or check `gpuctl gpus -f <name>` for the exact GPU string.[/]"
        )
        raise typer.Exit(1)
    console.print(_offer_table(offers, f"{r.title}"))


def _recipe_for(m, fit, *, context: int, fp8_kv: bool) -> "recipe_mod.Recipe":
    """Synthesise a launch recipe from a model and the offer chosen for it.

    Recipes are presets; this is the general case - parallelism comes from the
    offer's GPU count and the context from what the VRAM actually allows.
    """
    args = ["--max-model-len", str(context), "--gpu-memory-utilization", str(planner.GPU_MEM_UTIL)]
    if fit.num_gpus > 1:
        flag = "--tensor-parallel-size" if fit.parallel.startswith("TP") else "--pipeline-parallel-size"
        args += [flag, str(fit.num_gpus)]
    if fp8_kv:
        args += ["--kv-cache-dtype", "fp8"]
    # Room for the shards plus HF's staging copy, and never below 60 GB.
    disk = max(60, int(m.weights_gb * 2 + 30))
    return recipe_mod.Recipe(
        key=m.key, title=f"{m.label} on {fit.num_gpus}x {fit.gpu_name}",
        gpu_name=fit.gpu_name, num_gpus=fit.num_gpus, model=m.hf_id,
        vllm_args=args, disk_gb=disk, max_dph=fit.dph * 1.05,
        tool_parser=m.tool_parser,   # agent clients need these flags
        est_tokps=f"~{fit.est_tokps:.0f}" if fit.est_tokps else "?",
        doc_ref="planner", notes=m.notes,
    )


@app.command()
def launch(
    model: str = typer.Argument(..., help="model key (see `gpuctl models`)"),
    offer: Optional[int] = typer.Option(None, "--offer", help="specific offer id; default = cheapest fit"),
    context: int = typer.Option(32768, "--context", "-c"),
    fp8_kv: bool = typer.Option(False, "--fp8-kv"),
    max_dph: Optional[float] = typer.Option(None, "--max-dph"),
    min_tokps: Optional[float] = typer.Option(None, "--min-tokps", help="skip offers slower than this"),
    exclude_geo: str = typer.Option("", "--exclude-geo", help="comma-separated substrings to avoid, e.g. 'CN'"),
    ttl: float = typer.Option(DEFAULT_TTL_HOURS, "--ttl"),
    port: int = typer.Option(DEFAULT_PORT, "--port"),
    hf_token: Optional[str] = typer.Option(None, "--hf-token", envvar="HF_TOKEN"),
    yes: bool = typer.Option(False, "--yes", "-y"),
    watch_after: bool = typer.Option(True, "--watch/--no-watch"),
    set_default: bool = typer.Option(False, "--set-default"),
    conductor: bool = typer.Option(False, "--conductor", help="also set Conductor's default model once serving"),
) -> None:
    """Launch any catalogue model on the cheapest hardware that fits it."""
    try:
        m = model_mod.get(model)
    except KeyError as exc:
        _fail(str(exc))

    with _client() as c:
        with console.status("fetching the market…"):
            offers = planner.fetch_market(c)
        fits = planner.fit_model(m, offers, context=context, fp8_kv=fp8_kv, max_dph=max_dph)
        if min_tokps:
            fits = [f for f in fits if f.est_tokps and f.est_tokps >= min_tokps]
        for bad in [g.strip() for g in exclude_geo.split(",") if g.strip()]:
            fits = [f for f in fits if bad.lower() not in str(f.offer.get("geolocation", "")).lower()]
        if not fits:
            _fail(f"no offer fits {m.label} at {context:,} ctx under those constraints.")

        chosen = next((f for f in fits if int(f.offer["id"]) == offer), None) if offer else fits[0]
        if chosen is None:
            _fail(f"offer {offer} is not among the {len(fits)} that fit; re-run `gpuctl plan {model}`.")

        r = _recipe_for(m, chosen, context=context, fp8_kv=fp8_kv)
        s = provision.offer_summary(chosen.offer)
        plan_t = Table(box=None, pad_edge=False)
        plan_t.add_column(style="dim"); plan_t.add_column()
        plan_t.add_row("model", f"{m.label}  [dim]{m.hf_id}[/]")
        plan_t.add_row("weights", f"{m.weights_gib:.1f} GiB  ({m.quant})")
        plan_t.add_row("hardware", f"{chosen.num_gpus}x {chosen.gpu_name} — {chosen.vram_gib:.0f} GiB  ({s['geo']})")
        plan_t.add_row("parallel", chosen.parallel)
        plan_t.add_row("context", f"{context:,}{' (fp8 KV)' if fp8_kv else ''}  [dim]max {chosen.max_context:,}[/]")
        plan_t.add_row("est decode", f"{chosen.est_tokps:.0f} tok/s  [dim](modelled, ±25%)[/]" if chosen.est_tokps else "?")
        plan_t.add_row("disk / net", f"{r.disk_gb} GB  /  ↓{s['inet_down']:.0f} Mbps")
        plan_t.add_row("price", f"[bold]{_money(chosen.dph)}/hr[/]")
        plan_t.add_row("auto-destroy", f"after {ttl:g}h  →  ~{_money(chosen.dph * ttl)} max" if ttl > 0
                       else "[bold red]NONE — bills until destroyed[/]")
        console.print(Panel(plan_t, title="about to spend money", border_style="yellow"))

        if not yes and not typer.confirm("Rent this?", default=False):
            console.print("[dim]aborted — nothing rented.[/]")
            raise typer.Exit(1)
        try:
            dep = provision.launch(c, r, chosen.offer, port=port, disk_gb=r.disk_gb,
                                   ttl_hours=ttl, hf_token=hf_token)
        except VastError as exc:
            _fail(str(exc))

    state.save(dep)
    console.print(f"[green]rented[/] instance [bold]{dep.instance_id}[/] at {_money(dep.dph_at_launch)}/hr")
    if watch_after:
        _watch_loop(dep, interval=10.0, set_default=set_default, reap=True, conductor=conductor)
    else:
        console.print(f"[dim]track it with:[/] gpuctl watch {dep.instance_id}")


# ------------------------------------------------------------ models / plan


@app.command("models")
def list_models(
    context: int = typer.Option(32768, "--context", "-c", help="context window to size for"),
    fp8_kv: bool = typer.Option(False, "--fp8-kv", help="halve KV cache with fp8"),
) -> None:
    """Show servable models and how much VRAM each needs."""
    table = Table(title=f"models sized for {context:,}-token context"
                        f"{' (fp8 KV)' if fp8_kv else ''}", header_style="bold")
    table.add_column("key", style="bold cyan")
    table.add_column("model")
    table.add_column("quant")
    table.add_column("weights", justify="right")
    table.add_column("KV/tok", justify="right")
    table.add_column("VRAM need", justify="right", style="bold")
    table.add_column("active", justify="right", style="dim")
    for m in sorted(model_mod.MODELS.values(), key=lambda x: x.vram_gib(context, fp8_kv=fp8_kv)):
        need = m.vram_gib(context, fp8_kv=fp8_kv)
        table.add_row(
            m.key, m.label, m.quant,
            f"{m.weights_gib:.1f}G",
            f"{m.kv_bytes_per_token / 1024:.0f}K",
            f"{need:.1f}G",
            f"{m.active_b:.1f}B" if m.active_b else "dense",
        )
    console.print(table)
    console.print(f"[dim]{model_mod.KV_NOTE}[/]")


@app.command()
def plan(
    model: Optional[str] = typer.Argument(None, help="model key; default = price the whole catalogue"),
    context: int = typer.Option(32768, "--context", "-c"),
    fp8_kv: bool = typer.Option(False, "--fp8-kv"),
    limit: int = typer.Option(5, "--limit", "-n", help="offers to show for a single model"),
    max_dph: Optional[float] = typer.Option(None, "--max-dph"),
    min_cuda: float = typer.Option(planner.MIN_CUDA_FOR_LATEST, "--min-cuda",
                                   help="lower only if you pin an older vLLM image"),
) -> None:
    """Find the cheapest GPU offer that can actually host a model."""
    with _client() as c:
        with console.status("fetching the market…"):
            offers = planner.fetch_market(c, min_cuda=min_cuda)
    if not offers:
        _fail("no offers matched the baseline filters; try --min-cuda 12.1")
    console.print(f"[dim]{len(offers)} rentable offers (cuda ≥ {min_cuda}, verified, disk ≥ 120G)[/]\n")

    if model:
        try:
            m = model_mod.get(model)
        except KeyError as exc:
            _fail(str(exc))
        fits = planner.fit_model(m, offers, context=context, fp8_kv=fp8_kv, max_dph=max_dph)
        if not fits:
            _fail(f"nothing on the market can host {m.label} at {context:,} context "
                  f"({m.vram_gib(context, fp8_kv=fp8_kv):.0f} GiB needed).")
        table = Table(title=f"{m.label} @ {context:,} ctx — needs "
                            f"{m.vram_gib(context, fp8_kv=fp8_kv):.0f} GiB", header_style="bold")
        for col, kw in [("offer", {"style": "bold"}), ("gpu", {}), ("VRAM", {"justify": "right"}),
                        ("par", {}), ("$/hr", {"justify": "right"}), ("$/3h", {"justify": "right", "style": "dim"}),
                        ("est tok/s", {"justify": "right"}), ("max ctx", {"justify": "right"}),
                        ("location", {"style": "dim"})]:
            table.add_column(col, **kw)
        for f in fits[:limit]:
            table.add_row(
                str(f.offer.get("id")), f"{f.num_gpus}x {f.gpu_name}", f"{f.vram_gib:.0f}G",
                f.parallel, f"{f.dph:.3f}", f"{f.dph * 3:.2f}",
                f"{f.est_tokps:.0f}" if f.est_tokps else "—",
                f"{f.max_context // 1024}k", str(f.offer.get("geolocation"))[:22],
            )
        console.print(table)
        console.print("[dim]est tok/s = bandwidth model (docs/METHOD.md §2), ±25%. "
                      "`gpuctl bench` measures the truth.[/]")
        return

    table = Table(title=f"cheapest host for each model @ {context:,} ctx"
                        f"{' (fp8 KV)' if fp8_kv else ''}", header_style="bold")
    table.add_column("model", style="bold cyan")
    table.add_column("needs", justify="right")
    table.add_column("cheapest fit")
    table.add_column("VRAM", justify="right")
    table.add_column("$/hr", justify="right", style="bold")
    table.add_column("$/3h", justify="right", style="dim")
    table.add_column("est tok/s", justify="right")
    for m in sorted(model_mod.MODELS.values(), key=lambda x: x.vram_gib(context, fp8_kv=fp8_kv)):
        f = planner.cheapest(m, offers, context=context, fp8_kv=fp8_kv)
        need = f"{m.vram_gib(context, fp8_kv=fp8_kv):.0f}G"
        if not f:
            table.add_row(m.key, need, "[dim]— nothing available —[/]", "", "", "", "")
            continue
        table.add_row(m.key, need, f"{f.num_gpus}x {f.gpu_name} ({f.parallel})",
                      f"{f.vram_gib:.0f}G", f"{f.dph:.3f}", f"{f.dph * 3:.2f}",
                      f"{f.est_tokps:.0f}" if f.est_tokps else "—")
    console.print(table)
    console.print("[dim]`gpuctl plan <key>` for alternatives · `gpuctl up <recipe>` to launch[/]")


# -------------------------------------------------------------------------- up


@app.command()
def up(
    recipe: str = typer.Argument(..., help="recipe key (see `gpuctl recipes`)"),
    offer: Optional[int] = typer.Option(None, "--offer", help="specific offer id; default = cheapest"),
    max_dph: Optional[float] = typer.Option(None, "--max-dph", help="hard price ceiling, $/hr"),
    ttl: float = typer.Option(DEFAULT_TTL_HOURS, "--ttl", help="auto-destroy deadline in hours (0 = none)"),
    disk: Optional[int] = typer.Option(None, "--disk", help="disk GB (default: per recipe)"),
    port: int = typer.Option(DEFAULT_PORT, "--port", help="container port for vLLM"),
    model: Optional[str] = typer.Option(None, "--model", help="override the HuggingFace model id"),
    hf_token: Optional[str] = typer.Option(None, "--hf-token", envvar="HF_TOKEN", help="for gated models"),
    yes: bool = typer.Option(False, "--yes", "-y", help="skip the spend confirmation"),
    watch_after: bool = typer.Option(True, "--watch/--no-watch", help="track until it is serving"),
    set_default: bool = typer.Option(False, "--set-default", help="make this opencode's default model"),
    conductor: bool = typer.Option(False, "--conductor", help="also set Conductor's default model once serving"),
) -> None:
    """Rent a GPU box and start vLLM on it."""
    try:
        r = recipe_mod.get(recipe)
    except (KeyError, recipe_mod.RecipeError) as exc:
        _fail(str(exc))
    if model:
        r = dataclasses.replace(r, model=model)

    ceiling = max_dph if max_dph is not None else min(r.max_dph, DEFAULT_MAX_DPH)
    disk_gb = disk or r.disk_gb or DEFAULT_DISK_GB

    with _client() as c:
        try:
            offers = provision.search(c, r, max_dph=ceiling, limit=25)
        except VastError as exc:
            _fail(str(exc))
        if not offers:
            _fail(f"no offers for {r.num_gpus}x {r.gpu_name} under ${ceiling:.2f}/hr.")

        chosen = None
        if offer is not None:
            chosen = next((o for o in offers if int(o.get("id", -1)) == offer), None)
            if chosen is None:
                _fail(f"offer {offer} is not in the current result set; re-run `gpuctl search {r.key}`.")
        else:
            chosen = offers[0]

        s = provision.offer_summary(chosen)
        if s["dph"] > ceiling:
            _fail(f"offer {s['id']} is {_money(s['dph'])}/hr, above the ${ceiling:.2f} ceiling.")

        projected = s["dph"] * ttl if ttl > 0 else 0.0
        plan = Table(box=None, pad_edge=False)
        plan.add_column(style="dim")
        plan.add_column()
        plan.add_row("recipe", f"{r.key} — {r.title}")
        plan.add_row("hardware", f"{s['gpu']}  ({s['geo']})")
        plan.add_row("offer", f"{s['id']}  rel {s['reliability'] * 100:.1f}%  net ↓{s['inet_down']:.0f} Mbps")
        plan.add_row("model", r.model)
        plan.add_row("disk", f"{disk_gb} GB")
        plan.add_row("price", f"[bold]{_money(s['dph'])}/hr[/]")
        plan.add_row(
            "auto-destroy",
            f"after {ttl:g}h  →  ~{_money(projected)} max" if ttl > 0 else "[bold red]NONE — bills until you destroy it[/]",
        )
        console.print(Panel(plan, title="about to spend money", border_style="yellow"))

        if not yes and not typer.confirm("Rent this?", default=False):
            console.print("[dim]aborted — nothing rented.[/]")
            raise typer.Exit(1)

        try:
            dep = provision.launch(
                c, r, chosen,
                port=port, disk_gb=disk_gb, ttl_hours=ttl, hf_token=hf_token,
            )
        except VastError as exc:
            _fail(str(exc))

    state.save(dep)
    console.print(
        f"[green]rented[/] instance [bold]{dep.instance_id}[/] "
        f"({dep.gpu_label}) at {_money(dep.dph_at_launch)}/hr"
    )
    if ttl > 0:
        console.print(f"[dim]auto-destroy deadline: {time.strftime('%H:%M:%S', time.localtime(dep.deadline))}[/]")

    if watch_after:
        _watch_loop(dep, interval=10.0, set_default=set_default, reap=True, conductor=conductor)
    else:
        console.print(f"[dim]track it with:[/] gpuctl watch {dep.instance_id}")


# ------------------------------------------------------------------ ps / watch


def _status_table(snaps: list[Any]) -> Table:
    table = Table(header_style="bold", expand=False)
    table.add_column("instance", style="bold")
    table.add_column("recipe")
    table.add_column("gpu")
    table.add_column("phase")
    table.add_column("age", justify="right")
    table.add_column("spent", justify="right")
    table.add_column("$/hr", justify="right", style="dim")
    table.add_column("ttl left", justify="right")
    table.add_column("detail", style="dim", max_width=46, overflow="ellipsis")
    for snap in snaps:
        dep = snap.dep
        left = "—"
        if dep.ttl_hours > 0:
            remaining = dep.deadline - time.time()
            left = _dur(remaining) if remaining > 0 else "[red]expired[/]"
        table.add_row(
            str(dep.instance_id),
            dep.recipe,
            dep.gpu_label or "?",
            Text(snap.phase.value, style=PHASE_STYLE[snap.phase]),
            _dur(dep.age_hours() * 3600),
            _money(snap.cost),
            f"{dep.dph_at_launch:.3f}",
            left,
            snap.detail or snap.status_msg,
        )
    return table


@app.command()
def ps(
    all_: bool = typer.Option(False, "--all", "-a", help="include destroyed deployments"),
    deep: bool = typer.Option(True, "--deep/--shallow", help="probe the vLLM endpoint too"),
) -> None:
    """List tracked deployments, their phase and what they have cost so far."""
    deps = state.load_all(include_destroyed=all_)
    if not deps:
        console.print("[dim]nothing tracked. `gpuctl up smoke` to start.[/]")
        return
    with _client() as c:
        snaps = [snapshot(c, d, deep=deep) for d in deps]
    console.print(_status_table(snaps))
    total = sum(s.cost for s in snaps if not s.dep.destroyed_at)
    console.print(f"[dim]live spend so far: [/][bold]{_money(total)}[/]")


def _watch_loop(dep: Deployment, *, interval: float, set_default: bool, reap: bool,
                conductor: bool = False) -> None:
    """Poll until the box is serving, then wire it into opencode."""
    config_path = opencode.target_path()
    linked = False
    with _client() as c, Live(console=console, refresh_per_second=4) as live:
        while True:
            fresh = state.find(dep.instance_id) or dep
            snap = snapshot(c, fresh)

            body = _status_table([snap])
            hint = ""
            if snap.phase is Phase.LOADING:
                hint = "Vast is pulling the image — first boot on a cold host is the slow part."
            elif snap.phase is Phase.RUNNING:
                hint = f"container up; vLLM is downloading/loading {fresh.model}."
            elif snap.phase in (Phase.SERVING, Phase.LINKED):
                hint = f"endpoint: {snap.endpoint}"
            live.update(Panel(body, title=f"watching {dep.instance_id}", subtitle=hint, border_style="cyan"))

            if snap.phase is Phase.SERVING and not linked:
                result = link_opencode(snap, config_path=config_path, set_default=set_default)
                linked = True
                live.update(Panel(_status_table([snapshot(c, state.find(dep.instance_id) or fresh)]),
                                  title=f"watching {dep.instance_id}", border_style="green"))
                console.print(
                    f"[bold green]opencode configured[/] → provider [bold]{result.provider_id}[/] "
                    f"in {result.path}"
                )
                if result.backup:
                    console.print(f"[dim]backup: {result.backup}[/]")
                if set_default:
                    console.print("[dim]set as opencode's default model.[/]")
                if conductor:
                    _apply_conductor(state.find(dep.instance_id) or fresh)
                break

            if snap.phase in (Phase.ERROR, Phase.STOPPED, Phase.GONE):
                console.print(f"[bold red]{snap.phase.value}[/]: {snap.detail or snap.status_msg}")
                console.print(f"[dim]logs:[/] gpuctl logs {dep.instance_id}")
                break

            if reap and fresh.expired():
                console.print(f"[bold red]TTL reached[/] — destroying {dep.instance_id} to stop the meter.")
                _destroy(c, fresh)
                break

            time.sleep(interval)

    if linked:
        d = state.find(dep.instance_id)
        model_id = (d.notes.get("linked_model_id") if d else None) or dep.served_name
        console.print(
            Panel(
                f"Use it in opencode:\n\n"
                f"  [bold]opencode --model {dep.provider_id}/{model_id}[/]\n\n"
                f"or pick “{dep.provider_id}” from the model switcher.\n"
                f"When you are done: [bold]gpuctl down {dep.instance_id}[/]",
                title="ready", border_style="green",
            )
        )


@app.command()
def watch(
    ref: Optional[str] = typer.Argument(None, help="instance id or recipe key"),
    interval: float = typer.Option(10.0, "--interval", "-i", help="poll seconds"),
    set_default: bool = typer.Option(False, "--set-default"),
    reap: bool = typer.Option(True, "--reap/--no-reap", help="auto-destroy at the TTL deadline"),
    conductor: bool = typer.Option(False, "--conductor", help="also set Conductor's default model once serving"),
) -> None:
    """Track a deployment until it serves, then configure opencode."""
    _watch_loop(_need(ref), interval=interval, set_default=set_default, reap=reap,
                conductor=conductor)


# ------------------------------------------------------------- conductor

conductor_app = typer.Typer(
    no_args_is_help=False,
    help="Point Conductor's default model at a gpuctl deployment.",
)
app.add_typer(conductor_app, name="conductor")


@conductor_app.callback(invoke_without_command=True)
def conductor_default(
    ctx: typer.Context,
    settings: Optional[str] = typer.Option(
        None, "--settings",
        help="settings.toml to read (default: ~/.conductor/settings.toml; "
             "pass .conductor/settings.toml for a project's committed defaults)",
    ),
) -> None:
    """Show Conductor's current model settings."""
    if ctx.invoked_subcommand is not None:
        return
    try:
        view = conductor_mod.read(Path(settings).expanduser() if settings else None)
    except conductor_mod.ConductorError as exc:
        _fail(str(exc))
    if not view.exists:
        console.print(f"[yellow]no Conductor settings at[/] {view.path}")
        console.print("[dim]`gpuctl conductor set --create` will write a new file.[/]")
        return

    table = Table(box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("settings", str(view.path))
    for key in conductor_mod.MANAGED_KEYS:
        table.add_row(f"models.{key}", view.get(key) or "[dim]unset[/]")
    if view.opencode_executable:
        table.add_row("opencode path", view.opencode_executable)

    # Say whether the configured model is actually reachable right now.
    live = {d.provider_id: d for d in state.load_all() if d.linked_at}
    current = view.get("default") or ""
    provider = current.split("/")[0]
    if provider in live:
        table.add_row("status", f"[green]points at live deployment {live[provider].instance_id}[/]")
    elif provider.startswith("vast-"):
        table.add_row("status", "[bold red]points at a gpuctl provider that is no longer live[/]")
    console.print(Panel(table, title="conductor", border_style="cyan"))
    if provider.startswith("vast-") and provider not in live:
        console.print("[dim]`gpuctl conductor revert` restores the previous model.[/]")


@conductor_app.command("set")
def conductor_set(
    ref: Optional[str] = typer.Argument(None, help="instance id or recipe key"),
    review: bool = typer.Option(False, "--review", help="also set the code-review model"),
    create: bool = typer.Option(False, "--create", help="write a settings file if none exists"),
    settings: Optional[str] = typer.Option(
        None, "--settings",
        help="settings.toml to edit (default: ~/.conductor/settings.toml)",
    ),
) -> None:
    """Set Conductor's default model to a linked deployment."""
    dep = _need(ref)
    target = Path(settings).expanduser() if settings else conductor_mod.SETTINGS_PATH
    try:
        if create:
            target.parent.mkdir(parents=True, exist_ok=True)
        edit = link_conductor(dep, also_review=review, path=target)
    except (conductor_mod.ConductorError, RuntimeError) as exc:
        _fail(str(exc))
    for key, value in edit.changed.items():
        was = edit.previous.get(key)
        console.print(f"[green]set[/] models.{key} = [bold]{value}[/]"
                      f"[dim]{f'  (was {was})' if was else '  (was unset)'}[/]")
    console.print(f"[dim]{edit.path}[/]")
    if edit.backup:
        console.print(f"[dim]backup: {edit.backup}[/]")


@conductor_app.command("revert")
def conductor_revert(ref: Optional[str] = typer.Argument(None)) -> None:
    """Restore the Conductor model settings gpuctl changed."""
    dep = state.resolve(ref) if ref else None
    if dep is None:
        candidates = [d for d in state.load_all(include_destroyed=True) if d.conductor_prev]
        if not candidates:
            console.print("[dim]gpuctl has not changed Conductor's settings.[/]")
            return
        dep = candidates[-1]
    try:
        restored = unlink_conductor(dep)
    except conductor_mod.ConductorError as exc:
        _fail(str(exc))
    if restored:
        console.print(f"[green]reverted[/] models.{', models.'.join(sorted(set(restored)))}")
    else:
        console.print("[dim]nothing to revert.[/]")


# -------------------------------------------------------------- link / logs


@app.command()
def link(
    ref: Optional[str] = typer.Argument(None),
    set_default: bool = typer.Option(False, "--set-default"),
    config: Optional[str] = typer.Option(None, "--config", help="opencode config to edit"),
    project: bool = typer.Option(False, "--project", help="write ./opencode.json instead of the global config"),
    conductor: bool = typer.Option(False, "--conductor", help="also set Conductor's default model"),
) -> None:
    """Write a serving deployment into opencode's config now."""
    dep = _need(ref)
    with _client() as c:
        snap = snapshot(c, dep)
    if snap.phase not in (Phase.SERVING, Phase.LINKED):
        _fail(f"instance {dep.instance_id} is '{snap.phase.value}', not serving yet ({snap.detail}).")
    result = link_opencode(
        snap, config_path=opencode.target_path(config, project), set_default=set_default
    )
    console.print(f"[green]linked[/] {result.provider_id} → {result.path}")
    if result.backup:
        console.print(f"[dim]backup: {result.backup}[/]")
    if conductor:
        _apply_conductor(state.find(dep.instance_id) or dep)


@app.command()
def unlink(ref: Optional[str] = typer.Argument(None)) -> None:
    """Remove a deployment's provider block from opencode's config."""
    dep = _need(ref)
    if unlink_opencode(dep):
        console.print(f"[green]unlinked[/] {dep.provider_id} from {dep.opencode_target}")
    else:
        console.print("[dim]nothing to unlink.[/]")


@app.command()
def logs(
    ref: Optional[str] = typer.Argument(None),
    tail: int = typer.Option(200, "--tail", "-n"),
) -> None:
    """Fetch container logs (including vLLM's startup output)."""
    dep = _need(ref)
    with _client() as c:
        try:
            console.print(c.logs(dep.instance_id, tail=tail))
        except VastError as exc:
            _fail(str(exc))


@app.command()
def ssh(
    ref: Optional[str] = typer.Argument(None),
    exec_: bool = typer.Option(False, "--exec", help="run ssh instead of printing the command"),
    tunnel: bool = typer.Option(False, "--tunnel", help="forward the vLLM port to localhost"),
) -> None:
    """Print (or run) the ssh command for a deployment."""
    dep = _need(ref)
    with _client() as c:
        inst = c.get_instance(dep.instance_id)
    if not inst:
        _fail(f"instance {dep.instance_id} is not on the account any more.")
    target = ssh_target(inst)
    if not target:
        _fail("no ssh endpoint yet — the instance is probably still loading.")
    host, port = target
    cmd = ["ssh", "-p", str(port), f"root@{host}"]
    if tunnel:
        cmd[1:1] = ["-L", f"{dep.port}:localhost:{dep.port}"]
    if exec_:
        raise typer.Exit(subprocess.call(cmd))
    console.print(" ".join(cmd))


# ------------------------------------------------------------ down / reap


def _apply_conductor(dep: Deployment) -> None:
    """Best-effort: a Conductor edit must never fail a launch."""
    try:
        edit = link_conductor(dep)
    except (conductor_mod.ConductorError, RuntimeError) as exc:
        err.print(f"[yellow]Conductor not updated:[/] {exc}")
        return
    console.print(f"[green]Conductor default model[/] → [bold]{edit.changed['default']}[/]"
                  f"[dim]  (was {edit.previous.get('default') or 'unset'})[/]")


def _destroy(client: VastClient, dep: Deployment) -> None:
    try:
        client.destroy_instance(dep.instance_id)
    except VastError as exc:
        err.print(f"[yellow]destroy call failed:[/] {exc}")
    unlink_opencode(dep)
    # Never leave Conductor pointing at a destroyed instance.
    try:
        if unlink_conductor(dep):
            console.print("[dim]reverted Conductor's model setting.[/]")
    except conductor_mod.ConductorError as exc:
        err.print(f"[yellow]could not revert Conductor:[/] {exc}")
    dep.destroyed_at = time.time()
    state.save(dep)


@app.command()
def down(
    ref: Optional[str] = typer.Argument(None),
    all_: bool = typer.Option(False, "--all", help="destroy every tracked deployment"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Destroy an instance and remove it from opencode."""
    targets = state.load_all() if all_ else [_need(ref)]
    if not targets:
        console.print("[dim]nothing to destroy.[/]")
        return
    for dep in targets:
        if not yes:
            spent = _money(dep.accrued_cost())
            if not typer.confirm(f"Destroy {dep.instance_id} ({dep.gpu_label}, ~{spent} spent)?", default=True):
                continue
        with _client() as c:
            _destroy(c, dep)
        console.print(f"[green]destroyed[/] {dep.instance_id} — meter stopped (~{_money(dep.accrued_cost())} spent)")


@app.command()
def reap(
    yes: bool = typer.Option(False, "--yes", "-y", help="destroy expired deployments without asking"),
) -> None:
    """Destroy any tracked deployment past its TTL. Safe to run from cron."""
    expired = [d for d in state.load_all() if d.expired()]
    if not expired:
        console.print("[dim]nothing expired.[/]")
        return
    for dep in expired:
        over = _dur(time.time() - dep.deadline)
        if not yes and not typer.confirm(f"{dep.instance_id} is {over} past its TTL. Destroy?", default=True):
            continue
        with _client() as c:
            _destroy(c, dep)
        console.print(f"[green]reaped[/] {dep.instance_id} (~{_money(dep.accrued_cost())} spent)")


# ------------------------------------------------------------------- bench


@app.command()
def bench(
    ref: Optional[str] = typer.Argument(None),
    prompt: str = typer.Option(
        "Write a detailed technical explanation of how paged attention works in vLLM.",
        "--prompt", "-p",
    ),
    max_tokens: int = typer.Option(256, "--max-tokens"),
) -> None:
    """Measure real decode tok/s — the number every estimate above is missing."""
    import json

    import httpx

    dep = _need(ref)
    with _client() as c:
        snap = snapshot(c, dep)
    if snap.phase not in (Phase.SERVING, Phase.LINKED) or not snap.endpoint:
        _fail(f"instance {dep.instance_id} is '{snap.phase.value}', not serving.")

    model_id = (snap.probe.best_model if snap.probe else None) or dep.served_name
    ok, msg = completion_smoke(snap.endpoint, dep.serve_key, model_id)
    if not ok:
        _fail(f"smoke test failed: {msg}")

    url = snap.endpoint.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        # Count tokens the way the server counts them. One SSE chunk usually
        # carries one token, but that is a convention, not a guarantee.
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_token_at: float | None = None
    tokens = 0
    reported: int | None = None
    with httpx.stream(
        "POST", url, json=body, timeout=600.0,
        headers={"Authorization": f"Bearer {dep.serve_key}"},
    ) as r:
        if r.status_code != 200:
            _fail(f"HTTP {r.status_code}: {r.read()[:200]!r}")
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except ValueError:
                continue
            if event.get("usage"):
                reported = event["usage"].get("completion_tokens")
            choices = event.get("choices") or []
            if not (choices and (choices[0].get("delta") or {}).get("content")):
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter()
            tokens += 1
    finished = time.perf_counter()

    if not tokens or first_token_at is None:
        _fail("no tokens were streamed back.")
    ttft = first_token_at - started
    decode_s = max(finished - first_token_at, 1e-6)
    counted = reported or tokens
    tokps = (counted - 1) / decode_s if counted > 1 else 0.0

    est = dep.notes.get("est_tokps", "?")
    table = Table(box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("hardware", f"{dep.gpu_label}  ({_money(dep.dph_at_launch)}/hr)")
    table.add_row("model", model_id)
    table.add_row("TTFT", f"{ttft * 1000:.0f} ms")
    source = "server-reported" if reported else "chunk count"
    table.add_row("decode", f"[bold]{tokps:.1f} tok/s[/]  ({counted} tokens, {source}, in {decode_s:.1f}s)")
    table.add_row("ITL / TPOT", f"{1000 / tokps:.0f} ms" if tokps else "—")
    table.add_row("doc estimate", f"{est}   [dim]({dep.notes.get('doc_ref', '')})[/]")
    table.add_row("cost of this run", _money(dep.dph_at_launch * (finished - started) / 3600))
    console.print(Panel(table, title="measured", border_style="green"))


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        err.print("\n[dim]interrupted — instances are still running; `gpuctl ps` to check.[/]")
        sys.exit(130)


if __name__ == "__main__":
    main()
