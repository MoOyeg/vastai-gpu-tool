"""Launch recipes, loaded from TOML rather than hardcoded.

A recipe is one hardware+model configuration worth measuring before committing
to buying it (docs/METHOD.md §8). They are data, not code, so they live in
`*.toml` files and are read at runtime.

Search order — later directories override earlier ones on the same key:

  1. `recipes.d/` shipped inside the package
  2. `~/.config/gpuctl/recipes/`            (per-user; edit freely)
  3. every dir in `$GPUCTL_RECIPES_DIR`     (os.pathsep-separated)

`gpuctl recipes --paths` prints the live list. A recipe's key defaults to its
filename stem, so dropping `my-box.toml` into the user directory adds
`gpuctl up my-box`, and naming a file `build-a.toml` there replaces the builtin.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

from .config import CONFIG_DIR

BUILTIN_DIRNAME = "recipes.d"
USER_RECIPES_DIR = CONFIG_DIR / "recipes"
ENV_VAR = "GPUCTL_RECIPES_DIR"


class RecipeError(RuntimeError):
    """A recipe file is missing a required field or carries an unknown one."""


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
    source: str = ""        # where this recipe was loaded from, for provenance

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


# ------------------------------------------------------------------- loading

_FIELDS = {f.name for f in fields(Recipe)}
_REQUIRED = ("title", "gpu_name", "num_gpus")
# `model_key` is sugar: inherit the HF id and tool parser from the model
# catalogue so a recipe cannot silently drift from it (a missing tool parser is
# what makes a server unusable to an agent client).
_SUGAR = {"model_key"}

# Recipes are hand-written files now, so types are checked on the way in rather
# than left to fail later as a string in a Vast query or a format specifier.
_INT_FIELDS = {"num_gpus", "disk_gb", "min_inet_down", "min_gpu_ram_mb"}
_FLOAT_FIELDS = {"max_dph", "min_cuda"}
_STR_FIELDS = {"title", "gpu_name", "model", "est_tokps", "doc_ref", "notes", "tool_parser"}


def _coerce(name: str, value: Any, source: str) -> Any:
    """Normalise a TOML value to the type the dataclass declares."""
    where = f"{source}: field `{name}`"
    if name in _INT_FIELDS:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise RecipeError(f"{where} must be a whole number, got {value!r}")
        try:
            as_float = float(value)
        except ValueError:
            raise RecipeError(f"{where} must be a whole number, got {value!r}") from None
        if as_float != int(as_float):
            raise RecipeError(f"{where} must be a whole number, got {value!r}")
        return int(as_float)
    if name in _FLOAT_FIELDS:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise RecipeError(f"{where} must be a number, got {value!r}")
        try:
            return float(value)
        except ValueError:
            raise RecipeError(f"{where} must be a number, got {value!r}") from None
    if name in _STR_FIELDS:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        if not isinstance(value, str):
            raise RecipeError(f"{where} must be text, got {value!r}")
        return value
    if name == "vllm_args":
        if not isinstance(value, list):
            raise RecipeError(f"{where} must be an array of strings, got {value!r}")
        out = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (str, int, float)):
                raise RecipeError(f"{where} contains a non-string entry {item!r}")
            out.append(str(item))
        return out
    if name == "query_extra":
        if not isinstance(value, dict):
            raise RecipeError(f"{where} must be a table, got {value!r}")
        return value
    return value


def search_dirs() -> list[Path]:
    """Directories consulted for recipes, lowest precedence first."""
    dirs: list[Path] = []
    try:
        with as_file(files(__package__) / BUILTIN_DIRNAME) as builtin:
            dirs.append(Path(builtin))
    except (FileNotFoundError, ModuleNotFoundError):
        pass
    dirs.append(USER_RECIPES_DIR)
    for raw in os.environ.get(ENV_VAR, "").split(os.pathsep):
        if raw.strip():
            dirs.append(Path(raw.strip()).expanduser())
    return dirs


def _from_mapping(key: str, data: dict[str, Any], source: str) -> Recipe:
    data = dict(data)
    key = str(data.pop("key", key))

    unknown = set(data) - _FIELDS - _SUGAR
    if unknown:
        raise RecipeError(
            f"{source}: unknown field(s) {sorted(unknown)}. "
            f"Valid: {sorted(_FIELDS - {'key', 'source'}) + sorted(_SUGAR)}"
        )

    model_key = data.pop("model_key", None)
    if model_key:
        from .models import MODELS

        m = MODELS.get(str(model_key))
        if m is None:
            raise RecipeError(
                f"{source}: model_key {model_key!r} is not in the catalogue "
                f"({', '.join(sorted(MODELS))}). Set `model` directly instead."
            )
        data.setdefault("model", m.hf_id)
        data.setdefault("tool_parser", m.tool_parser)

    data = {k: _coerce(k, v, source) for k, v in data.items()}

    if "num_gpus" in data and data["num_gpus"] < 1:
        raise RecipeError(f"{source}: num_gpus must be at least 1")
    if "max_dph" in data and data["max_dph"] <= 0:
        raise RecipeError(f"{source}: max_dph must be greater than 0")

    missing = [f for f in _REQUIRED if not data.get(f)]
    if missing:
        raise RecipeError(f"{source}: missing required field(s) {missing}")
    if not data.get("model"):
        raise RecipeError(f"{source}: needs either `model_key` or `model`")
    if not data.get("vllm_args"):
        data["vllm_args"] = []
    if not data.get("tool_parser"):
        # Not fatal — some models genuinely have no parser — but it means any
        # agent client will fail against this recipe, so make it visible.
        data["tool_parser"] = ""

    try:
        return Recipe(key=key, source=source, **data)
    except TypeError as exc:
        raise RecipeError(f"{source}: {exc}") from None


def load_dir(directory: Path) -> dict[str, Recipe]:
    out: dict[str, Recipe] = {}
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.toml")):
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise RecipeError(f"{path}: {exc}") from None
        recipe = _from_mapping(path.stem, data, str(path))
        out[recipe.key] = recipe
    return out


@lru_cache(maxsize=1)
def all_recipes() -> dict[str, Recipe]:
    merged: dict[str, Recipe] = {}
    for directory in search_dirs():
        merged.update(load_dir(directory))
    return merged


def reload() -> dict[str, Recipe]:
    all_recipes.cache_clear()
    return all_recipes()


def get(key: str) -> Recipe:
    recipes = all_recipes()
    try:
        return recipes[key]
    except KeyError:
        raise KeyError(
            f"Unknown recipe {key!r}. Known: {', '.join(sorted(recipes))}"
        ) from None
