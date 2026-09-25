"""Recipes are user-authored TOML now, so loading, overriding and validation
all need to behave predictably."""
import pytest

from gpuctl import provision, recipes

VALID = '''
title = "Test box"
gpu_name = "RTX 3090"
num_gpus = 2
model_key = "llama70b"
vllm_args = ["--tensor-parallel-size", "2"]
'''


def write(d, name, body):
    (d / name).write_text(body)
    recipes.all_recipes.cache_clear()


def test_builtins_load():
    loaded = recipes.all_recipes()
    assert {"smoke", "build-a", "build-b", "build-f", "a6000", "pro6000"} <= set(loaded)
    for r in loaded.values():
        assert r.source.endswith(".toml")
        assert r.num_gpus >= 1 and r.model and r.title


def test_every_builtin_enables_tool_calling():
    """A server without these flags is unusable to an agent client."""
    for key, r in recipes.all_recipes().items():
        assert r.tool_parser, f"{key} has no tool parser"
        onstart = provision.build_onstart(r, port=8000)
        assert "--enable-auto-tool-choice" in onstart, key
        assert f"--tool-call-parser {r.tool_parser}" in onstart, key
        assert len(onstart) <= provision.ONSTART_LIMIT, key


def test_key_defaults_to_filename_stem(recipe_dir):
    write(recipe_dir, "my-rig.toml", VALID)
    assert "my-rig" in recipes.all_recipes()


def test_custom_dir_overrides_builtin(recipe_dir):
    write(recipe_dir, "build-a.toml", VALID + "max_dph = 0.45\n")
    r = recipes.get("build-a")
    assert r.max_dph == 0.45
    assert r.title == "Test box"
    # and still inherits the parser from the catalogue
    assert r.tool_parser == "llama3_json"


def test_model_key_inherits_id_and_parser(recipe_dir):
    write(recipe_dir, "inherit.toml", VALID)
    r = recipes.get("inherit")
    assert r.model == "casperhansen/llama-3.3-70b-instruct-awq"
    assert r.tool_parser == "llama3_json"


def test_explicit_model_overrides_catalogue(recipe_dir):
    write(recipe_dir, "explicit.toml", VALID + 'model = "some/other-model"\ntool_parser = "hermes"\n')
    r = recipes.get("explicit")
    assert r.model == "some/other-model"
    assert r.tool_parser == "hermes"
    assert r.served_name == "other-model"


@pytest.mark.parametrize("body,fragment", [
    (VALID + "tensor_paralel = 2\n", "unknown field"),
    ('title="x"\ngpu_name="RTX 3090"\nnum_gpus=1\nmodel_key="nope"\n', "not in the catalogue"),
    ('title="x"\nnum_gpus=1\nmodel_key="llama70b"\n', "missing required field"),
    ('title="x"\ngpu_name="RTX 3090"\nnum_gpus=1\n', "either `model_key` or `model`"),
    # tomllib's exact wording varies by Python version; None = any message,
    # but it must still be a RecipeError naming the offending file.
    ('title = "unterminated\n', None),
    ('title="x"\ngpu_name="RTX 3090"\nnum_gpus="two"\nmodel_key="llama70b"\n', "whole number"),
    ('title="x"\ngpu_name="RTX 3090"\nnum_gpus=2.5\nmodel_key="llama70b"\n', "whole number"),
    ('title="x"\ngpu_name="RTX 3090"\nnum_gpus=0\nmodel_key="llama70b"\n', "at least 1"),
    (VALID + 'max_dph = "cheap"\n', "must be a number"),
    (VALID.replace('["--tensor-parallel-size", "2"]', '"--tp 2"'), "array of strings"),
])
def test_bad_recipes_raise_clear_errors(recipe_dir, body, fragment):
    write(recipe_dir, "bad.toml", body)
    with pytest.raises(recipes.RecipeError) as exc:
        recipes.all_recipes()
    if fragment is not None:
        assert fragment in str(exc.value)
    assert "bad.toml" in str(exc.value)


def test_numeric_strings_are_accepted(recipe_dir):
    write(recipe_dir, "lenient.toml",
          'title="x"\ngpu_name="RTX 3090"\nnum_gpus="2"\nmax_dph="0.70"\nmodel_key="llama70b"\n')
    r = recipes.get("lenient")
    assert r.num_gpus == 2 and r.max_dph == 0.70


def test_search_query_uses_spaces_not_underscores():
    """gpu_name with underscores matches zero offers, with HTTP 200 and no error."""
    q = recipes.get("build-a").search_query()
    assert q["gpu_name"]["eq"] == "RTX 3090"
    assert q["num_gpus"]["eq"] == 2
    assert q["cuda_max_good"]["gte"] >= 13.0   # CUDA 13 image; see docs/METHOD.md §6


def test_reliability_floor_is_configurable(recipe_dir):
    """Scarce hardware may have exactly one offer; the floor must be relaxable."""
    write(recipe_dir, "scarce.toml", VALID + "min_reliability = 0.90\n")
    assert recipes.get("scarce").search_query()["reliability"]["gte"] == 0.90
    assert recipes.get("build-a").search_query()["reliability"]["gte"] == 0.97


def test_reliability_must_be_a_fraction(recipe_dir):
    write(recipe_dir, "bad.toml", VALID + "min_reliability = 97\n")
    with pytest.raises(recipes.RecipeError, match="fraction between 0 and 1"):
        recipes.all_recipes()


def test_recipe_can_pin_its_own_image(recipe_dir):
    """Kimi K3 ships in its own image; :latest cannot serve it."""
    write(recipe_dir, "pinned.toml", VALID + 'image = "vllm/vllm-openai:kimi-k3"\n')
    assert recipes.get("pinned").image == "vllm/vllm-openai:kimi-k3"
    assert recipes.get("build-a").image == "vllm/vllm-openai:latest"


def test_extra_env_reaches_the_container(recipe_dir):
    write(recipe_dir, "envy.toml", VALID + '\n[extra_env]\nVLLM_FLOAT32_MATMUL_PRECISION = "high"\n')
    env = provision.build_env(port=8000, serve_key="k",
                              extra=recipes.get("envy").extra_env)
    assert env["VLLM_FLOAT32_MATMUL_PRECISION"] == "high"
    assert env["-p 8000:8000"] == "1", "port request must survive the merge"


def test_shipped_reasoning_recipes_are_coherent():
    """The frontier-model recipes must carry their parsers and a sane image."""
    expect = {
        "qwen3.8-27b": ("qwen3_xml", "qwen3", "vllm/vllm-openai:latest"),
        "kimi-k3-max": ("kimi_k3", "kimi_k3", "vllm/vllm-openai:kimi-k3"),
        "mimo-v2.6-pro": ("mimo", "mimo", "vllm/vllm-openai:latest"),
        "minimax-m3": ("minimax_m3", "minimax_m3", "vllm/vllm-openai:latest"),
        "deepseek-v4.1-flash-max": ("deepseek_v41", "deepseek_v41", "vllm/vllm-openai:nightly"),
    }
    for key, (tool, reasoning, image) in expect.items():
        r = recipes.get(key)
        assert r.tool_parser == tool, key
        assert r.image == image, key
        args = r.vllm_args
        assert "--reasoning-parser" in args, key
        assert args[args.index("--reasoning-parser") + 1] == reasoning, key
        onstart = provision.build_onstart(r, port=8000)
        assert f"--tool-call-parser {tool}" in onstart, key
        assert len(onstart) <= provision.ONSTART_LIMIT, key


def test_minimax_keeps_its_mandatory_block_size():
    """--block-size 128 is mandatory on every platform for MiniMax-M3."""
    args = recipes.get("minimax-m3").vllm_args
    assert args[args.index("--block-size") + 1] == "128"


def test_json_chat_template_kwargs_survive_shell_quoting():
    """Where a recipe pins chat-template kwargs, the JSON must reach vllm as one
    argument rather than being split by bash."""
    onstart = provision.build_onstart(recipes.get("deepseek-v4.1-flash-max"), port=8000)
    assert "--default-chat-template-kwargs '{" in onstart
    assert '"reasoning_effort":100' in onstart


def test_every_onstart_has_balanced_quotes():
    """An unbalanced quote would silently swallow the rest of the command."""
    for key, r in recipes.all_recipes().items():
        onstart = provision.build_onstart(r, port=8000)
        assert onstart.count("'") % 2 == 0, f"{key}: unbalanced single quotes"
        assert onstart.count('"') % 2 == 0, f"{key}: unbalanced double quotes"


def test_unknown_recipe_lists_alternatives():
    with pytest.raises(KeyError) as exc:
        recipes.get("does-not-exist")
    assert "build-a" in str(exc.value)
