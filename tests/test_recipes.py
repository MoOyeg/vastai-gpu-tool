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


def test_unknown_recipe_lists_alternatives():
    with pytest.raises(KeyError) as exc:
        recipes.get("does-not-exist")
    assert "build-a" in str(exc.value)
