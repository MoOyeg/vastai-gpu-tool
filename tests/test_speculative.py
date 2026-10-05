"""Speculative decoding: config building, validation, and acceptance metrics."""
import json

import pytest

from gpuctl import provision, recipes, speculative as sp
from gpuctl.cli import _resolve_spec

BASE = '''
title = "spec"
gpu_name = "RTX 3090"
num_gpus = 2
model_key = "llama70b"
vllm_args = ["--tensor-parallel-size", "2"]
'''


def write(d, body, name="spec.toml"):
    (d / name).write_text(BASE + body)
    recipes.all_recipes.cache_clear()


# ------------------------------------------------------------------- building


def test_ngram_needs_no_draft_model():
    s = sp.from_mapping({"method": "ngram", "num_speculative_tokens": 5})
    assert not s.needs_draft_model and s.model is None
    assert json.loads(s.to_json()) == {"method": "ngram", "num_speculative_tokens": 5}


def test_json_is_passed_as_one_shell_word():
    args = sp.from_mapping({"method": "ngram", "num_speculative_tokens": 3}).cli_args()
    assert args[0] == "--speculative-config"
    assert args[1].startswith("'{") and args[1].endswith("}'")
    assert "'" not in args[1][1:-1], "inner payload must not contain single quotes"


def test_passthrough_options_are_preserved():
    s = sp.from_mapping({"method": "ngram", "num_speculative_tokens": 5,
                         "prompt_lookup_max": 4, "prompt_lookup_min": 2})
    body = json.loads(s.to_json())
    assert body["prompt_lookup_max"] == 4 and body["prompt_lookup_min"] == 2


def test_mtp_is_recognised_as_needing_no_draft():
    s = sp.from_mapping({"method": "deepseek_mtp", "num_speculative_tokens": 3})
    assert s.is_mtp and not s.needs_draft_model


@pytest.mark.parametrize("body,fragment", [
    ({"method": "nope", "num_speculative_tokens": 1}, "unknown method"),
    ({"method": "ngram"}, "num_speculative_tokens` is required"),
    ({"num_speculative_tokens": 2}, "`method` is required"),
    ({"method": "ngram", "num_speculative_tokens": 0}, "at least 1"),
    ({"method": "ngram", "num_speculative_tokens": "five"}, "whole number"),
    ({"method": "eagle3", "num_speculative_tokens": 2}, "needs a draft checkpoint"),
    ({"method": "draft_model", "num_speculative_tokens": 2}, "needs a draft checkpoint"),
    ({"method": "deepseek_mtp", "num_speculative_tokens": 1, "model": "x/y"},
     "must not be set"),
    ({"method": "ngram", "num_speculative_tokens": 1, "lookup_max": 4}, "unknown field"),
])
def test_bad_configs_are_rejected(body, fragment):
    with pytest.raises(sp.SpeculativeError) as exc:
        sp.from_mapping(body)
    assert fragment in str(exc.value)


# -------------------------------------------------------------- in a recipe


def test_recipe_speculative_table_reaches_the_onstart(recipe_dir):
    write(recipe_dir, '\n[speculative]\nmethod = "ngram"\nnum_speculative_tokens = 5\n')
    r = recipes.get("spec")
    assert r.speculative.method == "ngram"
    onstart = provision.build_onstart(r, port=8000)
    assert '--speculative-config \'{"method":"ngram","num_speculative_tokens":5}\'' in onstart
    assert onstart.count("'") % 2 == 0


def test_recipe_without_the_table_has_no_speculation(recipe_dir):
    write(recipe_dir, "")
    r = recipes.get("spec")
    assert r.speculative is None and r.spec_args == []


def test_bad_speculative_table_fails_with_the_filename(recipe_dir):
    write(recipe_dir, '\n[speculative]\nmethod = "eagle3"\nnum_speculative_tokens = 2\n')
    with pytest.raises(recipes.RecipeError) as exc:
        recipes.all_recipes()
    assert "spec.toml" in str(exc.value)
    assert "draft checkpoint" in str(exc.value)


def test_shipped_mtp_recipes_match_their_declared_heads():
    """Only set where config.json declares num_nextn_predict_layers > 0:
    MiniMax-M3 declares 1, DeepSeek V4.1 Flash declares 3."""
    assert recipes.get("minimax-m3").speculative.method == "minimax_m3_mtp"
    assert recipes.get("minimax-m3").speculative.num_speculative_tokens == 1
    ds = recipes.get("deepseek-v4.1-flash-max").speculative
    assert ds.method == "deepseek_mtp" and ds.num_speculative_tokens == 3


def test_models_without_declared_heads_claim_no_mtp():
    from gpuctl.models import MODELS
    assert all(m.mtp_method is None for m in MODELS.values())


# ------------------------------------------------------------- CLI resolution


def ngram5():
    return sp.from_mapping({"method": "ngram", "num_speculative_tokens": 5})


def test_no_flags_keeps_the_recipe_setting():
    cur = ngram5()
    assert _resolve_spec(current=cur, method=None, draft_model=None,
                         tokens=None, disable=False) is cur


def test_no_spec_disables_it():
    assert _resolve_spec(current=ngram5(), method=None, draft_model=None,
                         tokens=None, disable=True) is None


def test_tokens_override_keeps_the_method():
    got = _resolve_spec(current=ngram5(), method=None, draft_model=None,
                        tokens=2, disable=False)
    assert got.method == "ngram" and got.num_speculative_tokens == 2


def test_auto_uses_the_models_own_head_when_it_has_one():
    got = _resolve_spec(current=None, method="auto", draft_model=None, tokens=None,
                        disable=False, mtp_method="deepseek_mtp", mtp_tokens=3)
    assert got.method == "deepseek_mtp" and got.num_speculative_tokens == 3


def test_auto_falls_back_to_ngram_without_a_head():
    """ngram needs no draft checkpoint, so it is always available."""
    got = _resolve_spec(current=None, method="auto", draft_model=None,
                        tokens=None, disable=False)
    assert got.method == "ngram"


def test_changing_method_drops_options_that_belonged_to_the_old_one():
    cur = sp.from_mapping({"method": "ngram", "num_speculative_tokens": 5,
                           "prompt_lookup_max": 4})
    got = _resolve_spec(current=cur, method="suffix", draft_model=None,
                        tokens=None, disable=False)
    assert got.method == "suffix"
    assert "prompt_lookup_max" not in json.loads(got.to_json())


# ----------------------------------------------------------------- metrics


METRICS = """\
# HELP vllm:spec_decode_num_accepted_tokens Total accepted tokens.
# TYPE vllm:spec_decode_num_accepted_tokens counter
vllm:spec_decode_num_accepted_tokens_total{model_name="x"} 820.0
vllm:spec_decode_num_draft_tokens_total{model_name="x"} 1000.0
vllm:spec_decode_num_drafts_total{model_name="x"} 250.0
vllm:num_requests_running{model_name="x"} 0.0
"""


def test_parses_acceptance_counters():
    a = sp.parse_metrics(METRICS)
    assert a.accepted == 820.0 and a.drafted == 1000.0
    assert a.rate == pytest.approx(0.82)
    assert a.tokens_per_draft == pytest.approx(3.28)


def test_counters_without_the_total_suffix_also_parse():
    a = sp.parse_metrics('vllm:spec_decode_num_accepted_tokens 5.0\n'
                         'vllm:spec_decode_num_draft_tokens 10.0\n')
    assert a.rate == pytest.approx(0.5)


def test_metrics_without_speculation_return_none():
    assert sp.parse_metrics('vllm:num_requests_running{model_name="x"} 0.0\n') is None
    assert sp.parse_metrics("") is None


def test_zero_drafts_has_no_rate():
    a = sp.parse_metrics("vllm:spec_decode_num_draft_tokens_total 0.0\n")
    assert a.rate is None and a.tokens_per_draft is None


def test_delta_isolates_this_run():
    """Counters are cumulative, so the run's own rate needs a before/after diff."""
    from gpuctl.cli import _spec_delta
    before = sp.Acceptance(100.0, 200.0, 50.0)
    after = sp.Acceptance(180.0, 300.0, 75.0)
    d = _spec_delta(before, after)
    assert d.accepted == 80.0 and d.drafted == 100.0
    assert d.rate == pytest.approx(0.8)


def test_delta_falls_back_when_nothing_was_drafted():
    from gpuctl.cli import _spec_delta
    same = sp.Acceptance(100.0, 200.0, 50.0)
    assert _spec_delta(same, same).rate == pytest.approx(0.5), "lifetime, not 0/0"
