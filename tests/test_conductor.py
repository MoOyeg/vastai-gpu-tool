"""Conductor's settings.toml is a hand-maintained file, so edits must be
surgical: change the requested key and nothing else, comments included."""
import time
import tomllib

import pytest

from gpuctl import conductor, state, track
from gpuctl.state import Deployment

REAL_SHAPE = '''"$schema" = "https://conductor.build/schemas/settings.schema.json"
codex_provider = "default"

[git]
branch_prefix_type = "github_username"

# my model prefs — do not reformat
[models]
default = "opus"   # keep me

[models.codex]
default_thinking_level = "high"
review_thinking_level = "high"
'''


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """An isolated settings.toml, also installed as the module default so that
    nothing in the suite can reach the developer's real ~/.conductor."""
    p = tmp_path / "conductor" / "settings.toml"
    p.parent.mkdir()
    p.write_text(REAL_SHAPE)
    monkeypatch.setattr(conductor, "SETTINGS_PATH", p)
    return p


def parse(p):
    return tomllib.loads(p.read_text())


def test_read_reports_current_model(settings):
    view = conductor.read(settings)
    assert view.exists
    assert view.get("default") == "opus"
    assert view.get("review") is None


def test_set_default_leaves_everything_else_alone(settings):
    before = settings.read_text()
    conductor.set_models({"default": "vast-1/llama-70b"}, path=settings)
    after = parse(settings)
    assert after["models"]["default"] == "vast-1/llama-70b"
    # untouched neighbours
    assert after["git"]["branch_prefix_type"] == "github_username"
    assert after["models"]["codex"]["default_thinking_level"] == "high"
    assert after["codex_provider"] == "default"
    assert after["$schema"].endswith("settings.schema.json")
    # the standalone comment and the section order survive
    text = settings.read_text()
    assert "# my model prefs — do not reformat" in text
    assert text.index("[git]") < text.index("[models]") < text.index("[models.codex]")
    assert before != text


def test_inline_comment_is_preserved(settings):
    conductor.set_models({"default": "vast-1/m"}, path=settings)
    assert '# keep me' in settings.read_text()


def test_absent_key_is_appended_after_existing_ones(settings):
    conductor.set_models({"default": "vast-1/m", "review": "vast-1/m"}, path=settings)
    lines = [l.strip() for l in settings.read_text().splitlines()]
    assert lines.index('default = "vast-1/m"  # keep me') < lines.index('review = "vast-1/m"')
    assert parse(settings)["models"]["review"] == "vast-1/m"


def test_revert_restores_exactly(settings):
    original = settings.read_text()
    edit = conductor.set_models({"default": "vast-1/m", "review": "vast-1/m"}, path=settings)
    assert edit.previous == {"default": "opus", "review": None}
    conductor.revert(edit.previous, path=settings)
    after = parse(settings)
    assert after == tomllib.loads(original), "document must round-trip"
    assert "review" not in after["models"], "absent key must be removed, not blanked"


def test_backup_is_written(settings):
    edit = conductor.set_models({"default": "vast-1/m"}, path=settings)
    assert edit.backup and edit.backup.exists()
    assert edit.backup.read_text() == REAL_SHAPE


def test_creates_models_table_when_missing(tmp_path):
    p = tmp_path / "settings.toml"
    p.write_text('codex_provider = "default"\n')
    conductor.set_models({"default": "vast-1/m"}, path=p)
    d = parse(p)
    assert d["models"]["default"] == "vast-1/m"
    assert d["codex_provider"] == "default"


def test_refuses_to_create_file_without_permission(tmp_path):
    missing = tmp_path / "nope" / "settings.toml"
    with pytest.raises(conductor.ConductorError, match="does not exist"):
        conductor.set_models({"default": "x/y"}, path=missing, create=False)
    assert not missing.exists()


def test_refuses_unmanaged_keys(settings):
    with pytest.raises(conductor.ConductorError, match="unmanaged key"):
        conductor.set_models({"default_plan_mode": "true"}, path=settings)
    assert parse(settings)["models"]["default"] == "opus"


def test_rejects_unparseable_settings(tmp_path):
    p = tmp_path / "settings.toml"
    p.write_text('default = "unterminated\n')
    with pytest.raises(conductor.ConductorError, match="not readable TOML"):
        conductor.read(p)


def test_model_ref_is_provider_qualified():
    assert conductor.model_ref("vast-1", "llama-3.3-70b-instruct-awq") == \
        "vast-1/llama-3.3-70b-instruct-awq"


# ------------------------------------------------------- integration with state


def dep(**kw):
    base = dict(instance_id=7, recipe="build-a", model="m/x", served_name="x",
                offer_id=1, port=8000, serve_key="k", created_at=time.time(),
                ttl_hours=3.0, dph_at_launch=1.0, gpu_label="2x RTX 5090",
                opencode_provider="vast-7", opencode_target="/tmp/oc.json",
                linked_at=time.time(), notes={"linked_model_id": "llama-3.3-70b-instruct-awq"})
    base.update(kw)
    d = Deployment(**base)
    state.save(d)
    return d


def test_link_requires_an_opencode_link_first(settings):
    """Conductor discovers OpenCode models via opencode's own config."""
    with pytest.raises(RuntimeError, match="not linked into opencode"):
        track.link_conductor(dep(linked_at=None, opencode_provider=""), path=settings)


def test_link_uses_the_model_id_opencode_reported(settings):
    edit = track.link_conductor(dep(), path=settings)
    assert edit.changed["default"] == "vast-7/llama-3.3-70b-instruct-awq"
    assert parse(settings)["models"]["default"] == "vast-7/llama-3.3-70b-instruct-awq"


def test_previous_value_survives_repointing(settings):
    """Pointing twice must still remember 'opus', not our own first value."""
    d = dep()
    track.link_conductor(d, path=settings)
    track.link_conductor(state.find(7), path=settings)
    assert state.find(7).conductor_prev["default"] == "opus"
    track.unlink_conductor(state.find(7))
    assert parse(settings)["models"]["default"] == "opus"


def test_unlink_clears_tracking(settings):
    d = dep()
    track.link_conductor(d, path=settings)
    restored = track.unlink_conductor(state.find(7))
    assert "default" in restored
    after = state.find(7)
    assert after.conductor_target == "" and after.conductor_prev == {}
    assert track.unlink_conductor(after) == [], "second revert is a no-op"


# --------------------------------------------- opencode default follows through


@pytest.fixture
def oc_config(tmp_path):
    import json
    p = tmp_path / "opencode.json"
    p.write_text(json.dumps({
        "$schema": "https://opencode.ai/config.json",
        "provider": {"omlx": {"name": "oMLX"}},
        "model": "omlx/local-model",
    }, indent=2))
    return p


def linked_dep(oc_path, **kw):
    base = dict(instance_id=9, recipe="build-a", model="m/x", served_name="x",
                offer_id=1, port=8000, serve_key="k", created_at=time.time(),
                ttl_hours=3.0, dph_at_launch=1.0, gpu_label="1x RTX PRO 6000 WS",
                opencode_provider="vast-9", opencode_target=str(oc_path),
                linked_at=time.time(), notes={"linked_model_id": "Qwen3.8-27B-FP8"})
    base.update(kw)
    d = Deployment(**base)
    state.save(d)
    return d


def read(p):
    import json
    return json.loads(p.read_text())


def test_sets_opencode_default_model(oc_config):
    """Conductor delegates model choice to opencode, so opencode's own default
    must follow — otherwise it stays on whatever it was configured for before."""
    ref, previous = track.set_opencode_default(linked_dep(oc_config))
    assert ref == "vast-9/Qwen3.8-27B-FP8"
    assert previous == "omlx/local-model"
    assert read(oc_config)["model"] == "vast-9/Qwen3.8-27B-FP8"


def test_provider_block_is_untouched_by_a_default_change(oc_config):
    track.set_opencode_default(linked_dep(oc_config))
    assert read(oc_config)["provider"]["omlx"] == {"name": "oMLX"}


def test_previous_default_is_remembered_for_teardown(oc_config):
    d = linked_dep(oc_config)
    track.set_opencode_default(d)
    assert state.find(9).notes["previous_default_model"] == "omlx/local-model"


def test_repointing_does_not_overwrite_the_users_original(oc_config):
    """Setting twice must still remember the user's model, not ours."""
    d = linked_dep(oc_config)
    track.set_opencode_default(d)
    track.set_opencode_default(state.find(9))
    assert state.find(9).notes["previous_default_model"] == "omlx/local-model"


def test_requires_an_opencode_link_first(oc_config):
    with pytest.raises(RuntimeError, match="not linked into opencode"):
        track.set_opencode_default(linked_dep(oc_config, linked_at=None, opencode_provider=""))


def test_setting_the_same_default_twice_is_a_noop(oc_config):
    from gpuctl import opencode as oc
    track.set_opencode_default(linked_dep(oc_config))
    previous, backup = oc.set_default_model(path=oc_config,
                                           model_ref="vast-9/Qwen3.8-27B-FP8")
    assert previous == "vast-9/Qwen3.8-27B-FP8"
    assert backup is None, "no backup churn when nothing changes"


# ----------------------------------------------- the bundled-opencode trap


def test_sets_the_opencode_executable_path(settings):
    """Conductor's own opencode runs with its own data dir and never reads
    ~/.config/opencode/opencode.json, so a gpuctl provider is invisible to it and
    selecting that model fails with 'did not load the model … within 15s'."""
    edit = conductor.set_opencode_executable("/opt/homebrew/bin/opencode", path=settings)
    assert edit.changed[conductor.EXECUTABLE_KEY] == "/opt/homebrew/bin/opencode"
    assert parse(settings)["opencode_executable_path"] == "/opt/homebrew/bin/opencode"


def test_executable_path_lands_in_the_document_root(settings):
    """A root key written after a [table] header would be read as part of it."""
    conductor.set_opencode_executable("/usr/local/bin/opencode", path=settings)
    text = settings.read_text()
    assert text.index("opencode_executable_path") < text.index("[git]")
    # neighbours intact
    d = parse(settings)
    assert d["git"]["branch_prefix_type"] == "github_username"
    assert d["models"]["default"] == "opus"
    assert d["models"]["codex"]["default_thinking_level"] == "high"


def test_setting_the_same_executable_is_a_noop(settings):
    conductor.set_opencode_executable("/opt/homebrew/bin/opencode", path=settings)
    edit = conductor.set_opencode_executable("/opt/homebrew/bin/opencode", path=settings)
    assert edit.changed == {}
    assert edit.backup is None


# ------------------------------------------------- visible_provider_models


VISIBLE_REAL = ('{"claude":[],"codex":[],"cursor":[],'
                '"opencode":["opencode:vast-1/ModelA"],"pi":[]}')


def with_visible(settings, raw):
    settings.write_text(settings.read_text().replace(
        'default = "opus"   # keep me',
        'default = "opus"   # keep me\nvisible_provider_models = '
        + __import__("json").dumps(raw)))
    return settings


def test_reads_the_visible_list(settings):
    with_visible(settings, VISIBLE_REAL)
    assert conductor.visible_models(path=settings) == ["opencode:vast-1/ModelA"]


def test_no_visible_list_means_nothing_is_offered(settings):
    assert conductor.visible_models(path=settings) == []


def test_adding_a_model_preserves_other_harnesses(settings):
    import json as _json
    with_visible(settings, VISIBLE_REAL)
    raw = conductor.add_visible_model("vast-2/ModelB", path=settings)
    parsed = _json.loads(raw)
    assert parsed["opencode"] == ["opencode:vast-1/ModelA", "opencode:vast-2/ModelB"]
    assert set(parsed) == {"claude", "codex", "cursor", "opencode", "pi"}
    assert parsed["codex"] == []


def test_adding_an_existing_model_is_a_noop(settings):
    with_visible(settings, VISIBLE_REAL)
    assert conductor.add_visible_model("vast-1/ModelA", path=settings) is None


def test_removing_a_model(settings):
    import json as _json
    with_visible(settings, VISIBLE_REAL)
    raw = conductor.remove_visible_model("vast-1/ModelA", path=settings)
    assert _json.loads(raw)["opencode"] == []


def test_removing_an_absent_model_is_a_noop(settings):
    with_visible(settings, VISIBLE_REAL)
    assert conductor.remove_visible_model("vast-9/Nope", path=settings) is None


def test_unparseable_visible_list_is_treated_as_empty(settings):
    with_visible(settings, "not json at all")
    assert conductor.visible_models(path=settings) == []
    # and adding still produces a valid document
    raw = conductor.add_visible_model("vast-1/ModelA", path=settings)
    assert "opencode:vast-1/ModelA" in raw


def test_linking_makes_the_model_visible(settings, oc_config):
    import json as _json
    edit = track.link_conductor(linked_dep(oc_config), path=settings)
    assert conductor.VISIBLE_KEY in edit.changed
    listed = _json.loads(parse(settings)["models"][conductor.VISIBLE_KEY])
    assert listed["opencode"] == ["opencode:vast-9/Qwen3.8-27B-FP8"]
