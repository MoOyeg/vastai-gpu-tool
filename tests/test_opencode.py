"""Editing someone's opencode config must be lossless and reversible."""
import json

import pytest

from gpuctl import opencode

EXISTING = {
    "$schema": "https://opencode.ai/config.json",
    "provider": {"omlx": {"name": "oMLX", "options": {"baseURL": "http://127.0.0.1:8000/v1"}}},
    "model": "omlx/local-model",
}


@pytest.fixture
def config(tmp_path):
    p = tmp_path / "opencode.json"
    p.write_text(json.dumps(EXISTING, indent=2))
    return p


def block():
    return opencode.provider_block(
        display_name="Vast 2x RTX 5090 (1)", endpoint="http://1.2.3.4:41749",
        serve_key="sk-vast-TEST", model_id="llama-3.3-70b-instruct-awq", context_len=32768)


def test_link_preserves_existing_providers(config):
    opencode.link(path=config, provider_id="vast-1", block=block(),
                  model_id="llama-3.3-70b-instruct-awq", set_default=False)
    after = json.loads(config.read_text())
    assert after["provider"]["omlx"] == EXISTING["provider"]["omlx"]
    assert after["model"] == EXISTING["model"], "default model must not change"
    assert after["provider"]["vast-1"]["options"]["baseURL"].endswith("/v1")


def test_link_backs_up_first(config):
    result = opencode.link(path=config, provider_id="vast-1", block=block(),
                           model_id="m", set_default=False)
    assert result.backup and result.backup.exists()
    assert json.loads(result.backup.read_text()) == EXISTING


def test_unlink_restores_previous_default(config):
    result = opencode.link(path=config, provider_id="vast-1", block=block(),
                           model_id="llama-3.3-70b-instruct-awq", set_default=True)
    assert json.loads(config.read_text())["model"] == "vast-1/llama-3.3-70b-instruct-awq"
    assert opencode.unlink(path=config, provider_id="vast-1",
                           restore_model=result.previous_default)
    final = json.loads(config.read_text())
    assert final == EXISTING, "config must round-trip exactly"


def test_unlink_is_a_noop_when_absent(config):
    assert not opencode.unlink(path=config, provider_id="vast-nope")
    assert json.loads(config.read_text()) == EXISTING


def test_refuses_to_rewrite_non_plain_json(tmp_path):
    """A .jsonc with comments cannot be round-tripped, so refuse rather than clobber."""
    p = tmp_path / "opencode.jsonc"
    original = '{\n  // my notes\n  "provider": {}\n}'
    p.write_text(original)
    with pytest.raises(opencode.OpencodeError):
        opencode.link(path=p, provider_id="vast-1", block=block(), model_id="m", set_default=False)
    assert p.read_text() == original, "must not have been modified"


def test_creates_config_when_missing(tmp_path):
    p = tmp_path / "new" / "opencode.json"
    opencode.link(path=p, provider_id="vast-1", block=block(), model_id="m", set_default=False)
    assert "vast-1" in json.loads(p.read_text())["provider"]
