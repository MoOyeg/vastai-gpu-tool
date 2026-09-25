"""Keep every test off the developer's real config, state and recipe dirs."""
import importlib
import os

import pytest


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUCTL_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("GPUCTL_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("GPUCTL_RECIPES_DIR", raising=False)
    from gpuctl import config, recipes, state
    importlib.reload(config)
    importlib.reload(state)
    importlib.reload(recipes)
    yield
    recipes.all_recipes.cache_clear()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Nothing in the suite may touch the network.

    Without this, a phase-machine test that resolves an endpoint will really try
    to connect to it and sit there for the full timeout. Tests that need a
    serving probe override this with their own monkeypatch.
    """
    from gpuctl import health, track

    def refuse(*_a, **_k):
        return health.Probe(False, False, [], "stubbed: no network in tests")

    monkeypatch.setattr(track, "probe", refuse)
    monkeypatch.setattr(health, "probe", refuse)


@pytest.fixture
def recipe_dir(tmp_path, monkeypatch):
    """A custom recipe directory on the search path."""
    d = tmp_path / "recipes.custom"
    d.mkdir()
    monkeypatch.setenv("GPUCTL_RECIPES_DIR", str(d))
    from gpuctl import recipes
    recipes.all_recipes.cache_clear()
    return d
