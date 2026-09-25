"""Choosing between the cheapest offers instead of blindly taking #1."""
import sys

import pytest
import typer

from gpuctl import cli


def offer(oid, dph, *, geo="California, US", inet=900.0, rel=0.99,
          disk=600.0, cuda=13.0, cc=1200, gpus=2, name="RTX 3090"):
    return {"id": oid, "dph_total": dph, "geolocation": geo, "inet_down": inet,
            "reliability2": rel, "disk_space": disk, "cuda_max_good": cuda,
            "compute_cap": cc, "num_gpus": gpus, "gpu_name": name}


THREE = [offer(1, 0.30), offer(2, 0.40), offer(3, 0.50)]


@pytest.fixture
def terminal(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)


def answer(monkeypatch, *replies):
    """Feed successive answers to the prompt."""
    queue = list(replies)
    monkeypatch.setattr(typer, "prompt", lambda *a, **k: queue.pop(0))


def pick(**kw):
    kw.setdefault("title", "t")
    kw.setdefault("limit", 5)
    kw.setdefault("ttl_hours", 3.0)
    return cli._choose_offer(kw.pop("offers", THREE), **kw)


def test_selection_returns_the_chosen_offer(terminal, monkeypatch):
    answer(monkeypatch, "2")
    assert pick()["id"] == 2


def test_default_is_the_cheapest(terminal, monkeypatch):
    answer(monkeypatch, "1")
    assert pick()["id"] == 1


def test_q_aborts(terminal, monkeypatch):
    answer(monkeypatch, "q")
    assert pick() is None


def test_invalid_input_reprompts(terminal, monkeypatch):
    answer(monkeypatch, "9", "0", "banana", "3")
    assert pick()["id"] == 3


def test_limit_caps_the_shortlist(terminal, monkeypatch):
    """A choice of 4 must not be offered when only 2 were listed."""
    answer(monkeypatch, "3", "2")
    assert pick(offers=THREE, limit=2)["id"] == 2


def test_single_offer_needs_no_prompt(terminal, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should not prompt for a single option")

    monkeypatch.setattr(typer, "prompt", boom)
    assert pick(offers=[offer(7, 0.9)])["id"] == 7


def test_falls_back_to_cheapest_without_a_terminal(monkeypatch):
    """`--choose` in a script or CI must not hang waiting on stdin."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)

    def boom(*a, **k):
        raise AssertionError("should not prompt without a tty")

    monkeypatch.setattr(typer, "prompt", boom)
    assert pick()["id"] == 1


# ------------------------------------------------------------------- warnings


def test_flags_chinese_hosts_for_huggingface():
    assert any("CN" in f for f in cli._offer_flags(offer(1, 0.3, geo="Hebei, CN")))
    assert not cli._offer_flags(offer(1, 0.3, geo="California, US"))


def test_flags_slow_network_with_an_estimated_pull_time():
    flags = cli._offer_flags(offer(1, 0.3, inet=200.0), weights_gb=39.8)
    assert any("slow net" in f for f in flags)
    assert any("m pull" in f for f in flags), "should estimate the download time"


def test_fast_network_is_not_flagged():
    assert not cli._offer_flags(offer(1, 0.3, inet=2765.0), weights_gb=39.8)


def test_flags_shaky_reliability():
    assert any("reliability" in f for f in cli._offer_flags(offer(1, 0.3, rel=0.94)))


@pytest.mark.parametrize("name,flagged", [
    ("RTX 3090", True),            # GeForce: no forward compat -> error 804
    ("RTX 5090", True),
    ("RTX A6000", True),           # workstation is not datacenter either
    ("RTX PRO 6000 WS", True),
    ("H200", False),               # datacenter: forward compat works
    ("A100 SXM4", False),
    ("B300", False),
    ("Tesla V100", False),
])
def test_flags_old_cuda_only_where_forward_compat_is_unavailable(name, flagged):
    """compute_cap cannot express this: a GeForce 3090 is 860 while a datacenter
    A100 is 800, so the check keys on the product line."""
    flags = cli._offer_flags(offer(1, 0.3, cuda=12.4, name=name))
    assert any("804" in f for f in flags) is flagged


def test_current_cuda_is_never_flagged():
    assert not any("804" in f for f in cli._offer_flags(offer(1, 0.3, cuda=13.2, name="RTX 3090")))
