"""Streaming measurement, including reasoning models."""
from gpuctl.cli import _delta_token


def test_plain_content_counts():
    assert _delta_token({"content": "hello"}) == ("hello", False)


def test_reasoning_counts_as_a_token():
    """Regression: Qwen3.8 spent 30 of 34 tokens in delta.reasoning and emitted
    one empty content chunk, so counting only `content` measured zero tokens."""
    assert _delta_token({"reasoning": "The"}) == ("The", True)


def test_reasoning_content_variant_counts():
    """Different vLLM reasoning parsers name the field differently."""
    assert _delta_token({"reasoning_content": "hmm"}) == ("hmm", True)


def test_empty_content_is_not_a_token():
    """The first chunk is {"role":"assistant","content":""} — not a token."""
    assert _delta_token({"role": "assistant", "content": ""}) is None


def test_role_only_delta_is_not_a_token():
    assert _delta_token({"role": "assistant"}) is None


def test_empty_delta_is_not_a_token():
    assert _delta_token({}) is None


def test_non_string_values_are_ignored():
    assert _delta_token({"content": None}) is None
    assert _delta_token({"content": 42}) is None


def test_content_wins_when_both_are_present():
    assert _delta_token({"content": "answer", "reasoning": "think"}) == ("answer", False)
