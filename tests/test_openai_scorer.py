"""The API scorer, exercised with a fake client: no network, no key."""
import types

import pytest

import iclorder.openai_scorer as mod
from iclorder.openai_scorer import OpenAIScorer, match_label, price_for

LABELS = ["World", "Sports", "Business", "Technology"]


def _resp(text, prompt_tokens=100, completion_tokens=3, cached=0, reasoning=0):
    usage = types.SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        prompt_tokens_details=types.SimpleNamespace(cached_tokens=cached),
        completion_tokens_details=types.SimpleNamespace(reasoning_tokens=reasoning),
    )
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))], usage=usage)


class FakeClient:
    """client.chat.completions.create(**kw) -> response built from reply(user_text)."""

    def __init__(self, reply, fail_first=0):
        self.calls = []
        self.fail_first = fail_first
        outer = self

        class Completions:
            def create(self, **kw):
                outer.calls.append(kw)
                if outer.fail_first > 0:
                    outer.fail_first -= 1
                    exc = RuntimeError("rate limited")
                    exc.status_code = 429
                    raise exc
                return _resp(reply(kw["messages"][-1]["content"]))

        self.chat = types.SimpleNamespace(completions=Completions())


def test_match_label_rules():
    assert match_label("Sports", LABELS) == 1
    assert match_label("  sports\n", LABELS) == 1
    assert match_label("The topic is business.", LABELS) == 2
    assert match_label("Science/Technology", LABELS) == 3
    assert match_label("Business or Sports", LABELS) == 2      # earliest mention wins
    assert match_label("", LABELS) == -1
    assert match_label("Sportsmanship", LABELS) == -1         # whole words only
    assert match_label("Science", LABELS) == 3                # alias of Technology (AG News "Sci/Tech")
    assert match_label("Sci/Tech", LABELS) == 3
    assert match_label("Science", LABELS, aliases=None) == -1
    assert match_label("Science or Business", LABELS) == 2    # an exact label word beats an alias


def test_predict_uses_cache_and_counts_usage(tmp_path):
    client = FakeClient(lambda user: "Sports" if "ball" in user else "World")
    sc = OpenAIScorer("gpt-5-nano", workers=2, cache_dir=tmp_path, client=client)
    prompts = ["Article: a\nTopic: World\n\nArticle: ball game\nTopic:", "Article: a\nTopic: World\n\nArticle: summit\nTopic:"]
    assert sc.predict(prompts).tolist() == [1, 0]
    assert sc.usage["requests"] == 2 and sc.usage["cache_hits"] == 0 and sc.usage["unmatched"] == 0
    assert sc.usage["prompt_tokens"] == 200
    kw = client.calls[0]
    assert kw["model"] == "gpt-5-nano" and kw["reasoning_effort"] == "minimal" and kw["max_completion_tokens"] == 64
    assert "World, Sports, Business, Technology" in kw["messages"][0]["content"]
    assert kw["messages"][1]["content"] == prompts[0]

    # same prompts again: served from the sqlite cache, no new calls
    assert sc.predict(prompts).tolist() == [1, 0]
    assert sc.usage["requests"] == 2 and sc.usage["cache_hits"] == 2 and len(client.calls) == 2

    # a fresh scorer on the same cache dir also hits the cache
    sc2 = OpenAIScorer("gpt-5-nano", cache_dir=tmp_path, client=FakeClient(lambda u: "Business"))
    assert sc2.predict(prompts).tolist() == [1, 0]
    assert sc2.usage["requests"] == 0 and sc2.usage["cache_hits"] == 2


def test_unmatched_and_alias_tallies(tmp_path):
    sc = OpenAIScorer("gpt-5-nano", cache_dir=tmp_path, client=FakeClient(lambda u: "I am not sure."))
    assert sc.predict(["q1", "q2"]).tolist() == [-1, -1] and sc.usage["unmatched"] == 2
    sc = OpenAIScorer("gpt-5-nano", cache_dir=tmp_path / "b", client=FakeClient(lambda u: "Science"))
    assert sc.predict(["q1", "q2"]).tolist() == [3, 3]
    assert sc.usage["alias_matched"] == 2 and sc.usage["unmatched"] == 0


def test_retry_on_retryable_error(tmp_path, monkeypatch):
    monkeypatch.setattr("iclorder.openai_scorer.time.sleep", lambda s: None)
    client = FakeClient(lambda u: "World", fail_first=2)
    sc = OpenAIScorer("gpt-5-nano", cache_dir=tmp_path, client=client, retries=3)
    assert sc.complete("x", "sys", 8) == "World"
    assert sc.usage["retries"] == 2 and len(client.calls) == 3


def test_retries_exhausted_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("iclorder.openai_scorer.time.sleep", lambda s: None)
    sc = OpenAIScorer("gpt-5-nano", cache_dir=tmp_path, client=FakeClient(lambda u: "World", fail_first=5), retries=2)
    with pytest.raises(RuntimeError):
        sc.complete("x", "sys", 8)


def test_budget_exhausted_retries_with_larger_budget(tmp_path, monkeypatch):
    monkeypatch.setattr("iclorder.openai_scorer.time.sleep", lambda s: None)
    calls = []

    class Completions:
        def create(self, **kw):
            calls.append(kw)
            if kw["max_completion_tokens"] < 100:
                exc = RuntimeError("Could not finish the message because max_tokens or model output limit was reached.")
                exc.status_code = 400
                raise exc
            return _resp("Sports")

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Completions()))
    sc = OpenAIScorer("gpt-5-nano", cache_dir=tmp_path, client=client, max_completion_tokens=64)
    assert sc.predict(["q"]).tolist() == [1]
    assert [c["max_completion_tokens"] for c in calls] == [64, 256]
    assert sc.usage["budget_raised"] == 1 and sc.usage["retries"] == 0


def test_cost_estimate(tmp_path):
    sc = OpenAIScorer("gpt-5-nano-2025-08-07", cache_dir=tmp_path, client=FakeClient(lambda u: "World"))
    sc.usage.update(prompt_tokens=1_000_000, cached_prompt_tokens=200_000, completion_tokens=10_000)
    # 800k uncached * 0.05 + 200k cached * 0.005 + 10k out * 0.40, per million
    assert sc.estimated_cost_usd() == pytest.approx(0.04 + 0.001 + 0.004)
    assert price_for("unknown-model") is None
    assert sc.usage_summary()["estimated_cost_usd"] == pytest.approx(0.045)


def test_no_reasoning_effort_when_disabled(tmp_path):
    client = FakeClient(lambda u: "World")
    OpenAIScorer("gpt-4.1-mini", cache_dir=tmp_path, client=client, reasoning_effort=None).complete("x", "sys", 8)
    assert "reasoning_effort" not in client.calls[0]


def test_key_lookup_env_then_keychain(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(mod, "_api_key_from_keychain", lambda service=None: None)
    with pytest.raises(SystemExit):
        mod._make_client()
    monkeypatch.setattr(mod, "_api_key_from_keychain", lambda service=None: "sk-test-from-keychain")
    assert mod._make_client().api_key == "sk-test-from-keychain"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-from-env")
    assert mod._make_client().api_key == "sk-test-from-env"


def test_keychain_lookup_handles_missing_tool(monkeypatch):
    def boom(*a, **k):
        raise OSError("no security tool")
    monkeypatch.setattr(mod.subprocess, "run", boom)
    assert mod._api_key_from_keychain() is None
