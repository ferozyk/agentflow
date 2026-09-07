"""Tests for the foundation layer: config/pricing, models, and the TraceStore."""
from __future__ import annotations

import pytest

from agentflow.config import ModelPricing, PaidModelError, Settings, compute_cost, verify_models_are_free
from agentflow.models import WebsitePlan
from agentflow.observability.tracing import AGENT_NAMES, TraceStore


# -- cost computation (never hardcoded, always from catalog pricing) --------


def test_compute_cost_free_model_is_zero():
    catalog = {"dots-studio/dots-3-note-preview:free": ModelPricing(prompt=0.0, completion=0.0)}
    cost = compute_cost(
        "dots-studio/dots-3-note-preview:free", input_tokens=10_000, output_tokens=5_000, catalog=catalog
    )
    assert cost == 0.0


def test_compute_cost_unknown_model_never_fabricated():
    # A model absent from the catalog (e.g. offline-heuristic mode) costs 0.0 —
    # we never invent a price for something we couldn't verify.
    cost = compute_cost("some/unverified-model", input_tokens=1000, output_tokens=1000, catalog={})
    assert cost == 0.0


def test_compute_cost_uses_real_catalog_pricing():
    catalog = {"paid/model": ModelPricing(prompt=0.000002, completion=0.000004)}
    cost = compute_cost("paid/model", input_tokens=1000, output_tokens=1000, catalog=catalog)
    assert cost == pytest.approx(0.000002 * 1000 + 0.000004 * 1000)


# -- free-model verification (network mocked out) ---------------------------


_FREE_MODEL_IDS = [
    "dots-studio/dots-3-note-preview:free",
    "cohere/north-mini-code:free",
    "minimax/minimax-m2.7:free",
    "nvidia/nemotron-3.5-lightning:free",
]


def test_verify_models_are_free_raises_for_paid_model(monkeypatch):
    catalog = [
        {"id": mid, "pricing": {"prompt": "0", "completion": "0"}} for mid in _FREE_MODEL_IDS[:-1]
    ] + [{"id": _FREE_MODEL_IDS[-1], "pricing": {"prompt": "0.000001", "completion": "0"}}]
    monkeypatch.setattr("agentflow.config.fetch_model_catalog", lambda force_refresh=False: catalog)

    settings = Settings(openrouter_api_key="test-key", openrouter_fallback_models=_FREE_MODEL_IDS[2:])
    with pytest.raises(PaidModelError):
        verify_models_are_free(settings)


def test_verify_models_are_free_passes_when_all_zero_priced(monkeypatch):
    catalog = [{"id": mid, "pricing": {"prompt": "0", "completion": "0"}} for mid in _FREE_MODEL_IDS]
    monkeypatch.setattr("agentflow.config.fetch_model_catalog", lambda force_refresh=False: catalog)

    settings = Settings(openrouter_api_key="test-key", openrouter_fallback_models=_FREE_MODEL_IDS[2:])
    result = verify_models_are_free(settings)
    assert result.pricing_status == "FREE"
    assert settings.pricing_status == "FREE"


def test_verify_models_are_free_raises_when_model_missing_from_catalog(monkeypatch):
    monkeypatch.setattr("agentflow.config.fetch_model_catalog", lambda force_refresh=False: [])
    settings = Settings(openrouter_api_key="test-key")
    with pytest.raises(PaidModelError):
        verify_models_are_free(settings)


# -- models.py contracts -----------------------------------------------------


def test_website_plan_requires_between_4_and_6_sections():
    with pytest.raises(Exception):
        WebsitePlan(
            name="x",
            description="y",
            target_audience="z",
            sections=["only-one-section"],
            visual_style="modern",
        )

    plan = WebsitePlan(
        name="Agent Flow",
        description="AI engineering demo",
        target_audience="Enterprise leaders",
        sections=["Hero", "Services", "About", "Contact"],
        visual_style="modern, minimal",
    )
    assert len(plan.sections) == 4


# -- TraceStore ---------------------------------------------------------------


def test_run_trace_preseeds_all_five_spans_waiting():
    store = TraceStore()
    trace = store.start_run("build a site")
    assert [s.agent_name for s in trace.spans] == AGENT_NAMES
    assert all(s.status == "WAITING" for s in trace.spans)
    assert trace.run_id.startswith("af-")


async def test_span_context_manager_marks_running_then_completed():
    store = TraceStore()
    trace = store.start_run("build a site")

    async with store.span(trace.run_id, "Planner") as span:
        assert span.status == "RUNNING"
        assert span.start_time is not None

    updated = store.get(trace.run_id)
    planner_span = next(s for s in updated.spans if s.agent_name == "Planner")
    assert planner_span.status == "COMPLETED"
    assert planner_span.duration_ms is not None
    assert any("Planner" in e.message for e in updated.events)


async def test_span_context_manager_records_failure_and_reraises():
    store = TraceStore()
    trace = store.start_run("build a site")

    with pytest.raises(RuntimeError):
        async with store.span(trace.run_id, "Designer"):
            raise RuntimeError("boom")

    updated = store.get(trace.run_id)
    designer_span = next(s for s in updated.spans if s.agent_name == "Designer")
    assert designer_span.status == "FAILED"
    assert designer_span.error == "boom"


def test_latency_percentiles_none_under_five_samples():
    store = TraceStore()
    for i in range(4):
        trace = store.start_run(f"request {i}")
        trace.duration_ms = 100.0 + i

    result = store.latency_percentiles()
    assert result == {"p50": None, "p90": None, "samples": 4}


def test_latency_percentiles_computed_at_five_samples():
    store = TraceStore()
    for i, duration in enumerate([100.0, 200.0, 300.0, 400.0, 500.0]):
        trace = store.start_run(f"request {i}")
        trace.duration_ms = duration

    result = store.latency_percentiles()
    assert result["samples"] == 5
    assert result["p50"] is not None
    assert result["p90"] is not None


def test_latency_percentiles_never_fabricated_with_zero_samples():
    store = TraceStore()
    result = store.latency_percentiles()
    assert result == {"p50": None, "p90": None, "samples": 0}


# --- provider-side 400 failover (regression: Cohere tool_results incompatibility) ---------


def test_upstream_provider_400_is_retryable():
    """A 400 that came from the upstream provider should trigger free->free failover.

    Reproduces the live failure: cohere/north-mini-code:free intermittently returned
    400 "all elements in tool_results must have the 'outputs' property specified",
    which killed the Developer span outright because 400 wasn't failover-eligible.
    """
    from pydantic_ai.exceptions import ModelHTTPError

    from agentflow.llm import _is_retryable

    exc = ModelHTTPError(
        status_code=400,
        model_name="cohere/north-mini-code:free",
        body={
            "message": "Provider returned error",
            "code": 400,
            "metadata": {
                "raw": "invalid request: all elements in tool_results must have the 'outputs' property specified.",
                "provider_name": "Cohere",
            },
        },
    )
    assert _is_retryable(exc) is True


def test_client_side_400_is_not_retryable():
    """A malformed-request 400 from OpenRouter itself must NOT be masked by failover."""
    from pydantic_ai.exceptions import ModelHTTPError

    from agentflow.llm import _is_retryable

    exc = ModelHTTPError(
        status_code=400,
        model_name="dots-studio/dots-3-note-preview:free",
        body={"error": {"message": "invalid tool schema", "code": 400}},
    )
    assert _is_retryable(exc) is False


def test_429_and_403_still_retryable():
    from pydantic_ai.exceptions import ModelHTTPError

    from agentflow.llm import _is_retryable

    for code in (403, 429, 500, 503):
        exc = ModelHTTPError(status_code=code, model_name="x:free", body={})
        assert _is_retryable(exc) is True, f"{code} should be retryable"


# --- per-agent temperature control (regression: judge score noise) -----------------------
#
# Reviewer measured the SAME unchanged generated site scoring 6.0/6.5/7.5/8.0 across 4
# identical run_evaluator calls -- a 2.0-point spread straddling the pass threshold, because
# `temperature` was never set anywhere and every agent (including the judge) ran at the
# provider default. These tests lock in that Evaluator now runs at temperature=0.0 and that
# a caller-provided model_settings value is never silently overridden.


class _FakeUsage:
    input_tokens = 10
    output_tokens = 5
    total_tokens = 15
    cache_read_tokens = 0
    cache_write_tokens = 0


class _FakeAgent:
    """Stand-in for pydantic_ai.Agent: records the model_settings run_agent() built."""

    def __init__(self):
        self.captured_model_settings: dict | None = None

    async def run(self, user_prompt, *, model, **kwargs):
        from types import SimpleNamespace

        self.captured_model_settings = kwargs.get("model_settings")
        return SimpleNamespace(
            usage=_FakeUsage(), response=SimpleNamespace(model_name="fake/model:free")
        )


def _run_agent_no_network(monkeypatch):
    """Patch out the two things in run_agent() that would otherwise hit the network/disk."""
    import agentflow.llm as llm_mod
    from agentflow.config import VerificationResult

    monkeypatch.setattr(
        llm_mod, "_get_verification", lambda settings=None: VerificationResult(pricing_status="FREE")
    )
    monkeypatch.setattr(llm_mod, "build_model", lambda model_id, settings=None: object())
    return llm_mod


def test_agent_temperature_evaluator_is_zero():
    from agentflow.llm import AGENT_TEMPERATURE

    assert AGENT_TEMPERATURE["Evaluator"] == 0.0


async def test_run_agent_sets_evaluator_temperature_zero(monkeypatch):
    llm_mod = _run_agent_no_network(monkeypatch)
    from agentflow.observability.tracing import STORE

    trace = STORE.start_run("test request")
    fake_agent = _FakeAgent()

    await llm_mod.run_agent(
        run_id=trace.run_id,
        agent_name="Evaluator",
        agent=fake_agent,
        user_prompt="irrelevant",
        model_id="some/model:free",
        fallback_model_ids=[],
    )

    assert fake_agent.captured_model_settings["temperature"] == 0.0
    # AGENT_REASONING_EFFORT should also be merged in unless already set -- confirms the
    # two defaults dicts coexist rather than one clobbering the other's merge branch.
    assert fake_agent.captured_model_settings["openrouter_reasoning"] == {"effort": "low"}


async def test_run_agent_caller_temperature_is_never_overridden(monkeypatch):
    llm_mod = _run_agent_no_network(monkeypatch)
    from agentflow.observability.tracing import STORE

    trace = STORE.start_run("test request")
    fake_agent = _FakeAgent()

    await llm_mod.run_agent(
        run_id=trace.run_id,
        agent_name="Evaluator",
        agent=fake_agent,
        user_prompt="irrelevant",
        model_id="some/model:free",
        fallback_model_ids=[],
        model_settings={"temperature": 0.9},
    )

    # Caller explicitly asked for 0.9 -- run_agent must not silently clobber it with the
    # Evaluator default of 0.0.
    assert fake_agent.captured_model_settings["temperature"] == 0.9


async def test_run_agent_leaves_designer_temperature_unset(monkeypatch):
    """Designer/Content/Developer are deliberately left at the provider default (None ->
    no override) so creative variation in visual output isn't flattened."""
    llm_mod = _run_agent_no_network(monkeypatch)
    from agentflow.observability.tracing import STORE

    trace = STORE.start_run("test request")
    fake_agent = _FakeAgent()

    await llm_mod.run_agent(
        run_id=trace.run_id,
        agent_name="Designer",
        agent=fake_agent,
        user_prompt="irrelevant",
        model_id="some/model:free",
        fallback_model_ids=[],
    )

    assert "temperature" not in (fake_agent.captured_model_settings or {})
