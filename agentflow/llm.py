"""PydanticAI model factory + free/free failover + usage capture, all via OpenRouter.

Verified against pydantic-ai 2.36.0 (see report to the coordinating agent for the exact
inspection commands used). Key facts this module relies on:

- `pydantic_ai.providers.openrouter.OpenRouterProvider(api_key=..., app_url=..., app_title=...)`
  builds an `AsyncOpenAI`-backed provider pointed at https://openrouter.ai/api/v1 and sets the
  `HTTP-Referer`/`X-Title` attribution headers automatically from `app_url`/`app_title`.
- `pydantic_ai.models.openrouter.OpenRouterModel(model_id, provider=provider)` is a dedicated
  Model subclass (not the generic OpenAIChatModel) that also captures OpenRouter-specific
  `cache_write_tokens` from `usage.prompt_tokens_details`.
- `await agent.run(prompt, model=model, ...)` returns an `AgentRunResult`:
    - `result.output`            -> the validated structured output
    - `result.usage`             -> a `RequestUsage` (NOT a method call) with
                                     `.input_tokens/.output_tokens/.total_tokens/
                                      .cache_read_tokens/.cache_write_tokens`
    - `result.response.model_name` -> the model actually served by OpenRouter (can differ
                                       from the requested id when a router alias is used)
- Rate limit / server errors surface as `pydantic_ai.exceptions.ModelHTTPError` with a
  `.status_code`; network/read timeouts surface as `httpx.TimeoutException` subclasses.

Smoke-tested live against `dots-studio/dots-3-note-preview:free`: input_tokens=272,
output_tokens=198, total_tokens=470, cache_read_tokens=0, cache_write_tokens=0 (first call,
nothing to cache yet), served model == requested model, provider_details confirmed
`upstream_inference_cost: 0.0`.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import httpx
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.run import AgentRunResult

from .config import (
    ModelPricing,
    Settings,
    VerificationResult,
    compute_cost,
    get_settings,
    verify_models_are_free,
)
from .observability.tracing import STORE, LLMCall

# Output token budgets per docs/CONTRACT.md section 11 (raised 2026-08-31: the default
# model dots-studio/dots-3-note-preview:free is a REASONING model that burns 150-900+
# hidden reasoning tokens before emitting structured JSON. At the original budgets
# (Planner/Designer 500, Content 1200, Developer 4000, Evaluator 700) live testing by the
# agents subagent measured "Model token limit exceeded before any response was generated"
# in the large majority of ~8 live trials. Raised per reviewer-confirmed live evidence.
# Agents pass these as `model_settings={"max_tokens": AGENT_MAX_TOKENS[name]}`.
AGENT_MAX_TOKENS: dict[str, int] = {
    "Planner": 4000,
    "Designer": 4000,
    "Content": 6000,
    "Developer": 16000,
    "Evaluator": 5000,
}
# Raised 2026-08-31 after a live Slack run died with "Model token limit (1500) exceeded
# before any response was generated". Measured across 6 real Planner runs, output ranged
# 372-860 tokens (reasoning 266-407 of that) — so 1500 left too little headroom and failed
# INTERMITTENTLY, the worst failure mode for a live demo.
# `max_tokens` is a CEILING, not a reservation: you are billed only for tokens actually
# generated, so generous caps cost nothing and simply remove a variance-driven failure.
# Token discipline comes from context minimization (docs/CONTRACT.md section 7), never from
# starving output budgets — a truncated call wastes the whole request and forces a retry.

# OpenRouter reasoning-effort control (genuinely supported — see
# pydantic_ai.models.openrouter.OpenRouterReasoning / OpenRouterModelSettings.openrouter_reasoning,
# confirmed by inspecting the installed pydantic-ai 2.36.0 source). Suppressing hidden
# reasoning tokens on the non-Developer agents cuts latency + token burn on reasoning-heavy
# free models. "low" (not "none") is used deliberately: some reasoning-tuned models tie tool
# -call reliability to having *some* reasoning budget, and that trade-off is unverified live
# under the current quota constraints — this is a real, load-bearing parameter, not a no-op.
# Developer is left unset (None -> no override) since it targets a separate, non-reasoning
# code-tuned model (OPENROUTER_MODEL_DEVELOPER) and needs its full capability for codegen.
AGENT_REASONING_EFFORT: dict[str, str | None] = {
    "Planner": "low",
    "Designer": "low",
    "Content": "low",
    "Developer": None,
    "Evaluator": "low",
}

# Per-agent temperature control (added 2026-09-02: a reviewer measurement found the SAME
# UNCHANGED generated site scored 6.0/6.5/7.5/8.0 across 4 identical run_evaluator calls —
# a 2.0-point spread straddling the pass threshold, i.e. pass/fail was close to a coin flip
# regardless of actual site quality. Root cause: `temperature` was never set anywhere, so
# every agent — including the LLM-as-Judge — ran at the provider default (~0.7-1.0), so the
# judge was sampling, not measuring. `temperature` is a genuine, documented ModelSettings
# field explicitly listed as "Supported by: ... OpenRouter" in the installed pydantic-ai
# 2.36.0 (confirmed by inspecting pydantic_ai.settings.ModelSettings source) — it is
# forwarded on the wire, not silently dropped.
#   - Evaluator: 0.0 — judging must be as repeatable as we can make it; this is the point.
#   - Planner: 0.2 — emits structured facts (name/audience/sections), not prose.
#   - Designer/Content/Developer: None (provider default) — deliberately left alone. Their
#     output quality is already good and creative variation is desirable there; do not risk
#     regressing visual quality by flattening it to near-zero temperature.
AGENT_TEMPERATURE: dict[str, float | None] = {
    "Planner": 0.2,
    "Designer": None,
    "Content": None,
    "Developer": None,
    "Evaluator": 0.0,
}

# Failover on these — never to "fix" a bad/paid model, only transient unavailability.
# 403 is included alongside 429/5xx/timeout per a live-quota incident (2026-08-31): a
# free-tier key can get a confirmed-free model id back as "403 not entitled" for reasons
# unrelated to pricing (marketplace/entitlement gating), so it must not abort a run.
_FAILOVER_STATUS_CODES = {403, 429, 500, 502, 503, 504}
_MAX_TOTAL_ATTEMPTS = 5  # primary + up to 4 fallback hops (widened for daily-quota resilience)

_model_cache: dict[str, OpenRouterModel] = {}
_verification_cache: VerificationResult | None = None


def _get_verification(settings: Settings | None = None) -> VerificationResult:
    """Lazily verify (once per process) that every configured model is free.

    This is defense-in-depth: `cli.py doctor`/`serve` already verify at startup, but the
    LLM layer itself never trusts a model it hasn't independently confirmed is $0.
    """
    global _verification_cache
    if _verification_cache is None:
        _verification_cache = verify_models_are_free(settings)
    return _verification_cache


def build_model(model_id: str, settings: Settings | None = None) -> OpenRouterModel:
    """Build (and cache) an OpenRouterModel for `model_id`. Always routes through OpenRouter."""
    if model_id in _model_cache:
        return _model_cache[model_id]
    settings = settings or get_settings()
    provider = OpenRouterProvider(
        api_key=settings.openrouter_api_key,
        app_url=settings.public_base_url,
        app_title="Agent Flow",
    )
    model = OpenRouterModel(model_id, provider=provider)
    _model_cache[model_id] = model
    return model


def _served_model_name(result: AgentRunResult[Any]) -> str | None:
    return getattr(result.response, "model_name", None)


def _is_upstream_provider_error(exc: ModelHTTPError) -> bool:
    """True for a 400 raised by the UPSTREAM provider, not by our request being malformed.

    OpenRouter surfaces upstream failures as
        {'message': 'Provider returned error', 'metadata': {'provider_name': 'Cohere', ...}}
    Observed live: cohere/north-mini-code:free INTERMITTENTLY rejects the tool_results shape
    pydantic-ai sends, with "all elements in tool_results must have the 'outputs' property
    specified" — a provider-side incompatibility we cannot fix from here, and which failing
    over to another FREE model resolves.

    A genuine client-side 400 (our schema/request is wrong) carries no provider metadata and
    is deliberately NOT retried, so real bugs stay loud instead of being masked by failover.
    """
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return False
    metadata = body.get("metadata") or {}
    return bool(metadata.get("provider_name")) or body.get("message") == "Provider returned error"


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, ModelHTTPError):
        if exc.status_code in _FAILOVER_STATUS_CODES:
            return True
        # 400 only when it demonstrably came from the upstream provider (see above).
        return exc.status_code == 400 and _is_upstream_provider_error(exc)
    if isinstance(exc, httpx.TimeoutException):
        return True
    # A model that cannot satisfy the output schema after its own bounded retry is, for our
    # purposes, unusable — the same as unavailable. Observed live: dots-studio exhausted its
    # retries producing a DesignSpec ("Exceeded maximum output retries (1)") and the run died,
    # because this isn't an HTTP error so failover never engaged. Hopping to another FREE model
    # recovers it. Still bounded by _MAX_TOTAL_ATTEMPTS, so this cannot loop.
    if isinstance(exc, UnexpectedModelBehavior):
        return True
    return False


async def run_agent(
    *,
    run_id: str,
    agent_name: str,
    agent: Agent[Any, Any],
    user_prompt: str,
    model_id: str | None = None,
    fallback_model_ids: Sequence[str] | None = None,
    settings: Settings | None = None,
    catalog: dict[str, ModelPricing] | None = None,
    **run_kwargs: Any,
) -> AgentRunResult[Any]:
    """Run `agent` against OpenRouter with free->free failover, recording an LLMCall.

    On 403/429/5xx/timeout, hops to the next id in `fallback_model_ids` (up to
    `_MAX_TOTAL_ATTEMPTS` total attempts), logging a TraceEvent for each hop. Never
    substitutes a paid model. Raises the last error if every candidate is exhausted.

    If `agent_name` has a default reasoning effort in `AGENT_REASONING_EFFORT` and/or a
    default temperature in `AGENT_TEMPERATURE`, and the caller didn't already set
    `openrouter_reasoning`/`temperature` in `model_settings`, they're merged in
    automatically — callers (agents/*.py) don't need to know about either.
    """
    settings = settings or get_settings()
    verification = _get_verification(settings)
    if catalog is None:
        catalog = verification.catalog

    primary = model_id or settings.openrouter_model
    fallbacks = list(fallback_model_ids) if fallback_model_ids is not None else list(settings.openrouter_fallback_models)
    candidates = [primary, *fallbacks][:_MAX_TOTAL_ATTEMPTS]

    run_kwargs = dict(run_kwargs)
    effort = AGENT_REASONING_EFFORT.get(agent_name)
    temperature = AGENT_TEMPERATURE.get(agent_name)
    if effort is not None or temperature is not None:
        model_settings = dict(run_kwargs.get("model_settings") or {})
        if effort is not None:
            model_settings.setdefault("openrouter_reasoning", {"effort": effort})
        if temperature is not None:
            model_settings.setdefault("temperature", temperature)
        run_kwargs["model_settings"] = model_settings

    last_exc: Exception | None = None
    for attempt_idx, candidate_model_id in enumerate(candidates, start=1):
        model = build_model(candidate_model_id, settings)
        started = time.monotonic()
        try:
            result = await agent.run(user_prompt, model=model, **run_kwargs)
        except Exception as exc:  # noqa: BLE001 - inspect, then either failover or re-raise
            if _is_retryable(exc) and attempt_idx < len(candidates):
                STORE.event(
                    run_id,
                    f"{agent_name}: {candidate_model_id} unavailable ({exc}) — "
                    f"failing over to {candidates[attempt_idx]}",
                )
                last_exc = exc
                continue
            raise

        latency_ms = (time.monotonic() - started) * 1000
        served_model = _served_model_name(result) or candidate_model_id
        usage = result.usage

        call = LLMCall(
            agent=agent_name,
            model_requested=candidate_model_id,
            model_served=served_model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cached_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            latency_ms=latency_ms,
            cost_usd=compute_cost(candidate_model_id, usage.input_tokens, usage.output_tokens, catalog),
            attempt=attempt_idx,
            failover_from=candidates[attempt_idx - 2] if attempt_idx > 1 else None,
        )
        STORE.record_llm_call(run_id, agent_name, call)
        return result

    assert last_exc is not None  # pragma: no cover - loop always returns or raises
    raise last_exc
