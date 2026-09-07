"""Tests for agentflow.slackapp — message formatting + start/stop lifecycle.

These NEVER call the real Slack API. Message formatting is tested against fixture
RunTrace objects built from the real agentflow.observability.tracing / agentflow.models
classes (CONTRACT.md sections 5-6), so the fixtures are guaranteed to match the
authoritative schema rather than a hand-rolled duplicate.

Lifecycle tests (start_slack_if_configured/stop_slack) are hermetic: they inject a
Settings instance directly by monkeypatching `agentflow.slackapp.get_settings`
rather than deleting SLACK_BOT_TOKEN/SLACK_APP_TOKEN from os.environ. Deleting env
vars does NOT work here because config.load_settings() calls python-dotenv's
load_dotenv() on every call, which re-populates any var missing from os.environ
from the repo's real .env file — so a monkeypatch.delenv-based test would pass on
a clean checkout but silently start depending on whatever the developer's local
.env happens to contain (it would break the moment real Slack tokens are added to
.env, which is exactly the machine this must be reliable on). Injecting a Settings
object sidesteps environment/.env entirely, so results never depend on .env
contents. These tests never modify or read the real .env file.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agentflow import slackapp
from agentflow.config import Settings
from agentflow.models import DeterministicCheck, EvaluationResult, JudgeEvaluation
from agentflow.observability.tracing import AGENT_NAMES, AgentSpan, GuardrailEvent, LLMCall, RunTrace


def _fake_settings(**overrides) -> Settings:
    """A Settings instance built with NO dependency on the environment or .env —
    used to make Slack lifecycle tests hermetic regardless of what's in the real
    .env (including real Slack tokens on a developer's machine).
    """
    defaults: dict = dict(
        openrouter_api_key="test-key-not-real",
        slack_bot_token=None,
        slack_app_token=None,
        public_base_url="http://127.0.0.1:8000",
        pricing_status="FREE",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _make_trace(status: str = "RUNNING", **overrides) -> RunTrace:
    spans = [AgentSpan(span_id=f"af-test1-{name.lower()}", agent_name=name) for name in AGENT_NAMES]
    defaults = dict(
        run_id="af-test1",
        trace_id="trace-test1",
        request="Build a modern landing page for Agent Flow",
        status=status,
        created_at=datetime.now(timezone.utc),
        spans=spans,
        model="dots-studio/dots-3-note-preview:free",
        pricing_status="FREE",
    )
    defaults.update(overrides)
    return RunTrace(**defaults)


# ---------------------------------------------------------------------------
# Initial ack-time message
# ---------------------------------------------------------------------------


def test_initial_message_shows_planner_running_others_waiting():
    text = slackapp.format_initial_message("af-abcde")
    assert "af-abcde" in text
    assert "🧠 Planner — RUNNING" in text
    assert "🎨 Designer — WAITING" in text
    assert "✍️ Content — WAITING" in text
    assert "💻 Developer — WAITING" in text
    assert "🔍 Evaluator — WAITING" in text
    assert "Dashboard" in text
    assert "http://127.0.0.1:8000/run/af-abcde" in text


# ---------------------------------------------------------------------------
# Progress edits
# ---------------------------------------------------------------------------


def test_progress_message_reflects_real_span_statuses():
    trace = _make_trace()
    trace.spans[0].status = "COMPLETED"  # Planner
    trace.spans[1].status = "RUNNING"  # Designer
    trace.spans[2].status = "RUNNING"  # Content
    text = slackapp.format_progress_message(trace)
    assert "Planner — ✓" in text
    assert "Designer — RUNNING" in text
    assert "Content — RUNNING" in text
    assert "Developer — WAITING" in text
    assert "Evaluator — WAITING" in text


# ---------------------------------------------------------------------------
# Terminal: COMPLETED + PASS
# ---------------------------------------------------------------------------


def test_final_message_pass_reports_real_trace_numbers():
    trace = _make_trace(status="COMPLETED")
    trace.duration_ms = 24300.0
    trace.evaluation = EvaluationResult(
        score=8.4,
        passed=True,
        issues=[],
        suggestions=[],
        deterministic_checks=[DeterministicCheck(name="index.html exists", passed=True, detail="ok")],
        judge=JudgeEvaluation(score=8.4, issues=[], suggestions=[]),
    )
    trace.spans[3].status = "COMPLETED"  # Developer
    trace.spans[3].llm_calls.append(
        LLMCall(
            agent="Developer",
            model_requested="dots-studio/dots-3-note-preview:free",
            model_served="dots-studio/dots-3-note-preview:free",
            input_tokens=1820,
            output_tokens=2340,
            total_tokens=4160,
            latency_ms=5800.0,
            cost_usd=0.0,
        )
    )
    trace.site_path = "workspace/generated-site/af-test1"

    text = slackapp.format_final_message(trace)

    assert "🎉" in text
    assert "8.4/10" in text
    assert "PASS" in text
    assert "24.3s" in text
    assert "LLM calls: 1" in text
    assert "1,820" in text and "2,340" in text and "4,160" in text
    assert "Provider: OpenRouter" in text
    assert "Model: dots-studio/dots-3-note-preview:free" in text
    assert "Pricing: FREE" in text
    assert "LLM cost: $0.00" in text
    assert "Generated Website" in text
    assert "http://127.0.0.1:8000/site/af-test1/" in text
    assert "Dashboard" in text


def test_final_message_never_fabricates_cost_for_free_model():
    """Zero LLM calls recorded -> cost must be exactly $0.00, never invented."""
    trace = _make_trace(status="COMPLETED")
    trace.evaluation = EvaluationResult(score=9.0, passed=True, issues=[], suggestions=[], deterministic_checks=[])
    text = slackapp.format_final_message(trace)
    assert "LLM cost: $0.00" in text
    assert "LLM calls: 0" in text


# ---------------------------------------------------------------------------
# Terminal: COMPLETED + FAIL (evaluation did not pass, workflow still finished)
# ---------------------------------------------------------------------------


def test_final_message_reports_evaluation_fail_honestly():
    trace = _make_trace(status="COMPLETED")
    trace.evaluation = EvaluationResult(
        score=5.0,
        passed=False,
        issues=["Missing accessibility labels", "Weak CTA copy"],
        suggestions=["Add alt text"],
        deterministic_checks=[DeterministicCheck(name="css exists", passed=True, detail="ok")],
    )
    text = slackapp.format_terminal_message(trace)
    assert "FAIL" in text
    assert "evaluation FAILED" in text
    assert "🎉" not in text
    assert "5.0/10" in text


# ---------------------------------------------------------------------------
# Terminal: BLOCKED (input guardrail)
# ---------------------------------------------------------------------------


def test_blocked_message_shows_guardrail_reason_and_no_workflow_claim():
    trace = _make_trace(status="BLOCKED")
    trace.guardrail_events.append(
        GuardrailEvent(
            timestamp=datetime.now(timezone.utc),
            kind="input",
            tool="input_guard",
            target="request",
            reason="Detected prompt-injection pattern: 'ignore previous instructions'",
            blocked=True,
        )
    )
    text = slackapp.format_terminal_message(trace)
    assert "blocked" in text.lower()
    assert "ignore previous instructions" in text
    assert "No agent workflow was run" in text
    assert "🎉" not in text
    assert "Evaluation" not in text


# ---------------------------------------------------------------------------
# Terminal: FAILED (crash) — must never claim success
# ---------------------------------------------------------------------------


def test_failed_message_shows_error_never_claims_success():
    trace = _make_trace(status="FAILED")
    trace.duration_ms = 3200.0
    trace.spans[2].status = "FAILED"  # Content
    trace.spans[2].error = "ModelRetry: output validation failed twice"
    text = slackapp.format_terminal_message(trace)
    assert "failed" in text.lower()
    assert "ModelRetry" in text
    assert "3.2s" in text
    assert "🎉" not in text
    assert "completed" not in text.lower()


def test_format_terminal_message_dispatches_by_status():
    completed = _make_trace(status="COMPLETED")
    completed.evaluation = EvaluationResult(score=8.0, passed=True, issues=[], suggestions=[], deterministic_checks=[])
    assert slackapp.format_terminal_message(completed) == slackapp.format_final_message(completed)

    blocked = _make_trace(status="BLOCKED")
    assert slackapp.format_terminal_message(blocked) == slackapp.format_blocked_message(blocked)

    failed = _make_trace(status="FAILED")
    assert slackapp.format_terminal_message(failed) == slackapp.format_failed_message(failed)


# ---------------------------------------------------------------------------
# start_slack_if_configured / stop_slack lifecycle — never touches the real API,
# and never depends on what's actually in .env (see module docstring above).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_slack_if_configured_returns_false_without_tokens(monkeypatch):
    monkeypatch.setattr(slackapp, "get_settings", lambda: _fake_settings())

    result = await slackapp.start_slack_if_configured()
    assert result is False


@pytest.mark.asyncio
async def test_start_slack_if_configured_returns_false_with_partial_tokens(monkeypatch):
    monkeypatch.setattr(
        slackapp,
        "get_settings",
        lambda: _fake_settings(slack_bot_token="xoxb-fake", slack_app_token=None),
    )

    result = await slackapp.start_slack_if_configured()
    assert result is False


@pytest.mark.asyncio
async def test_start_slack_if_configured_connects_when_both_tokens_present(monkeypatch):
    """Mirror case: with both tokens present, start_slack_if_configured must
    actually attempt a Socket Mode connection — but the handler class is replaced
    with a fake so this never opens a real WebSocket to Slack. Also verifies
    stop_slack() disconnects the (fake) handler it started.
    """
    import slack_bolt.adapter.socket_mode.async_handler as async_handler_module

    monkeypatch.setattr(
        slackapp,
        "get_settings",
        lambda: _fake_settings(slack_bot_token="xoxb-fake", slack_app_token="xapp-fake"),
    )

    calls: dict = {"connected": False, "closed": False}

    class FakeSocketModeHandler:
        def __init__(self, app, app_token) -> None:
            self.app = app
            self.app_token = app_token

        async def connect_async(self) -> None:
            calls["connected"] = True

        async def close_async(self) -> None:
            calls["closed"] = True

    monkeypatch.setattr(async_handler_module, "AsyncSocketModeHandler", FakeSocketModeHandler)

    result = await slackapp.start_slack_if_configured()
    assert result is True
    assert calls["connected"] is True
    assert calls["closed"] is False

    await slackapp.stop_slack()
    assert calls["closed"] is True


@pytest.mark.asyncio
async def test_stop_slack_is_safe_when_never_started():
    await slackapp.stop_slack()
