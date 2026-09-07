"""Slack Socket Mode integration for Agent Flow (docs/CONTRACT.md sections 2, 3, 8).

Owns exactly two integration points that the dashboard's FastAPI lifespan calls,
in the SAME process/event loop as the dashboard (the trace store is in-memory —
see CONTRACT.md section 2, "SINGLE PROCESS RULE"):

    await start_slack_if_configured() -> bool   # False + log line if tokens missing; never raises
    await stop_slack() -> None

No HTTP endpoint / signing secret is used — this is Socket Mode only, per
slack/manifest.yaml (`socket_mode_enabled: true`), which this file does not modify.

Message formatting is exposed as small pure functions over `RunTrace`
(`format_initial_message`, `format_progress_message`, `format_final_message`,
`format_blocked_message`, `format_failed_message`, `format_terminal_message`) so
tests can build a fixture RunTrace conforming to CONTRACT.md section 6 and assert
on rendering without needing agentflow.orchestrator / agentflow.agents to exist.

`agentflow.orchestrator` is imported lazily (inside the command handler) because it
is owned by a different subagent and may not exist yet while this file is written.
"""
from __future__ import annotations

import asyncio
import logging
import time

from .config import get_settings
from .observability.tracing import AGENT_NAMES, STORE, RunTrace

logger = logging.getLogger("agentflow.slack")

# Emoji per INSTRUCTIONS.md "Main Demo Scenario" / "Slack Integration" examples.
AGENT_EMOJI = {
    "Planner": "🧠",
    "Designer": "🎨",
    "Content": "✍️",
    "Developer": "💻",
    "Evaluator": "🔍",
}

STATUS_LABEL = {
    "WAITING": "WAITING",
    "RUNNING": "RUNNING",
    "COMPLETED": "✓",
    "FAILED": "✗ FAILED",
    "BLOCKED": "BLOCKED",
}

TERMINAL_STATUSES = {"COMPLETED", "FAILED", "BLOCKED"}

# Slack's hard ack timeout is 3s; we edit the same message rather than spamming new
# ones, throttled so a burst of span transitions doesn't hit Slack's rate limits.
EDIT_THROTTLE_SECONDS = 1.0

_app = None
_handler = None
_watch_tasks: set[asyncio.Task] = set()


# ---------------------------------------------------------------------------
# URL helpers — everything is built from PUBLIC_BASE_URL (CONTRACT.md section 8),
# never hardcoded to localhost, so a cloudflared/ngrok tunnel URL renders correctly.
# ---------------------------------------------------------------------------


def _dashboard_url(run_id: str) -> str:
    base = get_settings().public_base_url.rstrip("/")
    return f"{base}/run/{run_id}"


def _site_url(run_id: str) -> str:
    base = get_settings().public_base_url.rstrip("/")
    return f"{base}/site/{run_id}/"


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _aggregate_llm_stats(trace: RunTrace) -> tuple[int, int, int, int, float]:
    """(call_count, input_tokens, output_tokens, total_tokens, cost_usd) across all spans."""
    calls = [call for span in trace.spans for call in span.llm_calls]
    input_tokens = sum(c.input_tokens for c in calls)
    output_tokens = sum(c.output_tokens for c in calls)
    total_tokens = sum(c.total_tokens for c in calls)
    cost_usd = sum(c.cost_usd for c in calls)
    return len(calls), input_tokens, output_tokens, total_tokens, cost_usd


# ---------------------------------------------------------------------------
# Message formatting — pure functions over RunTrace, no Slack client involved.
# ---------------------------------------------------------------------------


def format_initial_message(run_id: str) -> str:
    """The ack-time message: Planner RUNNING, the rest WAITING (INSTRUCTIONS.md
    "Main Demo Scenario"). Posted before we've received any real trace snapshot —
    the background workflow starts Planner immediately, so this is accurate as of
    the moment of posting and is superseded by format_progress_message on the
    first real update.
    """
    lines = ["*🚀 Agent Flow started*", "", f"Run ID: `{run_id}`", ""]
    for i, name in enumerate(AGENT_NAMES):
        status = "RUNNING" if i == 0 else "WAITING"
        lines.append(f"{AGENT_EMOJI.get(name, '•')} {name} — {status}")
    lines.append("")
    lines.append(f"Dashboard: <{_dashboard_url(run_id)}|Open Dashboard>")
    return "\n".join(lines)


def format_progress_message(trace: RunTrace) -> str:
    """In-flight progress edit, driven by the real AgentSpan statuses."""
    lines = ["*🚀 Agent Flow running*", "", f"Run ID: `{trace.run_id}`", ""]
    for span in trace.spans:
        label = STATUS_LABEL.get(span.status, span.status)
        lines.append(f"{AGENT_EMOJI.get(span.agent_name, '•')} {span.agent_name} — {label}")
    lines.append("")
    lines.append(f"Dashboard: <{_dashboard_url(trace.run_id)}|Open Dashboard>")
    return "\n".join(lines)


def format_blocked_message(trace: RunTrace) -> str:
    """BLOCKED: the input guardrail rejected the request. No LLM call was made —
    never imply a workflow ran.
    """
    reason = "request rejected by input guardrail"
    for ev in trace.guardrail_events:
        if ev.kind == "input" and ev.blocked:
            reason = ev.reason
            break
    lines = [
        "*🛡️ Request blocked*",
        "",
        f"Run ID: `{trace.run_id}`",
        f"Reason: {reason}",
        "",
        "No agent workflow was run — the request never reached the LLM.",
        "",
        f"Dashboard: <{_dashboard_url(trace.run_id)}|Open Dashboard>",
    ]
    return "\n".join(lines)


def format_failed_message(trace: RunTrace) -> str:
    """FAILED: the workflow crashed. Show the real error, never claim success."""
    error = trace.events[-1].message if trace.events else "unknown error"
    for span in trace.spans:
        if span.status == "FAILED" and span.error:
            error = f"{span.agent_name}: {span.error}"
            break
    latency = f"{trace.duration_ms / 1000:.1f}s" if trace.duration_ms is not None else "n/a"
    n_calls, _in, _out, _tot, _cost = _aggregate_llm_stats(trace)
    lines = [
        "*❌ Workflow failed*",
        "",
        f"Run ID: `{trace.run_id}`",
        f"Error: {error}",
        f"Latency: {latency}",
        f"LLM calls: {n_calls}",
        "",
        f"Provider: {trace.provider}",
        f"Model: {trace.model}",
        "",
        f"Dashboard: <{_dashboard_url(trace.run_id)}|Open Dashboard>",
    ]
    return "\n".join(lines)


def format_final_message(trace: RunTrace) -> str:
    """COMPLETED: report only what the trace actually recorded — score, latency,
    LLM call count, token breakdown, provider/model/pricing, and cost computed
    from the trace (never a hardcoded number). PASS/FAIL is evaluation.passed,
    not merely "it finished".
    """
    n_calls, in_tok, out_tok, tot_tok, cost = _aggregate_llm_stats(trace)
    latency = f"{trace.duration_ms / 1000:.1f}s" if trace.duration_ms is not None else "n/a"

    evaluation = trace.evaluation
    if evaluation is not None:
        passed = evaluation.passed
        headline = "*🎉 Website completed*" if passed else "*⚠️ Website completed — evaluation FAILED*"
        eval_line = f"Evaluation: {evaluation.score:.1f}/10 — {'PASS' if passed else 'FAIL'}"
    else:
        headline = "*⚠️ Website completed — no evaluation recorded*"
        eval_line = "Evaluation: not available"

    cost_line = f"LLM cost: ${cost:.2f}"
    if trace.pricing_status == "FREE" and cost != 0.0:
        # Should be mathematically impossible for a verified-free model — surface it
        # loudly rather than silently rounding it away to $0.00.
        cost_line += "  (unexpected non-zero cost for a FREE model — check catalog)"

    lines = [
        headline,
        "",
        f"Run ID: `{trace.run_id}`",
        eval_line,
        f"Latency: {latency}",
        f"LLM calls: {n_calls}",
        f"Tokens: input {_fmt_int(in_tok)} / output {_fmt_int(out_tok)} / total {_fmt_int(tot_tok)}",
        "",
        f"Provider: {trace.provider}",
        f"Model: {trace.model}",
        f"Pricing: {trace.pricing_status}",
        cost_line,
    ]
    if trace.retry_count:
        lines.append(f"Retries: {trace.retry_count}")
    lines.append("")
    lines.append(f"Open dashboard: <{_dashboard_url(trace.run_id)}|Dashboard>")

    developer_ran = any(span.agent_name == "Developer" and span.status == "COMPLETED" for span in trace.spans)
    if trace.site_path is not None or developer_ran:
        lines.append(f"Open website: <{_site_url(trace.run_id)}|Generated Website>")

    return "\n".join(lines)


def format_terminal_message(trace: RunTrace) -> str:
    """Dispatch to the right terminal formatter for trace.status."""
    if trace.status == "BLOCKED":
        return format_blocked_message(trace)
    if trace.status == "FAILED":
        return format_failed_message(trace)
    return format_final_message(trace)


# ---------------------------------------------------------------------------
# Slack Bolt wiring (Socket Mode — no HTTP endpoint, no signing secret).
# ---------------------------------------------------------------------------


def _build_app():
    from slack_bolt.async_app import AsyncApp

    settings = get_settings()
    app = AsyncApp(token=settings.slack_bot_token)

    @app.command("/build-website")
    async def handle_build_website(ack, body, client) -> None:
        # Slack's hard ack timeout is 3 seconds — this MUST be the first await, no
        # matter what happens afterwards.
        await ack()

        request_text = (body.get("text") or "").strip()
        channel_id = body.get("channel_id")
        user_id = body.get("user_id")

        if not request_text:
            try:
                await client.chat_postEphemeral(
                    channel=channel_id,
                    user=user_id,
                    text="Usage: `/build-website <request>`",
                )
            except Exception:  # noqa: BLE001
                logger.exception("failed to post usage message")
            return

        try:
            # Local import: agentflow.orchestrator is owned by another subagent and
            # may not exist yet while this file is developed/tested.
            from .orchestrator import start_workflow_background

            run_id = start_workflow_background(request_text)
        except Exception:  # noqa: BLE001
            logger.exception("failed to start workflow from /build-website")
            try:
                await client.chat_postEphemeral(
                    channel=channel_id,
                    user=user_id,
                    text="Sorry — Agent Flow could not start that workflow. Check server logs.",
                )
            except Exception:  # noqa: BLE001
                logger.exception("failed to post start-failure message")
            return

        try:
            posted = await client.chat_postMessage(channel=channel_id, text=format_initial_message(run_id))
        except Exception:  # noqa: BLE001
            logger.exception("failed to post initial Slack message for run %s", run_id)
            return

        task = asyncio.create_task(_watch_run(client, channel_id, posted["ts"], run_id))
        _watch_tasks.add(task)
        task.add_done_callback(_watch_tasks.discard)

    return app


async def _watch_run(client, channel: str, ts: str, run_id: str) -> None:
    """Subscribe to STORE updates for run_id and edit the same Slack message
    (chat_update) as spans change, throttled to ~1 edit/second. Terminal states
    (COMPLETED/FAILED/BLOCKED) are always sent immediately, never throttled away.
    """
    queue = STORE.subscribe(run_id)
    last_edit = 0.0
    try:
        while True:
            trace = await queue.get()
            terminal = trace.status in TERMINAL_STATUSES
            now = time.monotonic()
            if not terminal and (now - last_edit) < EDIT_THROTTLE_SECONDS:
                continue
            last_edit = now
            text = format_terminal_message(trace) if terminal else format_progress_message(trace)
            try:
                await client.chat_update(channel=channel, ts=ts, text=text)
            except Exception:  # noqa: BLE001
                logger.exception("failed to edit Slack message for run %s", run_id)
            if terminal:
                break
    finally:
        STORE.unsubscribe(run_id, queue)


async def start_slack_if_configured() -> bool:
    """Start the Slack Socket Mode client if both SLACK_BOT_TOKEN and SLACK_APP_TOKEN
    are set. Runs in the caller's event loop (the dashboard's FastAPI lifespan) — see
    CONTRACT.md section 2, the trace store is in-memory and single-process.

    NEVER raises: any missing config or startup failure is logged and this returns
    False so `agentflow serve` / the dashboard / CLI-triggered builds still work
    without Slack (per CONTRACT.md section 0: ".env may have no Slack tokens").
    """
    global _app, _handler

    settings = get_settings()
    if not settings.slack_bot_token or not settings.slack_app_token:
        logger.warning(
            "Slack disabled: SLACK_BOT_TOKEN and/or SLACK_APP_TOKEN not set in .env. "
            "Set both (see slack/README.md) to enable /build-website."
        )
        return False

    try:
        from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

        _app = _build_app()
        _handler = AsyncSocketModeHandler(_app, settings.slack_app_token)
        await _handler.connect_async()
    except Exception:  # noqa: BLE001 - Slack must never take the dashboard down with it
        logger.exception("Slack failed to start; continuing without Slack")
        _app = None
        _handler = None
        return False

    logger.info("Slack Socket Mode connected — /build-website is live")
    return True


async def stop_slack() -> None:
    """Disconnect the Socket Mode client, if running. Safe to call even if Slack was
    never started (e.g. tokens were missing).
    """
    global _app, _handler

    for task in list(_watch_tasks):
        task.cancel()
    _watch_tasks.clear()

    if _handler is not None:
        try:
            await _handler.close_async()
        except Exception:  # noqa: BLE001
            logger.exception("error while closing Slack Socket Mode handler")

    _handler = None
    _app = None
