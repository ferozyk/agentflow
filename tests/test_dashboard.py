"""Tests for the Agent Flow dashboard (agentflow/dashboard/app.py).

These tests build fixture RunTrace data directly through the real TraceStore
(agentflow.observability.tracing.STORE) — the same object the orchestrator will
mutate at runtime — and drive the FastAPI app through TestClient. No numbers are
invented here either: fixtures either set a real value or deliberately leave a
field at its "unknown" default (None / 0 / empty) so we can assert the dashboard's
truthful "not reported" / "Insufficient samples" fallbacks are actually reachable.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agentflow.dashboard.app import GENERATED_SITE_ROOT, STATIC_DIR, app
from agentflow.models import DeterministicCheck, EvaluationResult, JudgeEvaluation
from agentflow.observability.tracing import STORE, GuardrailEvent, LLMCall
from datetime import datetime, timezone

client = TestClient(app)


# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------


def _new_run_id() -> str:
    return f"af-{uuid.uuid4().hex[:5]}"


def _build_fixture_trace(*, with_evaluation: bool = True, with_guardrail: bool = True):
    """Populate STORE with one fully-shaped, COMPLETED RunTrace and return it.

    Mirrors what agentflow.orchestrator will do at runtime, but calls TraceStore's
    public mutation methods directly (record_llm_call / record_guardrail / event)
    instead of running real agents, since agents/orchestrator are owned by other
    subagents and may not exist yet.
    """
    run_id = _new_run_id()
    trace = STORE.start_run(request="Build a landing page for Agent Flow", run_id=run_id)

    planner = next(s for s in trace.spans if s.agent_name == "Planner")
    planner.status = "COMPLETED"
    planner.duration_ms = 1200.0
    STORE.record_llm_call(
        run_id,
        "Planner",
        LLMCall(
            agent="Planner",
            model_requested="dots-studio/dots-3-note-preview:free",
            model_served="dots-studio/dots-3-note-preview:free",
            input_tokens=300,
            output_tokens=120,
            total_tokens=420,
            latency_ms=1150.0,
            cost_usd=0.0,
        ),
    )

    designer = next(s for s in trace.spans if s.agent_name == "Designer")
    designer.status = "COMPLETED"
    designer.duration_ms = 1800.0
    STORE.record_llm_call(
        run_id,
        "Designer",
        LLMCall(
            agent="Designer",
            model_requested="dots-studio/dots-3-note-preview:free",
            model_served="minimax/minimax-m2.7:free",  # router served a different model
            failover_from="dots-studio/dots-3-note-preview:free",
            attempt=2,
            input_tokens=200,
            output_tokens=150,
            total_tokens=350,
            latency_ms=1700.0,
            cost_usd=0.0,
        ),
    )

    content = next(s for s in trace.spans if s.agent_name == "Content")
    content.status = "COMPLETED"
    content.duration_ms = 1900.0
    STORE.record_llm_call(
        run_id,
        "Content",
        LLMCall(
            agent="Content",
            model_requested="dots-studio/dots-3-note-preview:free",
            model_served=None,  # simulate "not reported" by the provider
            input_tokens=250,
            output_tokens=400,
            total_tokens=650,
            latency_ms=1850.0,
            cost_usd=0.0,
            cached_tokens=0,
        ),
    )

    developer = next(s for s in trace.spans if s.agent_name == "Developer")
    developer.status = "COMPLETED"
    developer.duration_ms = 5800.0
    STORE.record_llm_call(
        run_id,
        "Developer",
        LLMCall(
            agent="Developer",
            model_requested="cohere/north-mini-code:free",
            model_served="cohere/north-mini-code:free",
            input_tokens=1820,
            output_tokens=2340,
            total_tokens=4160,
            latency_ms=5700.0,
            cost_usd=0.0,
        ),
    )

    evaluator = next(s for s in trace.spans if s.agent_name == "Evaluator")
    evaluator.status = "COMPLETED"
    evaluator.duration_ms = 2000.0
    STORE.record_llm_call(
        run_id,
        "Evaluator",
        LLMCall(
            agent="Evaluator",
            model_requested="dots-studio/dots-3-note-preview:free",
            model_served="dots-studio/dots-3-note-preview:free",
            input_tokens=500,
            output_tokens=100,
            total_tokens=600,
            latency_ms=1900.0,
            cost_usd=0.0,
        ),
    )

    STORE.event(run_id, "Planner started")
    STORE.event(run_id, "Planner completed — 1.2s")
    STORE.event(run_id, "Designer started")
    STORE.event(run_id, "Content started")

    if with_guardrail:
        STORE.record_guardrail(
            run_id,
            GuardrailEvent(
                timestamp=datetime.now(timezone.utc),
                kind="filesystem",
                tool="filesystem.write",
                target="../../.env",
                reason="Path outside allowed workspace",
                blocked=True,
            ),
        )

    if with_evaluation:
        trace.evaluation = EvaluationResult(
            score=8.4,
            passed=True,
            issues=["Footer contact link is a placeholder"],
            suggestions=["Add a testimonials section"],
            deterministic_checks=[
                DeterministicCheck(name="index.html exists", passed=True, detail="found at site root"),
                DeterministicCheck(name="CSS exists", passed=True, detail="styles.css present"),
                DeterministicCheck(name="required sections present", passed=True, detail="6/6 sections found"),
            ],
            judge=JudgeEvaluation(
                score=8.4,
                issues=["Hero copy is slightly generic"],
                suggestions=["Tighten the value proposition in the hero"],
            ),
        )

    trace.status = "COMPLETED"
    trace.duration_ms = sum(s.duration_ms or 0 for s in trace.spans)
    trace.site_path = str(GENERATED_SITE_ROOT / run_id)
    trace.site_url = f"/site/{run_id}/"
    STORE.publish(run_id)
    return trace


@pytest.fixture
def fixture_run():
    trace = _build_fixture_trace()
    yield trace
    STORE._runs.pop(trace.run_id, None)
    STORE._subscribers.pop(trace.run_id, None)


@pytest.fixture
def generated_site(fixture_run):
    site_dir = GENERATED_SITE_ROOT / fixture_run.run_id
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "index.html").write_text("<html><body><h1>Agent Flow</h1></body></html>")
    (site_dir / "styles.css").write_text("body { background: black; }")
    yield fixture_run
    shutil.rmtree(site_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# health / basic wiring
# ---------------------------------------------------------------------------


def test_health_endpoint_shape():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["provider"] == "OpenRouter"
    assert "model" in body
    assert body["pricing_status"] in {"FREE", "OFFLINE-HEURISTIC", "UNVERIFIED"}


def test_static_and_templates_present():
    assert (STATIC_DIR / "style.css").exists()
    assert (STATIC_DIR / "app.js").exists()


# ---------------------------------------------------------------------------
# /api/runs, /api/run/{id}
# ---------------------------------------------------------------------------


def test_api_runs_includes_fixture(fixture_run):
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    ids = [r["run_id"] for r in resp.json()]
    assert fixture_run.run_id in ids


def test_api_run_detail_shape(fixture_run):
    resp = client.get(f"/api/run/{fixture_run.run_id}")
    assert resp.status_code == 200
    body = resp.json()
    for key in (
        "run_id",
        "trace_id",
        "request",
        "status",
        "spans",
        "guardrail_events",
        "events",
        "evaluation",
        "provider",
        "model",
        "pricing_status",
    ):
        assert key in body
    assert body["status"] == "COMPLETED"
    assert len(body["spans"]) == 5
    agent_names = {s["agent_name"] for s in body["spans"]}
    assert agent_names == {"Planner", "Designer", "Content", "Developer", "Evaluator"}
    # aggregate computed fields on AgentSpan must be present and summed from llm_calls
    developer_span = next(s for s in body["spans"] if s["agent_name"] == "Developer")
    assert developer_span["input_tokens"] == 1820
    assert developer_span["output_tokens"] == 2340
    assert developer_span["total_tokens"] == 4160


def test_api_run_not_found_is_404():
    resp = client.get("/api/run/af-doesnotexist")
    assert resp.status_code == 404


def test_model_served_and_failover_are_visible(fixture_run):
    resp = client.get(f"/api/run/{fixture_run.run_id}")
    body = resp.json()
    designer_span = next(s for s in body["spans"] if s["agent_name"] == "Designer")
    call = designer_span["llm_calls"][0]
    assert call["model_requested"] == "dots-studio/dots-3-note-preview:free"
    assert call["model_served"] == "minimax/minimax-m2.7:free"
    assert call["failover_from"] == "dots-studio/dots-3-note-preview:free"
    assert call["model_served"] != call["model_requested"]


def test_model_served_not_reported_when_absent(fixture_run):
    resp = client.get(f"/api/run/{fixture_run.run_id}")
    body = resp.json()
    content_span = next(s for s in body["spans"] if s["agent_name"] == "Content")
    call = content_span["llm_calls"][0]
    # foundation leaves this None (rather than fabricating it) when the provider
    # didn't report it; the dashboard JS renders "not reported" for exactly this.
    assert call["model_served"] is None


# ---------------------------------------------------------------------------
# guardrails
# ---------------------------------------------------------------------------


def test_guardrail_event_recorded_and_served(fixture_run):
    resp = client.get(f"/api/run/{fixture_run.run_id}")
    body = resp.json()
    assert len(body["guardrail_events"]) == 1
    g = body["guardrail_events"][0]
    assert g["blocked"] is True
    assert g["kind"] == "filesystem"
    assert g["target"] == "../../.env"
    assert "outside allowed workspace" in g["reason"]


def test_app_js_renders_guardrail_blocked_style():
    js = (STATIC_DIR / "app.js").read_text()
    assert "GUARDRAIL BLOCKED" in js


# ---------------------------------------------------------------------------
# evaluation: deterministic vs LLM-as-judge separation
# ---------------------------------------------------------------------------


def test_evaluation_keeps_deterministic_and_judge_separate(fixture_run):
    resp = client.get(f"/api/run/{fixture_run.run_id}")
    body = resp.json()
    ev = body["evaluation"]
    assert ev["passed"] is True
    assert ev["score"] == 8.4
    assert len(ev["deterministic_checks"]) == 3
    assert all("passed" in c and "detail" in c for c in ev["deterministic_checks"])
    assert ev["judge"]["score"] == 8.4
    assert ev["judge"]["issues"] == ["Hero copy is slightly generic"]


def test_app_js_visually_separates_deterministic_from_judge():
    js = (STATIC_DIR / "app.js").read_text()
    assert "Deterministic checks" in js
    assert "LLM-as-Judge" in js
    assert "eval-col deterministic" in js
    assert "eval-col judge" in js


def test_evaluation_absent_renders_not_yet_available():
    run_id = _new_run_id()
    trace = STORE.start_run(request="test with no evaluation yet", run_id=run_id)
    try:
        resp = client.get(f"/api/run/{run_id}")
        assert resp.json()["evaluation"] is None
        js = (STATIC_DIR / "app.js").read_text()
        assert "Evaluation not yet available" in js
    finally:
        STORE._runs.pop(run_id, None)
        STORE._subscribers.pop(run_id, None)


# ---------------------------------------------------------------------------
# latency percentiles: "Insufficient samples" must never be fabricated
# ---------------------------------------------------------------------------


def test_latency_percentiles_none_when_few_samples(monkeypatch):
    monkeypatch.setattr(STORE, "latency_percentiles", lambda: {"p50": None, "p90": None, "samples": 2})
    resp = client.get("/")
    assert resp.status_code == 200
    # bootstrap payload embedded in the page must faithfully carry the None/low-sample
    # state through to the client rather than a computed guess.
    assert '"p50": null' in resp.text or '"p50":null' in resp.text
    assert '"samples": 2' in resp.text or '"samples":2' in resp.text
    js = (STATIC_DIR / "app.js").read_text()
    assert "Insufficient samples for P50/P90" in js


def test_latency_percentiles_rendered_when_enough_samples(monkeypatch):
    monkeypatch.setattr(STORE, "latency_percentiles", lambda: {"p50": 4200.0, "p90": 6100.0, "samples": 7})
    resp = client.get("/")
    assert resp.status_code == 200
    assert "4200" in resp.text
    assert "6100" in resp.text


# ---------------------------------------------------------------------------
# prompt caching: never claim caching that isn't reported
# ---------------------------------------------------------------------------


def test_app_js_has_honest_prompt_caching_fallback():
    js = (STATIC_DIR / "app.js").read_text()
    assert "Prompt caching: not reported by selected free model — prompt structure is cache-ready" in js


def test_cached_tokens_reported_when_present():
    run_id = _new_run_id()
    STORE.start_run(request="cache test", run_id=run_id)
    STORE.record_llm_call(
        run_id,
        "Planner",
        LLMCall(
            agent="Planner",
            model_requested="x:free",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=500.0,
            cached_tokens=80,
            cache_write_tokens=20,
        ),
    )
    try:
        resp = client.get(f"/api/run/{run_id}")
        call = resp.json()["spans"][0]["llm_calls"][0]
        assert call["cached_tokens"] == 80
        assert call["cache_write_tokens"] == 20
    finally:
        STORE._runs.pop(run_id, None)
        STORE._subscribers.pop(run_id, None)


def test_large_tokens_survive_intact_alongside_a_failover_call():
    """Reproduces the exact reported scenario (run af-ef3d9): a call with a
    long failover chain AND large token counts (16,870 in / 5,962 out). The
    API layer must never truncate or round these — any clipping would have to
    be a rendering bug, not a data bug, and this pins the data side."""
    run_id = _new_run_id()
    STORE.start_run(request="failover + large tokens test", run_id=run_id)
    STORE.record_llm_call(
        run_id,
        "Developer",
        LLMCall(
            agent="Developer",
            model_requested="dots-studio/dots-3-note-preview:free",
            model_served="minimax/minimax-m2.7:free",
            failover_from="cohere/north-mini-code:free",
            attempt=2,
            input_tokens=16870,
            output_tokens=5962,
            total_tokens=22832,
            latency_ms=9400.0,
        ),
    )
    try:
        resp = client.get(f"/api/run/{run_id}")
        developer_span = next(s for s in resp.json()["spans"] if s["agent_name"] == "Developer")
        call = developer_span["llm_calls"][0]
        assert call["input_tokens"] == 16870
        assert call["output_tokens"] == 5962
        assert call["total_tokens"] == 22832
        assert call["failover_from"] == "cohere/north-mini-code:free"
    finally:
        STORE._runs.pop(run_id, None)
        STORE._subscribers.pop(run_id, None)


# ---------------------------------------------------------------------------
# pricing status visibility (OFFLINE-HEURISTIC must never render as bare FREE)
# ---------------------------------------------------------------------------


def test_app_js_calls_out_offline_heuristic_explicitly():
    js = (STATIC_DIR / "app.js").read_text()
    assert "OFFLINE-HEURISTIC" in js


# ---------------------------------------------------------------------------
# layout / overflow regression guards
#
# Round 1: a live run surfaced long OpenRouter model ids (~36 chars) blowing
# the per-call table past its agent card.
# Round 2: a live run with an actual failover (af-ef3d9) showed the fix from
# round 1 wasn't enough — the wide failover cell inside the <table> still
# forced the whole row wider than the card, which made the *numeric* columns
# clip mid-value ("16,870" rendered as "16,87") or scroll out of view
# entirely ("Out" not visible at all) even though the table "technically"
# scrolled. The real fix: per-call detail is no longer a <table> at all — the
# token/latency/cost stats live in their own flex-wrap row, structurally
# decoupled from model-id width, so a long id can only make *it* wrap; the
# numbers can never be clipped or pushed off screen.
# ---------------------------------------------------------------------------


def test_call_stats_are_not_a_table_and_are_decoupled_from_model_width():
    js = (STATIC_DIR / "app.js").read_text()
    # the old coupled-table design (one wide model cell forces every column
    # in the row wider) must be gone entirely, not just visually patched.
    assert "call-table" not in js
    assert "table-scroll" not in js
    assert 'class: "call-list"' in js
    assert 'class: "call-stats"' in js
    css = (STATIC_DIR / "style.css").read_text()
    assert ".call-stats" in css
    # flex-wrap, not overflow-x:auto, is what guarantees a number is never
    # clipped or scrolled out of view — it just wraps onto another line.
    call_stats_rule = css[css.index(".call-stats {") : css.index(".call-stats {") + 300]
    assert "flex-wrap: wrap" in call_stats_rule


def test_page_body_never_scrolls_horizontally():
    css = (STATIC_DIR / "style.css").read_text()
    # the html/body rule must declare overflow-x: hidden (not just individual
    # panels) so no combination of wide children can scroll the whole page.
    body_rule = css[css.index("html, body {") : css.index("html, body {") + 600]
    assert "overflow-x: hidden" in body_rule


def test_long_model_ids_collapse_when_requested_equals_served():
    js = (STATIC_DIR / "app.js").read_text()
    # common case (no failover, no router substitution) shows the id once.
    assert "same as requested" in js
    assert "served model not reported" in js
    assert "routed — see below" in js


def test_failover_detail_is_a_full_width_annotation_not_a_table_column():
    # The requested/served/failover distinction is still shown in FULL (not
    # hidden behind hover-only truncation) — just as a wrapping annotation
    # line below the row instead of a column that fights the numeric stats
    # for horizontal space.
    js = (STATIC_DIR / "app.js").read_text()
    assert "call-annotation" in js
    assert "failover from" in js
    assert "requested: ${c.model_requested}" in js
    assert "served: ${c.model_served" in js


def test_model_ids_are_truncated_with_full_id_in_title_attr():
    js = (STATIC_DIR / "app.js").read_text()
    assert "text-overflow: ellipsis" in (STATIC_DIR / "style.css").read_text()
    assert "title: text" in js  # truncatedSpan() always carries the full string as title=


def test_model_label_survives_a_call_missing_model_served():
    # A later LLMCall with model_served == None (provider didn't report it on
    # that hop) must not blank out an earlier, perfectly good model id.
    run_id = _new_run_id()
    STORE.start_run(request="robust model label test", run_id=run_id)
    STORE.record_llm_call(
        run_id,
        "Planner",
        LLMCall(agent="Planner", model_requested="dots-studio/dots-3-note-preview:free",
                model_served="dots-studio/dots-3-note-preview:free",
                input_tokens=10, output_tokens=5, total_tokens=15, latency_ms=100.0),
    )
    STORE.record_llm_call(
        run_id,
        "Planner",
        LLMCall(agent="Planner", model_requested="dots-studio/dots-3-note-preview:free",
                model_served=None, input_tokens=1, output_tokens=1, total_tokens=2, latency_ms=50.0),
    )
    try:
        resp = client.get(f"/api/run/{run_id}")
        calls = resp.json()["spans"][0]["llm_calls"]
        assert len(calls) == 2
        assert calls[-1]["model_served"] is None
        # the dashboard's lastKnownModel() helper (app.js) is expected to fall
        # back to the earlier call's known model rather than showing
        # "not reported" here — verified structurally, since app.js isn't
        # executed in this test process.
        js = (STATIC_DIR / "app.js").read_text()
        assert "function lastKnownModel" in js
    finally:
        STORE._runs.pop(run_id, None)
        STORE._subscribers.pop(run_id, None)


def test_agent_card_and_stat_panels_allow_text_to_wrap():
    css = (STATIC_DIR / "style.css").read_text()
    assert "overflow-wrap: anywhere" in css
    assert ".agent-card" in css and "min-width: 0" in css


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------


def test_index_page_renders(fixture_run):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Agent Flow" in resp.text
    assert "AI Website Builder" in resp.text


def test_run_detail_page_renders(fixture_run):
    resp = client.get(f"/run/{fixture_run.run_id}")
    assert resp.status_code == 200
    assert fixture_run.run_id in resp.text


# ---------------------------------------------------------------------------
# POST /api/build — must work standalone (Slack-free fallback)
# ---------------------------------------------------------------------------


def test_build_endpoint_rejects_empty_request():
    resp = client.post("/api/build", json={"request": "   "})
    assert resp.status_code == 400


def test_build_endpoint_behavior_depends_on_orchestrator_availability():
    resp = client.post("/api/build", json={"request": "Build a site for Acme"})
    try:
        import agentflow.orchestrator  # noqa: F401

        orchestrator_available = True
    except ImportError:
        orchestrator_available = False

    if orchestrator_available:
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]
        assert run_id.startswith("af-")
    else:
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /site/{run_id}/ — generated website hosting
# ---------------------------------------------------------------------------


def test_site_serves_generated_index(generated_site):
    resp = client.get(f"/site/{generated_site.run_id}/")
    assert resp.status_code == 200
    assert "Agent Flow" in resp.text


def test_site_serves_other_generated_assets(generated_site):
    resp = client.get(f"/site/{generated_site.run_id}/styles.css")
    assert resp.status_code == 200
    assert "background" in resp.text


def test_site_missing_run_is_404():
    resp = client.get("/site/af-neverexisted/")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# "run no longer in memory" handling (server restart between demo runs)
#
# The TraceStore is in-memory only (CONTRACT.md section 2): restarting
# `agentflow serve` clears every run. A browser left on /run/{old-id} then
# gets a DEFINITIVE 404 from /api/run/{id} forever. The backend correctly
# keeps returning 404 (that behavior is NOT changed here) — these guard the
# client's handling of it: stop polling/SSE instead of looping on it silently
# forever, and explain what happened instead of showing a blank/stale page.
# ---------------------------------------------------------------------------


def test_api_run_404_semantics_unchanged():
    # Backend behavior must NOT change — only client handling of it does.
    resp = client.get("/api/run/af-neverexisted")
    assert resp.status_code == 404
    resp = client.get("/api/run/af-neverexisted/events")
    assert resp.status_code == 404


def test_template_has_a_dedicated_run_gone_slot():
    resp = client.get("/")
    assert resp.status_code == 200
    assert 'id="run-gone"' in resp.text
    # hidden by default — only JS reveals it after a confirmed 404.
    assert '<div id="run-gone" class="run-gone-state" hidden>' in resp.text


def test_app_js_stops_polling_on_a_definitive_404():
    js = (STATIC_DIR / "app.js").read_text()
    assert "function showRunGone" in js
    # the fix must distinguish a definitive 404 from any other failure.
    assert "r.status === 404" in js
    assert js.count("r.status === 404") >= 2  # both startPolling and startLive check it
    # showRunGone must stop both polling and any open EventSource.
    show_run_gone_body = js[js.index("function showRunGone") : js.index("function showRunGone") + 400]
    assert "stopLive()" in show_run_gone_body


def test_app_js_transient_failures_keep_retrying():
    js = (STATIC_DIR / "app.js").read_text()
    # a non-404 failure (5xx, network drop) must NOT call showRunGone — the
    # polling interval body must still branch through it without stopping.
    poll_fn = js[js.index("function startPolling") : js.index("function startLive")]
    assert "showRunGone" in poll_fn
    assert "catch" in poll_fn  # network errors still swallowed -> next tick retries


def test_app_js_sse_error_path_checks_for_404_before_polling():
    # EventSource's error event carries no HTTP status — the fix must
    # explicitly re-check via fetch before falling back to a poll loop that
    # could otherwise spin on a 404 forever, same as the original bug.
    js = (STATIC_DIR / "app.js").read_text()
    onerror_body = js[js.index("eventSource.onerror") : js.index("eventSource.onerror") + 700]
    assert "404" in onerror_body
    assert "showRunGone" in onerror_body


def test_app_js_run_gone_message_explains_in_memory_design_and_offers_recovery():
    js = (STATIC_DIR / "app.js").read_text()
    assert "no longer in memory" in js
    assert "cleared whenever the server restarts" in js
    # recovery actions: back to dashboard, and a site link that is verified
    # (HEAD-checked) before being shown rather than assumed to work.
    assert 'href: "/"' in js  # a real "back to dashboard" link, not just any slash
    assert 'method: "HEAD"' in js
    assert "still on disk" in js


def test_show_run_gone_never_advertises_an_unchecked_site_link():
    # The site link must only be appended inside the HEAD-check's success
    # branch — never unconditionally alongside the "back to dashboard" link.
    js = (STATIC_DIR / "app.js").read_text()
    fn_start = js.index("function showRunGone")
    fn_end = js.index("function startPolling")
    body = js[fn_start:fn_end]
    unconditional_actions_block = body[: body.index("fetch(`/site/")]
    assert "still on disk" not in unconditional_actions_block


# ---------------------------------------------------------------------------
# SSE stream
# ---------------------------------------------------------------------------


async def test_sse_stream_emits_initial_snapshot(fixture_run):
    # The endpoint streams forever (one frame per mutation, else a keepalive comment
    # every SSE_HEARTBEAT_SECONDS) — a real EventSource just keeps that connection
    # open, but httpx's ASGITransport buffers the *entire* response body before
    # handing back a Response, so it never returns for a stream that never ends.
    # That's a test-harness limitation, not a product bug (confirmed: TraceStore
    # itself hands back the initial snapshot synchronously — see the two direct
    # STORE.subscribe()/queue.get() checks elsewhere in this file's history).
    # So we drive the ASGI callable directly at the protocol level instead: send()
    # is invoked by Starlette for every message, and we grab the first non-empty
    # "http.response.body" chunk the instant it's emitted, then cancel.
    run_id = fixture_run.run_id
    path = f"/api/run/{run_id}/events"
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "headers": [],
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "server": ("testserver", 80),
        "client": ("testclient", 12345),
        "root_path": "",
    }

    messages: list[dict] = []
    first_body_received = asyncio.Event()
    request_sent = False

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        # Simulate a client that stays connected (no more request messages, no
        # disconnect) until we cancel the task below — must actually suspend so
        # Starlette's disconnect-listener task doesn't spin the event loop.
        await asyncio.Event().wait()

    async def send(message):
        messages.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            first_body_received.set()

    task = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(first_body_received.wait(), timeout=5)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    start = next(m for m in messages if m["type"] == "http.response.start")
    assert start["status"] == 200
    headers = {k.decode().lower(): v.decode() for k, v in start["headers"]}
    assert "text/event-stream" in headers["content-type"]

    body_msg = next(m for m in messages if m["type"] == "http.response.body" and m.get("body"))
    text = body_msg["body"].decode()
    assert text.startswith("data:")
    payload = json.loads(text[len("data:") :].split("\n\n")[0].strip())
    assert payload["run_id"] == run_id


def test_sse_stream_unknown_run_is_404():
    resp = client.get("/api/run/af-neverexisted/events")
    assert resp.status_code == 404
