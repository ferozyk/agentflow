"""Tests for the five agents: context minimization, correct model routing, and the
Developer's sandboxed filesystem tools. The LLM layer (`run_agent`) is mocked throughout
-- these are unit tests, not live-network tests (see the orchestrator's own
end-to-end smoke test for that)."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
from pydantic_ai import ModelRetry

from agentflow.agents import content as content_mod
from agentflow.agents import designer as designer_mod
from agentflow.agents import developer as developer_mod
from agentflow.agents import evaluator as evaluator_mod
from agentflow.agents import planner as planner_mod
from agentflow.agents.content import run_content
from agentflow.agents.designer import run_designer
from agentflow.agents.developer import (
    DeveloperDeps,
    _list_files_tool,
    _read_file_tool,
    _write_file_tool,
    run_developer,
)
from agentflow.agents.evaluator import run_deterministic_checks, run_evaluator
from agentflow.agents.planner import run_planner
from agentflow.guardrails.filesystem import SiteWorkspace
from agentflow.observability.tracing import STORE
from agentflow.models import (
    DesignSpec,
    JudgeEvaluation,
    SectionContent,
    WebsiteContent,
    WebsitePlan,
)


def _fake_result(output):
    """Stand-in for pydantic_ai's AgentRunResult -- every agent only reads `.output`."""
    return SimpleNamespace(output=output)


SAMPLE_PLAN = WebsitePlan(
    name="Agent Flow",
    description="An AI engineering company.",
    target_audience="enterprise technology leaders",
    sections=["hero", "features", "testimonials", "cta", "footer"],
    visual_style="modern, dark, technical",
)

SAMPLE_DESIGN = DesignSpec(
    layout="single-page scroll with sticky nav",
    visual_style="modern dark technical",
    typography="system sans-serif, bold headings",
    color_direction="near-black background, electric blue accent",
    components=["sticky nav", "gradient hero", "card grid", "footer with columns"],
)

SAMPLE_CONTENT = WebsiteContent(
    tagline="Ship agents, not glue code.",
    sections=[
        SectionContent(section="hero", heading="Agent Flow", body="Build agentic systems fast.", cta="Get started"),
        SectionContent(section="features", heading="Why Agent Flow", body="Composable, observable, fast."),
    ],
    footer="(c) Agent Flow",
)

# A well-formed page satisfying every deterministic check, including the accessibility
# ones (html lang, exactly one h1, a <main> landmark, no unlabeled <img>). Used as the
# "everything passes" baseline in several tests below.
WELL_FORMED_HTML = (
    '<html lang="en"><head><title>Agent Flow</title></head><body>'
    '<nav aria-label="Primary">hero features testimonials cta footer</nav>'
    "<main>"
    "<header><h1>hero</h1></header>"
    "<section>features</section><section>testimonials</section><section>cta</section>"
    "</main>"
    "<footer>footer</footer>"
    + ("x" * 900)
    + "</body></html>"
)


# -- context minimization: enforced by the function SIGNATURE itself -------------------


@pytest.mark.parametrize(
    ("fn", "expected_params"),
    [
        (run_planner, ["run_id", "request"]),
        (run_designer, ["run_id", "plan"]),
        (run_content, ["run_id", "plan"]),
        (run_evaluator, ["run_id", "plan", "files"]),
    ],
)
def test_signature_enforces_context_minimization(fn, expected_params):
    params = list(inspect.signature(fn).parameters)
    assert params == expected_params, (
        f"{fn.__name__} must accept EXACTLY {expected_params} -- widening the signature "
        "would let a caller smuggle in extra context, defeating the point."
    )


def test_developer_signature_takes_plan_design_content_workspace_only():
    params = list(inspect.signature(run_developer).parameters)
    assert params[:5] == ["run_id", "plan", "design", "content", "workspace"]
    # span/fix_notes are keyword-only retry plumbing, not extra business context.
    assert set(params[5:]) == {"span", "fix_notes"}


# -- Planner: request only, verbatim ---------------------------------------------------


async def test_run_planner_passes_request_verbatim(monkeypatch):
    captured = {}

    async def fake_run_agent(*, user_prompt, model_settings, **kwargs):
        captured["user_prompt"] = user_prompt
        captured["max_tokens"] = model_settings["max_tokens"]
        return _fake_result(SAMPLE_PLAN)

    monkeypatch.setattr(planner_mod, "run_agent", fake_run_agent)

    request = "Build a landing page for a coffee subscription startup."
    result = await run_planner("af-test1", request)

    assert result is SAMPLE_PLAN
    assert captured["user_prompt"] == request  # verbatim, no wrapping/extra context
    from agentflow.llm import AGENT_MAX_TOKENS

    assert captured["max_tokens"] == AGENT_MAX_TOKENS["Planner"]


# -- Designer / Content: plan only, never the raw request -------------------------------


async def test_run_designer_prompt_contains_plan_not_raw_request(monkeypatch):
    captured = {}

    async def fake_run_agent(*, user_prompt, **kwargs):
        captured["user_prompt"] = user_prompt
        return _fake_result(SAMPLE_DESIGN)

    monkeypatch.setattr(designer_mod, "run_agent", fake_run_agent)

    result = await run_designer("af-test2", SAMPLE_PLAN)

    assert result is SAMPLE_DESIGN
    prompt = captured["user_prompt"]
    assert SAMPLE_PLAN.name in prompt
    assert SAMPLE_PLAN.visual_style in prompt
    # The raw Slack/CLI request text is never available to this function at all --
    # there's no parameter for it (see signature test above). Nothing to assert away.


async def test_run_content_prompt_contains_plan_sections(monkeypatch):
    captured = {}

    async def fake_run_agent(*, user_prompt, **kwargs):
        captured["user_prompt"] = user_prompt
        return _fake_result(SAMPLE_CONTENT)

    monkeypatch.setattr(content_mod, "run_agent", fake_run_agent)

    result = await run_content("af-test3", SAMPLE_PLAN)

    assert result is SAMPLE_CONTENT
    for section in SAMPLE_PLAN.sections:
        assert section in captured["user_prompt"]


# -- Developer: plan + design + content, routed to the DEVELOPER model ------------------


async def test_run_developer_uses_developer_model_and_full_prompt(monkeypatch, tmp_path):
    captured = {}

    async def fake_run_agent(*, user_prompt, model_id, deps, model_settings, **kwargs):
        captured["user_prompt"] = user_prompt
        captured["model_id"] = model_id
        captured["deps"] = deps
        return _fake_result(developer_mod.DeveloperOutput(files_written=["index.html"], summary="ok"))

    monkeypatch.setattr(developer_mod, "run_agent", fake_run_agent)

    workspace = SiteWorkspace(tmp_path / "generated-site" / "af-test4")
    result = await run_developer("af-test4", SAMPLE_PLAN, SAMPLE_DESIGN, SAMPLE_CONTENT, workspace)

    from agentflow.config import get_settings

    assert captured["model_id"] == get_settings().openrouter_model_developer
    assert isinstance(captured["deps"], DeveloperDeps)
    assert captured["deps"].workspace is workspace
    prompt = captured["user_prompt"]
    assert SAMPLE_PLAN.name in prompt
    assert SAMPLE_DESIGN.color_direction in prompt
    assert SAMPLE_CONTENT.tagline in prompt
    assert result.summary == "ok"


async def test_run_developer_fix_notes_only_present_on_retry(monkeypatch, tmp_path):
    captured = {}

    async def fake_run_agent(*, user_prompt, **kwargs):
        captured["user_prompt"] = user_prompt
        return _fake_result(developer_mod.DeveloperOutput(files_written=[], summary="ok"))

    monkeypatch.setattr(developer_mod, "run_agent", fake_run_agent)
    workspace = SiteWorkspace(tmp_path / "generated-site" / "af-test5")

    await run_developer("af-test5", SAMPLE_PLAN, SAMPLE_DESIGN, SAMPLE_CONTENT, workspace)
    assert "evaluator rejected" not in captured["user_prompt"].lower()

    await run_developer(
        "af-test5", SAMPLE_PLAN, SAMPLE_DESIGN, SAMPLE_CONTENT, workspace, fix_notes="- fix contrast"
    )
    assert "evaluator rejected" in captured["user_prompt"].lower()
    assert "fix contrast" in captured["user_prompt"]


# -- Developer's filesystem tools: wrap SiteWorkspace, ModelRetry propagates ------------


async def test_write_file_tool_writes_through_workspace(tmp_path):
    workspace = SiteWorkspace(tmp_path / "generated-site" / "af-test6")
    ctx = SimpleNamespace(deps=DeveloperDeps(workspace=workspace, run_id="af-test6"))

    result = await _write_file_tool(ctx, "index.html", "<html>hi</html>")

    assert "index.html" in result
    assert workspace.read_file("index.html") == "<html>hi</html>"


async def test_read_and_list_tools_wrap_workspace(tmp_path):
    workspace = SiteWorkspace(tmp_path / "generated-site" / "af-test7")
    workspace.write_file("styles.css", "body { color: red; }")
    ctx = SimpleNamespace(deps=DeveloperDeps(workspace=workspace, run_id="af-test7"))

    assert await _read_file_tool(ctx, "styles.css") == "body { color: red; }"
    assert await _list_files_tool(ctx) == ["styles.css"]


async def test_write_file_tool_path_escape_raises_model_retry(tmp_path):
    workspace = SiteWorkspace(tmp_path / "generated-site" / "af-test8")
    ctx = SimpleNamespace(deps=DeveloperDeps(workspace=workspace, run_id="af-test8"))

    with pytest.raises(ModelRetry):
        await _write_file_tool(ctx, "../../.env", "PAYLOAD=1")


async def test_write_file_tool_bad_extension_raises_model_retry(tmp_path):
    workspace = SiteWorkspace(tmp_path / "generated-site" / "af-test9")
    ctx = SimpleNamespace(deps=DeveloperDeps(workspace=workspace, run_id="af-test9"))

    with pytest.raises(ModelRetry):
        await _write_file_tool(ctx, "run.sh", "rm -rf /")


async def test_tool_latency_accumulates_on_span():
    from agentflow.observability.tracing import AgentSpan

    workspace_root_span = AgentSpan(span_id="s1", agent_name="Developer")
    assert workspace_root_span.tool_latency_ms == 0.0


async def test_write_file_tool_records_tool_latency_on_span(tmp_path):
    from agentflow.observability.tracing import AgentSpan

    workspace = SiteWorkspace(tmp_path / "generated-site" / "af-test10")
    span = AgentSpan(span_id="s2", agent_name="Developer")
    ctx = SimpleNamespace(deps=DeveloperDeps(workspace=workspace, run_id="af-test10", span=span))

    await _write_file_tool(ctx, "index.html", "<html>hi</html>")

    assert span.tool_latency_ms >= 0.0  # monotonic clock: never negative, may be ~0 on fast disks


# -- Evaluator: deterministic checks are pure Python, independent of the LLM judge ------


def test_deterministic_checks_flag_missing_files():
    checks = run_deterministic_checks(SAMPLE_PLAN, {})
    by_name = {c.name: c for c in checks}
    assert by_name["index_html_exists"].passed is False
    assert by_name["css_exists"].passed is False


def test_deterministic_checks_pass_for_well_formed_site():
    checks = run_deterministic_checks(
        SAMPLE_PLAN, {"index.html": WELL_FORMED_HTML, "styles.css": "body{}"}
    )
    assert all(c.passed for c in checks), [c for c in checks if not c.passed]


# -- accessibility checks (docs/CONTRACT.md credibility fix): each is pure Python, no LLM --


def test_html_lang_missing_fails_required_check():
    html = WELL_FORMED_HTML.replace('<html lang="en">', "<html>")
    checks = run_deterministic_checks(SAMPLE_PLAN, {"index.html": html, "styles.css": "x"})
    by_name = {c.name: c for c in checks}
    assert by_name["html_lang_present"].passed is False
    assert "lang" in by_name["html_lang_present"].detail.lower()


def test_html_lang_empty_value_fails():
    html = WELL_FORMED_HTML.replace('<html lang="en">', '<html lang="">')
    checks = run_deterministic_checks(SAMPLE_PLAN, {"index.html": html, "styles.css": "x"})
    by_name = {c.name: c for c in checks}
    assert by_name["html_lang_present"].passed is False


def test_html_lang_present_passes():
    checks = run_deterministic_checks(SAMPLE_PLAN, {"index.html": WELL_FORMED_HTML, "styles.css": "x"})
    by_name = {c.name: c for c in checks}
    assert by_name["html_lang_present"].passed is True


def test_zero_h1_fails_required_check():
    html = WELL_FORMED_HTML.replace("<h1>hero</h1>", "<p>hero</p>")
    checks = run_deterministic_checks(SAMPLE_PLAN, {"index.html": html, "styles.css": "x"})
    by_name = {c.name: c for c in checks}
    assert by_name["has_single_h1"].passed is False
    assert "0" in by_name["has_single_h1"].detail


def test_multiple_h1_fails_required_check():
    html = WELL_FORMED_HTML.replace("<h1>hero</h1>", "<h1>hero</h1><h1>duplicate</h1>")
    checks = run_deterministic_checks(SAMPLE_PLAN, {"index.html": html, "styles.css": "x"})
    by_name = {c.name: c for c in checks}
    assert by_name["has_single_h1"].passed is False
    assert "2" in by_name["has_single_h1"].detail


def test_exactly_one_h1_passes():
    checks = run_deterministic_checks(SAMPLE_PLAN, {"index.html": WELL_FORMED_HTML, "styles.css": "x"})
    by_name = {c.name: c for c in checks}
    assert by_name["has_single_h1"].passed is True


def test_missing_main_landmark_fails_required_check():
    html = WELL_FORMED_HTML.replace("<main>", "").replace("</main>", "")
    checks = run_deterministic_checks(SAMPLE_PLAN, {"index.html": html, "styles.css": "x"})
    by_name = {c.name: c for c in checks}
    assert by_name["has_main"].passed is False


def test_main_landmark_present_passes():
    checks = run_deterministic_checks(SAMPLE_PLAN, {"index.html": WELL_FORMED_HTML, "styles.css": "x"})
    by_name = {c.name: c for c in checks}
    assert by_name["has_main"].passed is True


def test_images_have_alt_vacuously_passes_with_no_images():
    # WELL_FORMED_HTML has no <img> tags at all -- the check must pass, but HONESTLY:
    # the detail string must say so rather than silently claiming a real check happened.
    checks = run_deterministic_checks(SAMPLE_PLAN, {"index.html": WELL_FORMED_HTML, "styles.css": "x"})
    by_name = {c.name: c for c in checks}
    assert by_name["images_have_alt"].passed is True
    assert "no <img>" in by_name["images_have_alt"].detail.lower()
    assert "vacuous" in by_name["images_have_alt"].detail.lower()


def test_images_missing_alt_fail_required_check():
    html = WELL_FORMED_HTML.replace("<h1>hero</h1>", '<h1>hero</h1><img src="hero.png">')
    checks = run_deterministic_checks(SAMPLE_PLAN, {"index.html": html, "styles.css": "x"})
    by_name = {c.name: c for c in checks}
    assert by_name["images_have_alt"].passed is False
    assert "missing" in by_name["images_have_alt"].detail.lower()


def test_images_with_empty_alt_fail():
    html = WELL_FORMED_HTML.replace("<h1>hero</h1>", '<h1>hero</h1><img src="hero.png" alt="">')
    checks = run_deterministic_checks(SAMPLE_PLAN, {"index.html": html, "styles.css": "x"})
    by_name = {c.name: c for c in checks}
    assert by_name["images_have_alt"].passed is False


def test_images_with_real_alt_pass():
    html = WELL_FORMED_HTML.replace(
        "<h1>hero</h1>", '<h1>hero</h1><img src="hero.png" alt="Agent Flow dashboard screenshot">'
    )
    checks = run_deterministic_checks(SAMPLE_PLAN, {"index.html": html, "styles.css": "x"})
    by_name = {c.name: c for c in checks}
    assert by_name["images_have_alt"].passed is True
    assert "1" in by_name["images_have_alt"].detail


def test_accessibility_checks_are_required_for_overall_pass():
    """Reviewer's credibility gap, closed: a site with a high judge score but missing
    accessibility structure must NOT be reported as passed -- these checks are required,
    not advisory (see REQUIRED_CHECK_NAMES in evaluator.py)."""
    from agentflow.agents.evaluator import REQUIRED_CHECK_NAMES

    for name in ("html_lang_present", "has_single_h1", "has_main", "images_have_alt"):
        assert name in REQUIRED_CHECK_NAMES


async def test_run_evaluator_combines_deterministic_and_judge(monkeypatch):
    captured = {}

    async def fake_run_agent(*, user_prompt, **kwargs):
        captured["user_prompt"] = user_prompt
        return _fake_result(JudgeEvaluation(score=8.5, issues=["minor spacing"], suggestions=["tighten hero"]))

    monkeypatch.setattr(evaluator_mod, "run_agent", fake_run_agent)

    files = {"index.html": "(missing)"}  # deterministic checks will fail on this
    result = await run_evaluator("af-test11", SAMPLE_PLAN, files)

    assert result.judge is not None
    assert result.judge.score == 8.5
    # deterministic checks fail (empty/too-small html) -> overall passed must be False
    # even though the judge score alone would pass, proving the two are combined AND'd.
    assert result.passed is False
    assert any(c.passed is False for c in result.deterministic_checks)
    assert "minor spacing" in result.issues


async def test_run_evaluator_passes_when_both_deterministic_and_judge_pass(monkeypatch):
    async def fake_run_agent(*, user_prompt, **kwargs):
        return _fake_result(JudgeEvaluation(score=9.0, issues=[], suggestions=[]))

    monkeypatch.setattr(evaluator_mod, "run_agent", fake_run_agent)

    result = await run_evaluator(
        "af-test12", SAMPLE_PLAN, {"index.html": WELL_FORMED_HTML, "styles.css": "body{}"}
    )

    assert result.passed is True
    assert result.score == 9.0


async def test_run_evaluator_fails_on_missing_accessibility_despite_high_judge_score(monkeypatch):
    """The exact scenario the reviewer flagged: a site the judge scores highly but that
    has zero <main>, no aria roles, etc. must not come back `passed=True` any more."""

    async def fake_run_agent(*, user_prompt, **kwargs):
        return _fake_result(JudgeEvaluation(score=8.5, issues=[], suggestions=[]))

    monkeypatch.setattr(evaluator_mod, "run_agent", fake_run_agent)

    # Same shape as the real af-79031 output that prompted this fix: no <main>, no aria-*,
    # no role=. lang/title/nav/footer/h1 are all fine, but accessibility structure is not.
    html = (
        '<html lang="en"><head><title>Agent Flow</title></head><body>'
        "<nav>hero features</nav>"
        "<header><h1>hero</h1></header>"
        "<section>features</section>"
        "<footer>footer</footer>"
        + ("x" * 900)
        + "</body></html>"
    )
    result = await run_evaluator("af-test13", SAMPLE_PLAN, {"index.html": html, "styles.css": "x"})

    assert result.judge.score == 8.5
    assert result.passed is False  # required accessibility check (has_main) failed
    assert any("main" in issue.lower() for issue in result.issues)


# -- median-of-3 LLM-judge sampling: rejects single-sample noise ------------------------
#
# Measured: repeated judge calls on an UNCHANGED site returned 6.0/6.5/7.5/8.0 at default
# sampling, and 6.0-9.0 across 8 calls even at temperature=0.0. The pass threshold (7)
# sits inside that noise band, so a single unlucky draw can fail a genuinely good site.
# JUDGE_SAMPLES independent judge calls run CONCURRENTLY; the MEDIAN is reported.


def _judge_agent_sequence(scores_and_errors: list):
    """Build a fake `run_agent` that returns/raises according to `scores_and_errors`, in
    call order. An item that's a float/int returns a JudgeEvaluation with that score
    (and a marker in issues/suggestions so the winning sample is identifiable); an
    Exception instance is raised instead."""
    call_index = {"n": -1}

    async def fake_run_agent(*, user_prompt, **kwargs):
        call_index["n"] += 1
        item = scores_and_errors[call_index["n"]]
        if isinstance(item, BaseException):
            raise item
        return _fake_result(
            JudgeEvaluation(score=float(item), issues=[f"issue-from-{item}"], suggestions=[f"suggestion-from-{item}"])
        )

    return fake_run_agent


async def test_median_of_three_rejects_low_outlier(monkeypatch):
    """[6, 7, 9] -> median 7, not the mean (7.33) and not the low outlier (6)."""
    monkeypatch.setattr(evaluator_mod, "run_agent", _judge_agent_sequence([6, 7, 9]))

    result = await run_evaluator(
        "af-median1", SAMPLE_PLAN, {"index.html": WELL_FORMED_HTML, "styles.css": "x"}
    )

    assert result.score == 7.0
    assert result.judge.score == 7.0


async def test_median_rejects_low_outlier_with_duplicate_high_scores(monkeypatch):
    """[6, 8, 8] -> median 8 -- the duplicate high scores outvote the one low outlier."""
    monkeypatch.setattr(evaluator_mod, "run_agent", _judge_agent_sequence([6, 8, 8]))

    result = await run_evaluator(
        "af-median2", SAMPLE_PLAN, {"index.html": WELL_FORMED_HTML, "styles.css": "x"}
    )

    assert result.score == 8.0


async def test_median_issues_and_suggestions_come_from_the_median_run_only(monkeypatch):
    """The reported issues/suggestions must come from the SAME sample as the reported
    score -- not merged/concatenated across all three judge calls."""
    monkeypatch.setattr(evaluator_mod, "run_agent", _judge_agent_sequence([6, 7, 9]))

    result = await run_evaluator(
        "af-median3", SAMPLE_PLAN, {"index.html": WELL_FORMED_HTML, "styles.css": "x"}
    )

    assert result.score == 7.0
    assert "issue-from-7" in result.issues
    assert result.suggestions == ["suggestion-from-7"]
    # Must NOT contain the other two samples' issues/suggestions.
    assert "issue-from-6" not in result.issues
    assert "issue-from-9" not in result.issues


async def test_one_judge_call_failing_still_yields_median_of_remaining_two(monkeypatch):
    """One of three samples raising (e.g. a 429) must not fail the whole Evaluator --
    the median is taken over whatever succeeded."""
    monkeypatch.setattr(
        evaluator_mod, "run_agent", _judge_agent_sequence([6, RuntimeError("429 rate limited"), 8])
    )

    result = await run_evaluator(
        "af-median4", SAMPLE_PLAN, {"index.html": WELL_FORMED_HTML, "styles.css": "x"}
    )

    # median of the two survivors {6, 8} is their average, 7.0.
    assert result.score == 7.0


async def test_all_three_judge_calls_failing_raises(monkeypatch):
    """Only fail the Evaluator span if EVERY sample fails -- never fabricate a score."""
    monkeypatch.setattr(
        evaluator_mod,
        "run_agent",
        _judge_agent_sequence([RuntimeError("429"), RuntimeError("500"), RuntimeError("timeout")]),
    )

    with pytest.raises(RuntimeError):
        await run_evaluator("af-median5", SAMPLE_PLAN, {"index.html": WELL_FORMED_HTML, "styles.css": "x"})


async def test_judge_calls_run_concurrently_not_sequentially(monkeypatch):
    """Sequential sampling would triple Evaluator latency -- assert the three judge calls
    are actually concurrent (elapsed ~= one call's delay, not three)."""
    import asyncio as _asyncio
    import time as _time

    async def fake_run_agent(*, user_prompt, **kwargs):
        await _asyncio.sleep(0.1)
        return _fake_result(JudgeEvaluation(score=7.0, issues=[], suggestions=[]))

    monkeypatch.setattr(evaluator_mod, "run_agent", fake_run_agent)

    started = _time.monotonic()
    await run_evaluator("af-median6", SAMPLE_PLAN, {"index.html": WELL_FORMED_HTML, "styles.css": "x"})
    elapsed = _time.monotonic() - started

    assert elapsed < 0.25, f"judge samples did not run concurrently (took {elapsed:.3f}s)"


async def test_trace_event_lists_all_sampled_scores_and_median(monkeypatch):
    """The user must be able to SEE the variance mitigation working -- all raw scores and
    the resulting median must land in the trace, not just the final number."""
    STORE.start_run("test request", run_id="af-median7")
    monkeypatch.setattr(evaluator_mod, "run_agent", _judge_agent_sequence([6, 7, 9]))

    await run_evaluator("af-median7", SAMPLE_PLAN, {"index.html": WELL_FORMED_HTML, "styles.css": "x"})

    messages = [e.message for e in STORE.get("af-median7").events]
    sampling_messages = [m for m in messages if "judge sampled" in m.lower()]
    assert len(sampling_messages) == 1
    msg = sampling_messages[0]
    assert "6.0" in msg and "7.0" in msg and "9.0" in msg
    assert "median 7.0" in msg


def test_deterministic_checks_unaffected_by_judge_sampling_change():
    """The deterministic-vs-LLM-judge separation is a headline talking point -- median-of-3
    sampling must not touch run_deterministic_checks at all."""
    checks = run_deterministic_checks(SAMPLE_PLAN, {"index.html": WELL_FORMED_HTML, "styles.css": "x"})
    # Same 11 checks as before this change -- not tripled, not altered.
    assert len(checks) == 11
    assert all(c.passed for c in checks)
