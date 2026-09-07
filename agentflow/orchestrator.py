"""Agent Flow orchestrator: Planner -> (Designer + Content concurrently) -> Developer ->
Evaluator, with at most one Developer-fix/Evaluator retry cycle. See
docs/CONTRACT.md section 7 for the exact public API this module must expose.

CONTEXT MINIMIZATION is mandatory and is visible right here, not just in the agent
modules: every `run_*` call below passes only the specific typed object(s) that agent's
signature accepts -- never `trace`, never the whole workflow state.

    Planner   <- request only
    Designer  <- WebsitePlan only
    Content   <- WebsitePlan only
    Developer <- WebsitePlan + DesignSpec + WebsiteContent (+ fix_notes on retry)
    Evaluator <- WebsitePlan + generated files
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from .agents.content import run_content
from .agents.designer import run_designer
from .agents.developer import run_developer
from .agents.evaluator import run_evaluator
from .agents.planner import run_planner
from .config import REPO_ROOT, get_settings
from .guardrails.filesystem import SiteWorkspace
from .guardrails.input_guard import check_input
from .models import EvaluationResult
from .observability.tracing import STORE, RunTrace

SITE_ROOT = REPO_ROOT / "workspace" / "generated-site"
GENERATED_EXTENSIONS = (".html", ".css")
MAX_RETRIES = 1  # CONTRACT section 7: at most one Developer-fix/Evaluator cycle, ever.


def _site_files(workspace: SiteWorkspace) -> dict[str, str]:
    """Read back only the generated HTML/CSS -- what the Evaluator is allowed to see."""
    return {
        rel: workspace.read_file(rel)
        for rel in workspace.list_files()
        if rel.endswith(GENERATED_EXTENSIONS)
    }


def _snapshot_site(workspace: SiteWorkspace) -> dict[str, str]:
    """Read EVERY current file in the sandboxed workspace into memory before the one
    allowed fix cycle runs, so a fix that regresses the score can be undone. Reads go
    through SiteWorkspace only -- no shell, nothing outside the sandbox."""
    return {rel: workspace.read_file(rel) for rel in workspace.list_files()}


def _restore_site(workspace: SiteWorkspace, snapshot: dict[str, str]) -> None:
    """Write a `_snapshot_site` result back verbatim -- used when the fix cycle makes the
    site WORSE than it was, so the higher-scoring version is never lost."""
    for rel, content in snapshot.items():
        workspace.write_file(rel, content)


def _fix_notes(evaluation: EvaluationResult) -> str:
    lines = [f"- {issue}" for issue in evaluation.issues]
    if evaluation.suggestions:
        lines.append("Suggestions:")
        lines.extend(f"- {s}" for s in evaluation.suggestions)
    return "\n".join(lines) or (
        "The evaluator gave a low score with no specific issues listed; "
        "improve overall visual polish and requirement coverage."
    )


async def _run_designer_spanned(run_id: str, plan):
    async with STORE.span(run_id, "Designer"):
        return await run_designer(run_id, plan)


async def _run_content_spanned(run_id: str, plan):
    async with STORE.span(run_id, "Content"):
        return await run_content(run_id, plan)


async def run_workflow(request: str, run_id: str | None = None) -> RunTrace:
    """Run the full workflow and return the final RunTrace.

    If `run_id` names a trace already started (e.g. by `start_workflow_background`), that
    trace is reused in place; otherwise a fresh run is started. Never blocks longer than
    the actual agent work -- there is no artificial delay anywhere in this function.
    """
    trace = STORE.get(run_id) if run_id else None
    if trace is None:
        trace = STORE.start_run(request, run_id=run_id)
    rid = trace.run_id

    # --- Input guardrail: deterministic, runs BEFORE any LLM call. -----------------
    guard_event = check_input(request)
    if guard_event is not None:
        STORE.record_guardrail(rid, guard_event)
        trace.status = "BLOCKED"
        trace.ended_at = datetime.now(timezone.utc)
        trace.duration_ms = 0.0
        STORE.event(rid, f"Input guardrail blocked this request: {guard_event.reason}")
        STORE.publish(rid)
        return trace

    started = time.monotonic()
    try:
        async with STORE.span(rid, "Planner"):
            plan = await run_planner(rid, request)  # Planner <- request only

        # Designer and Content run CONCURRENTLY -- both take only the WebsitePlan, so
        # neither has to wait on the other. return_exceptions=True lets each span close
        # itself out (COMPLETED/FAILED) independently before we re-raise the first error.
        design_result, content_result = await asyncio.gather(
            _run_designer_spanned(rid, plan),
            _run_content_spanned(rid, plan),
            return_exceptions=True,
        )
        for outcome in (design_result, content_result):
            if isinstance(outcome, BaseException):
                raise outcome
        design, content = design_result, content_result

        settings = get_settings()
        workspace = SiteWorkspace(SITE_ROOT / rid)

        async with STORE.span(rid, "Developer") as dev_span:
            await run_developer(rid, plan, design, content, workspace, span=dev_span)

        # The site exists on disk as soon as the FIRST Developer pass finishes -- set
        # site_path/site_url now (not only at the very end) so the dashboard/Slack never
        # show "no site" while the one allowed retry, if any, is still running. The site
        # exists on disk regardless of the final evaluation score -- COMPLETED means the
        # pipeline ran to completion without a real exception, not that it scored >= 7.
        # EvaluationResult.passed carries the pass/fail verdict separately.
        try:
            trace.site_path = str(workspace.root.relative_to(REPO_ROOT))
        except ValueError:
            # workspace.root isn't under REPO_ROOT -- only happens in tests that redirect
            # SITE_ROOT into a tmp_path; production always writes under REPO_ROOT.
            trace.site_path = str(workspace.root)
        trace.site_url = f"{settings.public_base_url}/site/{rid}/"
        STORE.publish(rid)

        async with STORE.span(rid, "Evaluator"):
            evaluation = await run_evaluator(rid, plan, _site_files(workspace))
        trace.evaluation = evaluation
        STORE.publish(rid)

        if not evaluation.passed and trace.retry_count < MAX_RETRIES:
            trace.retry_count += 1
            STORE.event(rid, "Evaluation failed -- running the one allowed fix cycle")

            # Snapshot BEFORE the fix cycle writes anything -- a fix that regresses the
            # score must be reversible. Developer writes into the SAME workspace, so
            # without this the better version would be destroyed irreversibly.
            snapshot = _snapshot_site(workspace)

            async with STORE.span(rid, "Developer") as dev_span:
                await run_developer(
                    rid,
                    plan,
                    design,
                    content,
                    workspace,
                    span=dev_span,
                    fix_notes=_fix_notes(evaluation),
                )
            async with STORE.span(rid, "Evaluator"):
                retry_evaluation = await run_evaluator(rid, plan, _site_files(workspace))

            if retry_evaluation.score < evaluation.score:
                # The "fix" made things worse -- restore the higher-scoring version and
                # keep reporting the FIRST evaluation. A retry that can regress with no
                # guard is worse than no retry at all.
                _restore_site(workspace, snapshot)
                STORE.event(
                    rid,
                    f"Fix cycle scored {retry_evaluation.score:.1f} vs "
                    f"{evaluation.score:.1f} -- regression detected, restored the "
                    "higher-scoring version",
                )
                # trace.evaluation intentionally left as the FIRST evaluation (set above).
            elif retry_evaluation.score > evaluation.score:
                trace.evaluation = retry_evaluation
                STORE.event(
                    rid,
                    f"Fix cycle improved the score {evaluation.score:.1f} -> "
                    f"{retry_evaluation.score:.1f} -- keeping the retried version",
                )
            else:
                trace.evaluation = retry_evaluation
                STORE.event(
                    rid,
                    f"Fix cycle re-scored {retry_evaluation.score:.1f} (unchanged) -- "
                    "keeping the retried version",
                )
            STORE.publish(rid)

        trace.status = "COMPLETED"
    except Exception as exc:  # noqa: BLE001 - a real failure must be recorded, never hidden
        trace.status = "FAILED"
        STORE.event(rid, f"Workflow failed: {exc}")
    finally:
        trace.ended_at = datetime.now(timezone.utc)
        trace.duration_ms = (time.monotonic() - started) * 1000
        STORE.publish(rid)

    return trace


def start_workflow_background(request: str) -> str:
    """Start the workflow and return the run_id IMMEDIATELY -- must never block on the
    workflow itself (Slack needs to ack the slash command within its own timeout)."""
    trace = STORE.start_run(request)
    asyncio.create_task(run_workflow(request, run_id=trace.run_id))
    return trace.run_id
