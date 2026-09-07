"""Developer agent: writes the actual static website through SiteWorkspace ONLY.

CONTEXT MINIMIZATION: `run_developer`'s signature accepts `(run_id, plan, design, content,
workspace)` plus optional retry-only `fix_notes` -- exactly WebsitePlan + DesignSpec +
WebsiteContent per docs/CONTRACT.md section 7, never the RunTrace.

FILESYSTEM GUARDRAIL: the agent has NO shell/exec tool anywhere. The only way it touches
disk is through the three tools below, which are thin wrappers around `SiteWorkspace`.
Every path rejection inside `SiteWorkspace` raises `pydantic_ai.ModelRetry`, which
propagates straight out of these wrapper tools so the calling LLM sees the refusal as a
tool error and can retry with a corrected path (bounded by the agent's own `retries=1`).

Tool functions are module-level (not closures) so they can be unit-tested directly by
constructing a `DeveloperDeps` and a minimal stand-in for `RunContext` (see
tests/test_agents.py) without spinning up a full pydantic-ai run.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from ..guardrails.filesystem import SiteWorkspace
from ..llm import AGENT_MAX_TOKENS, run_agent
from ..models import DesignSpec, WebsiteContent, WebsitePlan
from ..observability.tracing import STORE, AgentSpan

DEVELOPER_SYSTEM_PROMPT = (
    "You are the Developer agent for Agent Flow. You build a real static website using "
    "ONLY the write_file/read_file/list_files tools -- there is no shell access. Given a "
    "WebsitePlan, DesignSpec and WebsiteContent, write index.html then styles.css. "
    "index.html: semantic HTML5 with a <nav aria-label> linking every section, a hero "
    "<header> containing exactly one <h1> (no other <h1> anywhere) and a primary CTA "
    "button, a <main> landmark wrapping the hero and every content <section> (heading/"
    "body/cta each), alt text on any <img>, and a <footer>; link styles.css. styles.css: "
    "implement the DesignSpec's colors, typography and layout with real spacing, hover "
    "states, and responsive @media rules so the page looks professionally designed, not "
    "bare. Finish by returning a short summary of what you wrote."
)


class DeveloperOutput(BaseModel):
    files_written: list[str]
    summary: str


@dataclass
class DeveloperDeps:
    """Everything a Developer tool call needs -- the sandboxed workspace, plus enough
    identity to report tool latency back into the run's AgentSpan (never a full trace)."""

    workspace: SiteWorkspace
    run_id: str
    span: AgentSpan | None = None


def _record_tool_latency(deps: DeveloperDeps, started: float) -> None:
    if deps.span is not None:
        deps.span.tool_latency_ms += (time.monotonic() - started) * 1000
        STORE.publish(deps.run_id)


async def _write_file_tool(ctx: RunContext[DeveloperDeps], path: str, content: str) -> str:
    """Write a file inside the sandboxed site workspace. An out-of-sandbox, disallowed
    extension, oversized, or too-numerous path is rejected and raises a retryable error --
    fix the path/extension and try again."""
    started = time.monotonic()
    rel = ctx.deps.workspace.write_file(path, content)
    _record_tool_latency(ctx.deps, started)
    return f"wrote {rel} ({len(content)} bytes)"


async def _read_file_tool(ctx: RunContext[DeveloperDeps], path: str) -> str:
    """Read back a file already written to the sandboxed workspace (useful before a fix
    pass, to see what's there)."""
    started = time.monotonic()
    text = ctx.deps.workspace.read_file(path)
    _record_tool_latency(ctx.deps, started)
    return text


async def _list_files_tool(ctx: RunContext[DeveloperDeps]) -> list[str]:
    """List every file currently written in the sandboxed workspace."""
    started = time.monotonic()
    files = ctx.deps.workspace.list_files()
    _record_tool_latency(ctx.deps, started)
    return files


def build_developer_agent() -> Agent[DeveloperDeps, DeveloperOutput]:
    agent = Agent(
        output_type=DeveloperOutput,
        system_prompt=DEVELOPER_SYSTEM_PROMPT,
        deps_type=DeveloperDeps,
        retries=1,
    )
    agent.tool(_write_file_tool, name="write_file")
    agent.tool(_read_file_tool, name="read_file")
    agent.tool(_list_files_tool, name="list_files")
    return agent


def _build_prompt(
    plan: WebsitePlan,
    design: DesignSpec,
    content: WebsiteContent,
    fix_notes: str | None,
) -> str:
    sections_text = "\n".join(
        f"  - {s.section}: heading={s.heading!r} body={s.body!r} cta={s.cta!r}"
        for s in content.sections
    )
    prompt = (
        "WebsitePlan:\n"
        f"  name: {plan.name}\n"
        f"  description: {plan.description}\n"
        f"  target_audience: {plan.target_audience}\n"
        f"  sections: {', '.join(plan.sections)}\n"
        f"  visual_style: {plan.visual_style}\n\n"
        "DesignSpec:\n"
        f"  layout: {design.layout}\n"
        f"  visual_style: {design.visual_style}\n"
        f"  typography: {design.typography}\n"
        f"  color_direction: {design.color_direction}\n"
        f"  components: {', '.join(design.components)}\n\n"
        "WebsiteContent:\n"
        f"  tagline: {content.tagline}\n"
        f"{sections_text}\n"
        f"  footer: {content.footer}\n"
    )
    if fix_notes:
        prompt += (
            "\nAn evaluator rejected the previous version of this site -- it already exists "
            "and scored reasonably; do not throw it away. REQUIRED FIRST STEP: call "
            "read_file on index.html, then on styles.css, and actually read what comes "
            "back. Only after that, call write_file to make the smallest edits that fix "
            "every issue below -- keep every heading, section, and style rule that isn't "
            f"implicated in an issue untouched:\n{fix_notes}\n"
        )
    else:
        prompt += "\nWrite index.html, then styles.css, now.\n"
    return prompt


async def run_developer(
    run_id: str,
    plan: WebsitePlan,
    design: DesignSpec,
    content: WebsiteContent,
    workspace: SiteWorkspace,
    *,
    span: AgentSpan | None = None,
    fix_notes: str | None = None,
) -> DeveloperOutput:
    """Developer <- WebsitePlan + DesignSpec + WebsiteContent (+ optional fix_notes on the
    one allowed retry). Always uses `settings.openrouter_model_developer`, never the
    default agent model."""
    from ..config import get_settings

    agent = build_developer_agent()
    deps = DeveloperDeps(workspace=workspace, run_id=run_id, span=span)
    result = await run_agent(
        run_id=run_id,
        agent_name="Developer",
        agent=agent,
        user_prompt=_build_prompt(plan, design, content, fix_notes),
        model_id=get_settings().openrouter_model_developer,
        deps=deps,
        model_settings={"max_tokens": AGENT_MAX_TOKENS["Developer"]},
    )
    return result.output
