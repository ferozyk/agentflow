"""Content agent: turns a WebsitePlan into the actual on-page copy.

CONTEXT MINIMIZATION: `run_content`'s signature accepts only `(run_id, plan)` -- same
guarantee as the Designer agent, and deliberately identical input so both can run
concurrently off the same WebsitePlan (see orchestrator.py).
"""
from __future__ import annotations

from pydantic_ai import Agent

from ..llm import AGENT_MAX_TOKENS, run_agent
from ..models import WebsiteContent, WebsitePlan

CONTENT_SYSTEM_PROMPT = (
    "You are the Content agent for Agent Flow. Given a WebsitePlan, write the actual "
    "on-page copy: a short tagline, one SectionContent per section listed (section name, "
    "a heading, a 1-3 sentence body, and an optional short cta for hero/cta-like sections "
    "only), and a one-line footer. Match tone to the target audience and visual style. "
    "Keep copy tight and concrete -- no filler, no lorem ipsum, no bracket placeholders. "
    "Produce exactly one SectionContent per listed section, in the same order, and no "
    "extra sections. No commentary outside the structured fields."
)


def build_content_agent() -> Agent[None, WebsiteContent]:
    return Agent(output_type=WebsiteContent, system_prompt=CONTENT_SYSTEM_PROMPT, retries=1)


def _plan_prompt(plan: WebsitePlan) -> str:
    return (
        f"Website: {plan.name}\n"
        f"Description: {plan.description}\n"
        f"Target audience: {plan.target_audience}\n"
        f"Sections (write one SectionContent per item, in order): {', '.join(plan.sections)}\n"
        f"Visual style / tone cue: {plan.visual_style}"
    )


async def run_content(run_id: str, plan: WebsitePlan) -> WebsiteContent:
    """Content <- WebsitePlan only."""
    agent = build_content_agent()
    result = await run_agent(
        run_id=run_id,
        agent_name="Content",
        agent=agent,
        user_prompt=_plan_prompt(plan),
        model_settings={"max_tokens": AGENT_MAX_TOKENS["Content"]},
    )
    return result.output
