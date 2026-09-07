"""Designer agent: turns a WebsitePlan into a DesignSpec.

CONTEXT MINIMIZATION: `run_designer`'s signature accepts only `(run_id, plan)` -- the raw
user request, WebsiteContent, and RunTrace are structurally impossible to pass in here.
"""
from __future__ import annotations

from pydantic_ai import Agent

from ..llm import AGENT_MAX_TOKENS, run_agent
from ..models import DesignSpec, WebsitePlan

DESIGNER_SYSTEM_PROMPT = (
    "You are the Designer agent for Agent Flow. Given a WebsitePlan (name, description, "
    "audience, sections, visual style), produce a concrete DesignSpec: layout (page "
    "structure/grid approach), visual_style (refine the plan's style into something "
    "implementable), typography (a font pairing/scale using widely available web-safe or "
    "system fonts), color_direction (a specific palette: 2-3 colors plus an accent), and "
    "components (4-8 concrete UI patterns, e.g. sticky nav, gradient hero, card grid, "
    "testimonial carousel, multi-column footer). Be concrete enough that a developer could "
    "implement it without further questions. No commentary outside the structured fields."
)


def build_designer_agent() -> Agent[None, DesignSpec]:
    return Agent(output_type=DesignSpec, system_prompt=DESIGNER_SYSTEM_PROMPT, retries=1)


def _plan_prompt(plan: WebsitePlan) -> str:
    return (
        f"Website: {plan.name}\n"
        f"Description: {plan.description}\n"
        f"Target audience: {plan.target_audience}\n"
        f"Sections: {', '.join(plan.sections)}\n"
        f"Visual style direction: {plan.visual_style}"
    )


async def run_designer(run_id: str, plan: WebsitePlan) -> DesignSpec:
    """Designer <- WebsitePlan only."""
    agent = build_designer_agent()
    result = await run_agent(
        run_id=run_id,
        agent_name="Designer",
        agent=agent,
        user_prompt=_plan_prompt(plan),
        model_settings={"max_tokens": AGENT_MAX_TOKENS["Designer"]},
    )
    return result.output
