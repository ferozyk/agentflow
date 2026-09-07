"""Planner agent: turns the raw Slack/CLI request into a structured WebsitePlan.

CONTEXT MINIMIZATION (see docs/CONTRACT.md section 7): `run_planner`'s signature only
accepts `(run_id, request)` -- there is no parameter through which a RunTrace or any other
agent's output could leak in. The user prompt IS the raw request, verbatim; nothing else.
"""
from __future__ import annotations

from pydantic_ai import Agent

from ..llm import AGENT_MAX_TOKENS, run_agent
from ..models import WebsitePlan

# Module-level constant: static system prompt (prompt-caching structure per CONTRACT
# section 10). All dynamic data goes in the user prompt, never here.
PLANNER_SYSTEM_PROMPT = (
    "You are the Planner agent for Agent Flow, a website-builder pipeline. Given a short "
    "natural-language request, decide the site's purpose, audience, structure, and visual "
    "direction. Output a WebsitePlan: name (short site/brand name), description (1-2 "
    "sentences), target_audience (who it's for), sections (4-6 section names in the order "
    "they should appear, e.g. hero, about, features, testimonials, pricing, contact), and "
    "visual_style (a short concrete phrase describing look and feel). Be concise and "
    "concrete. Do not include commentary outside the structured fields."
)


def build_planner_agent() -> Agent[None, WebsitePlan]:
    return Agent(output_type=WebsitePlan, system_prompt=PLANNER_SYSTEM_PROMPT, retries=1)


async def run_planner(run_id: str, request: str) -> WebsitePlan:
    """Planner <- request only. No plan, no design, no content, no trace ever passed in."""
    agent = build_planner_agent()
    result = await run_agent(
        run_id=run_id,
        agent_name="Planner",
        agent=agent,
        user_prompt=request,
        model_settings={"max_tokens": AGENT_MAX_TOKENS["Planner"]},
    )
    return result.output
