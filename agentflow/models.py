"""Pydantic data contracts shared by every Agent Flow agent.

These are the typed handoffs between Planner -> Designer/Content -> Developer -> Evaluator.
Keep field names EXACTLY as specified in docs/CONTRACT.md section 5 — other subagents
(agents/, slackapp.py, dashboard/) import these directly.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AgentStatus = Literal["WAITING", "RUNNING", "COMPLETED", "FAILED", "BLOCKED"]
RunStatus = Literal["RUNNING", "COMPLETED", "FAILED", "BLOCKED"]


class WebsitePlan(BaseModel):
    name: str
    description: str
    target_audience: str
    sections: list[str] = Field(min_length=4, max_length=6)
    visual_style: str


class DesignSpec(BaseModel):
    layout: str
    visual_style: str
    typography: str
    color_direction: str
    components: list[str]


class SectionContent(BaseModel):
    section: str
    heading: str
    body: str
    cta: str | None = None


class WebsiteContent(BaseModel):
    tagline: str
    sections: list[SectionContent]
    footer: str


class DeterministicCheck(BaseModel):
    name: str
    passed: bool
    detail: str


class JudgeEvaluation(BaseModel):
    score: float  # 0-10
    issues: list[str]
    suggestions: list[str]


class EvaluationResult(BaseModel):
    score: float
    passed: bool  # passed = score >= 7 AND all required checks pass
    issues: list[str]
    suggestions: list[str]
    deterministic_checks: list[DeterministicCheck]
    judge: JudgeEvaluation | None = None
