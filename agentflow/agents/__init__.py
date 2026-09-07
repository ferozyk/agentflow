"""Agent Flow's five agents.

Each module owns exactly one static, module-level system prompt constant and one
`run_*` entrypoint whose *signature* enforces context minimization (docs/CONTRACT.md
section 7) -- e.g. `run_designer(run_id, plan)` structurally cannot receive the raw
request, WebsiteContent, or the RunTrace, because those parameters don't exist.
"""
from .content import CONTENT_SYSTEM_PROMPT, run_content
from .designer import DESIGNER_SYSTEM_PROMPT, run_designer
from .developer import DEVELOPER_SYSTEM_PROMPT, DeveloperOutput, run_developer
from .evaluator import EVALUATOR_SYSTEM_PROMPT, run_evaluator
from .planner import PLANNER_SYSTEM_PROMPT, run_planner

__all__ = [
    "run_planner",
    "run_designer",
    "run_content",
    "run_developer",
    "run_evaluator",
    "DeveloperOutput",
    "PLANNER_SYSTEM_PROMPT",
    "DESIGNER_SYSTEM_PROMPT",
    "CONTENT_SYSTEM_PROMPT",
    "DEVELOPER_SYSTEM_PROMPT",
    "EVALUATOR_SYSTEM_PROMPT",
]
