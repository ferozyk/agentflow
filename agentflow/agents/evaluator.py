"""Evaluator agent: deterministic Python checks + a separate LLM-as-Judge call, combined
into one EvaluationResult that keeps both parts explicitly separable (docs/CONTRACT.md
section 5) for the dashboard.

CONTEXT MINIMIZATION: `run_evaluator`'s signature accepts only `(run_id, plan, files)` --
the WebsitePlan and the generated site files. DesignSpec, WebsiteContent, and the RunTrace
are structurally impossible to pass in here.
"""
from __future__ import annotations

import asyncio
import re
import statistics

from pydantic_ai import Agent

from ..llm import AGENT_MAX_TOKENS, run_agent
from ..models import DeterministicCheck, EvaluationResult, JudgeEvaluation, WebsitePlan
from ..observability.tracing import STORE

EVALUATOR_SYSTEM_PROMPT = (
    "You are the Evaluator agent for Agent Flow, judging a generated static website "
    "against its plan. Score 0-10 on: requirement coverage, content quality, visual "
    "consistency (inferred from the CSS), accessibility basics (semantic tags, alt text, "
    "contrast cues), and technical correctness. Return a JudgeEvaluation: score, issues "
    "(concrete problems found), suggestions (concrete fixes). Be specific and concise -- a "
    "few short bullet points each, not essays. Do not repeat the input back to me."
)

MAX_HTML_CHARS = 6000
MAX_CSS_CHARS = 3000

# The LLM-as-Judge is a noisy estimator: measured against the SAME unchanged site,
# repeated judge calls returned 6.0/6.5/7.5/8.0 at default sampling, and even at
# temperature=0.0 (already the shipped setting in llm.py) 8 calls spread 6.0-9.0 per
# batch (spread ~1.5-2.0) -- pure model-sampling noise from the same model, zero
# failovers involved. The pass threshold (7) sits INSIDE that noise band, so a single
# unlucky low draw can fail a genuinely good site. JUDGE_SAMPLES independent judge calls
# are run concurrently and the MEDIAN score is used, because the median (unlike the
# mean) actually rejects a one-off outlier rather than being dragged toward it.
JUDGE_SAMPLES = 3

# Deterministic checks that must ALL pass for EvaluationResult.passed to be True,
# regardless of the judge score (docs/CONTRACT.md section 5: "passed = score >= 7 AND
# all required checks pass").
#
# The accessibility checks (html_lang_present, has_single_h1, has_main, images_have_alt)
# are deliberately REQUIRED, not advisory: the judge prompt explicitly claims to assess
# "accessibility basics", and a claim like that must rest on something a human can point
# to, not just an LLM's opinion. Each one is binary/objective (a tag either has a
# non-empty `lang` attribute or it doesn't) and low-risk of false-negative -- in
# particular images_have_alt passes vacuously when the page has no <img> tags at all
# (true for most of these CSS/icon-driven generated sites), so in practice it almost
# never blocks a pass on its own. Making them required is what actually closes the gap
# between "the judge prompt says it checks accessibility" and "something verifies that".
REQUIRED_CHECK_NAMES = frozenset(
    {
        "index_html_exists",
        "css_exists",
        "non_trivial_size",
        "has_title",
        "has_nav",
        "has_footer",
        "sections_present",
        "html_lang_present",
        "has_single_h1",
        "has_main",
        "images_have_alt",
    }
)

_H1_RE = re.compile(r"<h1[\s>]", re.IGNORECASE)
_MAIN_RE = re.compile(r"<main[\s>]", re.IGNORECASE)
_HTML_LANG_RE = re.compile(r"<html\b[^>]*\blang\s*=\s*[\"']([^\"']*)[\"']", re.IGNORECASE)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_IMG_TAG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_ALT_ATTR_RE = re.compile(r"\balt\s*=\s*[\"']([^\"']*)[\"']", re.IGNORECASE)


def build_evaluator_agent() -> Agent[None, JudgeEvaluation]:
    return Agent(output_type=JudgeEvaluation, system_prompt=EVALUATOR_SYSTEM_PROMPT, retries=1)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def _section_mentioned(section: str, lower_html: str) -> bool:
    words = [w for w in section.lower().replace("-", " ").replace("_", " ").split() if len(w) > 2]
    if not words:
        return section.lower() in lower_html
    return any(word in lower_html for word in words)


def _title_text(html: str) -> str:
    match = _TITLE_RE.search(html)
    return match.group(1).strip() if match else ""


def run_deterministic_checks(plan: WebsitePlan, files: dict[str, str]) -> list[DeterministicCheck]:
    """Pure Python, no LLM involved -- exists/size/structure checks the dashboard can show
    as ground truth alongside (and separate from) the LLM-as-Judge opinion below."""
    html = files.get("index.html", "")
    css = "\n".join(v for k, v in files.items() if k.endswith(".css"))
    lower_html = html.lower()
    title_text = _title_text(html)

    checks = [
        DeterministicCheck(
            name="index_html_exists",
            passed=bool(html.strip()),
            detail="index.html present" if html.strip() else "index.html is missing or empty",
        ),
        DeterministicCheck(
            name="css_exists",
            passed=bool(css.strip()),
            detail="a .css file is present" if css.strip() else "no .css file was written",
        ),
        DeterministicCheck(
            name="non_trivial_size",
            passed=len(html) >= 800,
            detail=f"index.html is {len(html)} bytes",
        ),
        DeterministicCheck(
            name="has_title",
            passed=bool(title_text),
            detail=f"<title>{title_text}</title>" if title_text else "missing or empty <title>",
        ),
        DeterministicCheck(
            name="has_nav",
            passed="<nav" in lower_html,
            detail="has <nav>" if "<nav" in lower_html else "missing <nav>",
        ),
        DeterministicCheck(
            name="has_footer",
            passed="<footer" in lower_html,
            detail="has <footer>" if "<footer" in lower_html else "missing <footer>",
        ),
    ]

    # -- accessibility basics: deterministic, objective, backing the judge's claim ------
    lang_match = _HTML_LANG_RE.search(html)
    lang_value = lang_match.group(1).strip() if lang_match else ""
    checks.append(
        DeterministicCheck(
            name="html_lang_present",
            passed=bool(lang_value),
            detail=f'<html lang="{lang_value}">' if lang_value else 'missing <html lang="...">',
        )
    )

    h1_count = len(_H1_RE.findall(html))
    checks.append(
        DeterministicCheck(
            name="has_single_h1",
            passed=h1_count == 1,
            detail=f"found {h1_count} <h1> element(s) (exactly 1 required)",
        )
    )

    has_main = bool(_MAIN_RE.search(html))
    checks.append(
        DeterministicCheck(
            name="has_main",
            passed=has_main,
            detail="has <main>" if has_main else "missing <main> landmark",
        )
    )

    img_tags = _IMG_TAG_RE.findall(html)
    if not img_tags:
        checks.append(
            DeterministicCheck(
                name="images_have_alt",
                passed=True,
                detail="no <img> tags found -- vacuous pass, nothing to check",
            )
        )
    else:
        missing = sum(
            1
            for tag in img_tags
            if not (m := _ALT_ATTR_RE.search(tag)) or not m.group(1).strip()
        )
        checks.append(
            DeterministicCheck(
                name="images_have_alt",
                passed=missing == 0,
                detail=(
                    f"all {len(img_tags)} <img> tag(s) have non-empty alt text"
                    if missing == 0
                    else f"{missing}/{len(img_tags)} <img> tag(s) missing non-empty alt text"
                ),
            )
        )

    covered = [s for s in plan.sections if _section_mentioned(s, lower_html)]
    ratio = len(covered) / max(len(plan.sections), 1)
    checks.append(
        DeterministicCheck(
            name="sections_present",
            passed=ratio >= 0.6,
            detail=f"{len(covered)}/{len(plan.sections)} planned sections found in HTML: "
            f"{covered or 'none'}",
        )
    )
    return checks


def _judge_prompt(plan: WebsitePlan, files: dict[str, str]) -> str:
    html = _truncate(files.get("index.html", "(missing)"), MAX_HTML_CHARS)
    css = _truncate(
        "\n".join(v for k, v in files.items() if k.endswith(".css")) or "(missing)",
        MAX_CSS_CHARS,
    )
    return (
        f"Plan: {plan.name} for {plan.target_audience}. "
        f"Required sections: {', '.join(plan.sections)}. Visual style: {plan.visual_style}.\n\n"
        f"--- index.html ---\n{html}\n\n--- styles.css ---\n{css}"
    )


async def _run_judge_once(run_id: str, agent: Agent[None, JudgeEvaluation], prompt: str) -> JudgeEvaluation:
    """One independent LLM-as-Judge sample. Each call is a separate LLMCall recorded by
    run_agent (token/cost totals for the Evaluator span rise ~JUDGE_SAMPLES x -- that is
    expected and must not be suppressed or deduped)."""
    result = await run_agent(
        run_id=run_id,
        agent_name="Evaluator",
        agent=agent,
        user_prompt=prompt,
        model_settings={"max_tokens": AGENT_MAX_TOKENS["Evaluator"]},
    )
    return result.output


def _median_judge(judges: list[JudgeEvaluation]) -> tuple[JudgeEvaluation, float]:
    """Return (a representative judge, the median score). The reported SCORE is always
    the exact statistics.median() of the successful samples. With the default odd
    JUDGE_SAMPLES that median lands exactly on one sample's score, so its issues/
    suggestions are used verbatim -- fully coherent with the reported score. In the
    degraded fallback case (one of three samples failed, leaving an even count of 2) the
    median is the average of both and matches no single sample exactly; we then use the
    judge CLOSEST to that value for issues/suggestions as the best available
    representative, rather than merging or concatenating issue lists across runs."""
    median_score = statistics.median(j.score for j in judges)
    judge = min(judges, key=lambda j: abs(j.score - median_score))
    return judge, median_score


async def run_evaluator(run_id: str, plan: WebsitePlan, files: dict[str, str]) -> EvaluationResult:
    """Evaluator <- WebsitePlan + generated files only."""
    deterministic_checks = run_deterministic_checks(plan, files)
    deterministic_passed = all(
        c.passed for c in deterministic_checks if c.name in REQUIRED_CHECK_NAMES
    )

    agent = build_evaluator_agent()
    prompt = _judge_prompt(plan, files)

    # CONCURRENT, not sequential -- sequential would triple Evaluator latency; concurrent
    # costs roughly the slowest of the JUDGE_SAMPLES calls. return_exceptions=True so one
    # 429/error doesn't take down the whole span: we take the median of whatever succeeds
    # and only fail if every sample fails.
    outcomes = await asyncio.gather(
        *(_run_judge_once(run_id, agent, prompt) for _ in range(JUDGE_SAMPLES)),
        return_exceptions=True,
    )
    judges = [o for o in outcomes if isinstance(o, JudgeEvaluation)]
    if not judges:
        errors = [str(o) for o in outcomes if isinstance(o, BaseException)]
        raise RuntimeError(f"All {JUDGE_SAMPLES} LLM-judge samples failed: {errors}")

    judge, median_score = _median_judge(judges)

    failed = len(outcomes) - len(judges)
    sample_note = f" ({failed} of {JUDGE_SAMPLES} samples failed)" if failed else ""
    STORE.event(
        run_id,
        f"LLM judge sampled {len(judges)}x{sample_note}: "
        f"{[round(j.score, 1) for j in judges]} -> median {round(median_score, 1)}",
    )

    passed = deterministic_passed and median_score >= 7
    issues = list(judge.issues) + [c.detail for c in deterministic_checks if not c.passed]

    return EvaluationResult(
        score=median_score,
        passed=passed,
        issues=issues,
        suggestions=list(judge.suggestions),
        deterministic_checks=deterministic_checks,
        judge=judge,
    )
