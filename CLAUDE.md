# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

This repository is currently empty of code — there is no implementation, no `pyproject.toml`, no README, and no commits on `main`. The only substantive file is `INSTRUCTIONS.md`, which is the full build spec for a project called **Agent Flow**. Treat `INSTRUCTIONS.md` as the authoritative source of requirements: read it in full before writing any code, and re-check it when priorities are unclear.

There are no build/lint/test commands yet because nothing has been scaffolded. Once the project is initialized (Python + `pyproject.toml`), add real commands here rather than guessing at conventions.

## What Agent Flow is

A ~3-hour interview demo (not a production system) that demonstrates agentic AI / multi-agent orchestration end-to-end: a Slack slash command triggers a background multi-agent workflow (via PydanticAI) that generates a static website, tracked with an observability dashboard (latency, tokens, tracing, guardrails, evaluation).

## The single hard constraint

**Every LLM call must use an OpenRouter FREE model** (default `OPENROUTER_MODEL=openrouter/free`, or an explicit `:free`-suffixed model). No OpenAI/Anthropic/Gemini/Groq/Together/Mistral or any paid OpenRouter model, ever. If the configured model isn't clearly free, **fail fast and loud** — never silently fall back to a paid model. Provider, model, and pricing status ($0.00 only when actually free) must be visible in Slack messages, the dashboard, and startup logs. Do not fabricate pricing, caching, or latency-percentile claims when data/support doesn't exist — display explicit "not available" messages instead (see INSTRUCTIONS.md's Prompt Caching and Latency sections for exact wording expectations).

## Intended architecture (per INSTRUCTIONS.md)

Suggested layout — simplify further if a smaller structure suffices:

```
agent-flow/
  agents/          planner.py, designer.py, content.py, developer.py, evaluator.py
  tools/           filesystem.py   (sandboxed FS access for the Developer agent)
  slack/           /build-website slash command handler
  dashboard/       lightweight web dashboard
  observability/   tracing.py      (in-memory trace/span/token/latency collection)
  workspace/generated-site/   ONLY place the Developer agent may write
  knowledge/optional/         RAG markdown docs (optional, P3)
  tests/
```

**Workflow (Pydantic models as the contract between every stage):**

```
Planner → (Designer + Content run concurrently) → Developer → Evaluator
                                                                  │
                                                        PASS → done
                                                        FAIL → Developer fixes ONCE → Evaluator (no further retries)
```

Each agent receives only the minimal input it needs (a key token-optimization technique called out in the spec) — not the entire workflow state:
- Planner ← raw Slack request text
- Designer ← WebsitePlan
- Content ← WebsitePlan
- Developer ← WebsitePlan + DesignSpec + WebsiteContent
- Evaluator ← requirements + generated website

**Guardrails are enforced in code, not by the LLM:**
- Input guardrail flags obviously malicious prompts (e.g. prompt-injection attempts).
- Filesystem guardrail: the Developer agent's tools must validate every path against `workspace/generated-site/` and reject anything outside it (including `.env`/credentials access) — implement this as a controlled tool, not raw filesystem access.
- Output guardrail: every structured agent output must pass Pydantic validation, with exactly one retry on failure (no unlimited retries).
- Blocked actions must surface in the dashboard.

**Evaluator** produces a typed result (score 0–10, pass at ≥7, issues, suggestions) and must clearly distinguish **deterministic checks** (files exist, required sections present) from **LLM-as-Judge** evaluation, both in code and in the dashboard UI.

**Observability**: in-memory tracing only (run_id/trace_id/span_id, per-agent start/end/duration/status/provider/model/token counts). Do not build distributed tracing. Show a chronological execution timeline. Only compute P50/P90 latency when enough historical runs exist; otherwise display "Insufficient samples for P50/P90" rather than fabricating numbers.

## Build priority (do not reorder)

P0 (must work end-to-end first) → P1 → P2 → P3 → P4 (do not build). See INSTRUCTIONS.md for the full breakdown. In short:
- **P0**: Slack → background workflow → Planner → Designer+Content → Developer → website → Evaluator → Slack result.
- **P1**: Dashboard, latency, token tracking, tracing, provider/model display.
- **P2**: Guardrails UI, evaluation UI.
- **P3 (nice to have, skip if it threatens P0/P1)**: RAG, prompt-caching demonstration, P50/P90.
- **P4 (explicitly out of scope)**: auth, persistence, distributed queues, Kubernetes, production deployment, real vector DB infra, advanced security systems.

Do not introduce Kafka, Redis, PostgreSQL, Kubernetes, microservices, or other heavyweight infrastructure — this runs on a laptop. Docker is optional and only worth adding if it genuinely simplifies setup.

## Working in this repo

- Before scaffolding, decide (or ask) whether to follow the suggested `agent-flow/` structure literally or flatten it — INSTRUCTIONS.md explicitly permits simplifying it.
- `.env` is gitignored; never read or expose its contents, and make sure the Developer agent's sandboxed filesystem tool cannot reach it.
- When in doubt about scope, default to the stated priority: **working demo > visual polish > observability > advanced features > production architecture**.
