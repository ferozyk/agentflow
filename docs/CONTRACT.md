# Agent Flow — Shared Build Contract (AUTHORITATIVE)

Every subagent codes against THIS file. Do not rename/move anything defined here.
`docs/ORIGINAL_SPEC.md` = product requirements. This file = the interfaces. On conflict, this file wins on *interfaces*, docs/ORIGINAL_SPEC.md wins on *requirements*.

## 0. Verified environment facts (measured 2026-08-31 — do NOT re-litigate or assume)

- `openrouter/free` IS a real zero-priced model id, but is EMPIRICALLY UNRELIABLE:
  it routed to `nvidia/nemotron-3.5-content-safety:free` (a classifier) and to a 2.6B model,
  and failed to emit a tool call in 1 of 2 trials. It is supported via config but is NOT the default.
- Measured tool-calling reliability on free models (2 trials each):
  - `dots-studio/dots-3-note-preview:free` 2/2, ~5-7s, 512k ctx  <- DEFAULT
  - `cohere/north-mini-code:free`          2/2, ~2-4s, code-tuned <- DEVELOPER AGENT
  - `minimax/minimax-m2.7:free`            2/2, ~10s              <- failover
  - `nvidia/nemotron-3.5-lightning:free`   ok                     <- failover
  - `google/gemma-4-31b-it:free`, `z-ai/glm-5.2:free` -> HTTP 429 (rate limited)
  - `thinkingmachines/inkling:free` -> HTTP 403
- OpenRouter `usage` returns REAL cache telemetry:
  `prompt_tokens_details.cached_tokens`, `.cache_write_tokens`, and `cost: 0`.
  Report these verbatim. Never fabricate cache numbers.
- The response body's `model` field reports the ACTUAL model that served the request
  (differs from requested when using a router). Record BOTH.
- `.env` currently has ONLY `OPENROUTER_API_KEY`. Slack tokens may be absent -> app MUST still run.
- System `python3` is 3.9.6 from an unrelated project venv. DO NOT USE IT. Use `uv` + Python 3.12.

## 1. Toolchain & commands

Python 3.12 via `uv`. Project root = repo root. Package = `agentflow/`.

    uv venv --python 3.12
    uv sync                       # or: uv pip install -e ".[dev]"
    uv run agentflow doctor       # prints provider/model/pricing + verifies free
    uv run agentflow serve        # dashboard + slack(if configured), ONE process
    uv run agentflow build "..."  # CLI trigger, no Slack needed
    uv run pytest -q
    uv run pytest tests/test_guardrails.py::test_path_escape -q

Deps: `pydantic-ai`, `fastapi`, `uvicorn[standard]`, `httpx`, `python-dotenv`,
`slack-bolt`, `aiohttp`, `jinja2`. Dev: `pytest`, `pytest-asyncio`, `anyio`.

## 2. SINGLE PROCESS RULE (critical)

The trace store is IN-MEMORY. Slack handler, orchestrator, and dashboard MUST run in the
same process and same asyncio event loop. `agentflow serve` starts uvicorn; the Slack
Socket-Mode client starts inside the FastAPI lifespan as a background task.
Never spawn a second process or the dashboard will show no runs.

## 3. Layout & FILE OWNERSHIP (do not edit files you do not own)

    agentflow/
      __init__.py
      config.py          [FOUNDATION]  Settings, free-model enforcement
      models.py          [FOUNDATION]  all Pydantic agent contracts
      llm.py             [FOUNDATION]  PydanticAI model factory + failover + usage capture
      observability/
        __init__.py
        tracing.py       [FOUNDATION]  TraceStore, RunTrace, AgentSpan, events
      guardrails/
        __init__.py
        input_guard.py   [FOUNDATION]
        filesystem.py    [FOUNDATION]  sandboxed FS tool
      agents/            [AGENTS]      planner/designer/content/developer/evaluator
      orchestrator.py    [AGENTS]      workflow + retry
      slackapp.py        [SLACK]
      dashboard/         [DASHBOARD]   app.py, templates/, static/
      cli.py             [FOUNDATION]  typer/argparse entrypoint (serve|build|doctor)
    workspace/generated-site/<run_id>/
    knowledge/
    tests/               each subagent adds tests/test_<their_area>.py only
    docs/CONTRACT.md

## 4. Config (`agentflow/config.py`)

Env vars:
- `OPENROUTER_API_KEY` (required)
- `OPENROUTER_MODEL` default `dots-studio/dots-3-note-preview:free`
- `OPENROUTER_MODEL_DEVELOPER` default `cohere/north-mini-code:free`
- `OPENROUTER_FALLBACK_MODELS` csv, default `minimax/minimax-m2.7:free,nvidia/nemotron-3.5-lightning:free`
- `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN` (both optional; Slack disabled if either missing)
- `DASHBOARD_HOST`=127.0.0.1, `DASHBOARD_PORT`=8000, `PUBLIC_BASE_URL` default http://127.0.0.1:8000
- `AGENTFLOW_ENABLE_RAG`=false

FREE ENFORCEMENT (`verify_models_are_free()`), run at startup and in `doctor`:
1. GET https://openrouter.ai/api/v1/models (cache 24h to `.cache/openrouter_models.json`).
2. For EVERY configured model id (primary, developer, fallbacks): it must exist AND
   `pricing.prompt == 0` AND `pricing.completion == 0` (parse as float).
3. On any violation raise `PaidModelError` with a clear message and EXIT NON-ZERO.
4. Never substitute a model to "fix" a failure. Free->free failover at REQUEST time is allowed
   (429/5xx/timeout only) and every failover must be recorded as a trace event.
5. If the catalog is unreachable: FAIL, unless `AGENTFLOW_ALLOW_OFFLINE_VERIFY=true`, in which
   case allow ids matching `:free$` or `openrouter/free` and set
   `pricing_verification="offline-heuristic"` which the dashboard MUST display as such.

Cost is COMPUTED from catalog pricing (`tokens * price`), never hardcoded. For free models this
yields exactly 0.0. Expose `pricing_status: "FREE" | "OFFLINE-HEURISTIC"`.

## 5. Agent data contracts (`agentflow/models.py`) — exact field names

    AgentStatus = Literal["WAITING","RUNNING","COMPLETED","FAILED","BLOCKED"]
    RunStatus   = Literal["RUNNING","COMPLETED","FAILED","BLOCKED"]

    class WebsitePlan(BaseModel):
        name: str; description: str; target_audience: str
        sections: list[str]          # 4-6 items
        visual_style: str

    class DesignSpec(BaseModel):
        layout: str; visual_style: str; typography: str
        color_direction: str; components: list[str]

    class SectionContent(BaseModel):
        section: str; heading: str; body: str; cta: str | None = None

    class WebsiteContent(BaseModel):
        tagline: str; sections: list[SectionContent]; footer: str

    class DeterministicCheck(BaseModel):
        name: str; passed: bool; detail: str

    class JudgeEvaluation(BaseModel):
        score: float           # 0-10
        issues: list[str]; suggestions: list[str]

    class EvaluationResult(BaseModel):
        score: float; passed: bool                 # passed = score >= 7 AND all required checks pass
        issues: list[str]; suggestions: list[str]
        deterministic_checks: list[DeterministicCheck]
        judge: JudgeEvaluation | None = None

`EvaluationResult` MUST keep deterministic vs LLM-judge separable for the dashboard.

## 6. Observability contracts (`agentflow/observability/tracing.py`)

    class LLMCall(BaseModel):
        agent: str; provider: str = "OpenRouter"
        model_requested: str; model_served: str | None = None
        input_tokens: int = 0; output_tokens: int = 0; total_tokens: int = 0
        cached_tokens: int = 0; cache_write_tokens: int = 0
        latency_ms: float; cost_usd: float = 0.0
        attempt: int = 1; failover_from: str | None = None

    class AgentSpan(BaseModel):
        span_id: str; agent_name: str          # Planner|Designer|Content|Developer|Evaluator
        status: AgentStatus = "WAITING"
        start_time: datetime | None; end_time: datetime | None
        duration_ms: float | None
        llm_calls: list[LLMCall] = []
        tool_latency_ms: float = 0.0
        error: str | None = None
        # aggregates (computed properties): input_tokens/output_tokens/total_tokens/cost_usd

    class GuardrailEvent(BaseModel):
        timestamp: datetime
        kind: Literal["input","filesystem","output"]
        tool: str; target: str; reason: str; blocked: bool

    class TraceEvent(BaseModel):
        timestamp: datetime; message: str          # human timeline line

    class RunTrace(BaseModel):
        run_id: str; trace_id: str; request: str
        status: RunStatus = "RUNNING"
        created_at: datetime; ended_at: datetime | None = None
        duration_ms: float | None = None
        spans: list[AgentSpan]                     # ALWAYS pre-seeded with all 5, WAITING
        guardrail_events: list[GuardrailEvent] = []
        events: list[TraceEvent] = []
        evaluation: EvaluationResult | None = None
        retry_count: int = 0
        site_path: str | None = None; site_url: str | None = None
        provider: str = "OpenRouter"; model: str; pricing_status: str

`run_id` format `af-<5 hex>`. Spans pre-seeded so the dashboard renders WAITING states immediately.

`TraceStore` (module-level singleton `STORE`):
    start_run(request, run_id=None) -> RunTrace
    get(run_id) -> RunTrace | None ; list_runs() -> list[RunTrace]  (newest first)
    span(run_id, agent_name) -> async context manager: sets RUNNING, times it,
        sets COMPLETED/FAILED, appends TraceEvents, records duration
    record_llm_call(run_id, agent, LLMCall) ; record_guardrail(run_id, GuardrailEvent)
    event(run_id, message)
    subscribe(run_id) -> asyncio.Queue  # for dashboard SSE; publish on every mutation
    latency_percentiles() -> {"p50": float|None, "p90": float|None, "samples": int}
        -> returns None values when samples < 5; dashboard prints
           "Insufficient samples for P50/P90". NEVER interpolate/fabricate.

## 7. Orchestrator public API (`agentflow/orchestrator.py`)

    async def run_workflow(request: str, run_id: str | None = None) -> RunTrace
    def start_workflow_background(request: str) -> str   # returns run_id immediately

Flow: input guardrail -> Planner -> asyncio.gather(Designer, Content) -> Developer ->
Evaluator -> if not passed and retry_count == 0: Developer(fix) -> Evaluator. MAX 1 RETRY.
If the input guardrail blocks: run status BLOCKED, no LLM calls, guardrail event recorded.

Context minimization is MANDATORY and must be visible in code:
Planner<-request | Designer<-plan | Content<-plan | Developer<-plan+design+content |
Evaluator<-plan + generated files. Never pass the whole RunTrace into a prompt.

## 8. Dashboard HTTP API (`agentflow/dashboard/app.py`) — Slack + CLI depend on these URLs

    GET  /                      -> dashboard (latest run, or run list)
    GET  /run/{run_id}          -> run detail page
    GET  /api/runs              -> [RunTrace...]
    GET  /api/run/{run_id}      -> RunTrace JSON
    GET  /api/run/{run_id}/events -> SSE stream of RunTrace snapshots
    POST /api/build  {"request": "..."} -> {"run_id": "af-xxxxx"}
    GET  /site/{run_id}/        -> serves workspace/generated-site/{run_id}/index.html
    GET  /api/health            -> {"status","provider","model","pricing_status"}

Canonical links: dashboard `{PUBLIC_BASE_URL}/run/{run_id}`, site `{PUBLIC_BASE_URL}/site/{run_id}/`.

## 9. Guardrails

INPUT (`input_guard.py`): `check_input(text) -> GuardrailEvent | None`. Deterministic regex/keyword
detection of prompt injection & secret exfiltration (e.g. "ignore previous instructions",
".env", "credentials", "api key", "system prompt", "../", "rm -rf", "password").
Blocked -> run BLOCKED, Slack + dashboard show the reason. No LLM call is made.

FILESYSTEM (`filesystem.py`): the ONLY way the Developer agent touches disk.
    class SiteWorkspace:
        def __init__(self, root: Path)              # workspace/generated-site/<run_id>
        def write_file(self, rel_path: str, content: str) -> str
        def read_file(self, rel_path: str) -> str
        def list_files(self) -> list[str]
Rules: resolve() the joined path and require it to be inside root (use `Path.resolve()` +
`is_relative_to`); reject absolute paths, `..`, symlinks, and any extension outside
{.html,.css,.js,.svg,.json,.txt,.md}; cap file size (256KB) and file count (20).
Every rejection records a `GuardrailEvent(kind="filesystem", blocked=True)` AND raises
`ModelRetry` so the LLM sees the refusal. No shell execution tool exists anywhere.

OUTPUT: Pydantic validation via PydanticAI. `retries=1` on every agent. On final validation
failure record `GuardrailEvent(kind="output")` and fail the span.

## 10. LLM layer (`agentflow/llm.py`)

- Build PydanticAI models against OpenRouter. VERIFY the installed pydantic-ai version's actual
  API before writing code (`uv run python -c "import pydantic_ai; print(pydantic_ai.__version__)"`
  and inspect available providers). Use OpenRouter's base_url `https://openrouter.ai/api/v1`.
  Do NOT guess import paths — check what exists.
- Send headers `HTTP-Referer` and `X-Title: Agent Flow` (OpenRouter attribution).
- After every agent run, extract usage and append an `LLMCall`. Capture `model_served` and cache
  token details from the raw response where pydantic-ai exposes it; if a field is unavailable,
  leave it 0 and let the dashboard say "not reported" — never invent it.
- Free->free failover ONLY on 429/5xx/timeout, max 2 hops through
  `OPENROUTER_FALLBACK_MODELS`, each hop recorded as a TraceEvent + `failover_from`.
- PROMPT CACHING: every agent's system prompt is a module-level CONSTANT (static) and all
  dynamic data goes in the user message. Dashboard shows real `cached_tokens` when > 0, else
  "Prompt caching: not reported by selected free model — prompt structure is cache-ready".

## 11. Output limits (token discipline)

REVISED 2026-08-31 after live measurement — the original budgets (500/500/1200/4000/700)
failed the MAJORITY of live runs. The default free models are REASONING models that burn
150-900+ hidden reasoning tokens before emitting structured JSON, so the budget must cover
reasoning overhead + the visible answer, or the call dies with
"Model token limit exceeded before any response was generated".

    Planner 4000, Designer 4000, Content 6000, Developer 16000, Evaluator 5000

RAISED AGAIN 2026-08-31 after a live Slack run failed with "Model token limit (1500)
exceeded before any response was generated". Measured over 6 real Planner runs, output
varied 372-860 tokens (266-407 of it reasoning) — 1500 failed INTERMITTENTLY, the worst
failure mode for a demo. `max_tokens` is a CEILING, not a reservation: you pay only for
tokens actually generated, so generous caps cost nothing. Verified end-to-end afterwards:
COMPLETED, 8.0/10, 11/11 deterministic checks, 5 calls, 70,033 tokens, $0.00, 114s.

Set via pydantic-ai model settings. Keep system prompts under ~120 words.
Token DISCIPLINE still comes primarily from context minimization (§7), not from starving
output budgets — capping output below the model's reasoning overhead does not save tokens,
it wastes an entire call and forces a retry, costing MORE.

## 12. Free-tier quota (operational constraint, measured 2026-08-31)

UPDATED 2026-08-31: the account now holds >=10 credits, so `is_free_tier: false` and the
cap is ~1000 free-model requests/DAY (was ~50). Verified after the change: dots-studio,
cohere/north-mini-code, nemotron-3-ultra and minimax all return 200 with `cost: 0`, and
account `usage` is 0 — the credit is UNSPENT balance that raises a rate limit; it is not
spent on inference. Every model remains zero-priced, so "FREE / $0.00" stays truthful and
the free-model-only guarantee is unchanged.
One workflow = 5-6 LLM calls => ~150+ runs/day. Rehearsal is no longer quota-constrained.
Consequences that the code MUST respect:
- CORRECTED 2026-08-31: the daily cap is ACCOUNT-WIDE, not per-model. OpenRouter returns
  `Rate limit exceeded: free-models-per-day`, so ALL free models share ONE daily pool.
  The multi-model fallback chain therefore buys resilience against a model being down /
  403 / 5xx / timing out — it does NOT buy extra quota. When the daily cap is hit, every
  free model fails together. Do not describe failover as a quota-multiplier anywhere.
- Concurrency is NOT rate-limited: 2 simultaneous free-model calls both succeeded (~1.5-1.9s),
  so the parallel Designer+Content step is safe.
- 403 (not entitled) is treated as failover-eligible, like 429/5xx.
- Never respond to a 429 by reaching for a paid model. Exhausted quota => fail clearly.
- Avoid gratuitous live end-to-end runs during development; they consume the demo budget.
