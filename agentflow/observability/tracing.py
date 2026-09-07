"""In-memory tracing: RunTrace / AgentSpan / LLMCall + the TraceStore singleton.

Single-process, single-event-loop by design (see docs/CONTRACT.md section 2) — this is
intentionally NOT a distributed tracing system. The dashboard reads snapshots via
STORE.get()/list_runs() and streams live updates via STORE.subscribe() (SSE).
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import asyncio
from typing import AsyncIterator, Literal

from pydantic import BaseModel, computed_field

from ..models import AgentStatus, EvaluationResult, RunStatus

AGENT_NAMES: list[str] = ["Planner", "Designer", "Content", "Developer", "Evaluator"]


class LLMCall(BaseModel):
    agent: str
    provider: str = "OpenRouter"
    model_requested: str
    model_served: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: float
    cost_usd: float = 0.0
    attempt: int = 1
    failover_from: str | None = None


class AgentSpan(BaseModel):
    span_id: str
    agent_name: str  # Planner|Designer|Content|Developer|Evaluator
    status: AgentStatus = "WAITING"
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_ms: float | None = None
    llm_calls: list[LLMCall] = []
    tool_latency_ms: float = 0.0
    error: str | None = None

    @computed_field  # type: ignore[misc]
    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.llm_calls)

    @computed_field  # type: ignore[misc]
    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.llm_calls)

    @computed_field  # type: ignore[misc]
    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.llm_calls)

    @computed_field  # type: ignore[misc]
    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.llm_calls)


class GuardrailEvent(BaseModel):
    timestamp: datetime
    kind: Literal["input", "filesystem", "output"]
    tool: str
    target: str
    reason: str
    blocked: bool


class TraceEvent(BaseModel):
    timestamp: datetime
    message: str  # human timeline line


class RunTrace(BaseModel):
    run_id: str
    trace_id: str
    request: str
    status: RunStatus = "RUNNING"
    created_at: datetime
    ended_at: datetime | None = None
    duration_ms: float | None = None
    spans: list[AgentSpan]  # ALWAYS pre-seeded with all 5, WAITING
    guardrail_events: list[GuardrailEvent] = []
    events: list[TraceEvent] = []
    evaluation: EvaluationResult | None = None
    retry_count: int = 0
    site_path: str | None = None
    site_url: str | None = None
    provider: str = "OpenRouter"
    model: str
    pricing_status: str


def new_run_id() -> str:
    return f"af-{uuid.uuid4().hex[:5]}"


class TraceStore:
    """In-memory store of every RunTrace this process has started. Not persisted."""

    def __init__(self) -> None:
        self._runs: dict[str, RunTrace] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    # -- lifecycle -----------------------------------------------------

    def start_run(self, request: str, run_id: str | None = None) -> RunTrace:
        from ..config import get_settings  # local import: avoid import cycle at module load

        settings = get_settings()
        run_id = run_id or new_run_id()
        spans = [AgentSpan(span_id=f"{run_id}-{name.lower()}", agent_name=name) for name in AGENT_NAMES]
        trace = RunTrace(
            run_id=run_id,
            trace_id=uuid.uuid4().hex,
            request=request,
            created_at=datetime.now(timezone.utc),
            spans=spans,
            model=settings.openrouter_model,
            pricing_status=settings.pricing_status,
        )
        self._runs[run_id] = trace
        self._subscribers.setdefault(run_id, [])
        return trace

    def get(self, run_id: str) -> RunTrace | None:
        return self._runs.get(run_id)

    def list_runs(self) -> list[RunTrace]:
        return sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)

    # -- pub/sub (dashboard SSE) ----------------------------------------

    def subscribe(self, run_id: str) -> asyncio.Queue:
        """Return a queue that receives a RunTrace snapshot on every mutation.

        An initial snapshot (current state) is enqueued immediately so a new SSE
        client doesn't have to wait for the next mutation to render something.
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(run_id, []).append(queue)
        trace = self._runs.get(run_id)
        if trace is not None:
            queue.put_nowait(trace.model_copy(deep=True))
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        subs = self._subscribers.get(run_id)
        if subs and queue in subs:
            subs.remove(queue)

    def publish(self, run_id: str) -> None:
        """Push the current snapshot to subscribers. Call after any direct mutation."""
        trace = self._runs.get(run_id)
        if trace is None:
            return
        for queue in self._subscribers.get(run_id, []):
            queue.put_nowait(trace.model_copy(deep=True))

    # -- mutations --------------------------------------------------------

    def _find_span(self, trace: RunTrace, agent_name: str) -> AgentSpan | None:
        for span in trace.spans:
            if span.agent_name == agent_name:
                return span
        return None

    def event(self, run_id: str, message: str) -> None:
        trace = self._runs.get(run_id)
        if trace is None:
            return
        trace.events.append(TraceEvent(timestamp=datetime.now(timezone.utc), message=message))
        self.publish(run_id)

    def record_llm_call(self, run_id: str, agent: str, call: LLMCall) -> None:
        trace = self._runs.get(run_id)
        if trace is None:
            return
        span = self._find_span(trace, agent)
        if span is not None:
            span.llm_calls.append(call)
        self.publish(run_id)

    def record_guardrail(self, run_id: str, event_: GuardrailEvent) -> None:
        trace = self._runs.get(run_id)
        if trace is None:
            return
        trace.guardrail_events.append(event_)
        self.publish(run_id)

    @asynccontextmanager
    async def span(self, run_id: str, agent_name: str) -> AsyncIterator[AgentSpan]:
        """Async context manager: RUNNING -> COMPLETED/FAILED, timed, with TraceEvents."""
        trace = self._runs.get(run_id)
        if trace is None:
            raise KeyError(f"unknown run_id: {run_id!r}")
        span = self._find_span(trace, agent_name)
        if span is None:
            raise KeyError(f"unknown agent_name: {agent_name!r} for run {run_id!r}")

        span.status = "RUNNING"
        span.start_time = datetime.now(timezone.utc)
        self.event(run_id, f"{agent_name} started")

        started = time.monotonic()
        try:
            yield span
        except Exception as exc:  # noqa: BLE001 - span failure must be recorded, then re-raised
            span.status = "FAILED"
            span.error = str(exc)
            span.end_time = datetime.now(timezone.utc)
            span.duration_ms = (time.monotonic() - started) * 1000
            self.event(run_id, f"{agent_name} failed after {span.duration_ms / 1000:.1f}s — {exc}")
            raise
        else:
            span.status = "COMPLETED"
            span.end_time = datetime.now(timezone.utc)
            span.duration_ms = (time.monotonic() - started) * 1000
            self.event(run_id, f"{agent_name} completed — {span.duration_ms / 1000:.1f}s")

    # -- aggregate stats ----------------------------------------------------

    def latency_percentiles(self) -> dict[str, float | int | None]:
        """P50/P90 of completed run durations (duration_ms). NEVER fabricated/interpolated
        beyond simple nearest-rank percentile — returns None when fewer than 5 samples exist.
        """
        samples = sorted(
            trace.duration_ms for trace in self._runs.values() if trace.duration_ms is not None
        )
        n = len(samples)
        if n < 5:
            return {"p50": None, "p90": None, "samples": n}

        def nearest_rank(pct: float) -> float:
            idx = min(n - 1, max(0, round(pct / 100 * (n - 1))))
            return samples[idx]

        return {"p50": nearest_rank(50), "p90": nearest_rank(90), "samples": n}


STORE = TraceStore()
