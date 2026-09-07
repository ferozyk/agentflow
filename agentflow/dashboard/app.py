"""Agent Flow dashboard — FastAPI app.

Exposes exactly the HTTP surface defined in docs/CONTRACT.md section 8. Renders
whatever is in the (foundation-owned) in-memory TraceStore; never invents numbers.

This module intentionally has NO hard import-time dependency on agentflow.slackapp,
agentflow.orchestrator, or agentflow.agents.* — those are owned by other subagents and
may not exist yet while this file is developed. Both are imported lazily (inside the
lifespan / inside the endpoint that needs them) and guarded with try/except ImportError
so `uv run agentflow serve` (or `uvicorn agentflow.dashboard.app:app`) still boots and
renders a usable (if inert) dashboard even before those layers land.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from agentflow.config import get_settings
from agentflow.observability.tracing import STORE, RunTrace

logger = logging.getLogger("agentflow.dashboard")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GENERATED_SITE_ROOT = REPO_ROOT / "workspace" / "generated-site"
DASHBOARD_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = DASHBOARD_DIR / "templates"
STATIC_DIR = DASHBOARD_DIR / "static"

GENERATED_SITE_ROOT.mkdir(parents=True, exist_ok=True)

SSE_HEARTBEAT_SECONDS = 15

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class BuildRequest(BaseModel):
    request: str


def _run_to_json(trace: RunTrace) -> dict:
    """Serialize a RunTrace to plain JSON-safe dict (datetimes -> ISO strings)."""
    return json.loads(trace.model_dump_json())


def _latency_snapshot() -> dict:
    """STORE.latency_percentiles() as a JSON-safe dict for template bootstrap."""
    return STORE.latency_percentiles()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # SINGLE PROCESS RULE (CONTRACT.md section 2): Slack runs in this same process /
    # event loop, started as a background task from the FastAPI lifespan. slackapp.py
    # is owned by another subagent and may not exist yet, or Slack tokens may be
    # unconfigured — either way the dashboard must still come up.
    try:
        from agentflow.slackapp import start_slack_if_configured

        await start_slack_if_configured()
    except ImportError:
        logger.info("agentflow.slackapp not available yet — Slack integration disabled")
    except Exception:  # noqa: BLE001 - Slack must never take the dashboard down with it
        logger.exception("start_slack_if_configured() raised — continuing without Slack")

    try:
        yield
    finally:
        try:
            from agentflow.slackapp import stop_slack

            await stop_slack()
        except ImportError:
            pass
        except Exception:  # noqa: BLE001
            logger.exception("stop_slack() raised during shutdown")


def create_app() -> FastAPI:
    app = FastAPI(title="Agent Flow Dashboard", lifespan=lifespan)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # /site/{run_id}/ -> workspace/generated-site/{run_id}/index.html, plus any other
    # generated asset (css/js/svg) referenced by that page, via StaticFiles(html=True).
    app.mount(
        "/site",
        StaticFiles(directory=str(GENERATED_SITE_ROOT), html=True, check_dir=False),
        name="site",
    )

    def _page_context(run_id: str | None) -> dict:
        runs = STORE.list_runs()
        settings = get_settings()
        return {
            "initial_run_id": run_id,
            "runs": [{"run_id": r.run_id, "status": r.status, "request": r.request} for r in runs],
            "latency_json": json.dumps(_latency_snapshot()),
            "public_base_url": settings.public_base_url,
        }

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        runs = STORE.list_runs()
        latest_id = runs[0].run_id if runs else None
        ctx = _page_context(latest_id)
        return templates.TemplateResponse(request, "dashboard.html", ctx)

    @app.get("/run/{run_id}", response_class=HTMLResponse)
    async def run_detail(request: Request, run_id: str) -> HTMLResponse:
        ctx = _page_context(run_id)
        return templates.TemplateResponse(request, "dashboard.html", ctx)

    @app.get("/api/runs")
    async def api_runs() -> list[dict]:
        return [_run_to_json(r) for r in STORE.list_runs()]

    @app.get("/api/run/{run_id}")
    async def api_run(run_id: str) -> dict:
        trace = STORE.get(run_id)
        if trace is None:
            raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
        return _run_to_json(trace)

    @app.get("/api/run/{run_id}/events")
    async def api_run_events(run_id: str) -> StreamingResponse:
        if STORE.get(run_id) is None:
            raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")

        async def event_stream() -> AsyncIterator[str]:
            queue = STORE.subscribe(run_id)
            try:
                while True:
                    try:
                        trace: RunTrace = await asyncio.wait_for(queue.get(), timeout=SSE_HEARTBEAT_SECONDS)
                    except asyncio.TimeoutError:
                        # Keep the connection alive through idle periods / proxies
                        # (e.g. a cloudflared tunnel) — not a data frame, just a comment.
                        yield ": keepalive\n\n"
                        continue
                    yield f"data: {trace.model_dump_json()}\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                STORE.unsubscribe(run_id, queue)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/build")
    async def api_build(payload: BuildRequest) -> dict:
        text = payload.request.strip()
        if not text:
            raise HTTPException(status_code=400, detail="request must not be empty")
        try:
            from agentflow.orchestrator import start_workflow_background
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail="orchestrator not available yet — try again once agentflow.orchestrator is deployed",
            ) from exc
        run_id = start_workflow_background(text)
        return {"run_id": run_id}

    @app.get("/api/health")
    async def api_health() -> dict:
        settings = get_settings()
        return {
            "status": "ok",
            "provider": "OpenRouter",
            "model": settings.openrouter_model,
            "pricing_status": settings.pricing_status,
        }

    return app


app = create_app()
