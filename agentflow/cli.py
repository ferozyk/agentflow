"""Agent Flow CLI entrypoint: `agentflow doctor|build|serve`.

Uses argparse rather than a third-party CLI framework — see docs/CONTRACT.md section 1,
which lists the project's dependencies and does not include typer/click.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from .config import ConfigError, PaidModelError, Settings, get_settings, verify_models_are_free


def _print_provider_banner(settings: Settings) -> None:
    print("Agent Flow")
    print(f"Provider: OpenRouter")
    print(f"Model:    {settings.openrouter_model}")


def _load_and_verify() -> tuple[Settings | None, int]:
    """Shared startup path for build/serve: load config, verify free models.

    Returns (settings, exit_code). exit_code is 0 on success; on failure settings is
    None and the caller should return exit_code immediately without proceeding.
    """
    try:
        settings = get_settings(reload=True)
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return None, 1

    try:
        verify_models_are_free(settings)
    except PaidModelError as exc:
        print("PRICING VERIFICATION FAILED — refusing to start.\n")
        print(str(exc))
        return None, 1

    return settings, 0


def cmd_doctor(args: argparse.Namespace) -> int:
    try:
        settings = get_settings(reload=True)
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 1

    _print_provider_banner(settings)
    print(f"Developer model: {settings.openrouter_model_developer}")
    print(f"Fallback models: {', '.join(settings.openrouter_fallback_models) or '(none)'}")
    print(f"Slack enabled:   {settings.slack_enabled}")
    print(f"RAG enabled:     {settings.agentflow_enable_rag}")
    print(f"Dashboard URL:   {settings.public_base_url}")
    print()

    try:
        result = verify_models_are_free(settings)
    except PaidModelError as exc:
        print("Pricing: FAILED")
        print(str(exc))
        return 1

    print(f"Pricing: {result.pricing_status}")
    for model_id in result.checked_models:
        pricing = result.catalog.get(model_id)
        if pricing is not None:
            print(f"  - {model_id}: FREE (prompt=$0/tok, completion=$0/tok)")
        else:
            print(f"  - {model_id}: accepted via offline heuristic (':free' suffix, catalog unreachable)")
    print("\nLLM Cost: $0.00 (every configured model verified free)")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    settings, code = _load_and_verify()
    if settings is None:
        return code

    try:
        from .orchestrator import run_workflow
    except ImportError as exc:
        print(
            "orchestrator not yet implemented — agentflow/orchestrator.py and agentflow/agents/ "
            "have not landed yet (this is expected until the AGENTS subagent finishes).\n"
            f"Import error: {exc}"
        )
        return 1

    async def _run() -> int:
        trace = await run_workflow(args.request)
        print(f"\nRun {trace.run_id}: {trace.status}")
        if trace.evaluation is not None:
            verdict = "PASS" if trace.evaluation.passed else "FAIL"
            print(f"Evaluation: {trace.evaluation.score}/10 ({verdict})")
        if trace.site_url:
            print(f"Website: {trace.site_url}")
        print(f"Dashboard: {settings.public_base_url}/run/{trace.run_id}")
        return 0 if trace.status == "COMPLETED" else 1

    return asyncio.run(_run())


def cmd_serve(args: argparse.Namespace) -> int:
    settings, code = _load_and_verify()
    if settings is None:
        return code

    try:
        from .dashboard.app import app
    except ImportError as exc:
        print(
            "dashboard not yet implemented — agentflow/dashboard/app.py has not landed yet "
            "(this is expected until the DASHBOARD subagent finishes).\n"
            f"Import error: {exc}"
        )
        return 1

    import logging

    import uvicorn

    # Surface agentflow's own INFO logs (notably "Slack Socket Mode connected") alongside
    # uvicorn's. Without this, uvicorn's logging config wins and agentflow.* records are
    # swallowed — during a demo that silence reads as "Slack is broken" when it is fine.
    logging.getLogger("agentflow").setLevel(logging.INFO)
    if not logging.getLogger("agentflow").handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     %(message)s"))
        logging.getLogger("agentflow").addHandler(handler)

    _print_provider_banner(settings)
    print(f"Pricing: {settings.pricing_status}")
    print(f"Serving on http://{settings.dashboard_host}:{settings.dashboard_port}")
    uvicorn.run(app, host=settings.dashboard_host, port=settings.dashboard_port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentflow", description="Agent Flow — AI website builder demo")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="Verify provider/model/pricing configuration")
    p_doctor.set_defaults(func=cmd_doctor)

    p_build = sub.add_parser("build", help="Run the website-builder workflow from the CLI (no Slack needed)")
    p_build.add_argument("request", help="Natural-language website request")
    p_build.set_defaults(func=cmd_build)

    p_serve = sub.add_parser("serve", help="Start the dashboard (+ Slack, if configured) in one process")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
