"""Centralized configuration + OpenRouter free-model enforcement.

Every LLM call in Agent Flow MUST go through an OpenRouter model that is verified
zero-priced against the live OpenRouter catalog. See docs/CONTRACT.md section 4.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = REPO_ROOT / ".cache" / "openrouter_models.json"
CACHE_TTL_SECONDS = 24 * 60 * 60
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

DEFAULT_MODEL = "dots-studio/dots-3-note-preview:free"
DEFAULT_MODEL_DEVELOPER = "cohere/north-mini-code:free"
# Widened 2026-08-31 (live-quota incident): this key is is_free_tier (50 free-model
# requests/day), and at the time both the primary and developer default models were
# returning HTTP 429. A 2-model fallback chain isn't enough resilience for a live demo —
# each of these was confirmed zero-priced in the live OpenRouter catalog. Order matters:
# minimax/minimax-m2.7:free was the only one still answering during the incident.
DEFAULT_FALLBACK_MODELS = (
    "minimax/minimax-m2.7:free,"
    "minimax/minimax-m3:free,"
    "nvidia/nemotron-3.5-lightning:free,"
    "google/gemma-4-31b-it:free,"
    "nvidia/nemotron-3-super-120b-a12b:free,"
    "z-ai/glm-5.2:free"
)


class ConfigError(Exception):
    """Raised for missing/invalid configuration (not a pricing violation)."""


class PaidModelError(Exception):
    """Raised when a configured model is missing from the catalog or is not free.

    Any code that catches this at the top level MUST exit the process non-zero —
    there is no safe way to continue once a paid model has been detected.
    """


class CatalogUnreachableError(Exception):
    """Internal: the OpenRouter catalog could not be fetched and no usable cache exists."""


@dataclass
class ModelPricing:
    prompt: float
    completion: float


@dataclass
class VerificationResult:
    pricing_status: str  # "FREE" | "OFFLINE-HEURISTIC"
    catalog: dict[str, ModelPricing] = field(default_factory=dict)
    checked_models: list[str] = field(default_factory=list)


@dataclass
class Settings:
    openrouter_api_key: str
    openrouter_model: str = DEFAULT_MODEL
    openrouter_model_developer: str = DEFAULT_MODEL_DEVELOPER
    openrouter_fallback_models: list[str] = field(default_factory=list)
    slack_bot_token: str | None = None
    slack_app_token: str | None = None
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8000
    public_base_url: str = "http://127.0.0.1:8000"
    agentflow_enable_rag: bool = False
    allow_offline_verify: bool = False
    pricing_status: str = "UNVERIFIED"

    @property
    def slack_enabled(self) -> bool:
        return bool(self.slack_bot_token and self.slack_app_token)

    @property
    def all_configured_models(self) -> list[str]:
        return [self.openrouter_model, self.openrouter_model_developer, *self.openrouter_fallback_models]


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _bool_env(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    """Read Settings from the environment (loads .env first, never overriding real env vars)."""
    load_dotenv(REPO_ROOT / ".env")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ConfigError("OPENROUTER_API_KEY is required — set it in .env")

    return Settings(
        openrouter_api_key=api_key,
        openrouter_model=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL),
        openrouter_model_developer=os.environ.get("OPENROUTER_MODEL_DEVELOPER", DEFAULT_MODEL_DEVELOPER),
        openrouter_fallback_models=_csv(os.environ.get("OPENROUTER_FALLBACK_MODELS", DEFAULT_FALLBACK_MODELS)),
        slack_bot_token=os.environ.get("SLACK_BOT_TOKEN") or None,
        slack_app_token=os.environ.get("SLACK_APP_TOKEN") or None,
        dashboard_host=os.environ.get("DASHBOARD_HOST", "127.0.0.1"),
        dashboard_port=int(os.environ.get("DASHBOARD_PORT", "8000")),
        public_base_url=os.environ.get("PUBLIC_BASE_URL", "http://127.0.0.1:8000"),
        agentflow_enable_rag=_bool_env(os.environ.get("AGENTFLOW_ENABLE_RAG", "false")),
        allow_offline_verify=_bool_env(os.environ.get("AGENTFLOW_ALLOW_OFFLINE_VERIFY", "false")),
    )


_settings_singleton: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    """Module-level Settings singleton. Pass reload=True to re-read the environment."""
    global _settings_singleton
    if _settings_singleton is None or reload:
        _settings_singleton = load_settings()
    return _settings_singleton


# ---------------------------------------------------------------------------
# OpenRouter model catalog (cached 24h) + free-pricing verification
# ---------------------------------------------------------------------------


def _load_cache() -> dict | None:
    if not CACHE_PATH.exists():
        return None
    try:
        return json.loads(CACHE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _cache_is_fresh(cache: dict) -> bool:
    fetched_at = cache.get("fetched_at", 0)
    return (time.time() - fetched_at) < CACHE_TTL_SECONDS


def _write_cache(models: list[dict]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps({"fetched_at": time.time(), "models": models}))


def _fetch_catalog_live() -> list[dict]:
    resp = httpx.get(OPENROUTER_MODELS_URL, timeout=10.0)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("data", [])


def fetch_model_catalog(force_refresh: bool = False) -> list[dict]:
    """Return the OpenRouter model list, using a 24h on-disk cache under .cache/.

    Raises CatalogUnreachableError if the network fetch fails and no cache
    (fresh or stale) is available to fall back on.
    """
    cache = _load_cache()
    if not force_refresh and cache is not None and _cache_is_fresh(cache):
        return cache["models"]

    try:
        models = _fetch_catalog_live()
    except (httpx.HTTPError, ValueError) as exc:
        if cache is not None:
            # Network is down but we have a (possibly stale) cache — better than failing
            # a demo outright. Freshness is best-effort per the 24h TTL.
            return cache["models"]
        raise CatalogUnreachableError(str(exc)) from exc

    _write_cache(models)
    return models


def verify_models_are_free(settings: Settings | None = None) -> VerificationResult:
    """Verify EVERY configured model (primary, developer, fallbacks) is zero-priced.

    On any violation raises PaidModelError. Callers at process boundaries (cli.py)
    must catch this, print it, and exit non-zero. Never substitutes a different model.
    """
    settings = settings or get_settings()
    model_ids = settings.all_configured_models

    try:
        raw_models = fetch_model_catalog()
    except CatalogUnreachableError as exc:
        if settings.allow_offline_verify:
            non_free_looking = [
                m for m in model_ids if not (m.endswith(":free") or m == "openrouter/free")
            ]
            if non_free_looking:
                raise PaidModelError(
                    "OpenRouter catalog unreachable and AGENTFLOW_ALLOW_OFFLINE_VERIFY heuristic "
                    f"rejects these model id(s) (no ':free' suffix): {non_free_looking}"
                ) from exc
            result = VerificationResult(pricing_status="OFFLINE-HEURISTIC", checked_models=model_ids)
            settings.pricing_status = result.pricing_status
            return result
        raise PaidModelError(
            f"Could not reach OpenRouter model catalog ({OPENROUTER_MODELS_URL}) and "
            f"AGENTFLOW_ALLOW_OFFLINE_VERIFY is not set: {exc}"
        ) from exc

    by_id = {m["id"]: m for m in raw_models if isinstance(m, dict) and "id" in m}
    violations: list[str] = []
    catalog: dict[str, ModelPricing] = {}

    for model_id in model_ids:
        entry = by_id.get(model_id)
        if entry is None:
            violations.append(f"{model_id!r}: not found in OpenRouter catalog")
            continue
        pricing = entry.get("pricing") or {}
        try:
            prompt_price = float(pricing.get("prompt", "0"))
            completion_price = float(pricing.get("completion", "0"))
        except (TypeError, ValueError):
            violations.append(f"{model_id!r}: unparseable pricing {pricing!r}")
            continue
        if prompt_price != 0.0 or completion_price != 0.0:
            violations.append(
                f"{model_id!r}: NOT FREE (prompt={prompt_price}, completion={completion_price})"
            )
            continue
        catalog[model_id] = ModelPricing(prompt=prompt_price, completion=completion_price)

    if violations:
        raise PaidModelError(
            "Free-model verification FAILED — refusing to start:\n  - " + "\n  - ".join(violations)
        )

    result = VerificationResult(pricing_status="FREE", catalog=catalog, checked_models=model_ids)
    settings.pricing_status = result.pricing_status
    return result


def compute_cost(model_id: str, input_tokens: int, output_tokens: int, catalog: dict[str, ModelPricing]) -> float:
    """Cost is ALWAYS computed from catalog pricing — never hardcoded.

    For a verified free model this is exactly 0.0 because both prices are 0.0.
    An unknown model (not in catalog, e.g. offline-heuristic mode) costs 0.0 —
    we never invent a price for a model we couldn't verify.
    """
    pricing = catalog.get(model_id)
    if pricing is None:
        return 0.0
    return input_tokens * pricing.prompt + output_tokens * pricing.completion
