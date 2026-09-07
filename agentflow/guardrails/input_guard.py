"""Deterministic input guardrail: prompt-injection / secret-exfiltration detection.

Intentionally simple regex/keyword matching — the point (see docs/CONTRACT.md section 9
and docs/ORIGINAL_SPEC.md "Why the LLM is NOT the security boundary") is that this runs in code
BEFORE any LLM call, so a malicious request never reaches a model at all.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from ..observability.tracing import GuardrailEvent

# Bare nouns like "password"/"credentials"/"api key" are NOT inherently malicious — this
# app's entire job is building websites, and legitimate requests for identity/security/
# dev-tool companies mention those words constantly ("a landing page for a password
# manager startup"). So secret-exfiltration terms are gated behind an actual exfiltration
# INTENT verb nearby, rather than matched as bare nouns. `.env`, path traversal, `rm -rf`,
# and the prompt-injection phrasings stay bare — those have no legitimate reading.
_EXFIL_VERBS = r"(?:read|show|print|reveal|display|cat|dump|leak|send|output|expose|fetch|access|open)"
_SENSITIVE_TERMS = r"(?:\.env|credentials?|api[\s_-]?keys?|passwords?|secrets?|tokens?)"

# (reason label, compiled pattern). Order doesn't matter; first match wins.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("prompt injection", re.compile(r"ignore\s+(all|any|the)?\s*previous\s+instructions", re.I)),
    ("prompt injection", re.compile(r"disregard\s+(all|any|the)?\s*(previous|prior)\s+instructions", re.I)),
    ("prompt injection", re.compile(r"system\s+prompt", re.I)),
    ("prompt injection", re.compile(r"you\s+are\s+now\s+in\s+\w+\s+mode", re.I)),
    ("secret exfiltration", re.compile(r"\.env\b")),
    ("path traversal", re.compile(r"\.\./")),
    ("shell injection", re.compile(r"rm\s+-rf", re.I)),
    (
        "secret exfiltration",
        re.compile(rf"\b{_EXFIL_VERBS}\b[^.\n]{{0,40}}?\b{_SENSITIVE_TERMS}\b", re.I),
    ),
]


def check_input(text: str) -> GuardrailEvent | None:
    """Return a blocking GuardrailEvent if `text` looks malicious, else None.

    Deterministic and side-effect free (does not touch the TraceStore) — the caller
    (orchestrator) decides what to do with the event, e.g. STORE.record_guardrail(...).
    """
    for reason_label, pattern in _PATTERNS:
        match = pattern.search(text)
        if match:
            return GuardrailEvent(
                timestamp=datetime.now(timezone.utc),
                kind="input",
                tool="guardrails.input_guard.check_input",
                target=text[:300],
                reason=f"Detected possible {reason_label} (matched {match.group(0)!r})",
                blocked=True,
            )
    return None
