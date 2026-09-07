"""Sandboxed filesystem access — the ONLY way the Developer agent touches disk.

Security model (docs/CONTRACT.md section 9): every path is resolved and must remain
inside the workspace root. Absolute paths, `..` traversal, symlinks (at any path
component, regardless of where they ultimately point), and disallowed extensions are all
rejected. Every rejection records a GuardrailEvent AND raises `pydantic_ai.ModelRetry` so
the calling LLM sees the refusal as a tool error and can try again (bounded by the
agent's own `retries=1`). No shell execution tool exists anywhere in this codebase.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic_ai import ModelRetry

from ..observability.tracing import STORE, GuardrailEvent

ALLOWED_EXTENSIONS = {".html", ".css", ".js", ".svg", ".json", ".txt", ".md"}
MAX_FILE_SIZE_BYTES = 256 * 1024
MAX_FILE_COUNT = 20


class SiteWorkspace:
    """Sandboxed access to `workspace/generated-site/<run_id>`.

    `root` IS `workspace/generated-site/<run_id>` per the contract — we derive the run_id
    from `root.name` purely to opportunistically record GuardrailEvents into the shared
    TraceStore; this is best-effort and never required for correctness (if the run_id
    isn't a known run, STORE.record_guardrail is a silent no-op).
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._run_id = self.root.name

    # -- guardrail plumbing --------------------------------------------------

    def _record_and_raise(self, tool: str, target: str, reason: str) -> None:
        event = GuardrailEvent(
            timestamp=datetime.now(timezone.utc),
            kind="filesystem",
            tool=tool,
            target=target,
            reason=reason,
            blocked=True,
        )
        STORE.record_guardrail(self._run_id, event)
        raise ModelRetry(f"Filesystem guardrail blocked {tool} on {target!r}: {reason}")

    def _has_symlink_component(self, candidate: Path) -> bool:
        """True if `candidate` or any path component under root is (already) a symlink.

        Checked on the UNRESOLVED join (before following links) — Path.resolve() would
        already follow symlinks, which is exactly what containment-checking needs, but a
        symlink that happens to resolve back inside the sandbox is still rejected here as
        a defense-in-depth measure (its target could later be swapped, TOCTOU-style).
        """
        try:
            rel_parts = candidate.relative_to(self.root).parts
        except ValueError:
            return True
        walked = self.root
        for part in rel_parts:
            walked = walked / part
            if walked.is_symlink():
                return True
        return False

    def _resolve_safe(self, rel_path: str, tool: str) -> Path:
        if not rel_path or not rel_path.strip():
            self._record_and_raise(tool, rel_path, "Empty path")

        raw = Path(rel_path)
        if raw.is_absolute():
            self._record_and_raise(tool, rel_path, "Absolute paths are not allowed")
        if ".." in raw.parts:
            self._record_and_raise(tool, rel_path, "Path traversal ('..') is not allowed")

        candidate = self.root / raw
        if self._has_symlink_component(candidate):
            self._record_and_raise(tool, rel_path, "Symlinks are not allowed in the workspace")

        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.root):
            self._record_and_raise(tool, rel_path, "Path escapes the allowed workspace")

        if resolved.suffix.lower() not in ALLOWED_EXTENSIONS:
            self._record_and_raise(
                tool, rel_path, f"File extension {resolved.suffix!r} is not allowed"
            )

        return resolved

    # -- public API (per docs/CONTRACT.md section 9) -------------------------

    def write_file(self, rel_path: str, content: str) -> str:
        resolved = self._resolve_safe(rel_path, "filesystem.write")

        data = content.encode("utf-8")
        if len(data) > MAX_FILE_SIZE_BYTES:
            self._record_and_raise(
                "filesystem.write", rel_path, f"File exceeds {MAX_FILE_SIZE_BYTES} byte limit"
            )

        rel_norm = resolved.relative_to(self.root).as_posix()
        existing = set(self.list_files())
        if rel_norm not in existing and len(existing) >= MAX_FILE_COUNT:
            self._record_and_raise(
                "filesystem.write", rel_path, f"Workspace file count limit ({MAX_FILE_COUNT}) reached"
            )

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return rel_norm

    def read_file(self, rel_path: str) -> str:
        resolved = self._resolve_safe(rel_path, "filesystem.read")
        if not resolved.exists():
            self._record_and_raise("filesystem.read", rel_path, "File does not exist")
        return resolved.read_text(encoding="utf-8")

    def list_files(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()
        )
