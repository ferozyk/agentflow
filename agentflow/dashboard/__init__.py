"""Agent Flow dashboard package (FastAPI app + templates + static assets).

See docs/CONTRACT.md section 8 for the exact HTTP surface this exposes.
"""
from .app import app, create_app

__all__ = ["app", "create_app"]
