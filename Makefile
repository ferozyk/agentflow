## Agent Flow — convenience targets. Everything here just wraps `uv run ...`
## so it works the same whether or not you have `uv` on your PATH as an alias.

.PHONY: serve build test doctor tunnel

# Start the single-process app: FastAPI dashboard (uvicorn) + Slack Socket Mode
# (if SLACK_BOT_TOKEN/SLACK_APP_TOKEN are set) in the SAME event loop.
# See docs/CONTRACT.md section 2 — never run these as separate processes.
serve:
	uv run agentflow serve

# Trigger a build from the CLI without Slack, e.g.:
#   make build REQUEST="Build a landing page for Agent Flow, an AI engineering company"
build:
	uv run agentflow build "$(REQUEST)"

test:
	uv run pytest -q

# Verifies provider/model/pricing (free-model enforcement) without starting anything.
doctor:
	uv run agentflow doctor

# Expose the local dashboard (default http://localhost:8000) to the public internet
# via a free Cloudflare Quick Tunnel, so Slack's /build-website command (and anyone
# you share the link with) can reach it without deploying anywhere.
#
# `cloudflared` prints a randomly generated URL on stdout that looks like:
#     https://some-random-words.trycloudflare.com
#
# Copy that URL and set it as PUBLIC_BASE_URL in .env, e.g.:
#     PUBLIC_BASE_URL=https://some-random-words.trycloudflare.com
#
# Then restart `make serve` so every Slack message and dashboard link agentflow
# builds points at the public URL instead of http://127.0.0.1:8000. Nothing in
# this codebase hardcodes localhost — it always reads PUBLIC_BASE_URL from config.
#
# Requires cloudflared: `brew install cloudflared` (macOS) or see
# https://developers.cloudflare.com/cloudflared/downloads/
tunnel:
	cloudflared tunnel --url http://localhost:8000
