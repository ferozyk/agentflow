// Agent Flow dashboard — pure client-side renderer over RunTrace JSON.
// Every number shown here comes straight from the RunTrace payload served by
// /api/run/{run_id} or pushed over /api/run/{run_id}/events (SSE). Nothing is
// invented client-side; unavailable values render literal "not reported" text.
(function () {
  "use strict";

  const bootstrap = JSON.parse(document.getElementById("bootstrap-data").textContent);
  let currentRunId = bootstrap.initial_run_id;
  const latencyBootstrap = bootstrap.latency || { p50: null, p90: null, samples: 0 };

  let eventSource = null;
  let pollTimer = null;

  const AGENT_ORDER = ["Planner", "Designer", "Content", "Developer", "Evaluator"];
  const STATUS_BADGE_CLASS = {
    WAITING: "badge-waiting",
    RUNNING: "badge-running",
    COMPLETED: "badge-completed",
    FAILED: "badge-failed",
    BLOCKED: "badge-blocked",
  };

  // ---------- formatting helpers ----------

  function fmtNum(n) {
    if (n === null || n === undefined) return "not reported";
    return Number(n).toLocaleString();
  }

  function fmtMs(ms) {
    if (ms === null || ms === undefined) return "not reported";
    if (ms < 1000) return `${Math.round(ms)} ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  }

  function fmtCost(usd) {
    if (usd === null || usd === undefined) return "not reported";
    return `$${Number(usd).toFixed(4)}`;
  }

  function fmtClock(iso) {
    if (!iso) return "--:--:--";
    const d = new Date(iso);
    return d.toLocaleTimeString("en-GB", { hour12: false });
  }

  function el(tag, opts) {
    const node = document.createElement(tag);
    opts = opts || {};
    if (opts.class) node.className = opts.class;
    if (opts.text !== undefined) node.textContent = opts.text;
    if (opts.html !== undefined) node.innerHTML = opts.html;
    if (opts.attrs) {
      for (const [k, v] of Object.entries(opts.attrs)) node.setAttribute(k, v);
    }
    return node;
  }

  function sumSpans(spans, field) {
    return spans.reduce((acc, s) => acc + (s[field] || 0), 0);
  }

  function totalLLMCalls(spans) {
    return spans.reduce((acc, s) => acc + (s.llm_calls ? s.llm_calls.length : 0), 0);
  }

  // Walk calls from the most recent backwards and return the first model id any
  // of them actually reported (model_served, falling back to model_requested for
  // that same call). A single call missing model_served (provider didn't report
  // it) must not blank out an otherwise-known model from an earlier call.
  function lastKnownModel(calls) {
    for (let i = calls.length - 1; i >= 0; i--) {
      const label = calls[i].model_served || calls[i].model_requested;
      if (label) return label;
    }
    return null;
  }

  // A "Label: value" row where the label never wraps/shrinks (it's always
  // short) and only the value is allowed to grow, shrink, and wrap.
  function metaRow(label, valueNode, opts) {
    opts = opts || {};
    const row = el("div", { class: "agent-meta" + (opts.class ? " " + opts.class : "") });
    row.appendChild(el("span", { class: "meta-label", text: label + ": " }));
    row.appendChild(valueNode);
    return row;
  }

  function truncatedSpan(text, opts) {
    opts = opts || {};
    return el("span", {
      class: "truncate" + (opts.class ? " " + opts.class : ""),
      text,
      attrs: { title: text },
    });
  }

  // ---------- rendering ----------

  function render(trace) {
    document.getElementById("empty-state").hidden = true;
    document.getElementById("run-gone").hidden = true;
    document.getElementById("run-content").hidden = false;

    renderStatusBadge(trace);
    renderOverview(trace);
    renderMetrics(trace);
    renderWorkflow(trace);
    renderTimeline(trace);
    renderGuardrails(trace);
    renderEvaluation(trace);
    renderLatency();
    renderCaching(trace);
    renderSiteLink(trace);
    document.title = `Agent Flow — ${trace.run_id} (${trace.status})`;
  }

  function renderStatusBadge(trace) {
    const badge = document.getElementById("status-badge");
    badge.textContent = trace.status;
    badge.className = "badge " + (STATUS_BADGE_CLASS[trace.status] || "badge-idle");
  }

  function statCard(label, value, opts) {
    opts = opts || {};
    const card = el("div", { class: "stat" });
    card.appendChild(el("div", { class: "stat-label", text: label }));
    card.appendChild(
      el("div", { class: "stat-value" + (opts.warn ? " warn" : "") + (opts.small ? " small" : ""), text: value })
    );
    return card;
  }

  function renderOverview(trace) {
    const grid = document.getElementById("overview-grid");
    grid.innerHTML = "";
    grid.appendChild(statCard("Provider", trace.provider || "not reported"));
    grid.appendChild(statCard("Model", trace.model || "not reported", { small: true }));

    let pricingLabel = trace.pricing_status || "not reported";
    let pricingWarn = false;
    if (trace.pricing_status === "OFFLINE-HEURISTIC") {
      pricingLabel = "OFFLINE-HEURISTIC (catalog unreachable — free assumed from ':free' suffix)";
      pricingWarn = true;
    } else if (trace.pricing_status === "FREE") {
      pricingLabel = "FREE (verified against OpenRouter catalog)";
    }
    grid.appendChild(statCard("Pricing status", pricingLabel, { small: true, warn: pricingWarn }));

    const totalCost = sumSpans(trace.spans, "cost_usd");
    grid.appendChild(statCard("Estimated cost", fmtCost(totalCost)));
  }

  function renderMetrics(trace) {
    const grid = document.getElementById("metrics-grid");
    grid.innerHTML = "";

    let latencyLabel;
    if (trace.duration_ms !== null && trace.duration_ms !== undefined) {
      latencyLabel = fmtMs(trace.duration_ms);
    } else if (trace.status === "RUNNING") {
      latencyLabel = "running…";
    } else {
      latencyLabel = "not reported";
    }
    grid.appendChild(statCard("Total latency", latencyLabel));
    grid.appendChild(statCard("LLM calls", fmtNum(totalLLMCalls(trace.spans))));
    grid.appendChild(statCard("Input tokens", fmtNum(sumSpans(trace.spans, "input_tokens"))));
    grid.appendChild(statCard("Output tokens", fmtNum(sumSpans(trace.spans, "output_tokens"))));
    grid.appendChild(statCard("Total tokens", fmtNum(sumSpans(trace.spans, "total_tokens"))));
    grid.appendChild(statCard("Estimated cost", fmtCost(sumSpans(trace.spans, "cost_usd"))));
  }

  function agentCard(span, opts) {
    opts = opts || {};
    const card = el("div", { class: "agent-card" + (opts.concurrent ? " concurrent" : "") });
    const head = el("div", { class: "agent-head" });
    head.appendChild(el("span", { class: "agent-name", text: span.agent_name }));
    const badge = el("span", { class: "badge " + (STATUS_BADGE_CLASS[span.status] || "badge-waiting"), text: span.status });
    head.appendChild(badge);
    card.appendChild(head);

    card.appendChild(metaRow("Duration", el("b", { text: fmtMs(span.duration_ms) })));

    const calls = span.llm_calls || [];
    const modelLabel = calls.length ? lastKnownModel(calls) || "not reported" : "not reported";
    // Plenty of width here (unlike the dense call table below), so wrap the full
    // id across lines rather than truncating it — this line is a glance-readable
    // talking point, not a tight column.
    card.appendChild(metaRow("Model", el("b", { class: "wrap-anywhere", text: modelLabel }), { class: "model-row" }));

    card.appendChild(
      metaRow(
        "Tokens",
        el("b", { text: `${fmtNum(span.input_tokens)} in / ${fmtNum(span.output_tokens)} out / ${fmtNum(span.total_tokens)} total` })
      )
    );
    if (span.error) {
      const b = el("b", { class: "err-text", text: span.error });
      b.style.color = "var(--red)";
      card.appendChild(metaRow("Error", b));
    }

    if (calls.length) {
      // Per-call detail is a compact chip list, NOT a table: token/latency/cost
      // stats live in a flex-wrap row that is fully decoupled from model-id
      // width. A long requested/served/failover id can never push a numeric
      // stat off-screen or truncate it mid-value — the stats row simply wraps
      // onto another line if the card is narrow. The full requested/served/
      // failover detail is always shown as a wrapping annotation line (not
      // hidden behind hover) whenever it actually differs from the common case.
      const list = el("div", { class: "call-list" });
      calls.forEach((c) => {
        const row = el("div", { class: "call-row" });

        const head = el("div", { class: "call-row-head" });
        head.appendChild(el("span", { class: "call-attempt", text: `#${c.attempt}` }));

        const differs = Boolean(c.model_served) && c.model_served !== c.model_requested;
        const routed = differs || Boolean(c.failover_from);
        const primaryLabel = c.model_served || c.model_requested;
        head.appendChild(truncatedSpan(primaryLabel, { class: "call-model" }));

        let badgeText;
        let badgeClass = "call-badge";
        if (routed) {
          badgeText = "routed — see below";
          badgeClass += " call-badge-routed";
        } else if (!c.model_served) {
          badgeText = "served model not reported";
        } else {
          badgeText = "same as requested";
        }
        head.appendChild(el("span", { class: badgeClass, text: badgeText }));
        row.appendChild(head);

        if (routed) {
          let note = `requested: ${c.model_requested}  →  served: ${c.model_served || "not reported"}`;
          if (c.failover_from) {
            note += `  (↩ failover from ${c.failover_from})`;
          }
          row.appendChild(el("div", { class: "call-annotation wrap-anywhere", text: note }));
        }

        const stats = el("div", { class: "call-stats" });
        const stat = (label, value) => {
          const s = el("span", { class: "call-stat" });
          s.appendChild(document.createTextNode(label + " "));
          s.appendChild(el("b", { text: value }));
          return s;
        };
        stats.appendChild(stat("In", fmtNum(c.input_tokens)));
        stats.appendChild(stat("Out", fmtNum(c.output_tokens)));
        stats.appendChild(stat("Cached", c.cached_tokens > 0 ? fmtNum(c.cached_tokens) : "—"));
        stats.appendChild(stat("Latency", fmtMs(c.latency_ms)));
        stats.appendChild(stat("Cost", fmtCost(c.cost_usd)));
        row.appendChild(stats);

        list.appendChild(row);
      });
      card.appendChild(list);
    } else {
      card.appendChild(el("div", { class: "agent-meta muted", text: "No LLM calls recorded." }));
    }

    return card;
  }

  function renderWorkflow(trace) {
    const container = document.getElementById("workflow-viz");
    container.innerHTML = "";
    const byName = {};
    trace.spans.forEach((s) => (byName[s.agent_name] = s));

    const wrap = el("div", { class: "workflow" });

    // Planner
    if (byName.Planner) {
      wrap.appendChild(agentCard(byName.Planner));
      wrap.appendChild(el("div", { class: "wf-arrow", text: "↓" }));
    }

    // Designer + Content run concurrently
    wrap.appendChild(el("div", { class: "wf-parallel-label", text: "runs concurrently (asyncio.gather)" }));
    const row = el("div", { class: "wf-row" });
    if (byName.Designer) row.appendChild(agentCard(byName.Designer, { concurrent: true }));
    if (byName.Content) row.appendChild(agentCard(byName.Content, { concurrent: true }));
    wrap.appendChild(row);
    wrap.appendChild(el("div", { class: "wf-arrow", text: "↓" }));

    if (byName.Developer) {
      wrap.appendChild(agentCard(byName.Developer));
      wrap.appendChild(el("div", { class: "wf-arrow", text: "↓" }));
    }
    if (byName.Evaluator) {
      wrap.appendChild(agentCard(byName.Evaluator));
    }

    if (trace.retry_count > 0) {
      wrap.appendChild(
        el("div", { class: "wf-parallel-label", text: `Evaluation failed once — Developer retried (retry_count=${trace.retry_count})` })
      );
    }

    container.appendChild(wrap);
  }

  function renderTimeline(trace) {
    const list = document.getElementById("timeline");
    list.innerHTML = "";
    const events = trace.events || [];
    if (!events.length) {
      list.appendChild(el("li", { class: "muted", text: "No events yet." }));
      return;
    }
    events.forEach((e) => {
      const li = el("li");
      li.appendChild(el("span", { class: "ts", text: fmtClock(e.timestamp) }));
      li.appendChild(el("span", { text: e.message }));
      list.appendChild(li);
    });
  }

  function renderGuardrails(trace) {
    const container = document.getElementById("guardrails");
    container.innerHTML = "";
    const events = trace.guardrail_events || [];
    if (!events.length) {
      container.appendChild(el("div", { class: "hint-box", text: "No guardrail events for this run." }));
      return;
    }
    events.forEach((g) => {
      const block = el("div", { class: "guardrail-block" });
      block.appendChild(
        el("div", { class: "grtitle", text: g.blocked ? "🛡️ GUARDRAIL BLOCKED" : "🛡️ guardrail event (not blocked)" })
      );
      block.appendChild(el("div", { text: `Kind: ${g.kind}` }));
      block.appendChild(el("div", { text: `Tool: ${g.tool}` }));
      block.appendChild(el("div", { text: `Path/target: ${g.target}` }));
      block.appendChild(el("div", { text: `Reason: ${g.reason}` }));
      block.appendChild(el("div", { class: "muted", text: fmtClock(g.timestamp) }));
      container.appendChild(block);
    });
  }

  function renderEvaluation(trace) {
    const container = document.getElementById("evaluation");
    container.innerHTML = "";
    const ev = trace.evaluation;
    if (!ev) {
      container.appendChild(el("div", { class: "hint-box", text: "Evaluation not yet available for this run." }));
      return;
    }

    const badge = el("div", {
      class: "pass-fail-badge " + (ev.passed ? "pass" : "fail"),
      text: `${ev.passed ? "PASS" : "FAIL"} — score ${ev.score}/10`,
    });
    container.appendChild(badge);

    const grid = el("div", { class: "eval-grid" });

    const detCol = el("div", { class: "eval-col deterministic" });
    detCol.appendChild(el("h4", { text: "Deterministic checks (code)" }));
    (ev.deterministic_checks || []).forEach((c) => {
      const row = el("div", { class: "check-row" });
      row.appendChild(el("span", { class: c.passed ? "check-pass" : "check-fail", text: c.passed ? "✓" : "✗" }));
      row.appendChild(el("span", { text: `${c.name} — ${c.detail}` }));
      detCol.appendChild(row);
    });
    if (!ev.deterministic_checks || !ev.deterministic_checks.length) {
      detCol.appendChild(el("div", { class: "muted", text: "not reported" }));
    }
    grid.appendChild(detCol);

    const judgeCol = el("div", { class: "eval-col judge" });
    judgeCol.appendChild(el("h4", { text: "LLM-as-Judge" }));
    if (ev.judge) {
      judgeCol.appendChild(el("div", { html: `Score: <b>${ev.judge.score}/10</b>` }));
      if (ev.judge.issues && ev.judge.issues.length) {
        judgeCol.appendChild(el("div", { class: "muted", text: "Issues:" }));
        const ul = el("ul", { class: "issue-list" });
        ev.judge.issues.forEach((i) => ul.appendChild(el("li", { text: i })));
        judgeCol.appendChild(ul);
      }
      if (ev.judge.suggestions && ev.judge.suggestions.length) {
        judgeCol.appendChild(el("div", { class: "muted", text: "Suggestions:" }));
        const ul = el("ul", { class: "issue-list" });
        ev.judge.suggestions.forEach((i) => ul.appendChild(el("li", { text: i })));
        judgeCol.appendChild(ul);
      }
    } else {
      judgeCol.appendChild(el("div", { class: "muted", text: "not reported" }));
    }
    grid.appendChild(judgeCol);

    container.appendChild(grid);
  }

  function renderLatency() {
    const container = document.getElementById("latency");
    container.innerHTML = "";
    if (latencyBootstrap.samples < 5 || latencyBootstrap.p50 === null || latencyBootstrap.p90 === null) {
      container.appendChild(el("div", { class: "hint-box", text: "Insufficient samples for P50/P90" }));
      return;
    }
    const grid = el("div", { class: "grid grid-4" });
    grid.appendChild(statCard("P50", fmtMs(latencyBootstrap.p50)));
    grid.appendChild(statCard("P90", fmtMs(latencyBootstrap.p90)));
    grid.appendChild(statCard("Samples", fmtNum(latencyBootstrap.samples)));
    container.appendChild(grid);
  }

  function renderCaching(trace) {
    const container = document.getElementById("caching");
    container.innerHTML = "";
    let cachedTotal = 0;
    let cacheWriteTotal = 0;
    (trace.spans || []).forEach((s) =>
      (s.llm_calls || []).forEach((c) => {
        cachedTotal += c.cached_tokens || 0;
        cacheWriteTotal += c.cache_write_tokens || 0;
      })
    );
    if (cachedTotal > 0 || cacheWriteTotal > 0) {
      const grid = el("div", { class: "grid grid-4" });
      grid.appendChild(statCard("Cached tokens (read)", fmtNum(cachedTotal)));
      grid.appendChild(statCard("Cache write tokens", fmtNum(cacheWriteTotal)));
      container.appendChild(grid);
    } else {
      container.appendChild(
        el("div", {
          class: "hint-box",
          text: "Prompt caching: not reported by selected free model — prompt structure is cache-ready",
        })
      );
    }
  }

  function renderSiteLink(trace) {
    const container = document.getElementById("site-link");
    container.innerHTML = "";
    const url = trace.site_url || `/site/${trace.run_id}/`;
    if (trace.status === "COMPLETED" || trace.site_path) {
      const a = el("a", { class: "btn", text: "Open generated website", attrs: { href: url, target: "_blank", rel: "noopener" } });
      container.appendChild(a);
    } else {
      container.appendChild(el("div", { class: "muted", text: "Site will be available once the Developer agent completes." }));
    }
  }

  function escapeHtml(s) {
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  // ---------- data fetching / live updates ----------
  //
  // The TraceStore is in-memory only (see docs/CONTRACT.md section 2): a
  // restarted `agentflow serve` starts with zero runs, so a browser tab left
  // open on /run/{old-id} will get a DEFINITIVE 404 from every endpoint for
  // that run forever. That is not a transient failure — retrying it is
  // pointless and produces endless log spam and a silently-stuck UI. A 5xx,
  // a dropped connection, or any other non-404 failure IS treated as
  // transient and keeps retrying exactly as before.

  function stopLive() {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  // Shown in place of the run view once /api/run/{runId} has definitively
  // 404'd. Never claims the site link works without checking first.
  function showRunGone(runId) {
    stopLive();

    document.getElementById("run-content").hidden = true;
    document.getElementById("empty-state").hidden = true;

    const badge = document.getElementById("status-badge");
    badge.textContent = "RUN NOT FOUND";
    badge.className = "badge badge-waiting";
    document.title = `Agent Flow — ${runId} (not found)`;

    const box = document.getElementById("run-gone");
    box.innerHTML = "";
    box.hidden = false;

    box.appendChild(el("h3", { text: "This run is no longer in memory" }));
    const p1 = el("p");
    p1.appendChild(document.createTextNode("Run "));
    p1.appendChild(el("span", { class: "run-gone-id", text: runId }));
    p1.appendChild(
      document.createTextNode(" is no longer in memory. Agent Flow keeps traces in memory only, so they are cleared whenever the server restarts — this is a deliberate design choice (see docs/CONTRACT.md section 2), not a crash. The workflow itself may well have finished normally before the restart.")
    );
    box.appendChild(p1);

    const actions = el("div", { class: "run-gone-actions" });
    actions.appendChild(el("a", { class: "btn", text: "Back to dashboard", attrs: { href: "/" } }));
    box.appendChild(actions);

    // Only offer the generated-site link if it verifiably still resolves —
    // /site/{run_id}/ reads straight from disk, independent of the
    // in-memory TraceStore, so the site usually outlives the trace. Never
    // show a link we know is dead.
    fetch(`/site/${runId}/`, { method: "HEAD" })
      .then((r) => {
        if (r.ok) {
          actions.appendChild(
            el("a", {
              class: "btn btn-secondary",
              text: "Open generated site (still on disk)",
              attrs: { href: `/site/${runId}/`, target: "_blank", rel: "noopener" },
            })
          );
        }
      })
      .catch(() => {});

    refreshRunList();
  }

  function startPolling(runId) {
    stopLive();
    pollTimer = setInterval(() => {
      fetch(`/api/run/${runId}`)
        .then((r) => {
          if (r.status === 404) {
            showRunGone(runId); // also stops this interval
            return null;
          }
          return r.ok ? r.json() : null;
        })
        .then((trace) => trace && render(trace))
        .catch(() => {
          // Network blip / server briefly down — transient, keep retrying.
        });
    }, 2000);
  }

  function startLive(runId) {
    stopLive();
    fetch(`/api/run/${runId}`)
      .then((r) => {
        if (r.status === 404) {
          showRunGone(runId);
          return null;
        }
        return r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`));
      })
      .then((trace) => {
        if (!trace) return; // already handled as "gone" above
        render(trace);
        openLiveConnection(runId);
      })
      .catch(() => {
        // Transient failure on the very first load — still try to open a
        // live connection; startPolling's own retries will keep going, and
        // a genuine 404 will be caught on the next poll instead.
        openLiveConnection(runId);
      });
  }

  // Opens the SSE stream, falling back to polling — but never for a run we
  // already know is gone (checked by the caller) or one that 404s here too.
  function openLiveConnection(runId) {
    if (typeof EventSource === "undefined") {
      startPolling(runId);
      return;
    }

    try {
      eventSource = new EventSource(`/api/run/${runId}/events`);
      eventSource.onmessage = (msg) => {
        try {
          render(JSON.parse(msg.data));
        } catch (e) {
          /* ignore malformed frame */
        }
      };
      eventSource.onerror = () => {
        if (eventSource) {
          eventSource.close();
          eventSource = null;
        }
        // EventSource's error event carries no HTTP status, so confirm
        // whether the run actually 404s before deciding to poll forever —
        // /api/run/{id}/events 404s for exactly the same unknown-run case as
        // /api/run/{id}, and a poll loop on a definitively-gone run is
        // exactly the bug this whole block exists to prevent.
        fetch(`/api/run/${runId}`)
          .then((r) => {
            if (r.status === 404) {
              showRunGone(runId);
            } else {
              // Transient — polling keeps the UI moving until SSE recovers.
              startPolling(runId);
            }
          })
          .catch(() => startPolling(runId));
      };
    } catch (e) {
      startPolling(runId);
    }
  }

  function refreshRunList() {
    fetch("/api/runs")
      .then((r) => (r.ok ? r.json() : []))
      .then((runs) => {
        const list = document.getElementById("run-list");
        list.innerHTML = "";
        if (!runs.length) {
          list.appendChild(el("li", { class: "muted", text: "No runs yet." }));
          return;
        }
        runs.forEach((r) => {
          const li = el("li");
          const a = el("a", {
            text: `${r.run_id} — ${r.status}`,
            attrs: { href: `/run/${r.run_id}` },
          });
          if (r.run_id === currentRunId) a.classList.add("active");
          li.appendChild(a);
          list.appendChild(li);
        });
      })
      .catch(() => {});
  }

  // ---------- build form ----------

  document.getElementById("build-form").addEventListener("submit", function (e) {
    e.preventDefault();
    const input = document.getElementById("build-input");
    const btn = document.getElementById("build-btn");
    const errorBox = document.getElementById("build-error");
    errorBox.hidden = true;
    const text = input.value.trim();
    if (!text) return;

    btn.disabled = true;
    fetch("/api/build", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request: text }),
    })
      .then(async (r) => {
        if (!r.ok) {
          const body = await r.json().catch(() => ({}));
          throw new Error(body.detail || `HTTP ${r.status}`);
        }
        return r.json();
      })
      .then((data) => {
        window.location.href = `/run/${data.run_id}`;
      })
      .catch((err) => {
        errorBox.textContent = err.message || "Failed to start build";
        errorBox.hidden = false;
      })
      .finally(() => {
        btn.disabled = false;
      });
  });

  // ---------- init ----------

  refreshRunList();
  setInterval(refreshRunList, 5000);

  if (currentRunId) {
    startLive(currentRunId);
  }
})();
