/* Localhost test UI (architecture §8.4). Vanilla JS; markdown via marked, sanitised by DOMPurify. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const form = $("ask-form"), qEl = $("question"), submit = $("submit"), bypass = $("bypass");
  const result = $("result"), answerEl = $("answer"), figuresEl = $("figures"), citesEl = $("citations");
  const modal = $("modal"), modalImg = $("modal-img"), modalTitle = $("modal-title");
  let last = null;

  marked.setOptions({ gfm: true, breaks: false });

  let adminEnabled = false;

  async function checkReady() {
    try {
      const r = await fetch("/readyz");
      const j = await r.json();
      const el = $("status");
      if (j.ready) {
        el.textContent = `ready · ${j.chunks} chunks · ${j.embedder} · ${j.model_profile}` + (j.cache_enabled ? "" : " · cache OFF");
        el.className = "status ok";
        adminEnabled = !!j.admin_enabled;
        refreshCache();
      }
      else { el.textContent = "not ready: " + (j.error || "loading"); el.className = "status bad"; setTimeout(checkReady, 3000); }
    } catch (e) { $("status").textContent = "server unreachable"; $("status").className = "status bad"; }
  }

  // ---- cache panel v4 (Phase 6) ----------------------------------------------------------
  const fmtTs = (s) => s ? new Date(s * 1000).toISOString().replace("T", " ").slice(0, 16) + "Z" : "–";
  const fmtDur = (s) => { if (s == null) return "–"; const a = Math.abs(s); if (a >= 86400) return (s / 86400).toFixed(1) + " d"; if (a >= 3600) return (s / 3600).toFixed(1) + " h"; if (a >= 60) return Math.round(s / 60) + " min"; return s + " s"; };

  async function admin(path, body) {
    const r = await fetch("/api/admin" + path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
    if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
    return r.json();
  }

  async function refreshCache() {
    const statsEl = $("cache-stats"), clockEl = $("clock-display");
    if (!adminEnabled) {
      statsEl.textContent = "admin routes disabled (APP_ENV is not dev) — cache stats and clock unavailable";
      document.querySelectorAll(".cache-controls button").forEach((b) => b.disabled = true);
      return;
    }
    try {
      const s = await admin("/cache/stats");
      const l1 = s.l1, l2 = s.l2, cls = Object.entries(l2.by_class).map(([k, v]) => `${k} ${v}`).join(", ") || "none";
      statsEl.textContent = `L1 ${l1.entries}/${l1.max_entries} · hits ${l1.hits} · misses ${l1.misses}  |  L2 ${l2.entries}/${l2.max_entries} (${cls}) · hits ${l2.hits} · misses ${l2.misses} · evictions ${l2.evictions} · swept ${l2.swept}` + (s.enabled ? "" : "  |  CACHE DISABLED");
      const off = s.clock.offset_s;
      clockEl.textContent = off ? `clock: ${s.clock.now_iso} (offset +${fmtDur(off)})` : `clock: ${s.clock.now_iso} (real time)`;
      clockEl.className = "clock" + (off ? " shifted" : "");
      if (!$("cache-entries").classList.contains("hidden")) await loadEntries();
    } catch (e) { statsEl.textContent = "cache stats unavailable: " + e.message; }
  }

  async function loadEntries() {
    const j = await admin("/cache/entries?limit=100");
    const tb = $("cache-entries-table").querySelector("tbody"); tb.innerHTML = "";
    for (const e of j.entries) {
      const tr = document.createElement("tr");
      const slots = [e.entity, e.periods_key || "–", e.metrics_key || "–", e.direction !== "none" ? e.direction : ""].filter(Boolean).join(" · ");
      const left = e.expires_at - j.now;
      tr.innerHTML = `<td>${e.answer_class}</td><td>${DOMPurify.sanitize(slots)}</td><td class="wrap">${DOMPurify.sanitize(e.document)}</td><td>${e.lfu_counter}</td><td>${e.hit_count}</td><td>${left > 0 ? "in " + fmtDur(left) : "EXPIRED"}</td><td>${fmtTs(e.created_at)}</td><td>${e.intent}</td>`;
      if (left <= 0) tr.className = "expanded";
      tb.appendChild(tr);
    }
    if (!j.entries.length) tb.innerHTML = `<tr><td colspan="8" class="wrap">no L2 entries</td></tr>`;
  }

  function renderCacheRequest(resp) {
    const c = resp.cache || { tier: resp.cache_tier }, el = $("cache-request");
    el.classList.remove("hidden"); el.innerHTML = "";
    const chip = (html, cls) => { const s = document.createElement("span"); s.className = cls || "slot"; s.innerHTML = html; el.appendChild(s); };
    chip(c.tier, "tier " + c.tier);
    if (c.tier === "L1" || c.tier === "L2") {
      chip(`similarity <b>${c.similarity == null ? "exact" : c.similarity.toFixed(4)}</b>${c.threshold ? " / threshold " + c.threshold : ""}`);
      chip(`class <b>${c.answer_class}</b>`); chip(`expires <b>${fmtTs(c.expires_at)}</b>`);
      chip(`lfu <b>${c.lfu_counter}</b>`); chip(`hits <b>${c.hit_count}</b>`); chip(`lookup ${c.lookup_ms} ms`);
      chip(`from request <code>${c.origin_request_id}</code> · no model call`);
    } else if (c.tier === "MISS") {
      chip(`L2 candidates passing the slot guard: <b>${c.l2_candidates}</b>` + (c.best_rejected_similarity != null ? ` · best similarity ${c.best_rejected_similarity.toFixed(4)} < ${c.threshold}` : "") + ` · lookup ${c.lookup_ms} ms`);
      const w = c.write;
      if (w) chip(w.admitted ? `ADMITTED → ${w.tiers.join(" + ")} · class <b>${w.answer_class}</b> · expires ${fmtTs(w.expires_at)}${w.evicted ? " · evicted " + w.evicted : ""}` : `NOT ADMITTED · ${DOMPurify.sanitize(w.reasons.join("; "))}`, w.admitted ? "slot" : "slot empty");
    } else {
      chip("bypassed — no cache read or write");
    }
    if (c.clock_offset_s) chip(`clock offset +${fmtDur(c.clock_offset_s)}`, "slot empty");
    refreshCache();
  }

  document.querySelectorAll(".cache-controls [data-adv]").forEach((b) => b.onclick = async () => { await admin("/clock", { advance_seconds: Number(b.dataset.adv) }); refreshCache(); });
  $("clock-reset").onclick = async () => { await admin("/clock", { reset: true }); refreshCache(); };
  $("cache-sweep").onclick = async () => { const r = await admin("/cache/sweep", {}); $("cache-stats").textContent = `sweep: ${JSON.stringify(r)}`; setTimeout(refreshCache, 1200); };
  $("cache-purge").onclick = async () => { if (!confirm("Purge every cache entry (L1 + L2)?")) return; await admin("/cache/purge", { scope: "all" }); refreshCache(); };
  $("cache-entries-toggle").onclick = async () => { const p = $("cache-entries"); p.classList.toggle("hidden"); if (!p.classList.contains("hidden")) await loadEntries(); };

  // ---- ops panel (Phase 7) ---------------------------------------------------------------
  const fmtNum = (n) => (n == null ? "–" : n.toLocaleString());

  async function loadOps() {
    if (!adminEnabled) { $("ops-cache").textContent = "admin routes disabled (APP_ENV is not dev)"; return; }
    let s;
    try { s = await admin("/stats?limit=500"); } catch (e) { $("ops-cache").textContent = "unavailable: " + e.message; return; }
    $("ops-window").textContent = s.window.traces;
    const c = s.cache;
    $("ops-cache").textContent =
      `served ${c.served_requests} · hit rate ${(c.hit_rate * 100).toFixed(1)}% · admitted ${c.admitted}\n` +
      `by tier ${JSON.stringify(c.by_tier)}\n` +
      `window ${s.window.first_ts || "–"} → ${s.window.last_ts || "–"} · profile ${s.profile} · env ${s.app_env}`;
    const spark = $("ops-spark"); spark.innerHTML = "";
    for (const b of c.over_time) {
      const bar = document.createElement("div"); bar.className = "bar";
      bar.style.height = Math.max(2, Math.round(b.hit_rate * 50)) + "px";
      bar.title = `${b.from_ts} → ${b.to_ts}\n${b.hits}/${b.requests} hits`;
      bar.innerHTML = `<span>${Math.round(b.hit_rate * 100)}</span>`;
      spark.appendChild(bar);
    }
    const mb = $("ops-models").querySelector("tbody"); mb.innerHTML = "";
    for (const [role, m] of Object.entries(s.models.by_role)) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${role}</td><td>${DOMPurify.sanitize(m.model || "–")}</td><td>${fmtNum(m.calls)}</td><td>${fmtNum(m.tokens_in)}</td><td>${fmtNum(m.tokens_out)}</td><td>${fmtNum(Math.round(m.pacing_wait_ms / 1000))} s</td><td>${m.retries}</td>`;
      mb.appendChild(tr);
    }
    const q = s.quality;
    $("ops-quality").textContent =
      `verification failures ${q.verification_failures}/${q.verified_requests} (${(q.failure_rate * 100).toFixed(1)}%) · regenerations ${q.regenerations} · pipeline errors ${q.errors}\n` +
      `call statuses ${JSON.stringify(s.models.statuses)}` +
      (q.last_issues.length ? "\nlast issues:\n" + q.last_issues.map((i) => `  ${i.request_id}: ${(i.issues || []).join("; ")}`).join("\n") : "");
    const lb = $("ops-latency").querySelector("tbody"); lb.innerHTML = "";
    const l = s.latency;
    for (const [node, v] of Object.entries(l.by_node)) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${node}</td><td>${v.n}</td><td>${v.p50 ?? "–"}</td><td>${v.p95 ?? "–"}</td>`;
      lb.appendChild(tr);
    }
    const tr = document.createElement("tr"); tr.className = "expanded";
    tr.innerHTML = `<td><b>total</b></td><td>–</td><td>${l.total_p50 ?? "–"}</td><td>${l.total_p95 ?? "–"}</td>`;
    lb.appendChild(tr);
    const cp = s.compression;
    $("ops-compression").textContent =
      `${cp.requests} logged requests · actions ${JSON.stringify(cp.actions)}\n` +
      `compressor calls ${cp.compressor_calls} · fidelity reverts ${cp.fidelity_reverts}\n` +
      `context tokens ${fmtNum(cp.context_tokens_before)} → ${fmtNum(cp.context_tokens_after)} (${cp.tokens_saved_pct}% saved)`;
    $("ops-process").textContent =
      `cache-hit p50 ${l.cache_hit_p50 ?? "–"} ms · miss p50 ${l.cache_miss_p50 ?? "–"} ms\n` +
      `RSS last ${s.memory.rss_mb_last ?? "–"} MB · peak ${s.memory.rss_mb_peak ?? "–"} MB · corpus ${s.corpus_version}`;
  }

  $("ops-refresh").onclick = loadOps;
  $("ops-toggle").onclick = () => {
    const b = $("ops-body"); b.classList.toggle("hidden");
    $("ops-toggle").textContent = b.classList.contains("hidden") ? "show" : "hide";
    if (!b.classList.contains("hidden")) loadOps();
  };

  function citeHtml(md) {
    // [C1] / [K2] → clickable chips (after markdown so the brackets survive)
    return md.replace(/\[([CK]\d+)\]/g, '<span class="cite" data-block="$1">$1</span>');
  }

  function openModal(title, src) {
    modalTitle.textContent = title; modalImg.src = src; modal.classList.remove("hidden");
  }
  $("modal-close").onclick = () => modal.classList.add("hidden");
  modal.onclick = (e) => { if (e.target === modal) modal.classList.add("hidden"); };

  function showCitation(blockId) {
    if (!last) return;
    const c = last.citations.concat(last.figures).find((x) => x.block_id === blockId)
      || last.debug.context_blocks.map(b => ({ block_id: b.block_id, page: b.page, label: b.header.replace(/^\[|\]$/g, ""), page_image_url: b.page ? `/api/pages/${b.page}` : null })).find((x) => x.block_id === blockId);
    if (!c) return;
    if (c.figure_image_url) openModal(c.label, c.figure_image_url);
    else if (c.page_image_url) openModal(c.label, c.page_image_url);
  }

  function render(resp) {
    last = resp;
    result.classList.remove("hidden");
    const tierEl = $("cache-tier"); tierEl.textContent = "cache: " + resp.cache_tier; tierEl.className = "badge " + resp.cache_tier;
    renderCacheRequest(resp);
    $("intent").textContent = resp.debug.intent;
    const conf = $("confidence"); conf.textContent = "confidence: " + resp.answer.confidence; conf.className = "badge " + resp.answer.confidence;
    $("answer-class").textContent = resp.answer.answer_class;
    $("latency").textContent = `${resp.debug.total_latency_ms} ms · ${resp.debug.generation_attempts} generation attempt(s) · ${resp.debug.generator_role} role`;
    const warn = $("warning");
    if (resp.warning) { warn.textContent = resp.warning; warn.classList.remove("hidden"); } else { warn.classList.add("hidden"); }
    // Phase 8 degrade mode: the answer is a retrieval-only view (no model), never cached.
    warn.classList.toggle("degraded", !!resp.degraded);
    if (resp.degraded) conf.textContent = "degraded · retrieval-only view";

    let html = DOMPurify.sanitize(marked.parse(resp.answer.answer_markdown));
    html = citeHtml(html);
    if (resp.answer.fiscal_year_interpretation) html += `<p class="meta">Fiscal-year interpretation: ${DOMPurify.sanitize(resp.answer.fiscal_year_interpretation)}</p>`;
    answerEl.innerHTML = html;
    answerEl.querySelectorAll(".cite").forEach((el) => el.onclick = () => showCitation(el.dataset.block));

    figuresEl.innerHTML = "";
    for (const f of resp.figures) {
      const fig = document.createElement("figure");
      const img = document.createElement("img"); img.src = f.figure_image_url; img.alt = f.label;
      img.onclick = () => openModal(f.label, f.figure_image_url);
      const cap = document.createElement("figcaption"); cap.textContent = `${f.block_id} · ${f.label}`;
      fig.append(img, cap); figuresEl.appendChild(fig);
    }

    citesEl.innerHTML = "";
    for (const c of resp.citations) {
      const chip = document.createElement("span"); chip.className = "chip";
      chip.innerHTML = `<b>${c.block_id}</b> ${DOMPurify.sanitize(c.label)}`;
      chip.onclick = () => showCitation(c.block_id);
      citesEl.appendChild(chip);
    }

    // debug panel v2 — query understanding (Phase 4)
    const d = resp.debug, sl = d.slots;
    const slotsEl = $("slots"); slotsEl.innerHTML = "";
    const slotRows = [
      ["entity", sl.entity], ["periods", sl.fiscal_periods.join(", ")], ["metrics", sl.metrics.join(", ")],
      ["formulas", sl.formulas.join(", ")], ["statement", sl.statement ? `${sl.statement} (${sl.statement_source})` : ""],
      ["direction", sl.direction], ["aggregation", sl.aggregation.join(", ")], ["time anchor", sl.time_anchor ? "yes" : ""],
    ];
    for (const [k, v] of slotRows) {
      const span = document.createElement("span"); span.className = "slot" + (v ? "" : " empty");
      span.innerHTML = `<b>${k}</b> ${DOMPurify.sanitize(v || "–")}`; slotsEl.appendChild(span);
    }
    if (sl.period_note) { const n = document.createElement("span"); n.className = "slot"; n.textContent = sl.period_note; slotsEl.appendChild(n); }
    const sc = d.scope;
    $("scope").textContent = `${sc.in_scope ? "IN SCOPE" : "OUT OF SCOPE"} · score ${sc.score} · rule ${sc.rule || "none"} · decided by ${sc.source}${sc.llm_called ? " (small model called)" : ""}\n${sc.reason}`;
    const an = d.analysis;
    $("analysis").textContent = `intent ${an.intent} · confidence ${an.confidence.toFixed(2)} · decided by ${an.source} · rule "${an.rule}"` +
      (an.llm_called ? `\nsmall model called: intent ${an.llm_intent || "?"} · paraphrases ${an.paraphrases.length} · sub-questions ${an.sub_questions.length} · section hints ${an.section_hints.join(", ") || "none"}` : "\nno model call (rule-confident)") +
      (an.hyde_passage ? `\nHyDE: ${an.hyde_passage}` : (an.hyde_rejected ? "\nHyDE rejected (contained digits) and dropped" : "")) +
      (an.notes.length ? `\nnotes: ${an.notes.join(" | ")}` : "");
    const qb = $("queries-table").querySelector("tbody"); qb.innerHTML = "";
    const retried = d.retrieval.filtered_retries || [];
    an.queries.forEach((q, i) => {
      const tr = document.createElement("tr");
      const f = q.where ? JSON.stringify(q.where).replace(/"/g, "") + (retried.includes(i) ? " → < 3 results, retried unfiltered" : "") : "–";
      tr.innerHTML = `<td>${q.kind}</td><td class="wrap">${DOMPurify.sanitize(q.text)}</td><td class="wrap">${DOMPurify.sanitize(f)}</td>`;
      qb.appendChild(tr);
    });
    const rr = d.rerank;
    $("rerank").textContent = rr.applied
      ? `RERANKED with ${rr.model} · kept ${rr.kept} · dropped ${rr.dropped.length ? rr.dropped.join(", ") : "none"} · ${rr.latency_ms} ms`
      : `SKIPPED · gate ${rr.gate || "-"}\n${rr.skip_reason || ""}` + (rr.required_metrics.length ? `\nrequired metrics: ${rr.required_metrics.join(", ")}` : "");

    // debug panel v3 — compression (Phase 5)
    const cp = d.compression || {};
    $("compression").textContent = !cp.enabled
      ? `OFF (${cp.mode})`
      : (cp.query_needs_compression
        ? `COMPRESSED · ${cp.tokens_before} → ${cp.tokens_after} tokens · actions ${JSON.stringify(cp.actions)} · small-model calls ${cp.llm_calls} · violations ${cp.violations.length ? cp.violations.join(" | ") : "none"} · ${cp.latency_ms} ms`
        : `NOT NEEDED · ${cp.skip_reason || "every chunk kept"} · ${cp.tokens_before} tokens · ${cp.latency_ms} ms`)
      + (cp.query ? `\nbudget ratio ${cp.query.budget_ratio} · ${cp.query.n_chunks} chunks · ${cp.query.total_tokens} tokens · intent ${cp.query.intent}` : "");
    const cb = $("compression-table").querySelector("tbody"); cb.innerHTML = "";
    (cp.chunks || []).forEach((c) => {
      const tr = document.createElement("tr"); if (c.applied === "DEDUPE" || c.applied === "DROP") tr.className = "expanded";
      const fid = c.fidelity_ok == null ? "–" : (c.fidelity_ok ? "✓" : "✗ reverted");
      tr.innerHTML = `<td>${c.chunk_id}</td><td>${c.modality}</td><td>${c.action}</td><td>${c.applied}${c.reverted ? " (reverted)" : ""}</td><td>${c.tokens_before}→${c.tokens_after}</td><td>${c.relevance_density}</td><td>${c.max_dup_sim > 0 ? c.max_dup_sim.toFixed(2) : "–"}</td><td>${fid}</td><td class="wrap">${DOMPurify.sanitize(c.reason)}${c.note ? " · " + DOMPurify.sanitize(c.note) : ""}</td>`;
      cb.appendChild(tr);
    });

    const inCtx = new Set(d.context_blocks.map((b) => b.chunk_id));
    const tbody = $("retrieval-table").querySelector("tbody"); tbody.innerHTML = "";
    d.candidates.forEach((c, i) => {
      const tr = document.createElement("tr"); if (c.source === "expanded") tr.className = "expanded";
      const dense = c.dense_rank ? `#${c.dense_rank} (${c.dense_score})` : "–";
      const bm25 = c.bm25_rank ? `#${c.bm25_rank} (${c.bm25_score})` : "–";
      const rers = c.rerank_score != null ? `#${c.rerank_rank} (${c.rerank_score.toFixed(2)})` : "–";
      tr.innerHTML = `<td>${i + 1}</td><td>${c.chunk_id}</td><td>${c.modality}</td><td>${c.page}</td><td>${dense}</td><td>${bm25}</td><td>${c.rrf.toFixed(4)}</td><td>${rers}</td><td>${(c.hit_by || []).join(",") || "–"}</td><td>${inCtx.has(c.chunk_id) ? "✓" : ""}</td>`;
      tbody.appendChild(tr);
    });
    $("calc").textContent = resp.debug.calculations.length
      ? resp.debug.calculations.map((k) => `${k.formula} (FY${k.fiscal_year}${k.metric ? ", " + k.metric : ""}): ${k.status === "ok" ? k.rounded : k.status + " — " + k.message}\n` + k.inputs.map((i) => `   ${i.name} = ${i.value} ← ${i.chunk_id} p.${i.page}`).join("\n")).join("\n\n")
      : "(no calculation for this intent)";
    const v = resp.debug.verify;
    $("verify").textContent = `passed: ${v.passed}\nnumbers checked: ${v.numbers_checked}\nunmatched: ${v.unmatched_numbers.join(", ") || "none"}\ninvalid citations: ${v.invalid_citations.join(", ") || "none"}\nissues: ${v.issues.join(" | ") || "none"}`;
    const tokens = Object.entries(resp.debug.tokens_by_model).map(([m, t]) => `${m}: in ${t.in} · out ${t.out} · calls ${t.calls}${t.pacing_wait_ms ? ` · pacing wait ${t.pacing_wait_ms} ms` : ""}`).join("\n") || "no model calls";
    const lat = Object.entries(resp.debug.latency_ms_by_node).map(([n, ms]) => `${n} ${ms} ms`).join(" · ");
    $("tokens").textContent = `${tokens}\n${lat}\ntotal ${resp.debug.total_latency_ms} ms · context ${resp.debug.context_tokens}/${resp.debug.context_budget} tokens · dropped ${resp.debug.dropped.length}`;
    $("ctx-tokens").textContent = resp.debug.context_tokens;
    const blocks = $("context-blocks"); blocks.innerHTML = "";
    for (const b of resp.debug.context_blocks) {
      const div = document.createElement("div"); div.className = "block";
      div.innerHTML = `<div class="hdr">${DOMPurify.sanitize(b.header)} · ${b.token_count} tok${b.compression ? " · " + b.compression : ""}</div><div class="body">${DOMPurify.sanitize(b.text)}</div>`;
      blocks.appendChild(div);
    }
    $("request-id").textContent = resp.request_id;
    $("trace-link").href = `/api/trace/${resp.request_id}`;
    $("corpus-version").textContent = resp.debug.corpus_version;
  }

  async function ask(question) {
    submit.disabled = true; submit.textContent = "Thinking…";
    try {
      const r = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, bypass_cache: bypass.checked }) });
      if (!r.ok) { const t = await r.text(); throw new Error(`${r.status}: ${t}`); }
      render(await r.json());
    } catch (e) {
      result.classList.remove("hidden");
      answerEl.textContent = "Request failed: " + e.message;
    } finally { submit.disabled = false; submit.textContent = "Ask"; }
  }

  form.onsubmit = (e) => { e.preventDefault(); const q = qEl.value.trim(); if (q) ask(q); };
  qEl.onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); } };
  $("examples").querySelectorAll("button").forEach((b) => b.onclick = () => { qEl.value = b.dataset.q; form.requestSubmit(); });
  checkReady();
})();
