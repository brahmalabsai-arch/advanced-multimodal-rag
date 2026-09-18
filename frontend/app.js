/* Localhost test UI (architecture §8.4). Vanilla JS; markdown via marked, sanitised by DOMPurify. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const form = $("ask-form"), qEl = $("question"), submit = $("submit"), bypass = $("bypass");
  const result = $("result"), answerEl = $("answer"), figuresEl = $("figures"), citesEl = $("citations");
  const modal = $("modal"), modalImg = $("modal-img"), modalTitle = $("modal-title");
  let last = null;

  marked.setOptions({ gfm: true, breaks: false });

  async function checkReady() {
    try {
      const r = await fetch("/readyz");
      const j = await r.json();
      const el = $("status");
      if (j.ready) { el.textContent = `ready · ${j.chunks} chunks · ${j.embedder} · ${j.model_profile}`; el.className = "status ok"; }
      else { el.textContent = "not ready: " + (j.error || "loading"); el.className = "status bad"; setTimeout(checkReady, 3000); }
    } catch (e) { $("status").textContent = "server unreachable"; $("status").className = "status bad"; }
  }

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
    $("cache-tier").textContent = "cache: " + resp.cache_tier;
    $("intent").textContent = resp.debug.intent;
    const conf = $("confidence"); conf.textContent = "confidence: " + resp.answer.confidence; conf.className = "badge " + resp.answer.confidence;
    $("answer-class").textContent = resp.answer.answer_class;
    $("latency").textContent = `${resp.debug.total_latency_ms} ms · ${resp.debug.generation_attempts} generation attempt(s) · ${resp.debug.generator_role} role`;
    const warn = $("warning");
    if (resp.warning) { warn.textContent = resp.warning; warn.classList.remove("hidden"); } else { warn.classList.add("hidden"); }

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

    // debug panel
    const inCtx = new Set(resp.debug.context_blocks.map((b) => b.chunk_id));
    const tbody = $("retrieval-table").querySelector("tbody"); tbody.innerHTML = "";
    resp.debug.candidates.forEach((c, i) => {
      const tr = document.createElement("tr"); if (c.source === "expanded") tr.className = "expanded";
      const dense = c.dense_rank ? `#${c.dense_rank} (${c.dense_score})` : "–";
      const bm25 = c.bm25_rank ? `#${c.bm25_rank} (${c.bm25_score})` : "–";
      tr.innerHTML = `<td>${i + 1}</td><td>${c.chunk_id}</td><td>${c.modality}</td><td>${c.page}</td><td>${dense}</td><td>${bm25}</td><td>${c.rrf.toFixed(4)}</td><td>${inCtx.has(c.chunk_id) ? "✓" : ""}</td>`;
      tbody.appendChild(tr);
    });
    $("calc").textContent = resp.debug.calculations.length
      ? resp.debug.calculations.map((k) => `${k.formula} (FY${k.fiscal_year}${k.metric ? ", " + k.metric : ""}): ${k.status === "ok" ? k.rounded : k.status + " — " + k.message}\n` + k.inputs.map((i) => `   ${i.name} = ${i.value} ← ${i.chunk_id} p.${i.page}`).join("\n")).join("\n\n")
      : "(no calculation for this intent)";
    const v = resp.debug.verify;
    $("verify").textContent = `passed: ${v.passed}\nnumbers checked: ${v.numbers_checked}\nunmatched: ${v.unmatched_numbers.join(", ") || "none"}\ninvalid citations: ${v.invalid_citations.join(", ") || "none"}\nissues: ${v.issues.join(" | ") || "none"}`;
    const tokens = Object.entries(resp.debug.tokens_by_model).map(([m, t]) => `${m}: in ${t.in} · out ${t.out} · calls ${t.calls}`).join("\n") || "no model calls";
    const lat = Object.entries(resp.debug.latency_ms_by_node).map(([n, ms]) => `${n} ${ms} ms`).join(" · ");
    $("tokens").textContent = `${tokens}\n${lat}\ntotal ${resp.debug.total_latency_ms} ms · context ${resp.debug.context_tokens}/${resp.debug.context_budget} tokens · dropped ${resp.debug.dropped.length}`;
    $("ctx-tokens").textContent = resp.debug.context_tokens;
    const blocks = $("context-blocks"); blocks.innerHTML = "";
    for (const b of resp.debug.context_blocks) {
      const div = document.createElement("div"); div.className = "block";
      div.innerHTML = `<div class="hdr">${DOMPurify.sanitize(b.header)} · ${b.token_count} tok</div><div class="body">${DOMPurify.sanitize(b.text)}</div>`;
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
