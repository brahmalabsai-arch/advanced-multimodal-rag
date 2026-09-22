/* LedgerLens frontend
 * ---------------------------------------------------------------------------
 * Talks to the FastAPI backend on the same origin.
 *
 * Expected API
 *   POST /api/key/test      headers: X-Provider, X-Provider-Key      -> 200 {model}
 *   POST /api/ask           headers: X-Provider, X-Provider-Key
 *                           body:    {question}                      -> 200 AnswerPayload
 *   GET  /api/pages/{n}                                              -> image
 *
 * AnswerPayload (every field optional except answer_markdown)
 *   {
 *     answer_markdown: "…",
 *     headline:  {value: "3.91", caption: "current ratio, as of 25 January 2026",
 *                 working: "125,605 / 32,163 — USD millions"},
 *     figures_used: [{label: "Total current assets", value: "125,605"}],
 *     citations:    [{label: "Consolidated Balance Sheets", page: 141}],
 *     confidence:   "high" | "medium" | "low",
 *     degraded:     false,
 *     disclaimer:   "…",
 *     trace: {
 *       cache_tier: "miss" | "L1" | "L2",
 *       model: "llama-3.3-70b",
 *       total_seconds: 4.8,
 *       verified: true,
 *       slots: {entity: "…", period: "FY2026", formula: "current_ratio"},
 *       steps: [{label, detail, seconds, tags: [], kind: "normal|computed|checked"}]
 *     }
 *   }
 * If trace.steps is missing the rail is built from whatever trace fields exist.
 * ------------------------------------------------------------------------- */

(function () {
  "use strict";

  var API = "";
  var DEMO = new URLSearchParams(location.search).has("demo");

  var STORE = { provider: "ll.provider", key: "ll.key" };
  var session = { provider: "groq", key: "" };

  var el = function (id) { return document.getElementById(id); };
  var gate = el("gate");
  var workspace = el("workspace");

  /* ── key handling ─────────────────────────────────────────────────────── */

  function loadKey() {
    try {
      var p = sessionStorage.getItem(STORE.provider);
      var k = sessionStorage.getItem(STORE.key);
      if (p) { session.provider = p; }
      if (k) { session.key = k; }
    } catch (e) { /* private mode: keep the key in memory only */ }
  }

  function saveKey(remember) {
    try {
      if (remember) {
        sessionStorage.setItem(STORE.provider, session.provider);
        sessionStorage.setItem(STORE.key, session.key);
      } else {
        sessionStorage.removeItem(STORE.provider);
        sessionStorage.removeItem(STORE.key);
      }
    } catch (e) { /* ignore */ }
  }

  function clearKey() {
    session.key = "";
    try {
      sessionStorage.removeItem(STORE.key);
    } catch (e) { /* ignore */ }
  }

  function authHeaders() {
    return { "X-Provider": session.provider, "X-Provider-Key": session.key };
  }

  /* ── screens ──────────────────────────────────────────────────────────── */

  function showWorkspace(model) {
    gate.hidden = true;
    workspace.hidden = false;
    el("change-key").hidden = false;
    var pill = el("model-pill");
    pill.hidden = false;
    pill.textContent = (model || session.provider) + " · your key";
    el("question").focus();
  }

  function showGate() {
    workspace.hidden = true;
    gate.hidden = false;
    el("change-key").hidden = true;
    el("model-pill").hidden = true;
    el("api-key").value = "";
    el("api-key").focus();
  }

  function only(id) {
    ["state-empty", "state-working", "state-error", "state-answer"].forEach(function (s) {
      el(s).hidden = s !== id;
    });
    if (id !== "state-answer") { el("rail").hidden = true; }
  }

  /* ── gate ─────────────────────────────────────────────────────────────── */

  el("provider-group").addEventListener("click", function (ev) {
    var btn = ev.target.closest("[data-provider]");
    if (!btn) { return; }
    session.provider = btn.dataset.provider;
    Array.prototype.forEach.call(this.querySelectorAll(".segment"), function (b) {
      var on = b === btn;
      b.classList.toggle("is-selected", on);
      b.setAttribute("aria-checked", on ? "true" : "false");
    });
    var hints = { groq: "gsk_…", anthropic: "sk-ant-…", gemini: "AIza…", openrouter: "sk-or-…" };
    el("api-key").placeholder = hints[session.provider] || "";
  });

  el("key-form").addEventListener("submit", function (ev) {
    ev.preventDefault();
    var input = el("api-key");
    var error = el("key-error");
    var button = el("connect");

    error.hidden = true;
    session.key = input.value.trim();

    if (!session.key) {
      error.textContent = "Paste a key to continue.";
      error.hidden = false;
      input.focus();
      return;
    }

    if (DEMO) {
      saveKey(el("remember-key").checked);
      showWorkspace("demo");
      return;
    }

    button.disabled = true;
    button.textContent = "Testing the key";

    fetch(API + "/api/key/test", { method: "POST", headers: authHeaders() })
      .then(function (res) {
        if (res.ok) { return res.json(); }
        if (res.status === 401 || res.status === 403) {
          throw new Error("That key was rejected by " + session.provider + ". Check it and paste it again.");
        }
        throw new Error("The server could not reach " + session.provider + " (status " + res.status + "). Try again in a moment.");
      })
      .then(function (data) {
        saveKey(el("remember-key").checked);
        showWorkspace(data && data.model);
      })
      .catch(function (err) {
        error.textContent = err.message || "The key could not be checked.";
        error.hidden = false;
      })
      .finally(function () {
        button.disabled = false;
        button.textContent = "Test key and start";
      });
  });

  el("change-key").addEventListener("click", function () { clearKey(); showGate(); });
  el("error-key").addEventListener("click", function () { clearKey(); showGate(); });

  /* ── asking ───────────────────────────────────────────────────────────── */

  var STAGES = [
    "Checking answered questions",
    "Reading the question",
    "Searching the filing",
    "Trimming the context",
    "Computing the figures",
    "Writing the explanation"
  ];

  var timers = [];
  var lastQuestion = "";

  function runProgress() {
    var list = el("progress");
    list.innerHTML = "";
    STAGES.forEach(function (label, i) {
      var li = document.createElement("li");
      li.dataset.state = i === 0 ? "active" : "waiting";
      li.innerHTML = '<span class="dot"></span>';
      li.appendChild(document.createTextNode(label));
      list.appendChild(li);
    });

    timers.forEach(clearTimeout);
    timers = [];
    STAGES.forEach(function (_, i) {
      if (i === 0) { return; }
      timers.push(setTimeout(function () {
        var items = list.children;
        for (var j = 0; j < i; j++) { items[j].dataset.state = "done"; }
        items[i].dataset.state = "active";
      }, i * 900));
    });
    timers.push(setTimeout(function () { el("warming").hidden = false; }, 7000));
  }

  function stopProgress() {
    timers.forEach(clearTimeout);
    timers = [];
    el("warming").hidden = true;
  }

  function ask(question) {
    if (!question) { return; }
    lastQuestion = question;
    el("question").value = question;
    only("state-working");
    runProgress();

    var request = DEMO
      ? fetch("./demo-answer.json").then(function (r) { return r.json(); })
      : fetch(API + "/api/ask", {
          method: "POST",
          headers: Object.assign({ "Content-Type": "application/json" }, authHeaders()),
          body: JSON.stringify({ question: question })
        }).then(handleResponse);

    request
      .then(function (payload) { stopProgress(); renderAnswer(payload); })
      .catch(function (err) { stopProgress(); renderError(err); });
  }

  function handleResponse(res) {
    if (res.ok) { return res.json(); }
    var err = new Error();
    err.status = res.status;
    return res.json().then(function (body) {
      err.detail = body && (body.detail || body.message);
      throw err;
    }, function () { throw err; });
  }

  el("ask-form").addEventListener("submit", function (ev) {
    ev.preventDefault();
    ask(el("question").value.trim());
  });

  el("suggestions").addEventListener("click", function (ev) {
    var chip = ev.target.closest(".chip");
    if (chip) { ask(chip.textContent.trim()); }
  });

  el("error-retry").addEventListener("click", function () { ask(lastQuestion); });

  /* ── rendering ────────────────────────────────────────────────────────── */

  function toHtml(markdown) {
    var text = markdown || "";
    if (window.marked && window.DOMPurify) {
      return window.DOMPurify.sanitize(window.marked.parse(text));
    }
    var escaped = text.replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
    return escaped.split(/\n{2,}/).map(function (p) {
      return "<p>" + p.replace(/\n/g, "<br>") + "</p>";
    }).join("");
  }

  function renderAnswer(payload) {
    var head = payload.headline || {};
    el("answer-figure").textContent = head.value || "";
    el("answer-caption").textContent = head.caption || "";
    el("answer-working").textContent = head.working || "";
    el("answer-prose").innerHTML = toHtml(payload.answer_markdown);

    var note = el("answer-disclaimer");
    if (payload.degraded) {
      note.textContent = "The model could not be reached, so this shows the cited passages and computed figures without a written explanation.";
      note.hidden = false;
    } else if (payload.confidence === "low") {
      note.textContent = "Low confidence: not every figure could be traced back to a cited passage. Check the sources before using this.";
      note.hidden = false;
    } else if (payload.disclaimer) {
      note.textContent = payload.disclaimer;
      note.hidden = false;
    } else {
      note.hidden = true;
    }

    var cites = el("citations");
    cites.innerHTML = "";
    (payload.citations || []).forEach(function (c) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "citation";
      b.textContent = c.label + (c.page ? ", page " + c.page : "");
      if (c.page) { b.addEventListener("click", function () { openPage(c); }); }
      cites.appendChild(b);
    });

    var body = el("figures-body");
    body.innerHTML = "";
    var figures = payload.figures_used || [];
    el("figures-table").hidden = figures.length === 0;
    figures.forEach(function (f) {
      var tr = document.createElement("tr");
      var label = document.createElement("td");
      label.textContent = f.label;
      var value = document.createElement("td");
      value.textContent = f.value + (f.unit ? " " + f.unit : "");
      tr.appendChild(label);
      tr.appendChild(value);
      body.appendChild(tr);
    });

    renderRail(payload.trace || {});
    only("state-answer");
    el("rail").hidden = false;
  }

  function renderRail(trace) {
    var total = el("rail-total");
    total.textContent = trace.total_seconds ? trace.total_seconds.toFixed(1) + "s" : "";
    total.hidden = !trace.total_seconds;

    var steps = trace.steps && trace.steps.length ? trace.steps : buildSteps(trace);
    var list = el("steps");
    list.innerHTML = "";

    steps.forEach(function (s) {
      var li = document.createElement("li");
      li.className = "step" + (s.kind === "computed" ? " step--computed" : s.kind === "checked" ? " step--checked" : "");

      var head = document.createElement("div");
      head.className = "step__head";
      var label = document.createElement("span");
      label.textContent = s.label;
      head.appendChild(label);
      if (s.seconds || s.note) {
        var time = document.createElement("span");
        time.className = "step__time";
        time.textContent = s.note || (s.seconds.toFixed(2) + "s");
        head.appendChild(time);
      }
      li.appendChild(head);

      if (s.detail) {
        var p = document.createElement("p");
        p.className = "step__body";
        p.textContent = s.detail;
        li.appendChild(p);
      }

      if (s.tags && s.tags.length) {
        var tags = document.createElement("div");
        tags.className = "step__tags";
        s.tags.forEach(function (t) {
          var span = document.createElement("span");
          span.className = "tag";
          span.textContent = t;
          tags.appendChild(span);
        });
        li.appendChild(tags);
      }

      list.appendChild(li);
    });

    var foot = el("rail-foot");
    if (trace.tokens) {
      foot.textContent = trace.tokens["in"] + " tokens in, " + trace.tokens.out + " out, billed to your key.";
      foot.hidden = false;
    } else {
      foot.hidden = true;
    }
  }

  function buildSteps(trace) {
    var steps = [];
    var tier = trace.cache_tier || "miss";

    steps.push({
      label: "Memory checked",
      note: tier === "miss" ? "no match" : tier + " hit",
      detail: tier === "miss"
        ? "No stored answer matched on period, metric and formula. Similarity alone is never enough to reuse one."
        : "A stored answer matched on every guard key, so no model was called."
    });

    if (trace.slots) {
      steps.push({
        label: "Question read",
        tags: Object.keys(trace.slots).map(function (k) { return trace.slots[k]; }).filter(Boolean)
      });
    }
    if (trace.retrieval) {
      steps.push({ label: "Passages found", seconds: trace.retrieval.seconds, detail: trace.retrieval.detail });
    }
    if (trace.compression) {
      steps.push({ label: "Context trimmed", seconds: trace.compression.seconds, detail: trace.compression.detail });
    }
    if (trace.calculator) {
      steps.push({ label: "Arithmetic in code", note: "exact", detail: trace.calculator.detail, kind: "computed" });
    }
    if (trace.generation) {
      steps.push({ label: "Explanation written", seconds: trace.generation.seconds, detail: trace.generation.detail });
    }
    if (trace.verified !== undefined) {
      steps.push({
        label: trace.verified ? "Every number checked" : "Check incomplete",
        kind: "checked",
        detail: trace.verified
          ? "Each figure traces back to a cited passage or to the calculator."
          : "At least one figure could not be traced. Treat this answer with care."
      });
    }
    return steps;
  }

  function renderError(err) {
    var title = "Something went wrong";
    var body = err.detail || "The request did not complete. Try again.";
    var offerKey = false;

    if (err.status === 401 || err.status === 403) {
      title = "That key was rejected";
      body = "The provider turned the key down. Paste it again, or switch provider.";
      offerKey = true;
    } else if (err.status === 402) {
      title = "The key has no credit left";
      body = "Top up the account behind this key, or switch to another provider.";
      offerKey = true;
    } else if (err.status === 429) {
      title = "Rate limit reached";
      body = "The provider is throttling this key. Wait about a minute and ask again.";
    } else if (err.status === 502 || err.status === 503 || err.status === 504) {
      title = "The server is still waking up";
      body = "Free hosting puts it to sleep after a quiet spell. Give it a few seconds and try again.";
    } else if (!err.status) {
      title = "No connection";
      body = "The page could not reach the server. Check your network and try again.";
    }

    el("error-title").textContent = title;
    el("error-body").textContent = body;
    el("error-key").hidden = !offerKey;
    only("state-error");
  }

  /* ── page viewer ──────────────────────────────────────────────────────── */

  var viewer = el("viewer");

  function openPage(citation) {
    el("viewer-title").textContent = citation.label + ", page " + citation.page;
    var img = el("viewer-img");
    img.src = API + "/api/pages/" + citation.page;
    img.alt = "Page " + citation.page + " of the filing";
    if (typeof viewer.showModal === "function") { viewer.showModal(); }
  }

  el("viewer-close").addEventListener("click", function () { viewer.close(); });
  viewer.addEventListener("click", function (ev) { if (ev.target === viewer) { viewer.close(); } });

  /* ── start ────────────────────────────────────────────────────────────── */

  loadKey();
  only("state-empty");
  if (session.key) { showWorkspace(); } else { el("api-key").focus(); }
})();
