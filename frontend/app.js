/* LedgerLens frontend
 * ---------------------------------------------------------------------------
 * Talks to the FastAPI backend on the same origin.
 *
 * Expected API
 *   POST /api/key/test      headers: X-Provider, X-Provider-Key      -> 200 {model}
 *   POST /api/ask           headers: X-Provider, X-Provider-Key
 *                           body:    {question, history}             -> 200 AnswerPayload
 *                           history: the last turns of this tab's conversation, compact:
 *                                    [{question, standalone, summary}] (in-session memory)
 *   GET  /api/pages/{n}                                              -> page image
 *   GET  /api/figures/{id}                                           -> figure crop
 *
 * AnswerPayload (every field optional except answer_markdown)
 *   {
 *     answer_markdown: "…",
 *     headline:  {value: "3.91", caption: "current ratio, as of 25 January 2026",
 *                 working: "125,605 / 32,163 USD millions"},
 *     figures_used: [{label: "Total current assets", value: "125,605", unit: ""}],
 *     figures:      [{url: "/api/figures/p3_0", caption: "AI Is a Five-Layer Cake", page: 3}],
 *     citations:    [{label: "Consolidated Balance Sheets", page: 141}],
 *     confidence:   "high" | "medium" | "low",
 *     degraded:     false,
 *     disclaimer:   "…",
 *     trace: {
 *       cache_tier: "miss" | "L1" | "L2",
 *       model: "llama-3.3-70b",
 *       total_seconds: 4.8,
 *       verified: true,
 *       tokens: {in, out},
 *       slots: {entity, period, formula},
 *       steps: [{label, detail, seconds, tags: [], kind: "normal|computed|checked"}],
 *       metrics: {cache: {tier, similarity, threshold, best_rejected_similarity, lookup_ms, written},
 *                 retrieval: {queries, top_similarity, top_rrf, passages: [{label, page, similarity, rrf}]},
 *                 context_tokens}
 *     }
 *   }
 * If trace.steps is missing the step list is built from whatever trace fields exist.
 * ------------------------------------------------------------------------- */

(function () {
  "use strict";

  var API = "";
  var DEMO = new URLSearchParams(location.search).has("demo");

  var PROVIDER_LABELS = { groq: "Groq", anthropic: "Anthropic", gemini: "Gemini" };
  var STORE = { provider: "ll.provider", key: "ll.key", model: "ll.model", answered: "ll.answered", thread: "ll.thread" };
  var session = { provider: "groq", key: "", model: "" };

  var el = function (id) { return document.getElementById(id); };
  var make = function (tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) { n.className = cls; }
    if (text !== undefined && text !== null) { n.textContent = text; }
    return n;
  };

  var store = {
    get: function (k) { try { return sessionStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { sessionStorage.setItem(k, v); } catch (e) { /* private mode */ } },
    del: function (k) { try { sessionStorage.removeItem(k); } catch (e) { /* ignore */ } }
  };

  /* ── key handling ─────────────────────────────────────────────────────── */

  function loadKey() {
    session.provider = store.get(STORE.provider) || session.provider;
    session.key = store.get(STORE.key) || "";
    session.model = store.get(STORE.model) || "";
  }

  function saveKey(remember) {
    if (remember) {
      store.set(STORE.provider, session.provider);
      store.set(STORE.key, session.key);
      store.set(STORE.model, session.model);
    } else {
      store.del(STORE.provider);
      store.del(STORE.key);
      store.del(STORE.model);
    }
  }

  function authHeaders() {
    return { "X-Provider": session.provider, "X-Provider-Key": session.key };
  }

  function selectProvider(name) {
    session.provider = name;
    Array.prototype.forEach.call(el("provider-group").querySelectorAll(".segment"), function (b) {
      var on = b.dataset.provider === name;
      b.classList.toggle("is-selected", on);
      b.setAttribute("aria-checked", on ? "true" : "false");
    });
    var hints = { groq: "gsk_…", anthropic: "sk-ant-…", gemini: "AIza…" };
    el("api-key").placeholder = hints[name] || "";
  }

  function showConnected() {
    var connected = Boolean(session.key);
    var pill = el("model-pill");
    pill.hidden = !connected;
    // the provider, not the key test's model: that call uses the small role, answers do not
    var label = DEMO ? "Demo" : (PROVIDER_LABELS[session.provider] || session.provider);
    pill.textContent = connected ? label + " · your key" : "";
    el("key-status").textContent = connected ? session.provider + " · connected" : "not connected";
    el("key-status").classList.toggle("is-on", connected);
    el("connect").textContent = connected ? "Replace key" : "Test key and start";
  }

  /* The one key form lives in the modal on first launch and in the sidebar after that. */
  function placeForm(where) {
    var form = el("key-form");
    var slot = where === "modal" ? el("modal-slot") : el("key-slot");
    if (form.parentNode !== slot) { slot.appendChild(form); }
  }

  var keyModal = el("key-modal");

  function openKeyModal() {
    placeForm("modal");
    el("key-error").hidden = true;
    if (!keyModal.open && typeof keyModal.showModal === "function") { keyModal.showModal(); }
    el("api-key").focus();
  }

  function closeKeyModal() {
    if (keyModal.open) { keyModal.close(); }
    placeForm("drawer");
  }

  keyModal.addEventListener("close", function () { placeForm("drawer"); });
  el("key-later").addEventListener("click", closeKeyModal);

  el("provider-group").addEventListener("click", function (ev) {
    var btn = ev.target.closest("[data-provider]");
    if (btn) { selectProvider(btn.dataset.provider); }
  });

  el("key-form").addEventListener("submit", function (ev) {
    ev.preventDefault();
    var input = el("api-key");
    var error = el("key-error");
    var button = el("connect");
    var key = input.value.trim();

    error.hidden = true;
    if (!key) {
      error.textContent = "Paste a key to continue.";
      error.hidden = false;
      input.focus();
      return;
    }

    function done(model) {
      session.key = key;
      session.model = model || "";
      saveKey(el("remember-key").checked);
      input.value = "";
      showConnected();
      closeKeyModal();
      el("question").focus();
      if (pendingQuestion) { var q = pendingQuestion; pendingQuestion = ""; ask(q); }
    }

    if (DEMO) { done("demo"); return; }

    button.disabled = true;
    button.textContent = "Testing the key";
    var tried = session.provider;

    fetch(API + "/api/key/test", {
      method: "POST",
      headers: { "X-Provider": tried, "X-Provider-Key": key }
    })
      .then(function (res) {
        if (res.ok) { return res.json(); }
        if (res.status === 429) {
          return res.json().catch(function () { return {}; }).then(function (body) {
            throw new Error(body.detail || "Too many attempts. Wait a minute and try again.");
          });
        }
        if (res.status === 401 || res.status === 403) {
          throw new Error("That key was rejected by " + tried + ". Check it and paste it again.");
        }
        throw new Error("The server could not reach " + tried + " (status " + res.status + "). Try again in a moment.");
      })
      .then(function (data) { done(data && data.model); })
      .catch(function (err) {
        error.textContent = err.message || "The key could not be checked.";
        error.hidden = false;
      })
      .finally(function () {
        button.disabled = false;
        showConnected();
      });
  });

  /* ── sidebar ──────────────────────────────────────────────────────────── */

  var drawer = el("drawer");
  var scrim = el("scrim");
  var lastFocus = null;

  function openDrawer(section) {
    var target = el("section-" + section);
    if (target) { target.open = true; }
    if (drawer.classList.contains("is-open")) {
      if (target) { target.scrollIntoView({ block: "nearest" }); }
      return;
    }
    lastFocus = document.activeElement;
    scrim.hidden = false;
    drawer.inert = false;
    drawer.setAttribute("aria-hidden", "false");
    // next frame, so the closed transform is painted before the open one is applied
    requestAnimationFrame(function () {
      drawer.classList.add("is-open");
      scrim.classList.add("is-open");
    });
    if (target) { target.querySelector("summary").focus({ preventScroll: true }); }
  }

  function closeDrawer() {
    if (!drawer.classList.contains("is-open")) { return; }
    drawer.classList.remove("is-open");
    scrim.classList.remove("is-open");
    drawer.inert = true;
    drawer.setAttribute("aria-hidden", "true");
    setTimeout(function () { if (!drawer.classList.contains("is-open")) { scrim.hidden = true; } }, 400);
    if (lastFocus && lastFocus.focus) { lastFocus.focus({ preventScroll: true }); }
  }

  document.addEventListener("click", function (ev) {
    var opener = ev.target.closest("[data-open]");
    if (opener) { openDrawer(opener.dataset.open); }
  });
  el("drawer-close").addEventListener("click", closeDrawer);
  scrim.addEventListener("click", closeDrawer);
  document.addEventListener("keydown", function (ev) {
    if (ev.key === "Escape" && drawer.classList.contains("is-open")) { closeDrawer(); }
  });

  /* ── session cache list ───────────────────────────────────────────────── */

  var answered = [];
  try { answered = JSON.parse(store.get(STORE.answered) || "[]") || []; } catch (e) { answered = []; }

  function recordAnswer(question, payload) {
    var trace = payload.trace || {};
    var cache = (trace.metrics && trace.metrics.cache) || {};
    var tier = trace.cache_tier || cache.tier || "miss";
    answered.unshift({
      q: question,
      tier: tier === "L1" || tier === "L2" ? tier : "miss",
      sim: typeof cache.similarity === "number" ? cache.similarity : null,
      written: Boolean(cache.written),
      value: payload.headline && payload.headline.value ? payload.headline.value : "",
      t: Date.now()
    });
    answered = answered.slice(0, 40);
    store.set(STORE.answered, JSON.stringify(answered));
    renderHistory();
  }

  function renderHistory() {
    var list = el("cache-list");
    list.innerHTML = "";
    el("cache-empty").hidden = answered.length > 0;
    el("cache-count").textContent = String(answered.length);
    var count = el("rail-count");
    count.hidden = answered.length === 0;
    count.textContent = answered.length > 9 ? "9+" : String(answered.length);

    answered.forEach(function (h) {
      var li = make("li");
      var b = make("button", "cache-item");
      b.type = "button";
      b.title = "Ask this again";
      b.appendChild(make("span", "cache-item__q", h.q));
      var meta = make("span", "cache-item__meta");
      var badge;
      if (h.tier !== "miss") {
        badge = make("span", "badge badge--hit", "Served from cache · " + h.tier);
      } else if (h.written) {
        badge = make("span", "badge badge--saved", "Saved to cache");
      } else {
        badge = make("span", "badge", "Not cached");
      }
      meta.appendChild(badge);
      if (h.sim !== null && h.tier !== "miss") { meta.appendChild(make("span", "num", h.sim.toFixed(3) + " match")); }
      if (h.value) { meta.appendChild(make("span", "num", h.value)); }
      b.appendChild(meta);
      b.addEventListener("click", function () { closeDrawer(); ask(h.q); });
      li.appendChild(b);
      list.appendChild(li);
    });
  }

  /* ── asking ───────────────────────────────────────────────────────────── */

  var STAGES = [
    "Checking answered questions",
    "Reading the question",
    "Searching the filing",
    "Trimming the context",
    "Computing the figures",
    "Writing the explanation"
  ];

  var thread = el("thread");
  var questionBox = el("question");
  var pendingQuestion = "";
  var busy = false;
  var turns = [];               // {question, payload} per answered turn
  var current = -1;             // the turn the floating bubble explains: the answer most in view
  var visible = {};

  var inView = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) { visible[e.target.dataset.turn] = e.intersectionRatio; });
    var best = -1, ratio = 0;
    Object.keys(visible).forEach(function (k) {
      if (visible[k] > ratio || (visible[k] === ratio && Number(k) > best)) { best = Number(k); ratio = visible[k]; }
    });
    if (best >= 0 && ratio > 0) { current = best; }
  }, { root: thread, threshold: [0, 0.25, 0.5, 0.75, 1] });

  function scrollToEnd() {
    thread.scrollTo({ top: thread.scrollHeight, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  }

  function newTurn(question) {
    el("welcome").hidden = true;
    el("new-chat").hidden = false;
    var node = el("turn-tpl").content.firstElementChild.cloneNode(true);
    node.querySelector(".msg-user").textContent = question;
    thread.appendChild(node);
    return node;
  }

  function runProgress(turn) {
    var list = turn.querySelector(".progress");
    STAGES.forEach(function (label, i) {
      var li = make("li");
      li.dataset.state = i === 0 ? "active" : "waiting";
      li.style.setProperty("--i", i);
      li.appendChild(make("span", "dot"));
      li.appendChild(document.createTextNode(label));
      list.appendChild(li);
    });
    var timers = [];
    STAGES.forEach(function (_, i) {
      if (i === 0) { return; }
      timers.push(setTimeout(function () {
        var items = list.children;
        for (var j = 0; j < i; j++) { items[j].dataset.state = "done"; }
        items[i].dataset.state = "active";
      }, i * 900));
    });
    // a fresh answer takes 10-15 s on the free instance, so only a wait well past that suggests
    // the instance was asleep (waking takes up to a minute)
    timers.push(setTimeout(function () { turn.querySelector(".working__warm").hidden = false; }, 20000));
    return function stop() { timers.forEach(clearTimeout); };
  }

  /* ── in-session memory ────────────────────────────────────────────────
     The conversation lives in this tab only (sessionStorage): the server keeps nothing. Each
     question carries the last turns in compact form, so a follow-up such as "and gross
     profit?" can be read against them. A reload keeps the thread; New chat forgets it. */

  var MEMORY_TURNS = 10;      // sent with a question; the server caps them again (app.yaml)
  var THREAD_KEEP = 30;       // kept in the tab to redraw the thread after a reload

  function storedThread() {
    try { return JSON.parse(store.get(STORE.thread) || "[]") || []; } catch (e) { return []; }
  }

  function rememberTurn(question, payload) {
    var copy = Object.assign({}, payload);
    delete copy.debug;                       // dev-only and large; the page never needs it back
    var thread = storedThread();
    thread.push({ question: question, payload: copy });
    store.set(STORE.thread, JSON.stringify(thread.slice(-THREAD_KEEP)));
  }

  function summaryOf(payload) {
    var text = (payload.answer_markdown || "")
      .replace(/\[[CK]\d+(?:\s*,\s*[CK]\d+)*\]/g, "")
      .replace(/[*_`#>|]/g, "")
      .replace(/\s+/g, " ")
      .trim();
    var head = payload.headline;
    var lead = head && head.value ? head.value + " (" + (head.caption || "") + "). " : "";
    return (lead + text).slice(0, 280);
  }

  function conversation() {
    return storedThread().slice(-MEMORY_TURNS).map(function (t) {
      return {
        question: t.question,
        standalone: t.payload.standalone_question || null,
        summary: summaryOf(t.payload)
      };
    });
  }

  function newChat() {
    if (busy) { return; }
    store.del(STORE.thread);
    Array.prototype.forEach.call(thread.querySelectorAll(".turn"), function (n) { n.remove(); });
    inView.disconnect();
    turns = [];
    current = -1;
    visible = {};
    bubble.classList.remove("is-shown");
    bubble.hidden = true;
    el("welcome").hidden = false;
    el("new-chat").hidden = true;
    questionBox.value = "";
    autosize();
    questionBox.focus();
  }

  function restoreThread() {
    var saved = storedThread();
    if (!saved.length) { return; }
    thread.classList.add("is-restoring");    // redraw without the entrance motion
    saved.forEach(function (t) { renderAnswer(newTurn(t.question), t.question, t.payload); });
    thread.scrollTop = thread.scrollHeight;
    requestAnimationFrame(function () { thread.classList.remove("is-restoring"); });
  }

  function ask(question) {
    question = (question || "").trim();
    if (!question || busy) { return; }
    if (!session.key && !DEMO) {
      pendingQuestion = question;
      openKeyModal();
      return;
    }
    busy = true;
    el("ask").disabled = true;
    questionBox.value = "";
    autosize();

    var turn = newTurn(question);
    var stop = runProgress(turn);
    scrollToEnd();

    var request = DEMO
      ? fetch("./demo-answer.json").then(function (r) { return r.json(); })
      : fetch(API + "/api/ask", {
          method: "POST",
          headers: Object.assign({ "Content-Type": "application/json" }, authHeaders()),
          body: JSON.stringify({ question: question, history: conversation() })
        }).then(handleResponse);

    request
      .then(function (payload) {
        stop();
        renderAnswer(turn, question, payload);
        recordAnswer(question, payload);
        // a retrieval-only view (model failed, or the key's daily limit) is not a turn a
        // follow-up should build on
        if (!payload.degraded) { rememberTurn(question, payload); }
      })
      .catch(function (err) { stop(); renderError(turn, question, err); })
      .finally(function () {
        busy = false;
        el("ask").disabled = false;
        questionBox.focus({ preventScroll: true });
      });
  }

  function handleResponse(res) {
    if (res.ok) { return res.json(); }
    var err = new Error();
    err.status = res.status;
    return res.json().then(function (body) {
      err.detail = body && (body.detail || body.message);
      if (typeof err.detail !== "string") { err.detail = ""; }
      // "rate" | "busy" when this server refused it; "daily" when the key's daily quota is spent
      err.limit = (body && body.limit) || res.headers.get("X-Limit");
      throw err;
    }, function () { throw err; });
  }

  function autosize() {
    questionBox.style.height = "auto";
    questionBox.style.height = Math.min(questionBox.scrollHeight, 160) + "px";
  }

  questionBox.addEventListener("input", autosize);
  questionBox.addEventListener("keydown", function (ev) {
    if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) {
      ev.preventDefault();
      ask(questionBox.value);
    }
  });
  el("ask-form").addEventListener("submit", function (ev) { ev.preventDefault(); ask(questionBox.value); });
  el("suggestions").addEventListener("click", function (ev) {
    var chip = ev.target.closest(".chip");
    if (chip) { ask(chip.textContent); }
  });

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

  /* The model cites context ids inline ("[C1] [C3]"). Each becomes a small page marker that
     opens the page it points to; an id with no page (the arithmetic, say) is dropped. */
  var MARKERS = /(?:\s*\[[CK]\d+(?:\s*,\s*[CK]\d+)*\])+/g;

  function linkMarkers(markdown, citations) {
    var pageOf = {};
    (citations || []).forEach(function (c) {
      (c.refs || []).forEach(function (r) { pageOf[r] = c.page; });
    });
    return (markdown || "").replace(MARKERS, function (run) {
      var pages = [];
      (run.match(/[CK]\d+/g) || []).forEach(function (id) {
        var page = pageOf[id];
        if (page && pages.indexOf(page) === -1) { pages.push(page); }
      });
      return pages.length
        ? " " + pages.map(function (n) {
            return '<a class="ref" href="#page-' + n + '" data-page="' + n + '" title="Open page ' + n + '">p.' + n + "</a>";
          }).join(" ")
        : "";
    });
  }

  thread.addEventListener("click", function (ev) {
    var ref = ev.target.closest(".ref");
    if (!ref) { return; }
    ev.preventDefault();
    var n = ref.dataset.page;
    openImage(API + "/api/pages/" + n, "Page " + n, "Page " + n + " of the filing");
  });

  function quotaMessage(payload) {
    return payload.quota && payload.quota.limit === "daily" ? payload.quota.message : "";
  }

  function noteFor(payload) {
    if (quotaMessage(payload)) { return quotaMessage(payload); }
    if (payload.incomplete && payload.incomplete.length) {
      return "This answer does not cover every part of your question. Not answered: " + payload.incomplete.join("; ") + ". Try asking for it on its own.";
    }
    if (payload.degraded) {
      return "The model could not be reached, so this shows the cited passages and computed figures without a written explanation.";
    }
    if (payload.confidence === "low") {
      return "Low confidence: not every figure could be traced back to a cited passage. Check the sources before using this.";
    }
    return payload.disclaimer || "";
  }

  function renderAnswer(turn, question, payload) {
    var card = turn.querySelector(".msg-agent");
    card.innerHTML = "";
    card.classList.add("is-answer");

    var head = payload.headline;
    if (head && head.value) {
      var h = make("header", "answer__head reveal");
      h.appendChild(make("p", "answer__figure num", head.value));
      var meta = make("div", "answer__meta");
      if (head.caption) { meta.appendChild(make("p", "answer__caption", head.caption)); }
      if (head.working) { meta.appendChild(make("p", "answer__working num", head.working)); }
      h.appendChild(meta);
      card.appendChild(h);
    }

    var prose = make("div", "prose reveal");
    var markdown = payload.answer_markdown || "";
    if (quotaMessage(payload)) { markdown = markdown.replace(quotaMessage(payload), "").trim(); }
    prose.innerHTML = toHtml(linkMarkers(markdown, payload.citations));
    card.appendChild(prose);

    (payload.figures || []).forEach(function (f) { card.appendChild(figureNode(f)); });

    var used = payload.figures_used || [];
    if (used.length) {
      var details = make("details", "used reveal");
      details.appendChild(make("summary", null, "Figures used (" + used.length + ")"));
      var table = make("table", "used__table");
      var body = make("tbody");
      used.forEach(function (f) {
        var tr = make("tr");
        tr.appendChild(make("td", null, f.label));
        tr.appendChild(make("td", "num", f.value + (f.unit ? " " + f.unit : "")));
        body.appendChild(tr);
      });
      table.appendChild(body);
      details.appendChild(table);
      card.appendChild(details);
    }

    var note = noteFor(payload);
    if (note) {
      var warn = payload.degraded || payload.confidence === "low" || (payload.incomplete && payload.incomplete.length) || quotaMessage(payload);
      var n = make("p", "disclaimer reveal" + (warn ? " disclaimer--warn" : ""), note);
      card.appendChild(n);
    }

    var foot = make("footer", "answer__foot reveal");
    var cites = payload.citations || [];
    if (cites.length) {
      var wrap = make("div", "citations");
      wrap.appendChild(make("span", "citations__label", "Read from"));
      cites.forEach(function (c) {
        var b = make("button", "citation");
        b.type = "button";
        b.appendChild(document.createTextNode(c.label));
        if (c.page) {
          b.appendChild(make("span", "num", "p." + c.page));
          b.title = "Open page " + c.page + " of the filing";
          b.addEventListener("click", function () {
            openImage(API + "/api/pages/" + c.page, c.label + ", page " + c.page, "Page " + c.page + " of the filing");
          });
        }
        wrap.appendChild(b);
      });
      foot.appendChild(wrap);
    }
    if (foot.childNodes.length) { card.appendChild(foot); }

    card.dataset.turn = String(turns.length);
    turns.push({ question: question, payload: payload });
    current = turns.length - 1;
    inView.observe(card);
    showBubble();
    scrollToEnd();
  }

  function figureNode(f) {
    var fig = make("figure", "answer__image reveal");
    fig.hidden = true;                       // shown once the crop has actually loaded
    var frame = make("button", "answer__image-frame");
    frame.type = "button";
    frame.title = "Open the figure";
    var img = make("img");
    img.alt = f.caption || "Figure from the filing";
    img.decoding = "async";
    img.addEventListener("load", function () { fig.hidden = false; requestAnimationFrame(function () { fig.classList.add("is-loaded"); }); });
    img.addEventListener("error", function () { fig.remove(); });
    img.src = API + f.url;
    frame.appendChild(img);
    frame.addEventListener("click", function () {
      openImage(API + f.url, (f.caption || "Figure") + (f.page ? ", page " + f.page : ""), img.alt);
    });
    fig.appendChild(frame);
    var cap = make("figcaption");
    cap.appendChild(document.createTextNode(f.caption || "Figure from the filing"));
    if (f.page) {
      var open = make("button", "figure-page", "p." + f.page);
      open.type = "button";
      open.title = "Open page " + f.page + " of the filing";
      open.addEventListener("click", function () {
        openImage(API + "/api/pages/" + f.page, "Page " + f.page, "Page " + f.page + " of the filing");
      });
      cap.appendChild(open);
    }
    fig.appendChild(cap);
    return fig;
  }

  function renderError(turn, question, err) {
    var title = "Something went wrong";
    var body = err.detail || "The request did not complete. Try again.";
    var offerKey = false;

    if (err.limit === "daily") {
      title = "Daily limit reached";
      body = err.detail;
      offerKey = true;                       // "Change key" opens the provider choice
    } else if (err.limit === "rate") {
      title = "One moment";
      body = err.detail;
    } else if (err.limit === "busy") {
      title = "The demo is busy";
      body = err.detail;
    } else if (err.status === 401 || err.status === 403) {
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
      body = err.detail || "Free hosting puts it to sleep after a quiet spell. Give it a few seconds and try again.";
    } else if (!err.status) {
      title = "No connection";
      body = "The page could not reach the server. Check your network and try again.";
    }

    var card = turn.querySelector(".msg-agent");
    card.innerHTML = "";
    card.classList.add("is-error");
    card.setAttribute("role", "alert");
    card.appendChild(make("p", "error__title", title));
    card.appendChild(make("p", "muted", body));
    var actions = make("div", "error__actions");
    var retry = make("button", "button button--quiet", "Try again");
    retry.type = "button";
    retry.addEventListener("click", function () { turn.remove(); ask(question); });
    actions.appendChild(retry);
    if (offerKey) {
      var key = make("button", "button button--quiet", "Change key");
      key.type = "button";
      key.addEventListener("click", function () { openKeyModal(); });
      actions.appendChild(key);
    }
    card.appendChild(actions);
    scrollToEnd();
  }

  /* ── how the agent answered ───────────────────────────────────────────── */

  var bubble = el("how-bubble");
  var howDialog = el("how-dialog");

  function showBubble() {
    if (!bubble.hidden) { return; }
    bubble.hidden = false;
    requestAnimationFrame(function () { bubble.classList.add("is-shown"); });
  }

  bubble.addEventListener("click", function () {
    var t = turns[current];
    if (t) { openHow(t.question, t.payload); }
  });

  function fmt(n, digits) {
    return typeof n === "number" && isFinite(n) ? n.toFixed(digits) : "–";
  }

  function metric(label, value, sub, tone) {
    var d = make("div", "metric" + (tone ? " metric--" + tone : ""));
    d.appendChild(make("p", "metric__label", label));
    d.appendChild(make("p", "metric__value num", value));
    if (sub) { d.appendChild(make("p", "metric__sub", sub)); }
    return d;
  }

  function openHow(question, payload) {
    var trace = payload.trace || {};
    var m = trace.metrics || {};
    var cache = m.cache || {};
    var ret = m.retrieval || {};
    var tier = trace.cache_tier || cache.tier || "miss";

    el("how-question").textContent = question;

    var grid = el("how-metrics");
    grid.innerHTML = "";
    grid.appendChild(metric("Latency", trace.total_seconds ? fmt(trace.total_seconds, 2) + "s" : "–",
      trace.model ? trace.model : ""));

    var cacheSim = typeof cache.similarity === "number" ? cache.similarity : cache.best_rejected_similarity;
    var cacheSub = tier === "L1" ? "exact match, no model call"
      : tier === "L2" ? "semantic hit, no model call"
      : typeof cache.best_rejected_similarity === "number" ? "closest entry, below threshold"
      : "no comparable entry";
    if (typeof cache.threshold === "number") { cacheSub += " · needs " + fmt(cache.threshold, 2); }
    grid.appendChild(metric("Similarity to cache", fmt(cacheSim, 3), cacheSub, tier === "miss" ? "" : "good"));

    if (ret.passages) {
      grid.appendChild(metric("Top semantic match", fmt(ret.top_similarity, 3),
        "cosine, best of " + ret.queries + " quer" + (ret.queries === 1 ? "y" : "ies")));
      grid.appendChild(metric("Top fused rank score", fmt(ret.top_rrf, 4), "reciprocal rank fusion"));
    } else {
      var why = tier === "miss" ? "no search ran" : "served from cache, no search ran";
      grid.appendChild(metric("Top semantic match", "skipped", why, "quiet"));
      grid.appendChild(metric("Top fused rank score", "skipped", why, "quiet"));
    }

    var steps = el("how-steps");
    steps.innerHTML = "";
    var list = trace.steps && trace.steps.length ? trace.steps : buildSteps(trace);
    list.forEach(function (s, i) {
      var li = make("li", "step" + (s.kind === "computed" ? " step--computed" : s.kind === "checked" ? " step--checked" : ""));
      li.style.setProperty("--i", i);
      var row = make("div", "step__head");
      row.appendChild(make("span", null, s.label));
      if (s.seconds || s.note) { row.appendChild(make("span", "step__time num", s.note || s.seconds.toFixed(2) + "s")); }
      li.appendChild(row);
      if (s.detail) { li.appendChild(make("p", "step__body", s.detail)); }
      steps.appendChild(li);
    });

    var passages = ret.passages || [];
    el("how-passages-wrap").hidden = passages.length === 0;
    document.querySelector(".how__cols").classList.toggle("is-single", passages.length === 0);
    var tbody = el("how-passages");
    tbody.innerHTML = "";
    passages.forEach(function (p) {
      var tr = make("tr");
      var label = make("td");
      label.appendChild(make("span", "passage__label", p.label));
      label.appendChild(make("span", "passage__page num", "p." + p.page + (p.modality && p.modality !== "text" ? " · " + p.modality.replace("_", " ") : "")));
      tr.appendChild(label);
      tr.appendChild(make("td", "num", fmt(p.similarity, 3)));
      tr.appendChild(make("td", "num", fmt(p.rrf, 4)));
      tbody.appendChild(tr);
    });

    var foot = [];
    if (trace.tokens) { foot.push(trace.tokens["in"] + " tokens in, " + trace.tokens.out + " out, billed to your key"); }
    if (m.context_tokens) { foot.push(m.context_tokens + " tokens of context"); }
    if (cache.written) { foot.push("saved to the answer cache"); }
    el("how-foot").textContent = foot.join(" · ");
    el("how-foot").hidden = foot.length === 0;

    if (!howDialog.open && typeof howDialog.showModal === "function") { howDialog.showModal(); }
  }

  function buildSteps(trace) {
    var steps = [];
    var tier = trace.cache_tier || "miss";
    steps.push({ label: "Checked the cache", note: tier === "miss" ? "no match" : tier + " hit" });
    if (trace.slots) {
      steps.push({ label: "Read the question", detail: Object.keys(trace.slots).map(function (k) { return k + ": " + trace.slots[k]; }).join(", ") });
    }
    if (trace.retrieval) { steps.push({ label: "Searched the filing", seconds: trace.retrieval.seconds, detail: trace.retrieval.detail }); }
    if (trace.compression) { steps.push({ label: "Trimmed the context", seconds: trace.compression.seconds, detail: trace.compression.detail }); }
    if (trace.calculator) { steps.push({ label: "Computed the ratio", note: "exact", detail: trace.calculator.detail, kind: "computed" }); }
    if (trace.generation) { steps.push({ label: "Wrote the answer", seconds: trace.generation.seconds, detail: trace.generation.detail }); }
    if (trace.verified !== undefined) {
      steps.push({
        label: trace.verified ? "Verified every number" : "Check incomplete",
        kind: "checked",
        detail: trace.verified ? "every number traced to the filing" : "at least one figure could not be traced"
      });
    }
    return steps;
  }

  el("how-close").addEventListener("click", function () { howDialog.close(); });

  /* ── page / figure viewer ─────────────────────────────────────────────── */

  var viewer = el("viewer");

  function openImage(src, title, alt) {
    el("viewer-title").textContent = title;
    var img = el("viewer-img");
    img.hidden = true;                       // never show a broken-image box
    img.onload = function () { img.hidden = false; };
    img.onerror = function () { el("viewer-title").textContent = title + " could not be loaded"; };
    img.src = src;
    img.alt = alt;
    if (typeof viewer.showModal === "function") { viewer.showModal(); }
  }

  el("viewer-close").addEventListener("click", function () { viewer.close(); });

  // a click on the backdrop closes any of the dialogs
  [viewer, howDialog, keyModal].forEach(function (d) {
    d.addEventListener("click", function (ev) { if (ev.target === d) { d.close(); } });
  });

  /* ── the filing itself, shown in "About" only if the page image is there ── */

  var specimen = el("specimen");
  var specimenImg = el("specimen-img");
  if (specimenImg.complete && specimenImg.naturalWidth > 0) {
    specimen.hidden = false;
  } else {
    specimenImg.addEventListener("load", function () { specimen.hidden = false; });
    specimenImg.addEventListener("error", function () { specimen.remove(); });
  }

  /* ── start ────────────────────────────────────────────────────────────── */

  el("new-chat").addEventListener("click", newChat);

  loadKey();
  selectProvider(session.provider);
  showConnected();
  renderHistory();
  restoreThread();
  placeForm("drawer");
  if (session.key) {
    questionBox.focus();
  } else {
    openKeyModal();
  }
})();
