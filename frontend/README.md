# frontend/

Static single-page interface served by FastAPI. No build step, no framework, no package manager.

## Layout

- **Chat** is the whole page: suggestions on first load, then a thread of questions and answers,
  with the composer pinned at the bottom. An answer shows its headline figure, the prose with
  inline page markers (`p.141`), any figure crop it drew on (under the prose, click to enlarge),
  the figures used (collapsed), and "Read from" chips that open the cited page.
- **Sidebar**, collapsed to a thin rail of icons; it expands as a drawer with three sections:
  *Model key* (provider, key, replace), *Cache this session* (every answer in this tab, and
  whether it came from the answer cache or was saved to it; click to ask again) and *About this
  build* (the evaluation numbers and the filing specimen).
- **Key modal** on first launch. "Look around first" dismisses it; asking without a key reopens it.
- **"How the agent answered this"**, a floating bubble that opens the method for the answer most
  in view: latency, similarity to the cache, top semantic (cosine) match, top RRF score, the
  pipeline steps with timings, and the top passages retrieved.

Light theme only. Motion uses the curves in `app.css` (`--ease-out`, `--ease-drawer`) and every
animation has a `prefers-reduced-motion` fallback that keeps fades and drops movement.

```
frontend/
├── index.html
├── app.css
├── app.js
├── demo-answer.json      # canned response for demo mode
├── fonts/                # self-hosted faces, committed
│   ├── Archivo-Variable.woff2          35 KB, OFL
│   └── SplineSansMono-Variable.woff2   36 KB, OFL
└── vendor/
    ├── marked.min.js     # download once, commit
    └── purify.min.js     # download once, commit
```

## Serving it

```python
app.mount("/", Frontend(directory="frontend", html=True), name="frontend")
```

Mount it **after** the API routes so `/api/*` is matched first. Same origin means no CORS
configuration. `Frontend` is a thin `StaticFiles` subclass in `rag/api/main.py` that pins the
web-font media types: `mimetypes` has no `.woff2` entry on a stock Windows install, so the faces
would otherwise be served as `application/octet-stream`.

## Typefaces

Archivo for language, Spline Sans Mono for anything measured. Both are variable `woff2`, both
SIL Open Font License (licence text sits beside them), and both are **self-hosted**: the page
makes no third-party request at runtime, which is the point on a screen that asks for an API
key. Loading them from Google Fonts would break that promise for the sake of two files.

```bash
mkdir -p frontend/fonts
curl -Lo frontend/fonts/Archivo-Variable.woff2   https://cdn.jsdelivr.net/npm/@fontsource-variable/archivo/files/archivo-latin-wght-normal.woff2
curl -Lo frontend/fonts/SplineSansMono-Variable.woff2   https://cdn.jsdelivr.net/npm/@fontsource-variable/spline-sans-mono/files/spline-sans-mono-latin-wght-normal.woff2
```

## Vendored libraries

`marked` renders the answer Markdown and `DOMPurify` sanitises the result before it reaches the DOM — model output is untrusted, because the source PDF could contain injected text.

```bash
mkdir -p frontend/vendor
curl -Lo frontend/vendor/marked.min.js  https://cdn.jsdelivr.net/npm/marked/marked.min.js
curl -Lo frontend/vendor/purify.min.js  https://cdn.jsdelivr.net/npm/dompurify/dist/purify.min.js
```

Pin the versions you download and commit both files, so the page works offline and makes no third-party calls at runtime. If either script is missing, `app.js` falls back to escaped plain-text paragraphs rather than rendering unsanitised HTML.

## API contract

| Endpoint | Headers | Body | Returns |
|---|---|---|---|
| `POST /api/key/test` | `X-Provider`, `X-Provider-Key` | — | `200 {"model": "…"}` |
| `POST /api/ask` | `X-Provider`, `X-Provider-Key` | `{"question": "…"}` | `200` answer payload |
| `GET /api/pages/{n}` | — | — | page image |

Answer payload — only `answer_markdown` is required; every other field degrades gracefully:

```jsonc
{
  "answer_markdown": "…",
  "headline":  {"value": "3.91",
                "caption": "current ratio, as of 25 January 2026 (FY2026)",
                "working": "125,605 / 32,163 — USD millions"},
  "figures_used": [{"label": "Total current assets", "value": "125,605", "unit": ""}],
  "citations":    [{"label": "Consolidated Balance Sheets", "page": 141,
                    "refs": ["C1", "K1"]}],   // the inline [C1] markers that point here
  "figures":      [{"url": "/api/figures/p3_0", "caption": "AI Is a Five-Layer Cake", "page": 3}],
  "confidence":   "high",          // low → a caution line appears above the citations
  "degraded":     false,           // true → retrieval-only notice
  "incomplete":   ["net income: change FY2025 to FY2026"],  // parts not answered → caution line
  "disclaimer":   "…",             // optional, shown when confidence is not low
  "trace": {
    "cache_tier": "miss",          // or "L1" / "L2"
    "model": "llama-3.3-70b",
    "total_seconds": 4.8,
    "verified": true,
    "tokens": {"in": 2140, "out": 260},
    "slots": {"entity": "…", "period": "FY2026", "formula": "current_ratio"},
    "retrieval":   {"seconds": 0.31, "detail": "…"},
    "compression": {"seconds": 0.02, "detail": "…"},
    "calculator":  {"detail": "current_ratio = 125,605 / 32,163"},
    "generation":  {"seconds": 4.3,  "detail": "…"},
    "metrics": {
      "cache": {"tier": "miss", "similarity": null, "threshold": 0.9,
                "best_rejected_similarity": 0.71, "lookup_ms": 30, "written": true},
      "retrieval": {"queries": 2, "top_similarity": 0.874, "top_rrf": 0.0325,
                    "passages": [{"label": "Total current assets", "page": 141,
                                  "modality": "row_fact", "similarity": 0.874, "rrf": 0.0325}]},
      "context_tokens": 1620
    }
  }
}
```

`figures` lists at most two crops, from the strongest tier that has any: figures the answer cited,
else figures on a page it cited, else figures attached to the model as images. A crop that fails
to load removes itself rather than leaving a broken box. `metrics.retrieval` is absent on a cache
hit (no search ran) and the panel says so.

The method panel prefers `trace.steps` if you send it:

```jsonc
"steps": [{"label": "…", "detail": "…", "seconds": 0.3, "tags": ["FY2026"],
           "kind": "normal" | "computed" | "checked"}]
```

Otherwise it builds the step list from the fields above, so partial traces still render.

### Error codes the UI already handles

| Status | What the user sees |
|---|---|
| 401 / 403 | Key rejected, with a button back to the key screen |
| 402 | Key out of credit |
| 429 | Provider rate limit, wait a minute |
| 502 / 503 / 504 | Server still waking up |
| network failure | No connection |

Send a plain-text reason in `{"detail": "…"}` and it replaces the default body text.

## Key handling

The key is read from the form, held in `sessionStorage` when "keep it for this browser tab only" is ticked, and sent as `X-Provider-Key` on each request. It is never placed in a URL. On the server, keep it in memory for the duration of the request and make sure it is redacted from the trace log, the usage ledger and the answer cache.

## Demo mode

Append `?demo=1` to the URL. Any key is accepted without a network call, and questions resolve from `demo-answer.json`. Useful for:

- recording a GIF for the repository README without spending tokens,
- publishing a clickable demo on GitHub Pages (the static files work on their own; the page viewer will 404, which is expected).

To record the GIF: open `?demo=1`, ask one question, expand a citation, and keep it under 15 seconds and 5 MB so GitHub renders it inline.
