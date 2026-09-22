# frontend/

Static single-page interface served by FastAPI. No build step, no framework, no package manager.

```
frontend/
├── index.html
├── app.css
├── app.js
├── demo-answer.json      # canned response for demo mode
└── vendor/
    ├── marked.min.js     # download once, commit
    └── purify.min.js     # download once, commit
```

## Serving it

```python
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
```

Mount it **after** the API routes so `/api/*` is matched first. Same origin means no CORS configuration.

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
  "citations":    [{"label": "Consolidated Balance Sheets", "page": 141}],
  "confidence":   "high",          // low → a caution line appears above the citations
  "degraded":     false,           // true → retrieval-only notice
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
    "generation":  {"seconds": 4.3,  "detail": "…"}
  }
}
```

The method rail prefers `trace.steps` if you send it:

```jsonc
"steps": [{"label": "…", "detail": "…", "seconds": 0.3, "tags": ["FY2026"],
           "kind": "normal" | "computed" | "checked"}]
```

Otherwise it builds the rail from the fields above, so partial traces still render.

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
