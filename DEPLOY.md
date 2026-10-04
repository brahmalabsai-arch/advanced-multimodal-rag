# Deploying the public demo (F2)

The localhost build answers from the key in your `.env`. A public URL cannot: every visitor's
question would be billed to you. This deployment therefore runs **bring-your-own-key** — the
page asks each visitor for a provider key, uses it for that one request, and keeps nothing.

| | Localhost (`make serve`) | Deployed (`deploy/`) |
|---|---|---|
| Key | yours, from `.env` | the visitor's, from a request header |
| `APP_ENV` | `dev` | `prod` — admin and dev-clock routes answer 404 |
| `BYOK_ONLY` | unset | `true` — a request without a key is refused with 401 |
| `PUBLIC_DEPLOY` | unset | `true` — the loopback-only request guard is off |
| Request guard | non-loopback peers get 403 (NFR-12) | the platform's proxy is the boundary |
| Debug payload | included in `/api/ask` | omitted |
| Index | built by `make ingest` | downloaded at image build time |

Startup refuses to serve if `PUBLIC_DEPLOY` is set without `BYOK_ONLY`, or with admin routes
still enabled (`check_public_safety`). That interlock is the one thing standing between a public
URL and strangers spending your quota, so it is fatal rather than a warning.

---

## 1. Publish the index

Render's build cannot run ingestion — Docling and PyTorch are not in the serving image, and the
parse takes ~35 minutes — so the index travels as a release asset.

```bash
.venv/Scripts/python scripts/package_index.py
# index    82.6 MB on disk
# figures kept 16, dropped 27 unreferenced, 3.1 MB saved
# → dist/index.tar.gz  71.6 MB

gh release create v1.1-index dist/index.tar.gz --title "v1.1-index" \
  --notes "Prebuilt index for the deployed demo (corpus_version 747912fc5da6)"
gh release view v1.1-index --json assets -q '.assets[0].url'
```

Without the `gh` CLI: **Releases → Draft a new release**, tag `v1.1-index`, drag
`dist/index.tar.gz` onto the assets box, untick **Set as the latest release** (this is a data
asset, not a release of the software), publish, then copy the asset's link address. A public
repository's asset URL needs no token; a private one does, which is a reason to keep this
repository public or to host the tarball somewhere the build can reach unauthenticated.

Paste that URL into `deploy/render.yaml` as `INDEX_URL`, or set it in the Render dashboard.
Optionally pass `--build-arg INDEX_SHA256=$(sha256sum dist/index.tar.gz | cut -d' ' -f1)` so a
truncated download fails the build instead of the first question.

The packaging step drops the 27 figure crops ingestion discarded (logos, photographs) and
re-encodes the 16 that a chunk can actually cite. Page thumbnails are kept whole — chunks cite
168 of the 175, so pruning them saves nothing. Everything else is copied verbatim, because
`corpus_version` has to recompute at startup or `check_index` fails the deploy.

## 2. Build and test at the memory limit

This is the step that catches the real failure. The free instance is 512 MB and the measured
resting footprint is **373 MB** after three questions, so there is ~138 MB of headroom — enough,
but not enough to guess about.

**You do not need Docker on your machine for any of this.** Render builds the image on its own
infrastructure, and the pre-flight test runs in CI on Linux, which is the platform that actually
matters — a Windows or macOS RSS figure would not tell you whether the container survives.

**Actions → deploy check → Run workflow**, paste the release asset URL. The job builds the image,
runs it capped to the same 512 MB / 0.1 CPU, and fails if the container is killed (exit 137), if
a startup check fails, if an admin route answers, if a question is answered without a key, or if
a key shape appears in the logs. It prints the memory figure at rest and after three questions.

With Docker installed, the same test locally:

```bash
docker build -f deploy/Dockerfile --build-arg INDEX_URL="<release asset url>" -t amr .
docker run --rm -p 8000:8000 --memory=512m --cpus=0.1 amr

curl -s localhost:8000/readyz | python -m json.tool     # every check ok, byok_only true
open http://localhost:8000
```

Where the memory goes, measured on this index (`298 MB` before the web stack loads):

| Component | MB |
|---|---:|
| bge-small ONNX session | 136 |
| chromadb import | 55 |
| Python baseline | 29 |
| fastembed import | 32 |
| Chroma client + reading the vectors | 31 |
| BM25 index | 6 |
| chunks + sentences + vectors | 6 |

If the container exits with **137** it was killed for memory. In order of effect:

1. `--cpus=0.1` with two uvicorn workers doubles the ONNX session — keep `--workers 1`.
2. The sentence embeddings are already memory-mapped and the Chroma read is already reclaimed
   (`store.py`); next is dropping `sentences.jsonl` from the image, which costs the
   sentence-extraction compressor (`compression.mode: never` in `thresholds.yaml` first).
3. Move to the 2 GB instance. Switching the reranker will **not** help — it is off by default
   (D-24) and never loaded.

## 3. Deploy

Render → **New** → **Blueprint** → point it at this repository. `deploy/render.yaml` declares
everything; the only value to supply is `INDEX_URL` if you did not hard-code it. There are no
secrets to set, which is the point.

`autoDeploy` is off: deploys happen when you ask, not on every push to `main`.

## 4. Verify

| Check | Expected |
|---|---|
| `GET /healthz` | `200 {"status":"ok"}` |
| `GET /readyz` | `200`, `ready: true`, `byok_only: true`, `public_deploy: true`, all five checks `ok` |
| **`GET /api/admin/cache/stats`** | **`404`** — admin routes must not exist in production |
| `GET /api/admin/clock` | `404` |
| `POST /api/ask` with no headers | `401`, "Add your model key…" |
| `POST /api/key/test` with a bad key | `401`, "The provider rejected that key." |
| `POST /api/key/test` with a real key | `200 {"model": "...", "provider": "..."}` |
| Ask a question in the page | headline number, citation chips that open page images, method rail |
| **DevTools → Network** | **the key appears only as the `X-Provider-Key` request header** — never in a URL, never in a response body |
| Ask the same question twice | second answer returns in ~40 ms, `trace.cache_tier` is `L1` or `L2` |
| Render logs | no key, no `gsk_…` / `sk-ant-…` / `AIza…` anywhere |

The last two are the ones worth not skipping. The redaction promise is enforced in three places
(`core/logging.py`): the logging handlers mask key shapes, the in-flight key is masked by exact
value through a context variable, and both the trace writer and the usage ledger redact before
writing. `tests/test_byok.py` holds them to it.

## 5. What to know once it is live

- **Cold starts.** The free instance sleeps after 15 idle minutes; the next request waits ~40 s
  while the image wakes and the index loads. The page already shows "The server was asleep" when
  a request takes unusually long.
- **The cache is shared between visitors, deliberately.** That is where the cost saving comes
  from: the second person to ask about the current ratio gets an answer with no model call at
  all. It also means one visitor's question and answer can be served to another. Both are public
  filings, so this is acceptable — but it is a property worth stating on the page rather than
  letting someone discover it. Answers are separated by provider (`generator_model` carries it),
  so a Groq answer is never served to an Anthropic request.
- **There is no rate limit.** A visitor's key pays for their tokens, but your instance pays for
  the CPU, and nothing stops a script from asking a thousand questions. D-43 (demo access key,
  rate limit, daily cap) is still deferred; on the free tier the blast radius is one sleeping
  instance, but this is the first thing to add if the URL gets attention.
- **Disk is ephemeral.** The semantic cache, traces and usage ledger live in the container and
  vanish on redeploy or restart. That is fine for a demo — and it means visitor questions are
  not retained beyond the instance's life.
- **Questions are logged while the instance lives.** `data/logs/traces.jsonl` records each
  question, its slots and its timings (never the key). If you would rather not keep even that,
  point `TraceWriter` at `os.devnull` in the deployed build.

## 6. Rolling back

`autoDeploy: false` means the live service stays where it is until you deploy. Render keeps the
previous image: **Deploys → the last good one → Redeploy**. The index is pinned by the release
tag in `INDEX_URL`, so a rebuild reproduces the same corpus; publish `v1.1-index` rather than
replacing the asset on an existing tag, or old images stop reproducing.
