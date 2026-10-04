"""FastAPI app: API + the localhost HTML test page in one process (architecture §8.3, §14).

    make serve   →   uvicorn rag.api.main:app --host 127.0.0.1 --port 8000 --workers 1

Startup runs the Phase 8 checks (`checks.py`: bind address, admin gating, provider keys, index
manifest consistency), then loads the index and the model client once (lifespan). A fatal check
— a non-loopback bind or an inconsistent index — leaves the pipeline unloaded, `/readyz` answers
503 with the failing check, and `/api/ask` answers 503. The server is meant for 127.0.0.1 only
(NFR-12): a request from a routable peer address is refused with 403. Admin and dev-clock routes
(`routes_admin.py`, Phase 6) answer 404 unless `app.yaml` puts the environment in
`dev_clock.enabled_in`; the cache sweeper runs on startup and every
`cache.sweep_interval_seconds` as a lifespan background task (§5.5).
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from rag import __version__
from rag.api.byok import router as byok_router
from rag.api.checks import CheckResult, fatal_failures, is_loopback_client, run_startup_checks
from rag.api.ratelimit import RateLimiter, RateLimitMiddleware
from rag.api.routes_admin import router as admin_router
from rag.api.routes_ask import router as ask_router
from rag.core.config import load_app_config, load_models_config, load_thresholds_config
from rag.core.logging import configure_logging, get_logger
from rag.core.settings import PROJECT_ROOT, SettingsError, get_settings

log = get_logger(__name__)


def _load_pipeline(app: FastAPI) -> None:
    """Load the index once and build the default pipeline over it.

    The registry owns the store, so the per-provider pipelines a BYOK request needs (F2) reuse
    these same vectors instead of loading the index again.
    """
    from rag.api.pipelines import PipelineRegistry

    started = time.perf_counter()
    registry = PipelineRegistry(settings=app.state.settings)
    app.state.registry = registry
    app.state.pipeline = registry.default()
    app.state.load_error = None
    app.state.startup_seconds = round(time.perf_counter() - started, 1)
    log.info(
        "Pipeline ready in %.1fs (provider=%s, byok_only=%s)",
        app.state.startup_seconds,
        registry.default_provider,
        app.state.settings.byok_only,
    )


def _run_checks(app: FastAPI) -> list[CheckResult]:
    """Startup checks (Phase 8). Config errors are reported as a failed check, not a crash."""
    settings = app.state.settings
    try:
        results = run_startup_checks(
            settings,
            app.state.app_config,
            load_models_config(settings=settings),
            load_thresholds_config(settings=settings),
        )
    except (SettingsError, ValueError, OSError) as exc:
        results = [CheckResult("config", False, str(exc)[:300], fatal=True)]
    for r in results:
        (log.info if r.ok else (log.error if r.fatal else log.warning))(
            "startup check %-7s %s  %s", r.name, "ok" if r.ok else "FAIL", r.detail
        )
    return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, secrets=settings.secret_values())
    app_cfg = load_app_config(settings=settings)
    app.state.settings = settings
    app.state.app_config = app_cfg
    app.state.figures_root = settings.data_dir / "index"
    app.state.pipeline = None
    app.state.registry = None
    app.state.load_error = None
    app.state.startup_seconds = None
    app.state.admin_enabled = app_cfg.dev_clock_enabled
    app.state.public_deploy = settings.public_deploy
    limits = app_cfg.server.rate_limit
    app.state.rate_limiter = (
        RateLimiter(limits) if settings.public_deploy and limits.enabled else None
    )
    if app.state.rate_limiter is not None:
        log.info(
            "rate limit on: ask %d/min %d/h, key test %d/min, %d in flight",
            limits.ask_per_minute,
            limits.ask_per_hour,
            limits.key_test_per_minute,
            limits.max_in_flight,
        )
    app.state.checks = _run_checks(app)
    sweep_task: asyncio.Task | None = None
    fatal = fatal_failures(app.state.checks)
    if fatal:
        app.state.load_error = "startup check failed: " + "; ".join(
            f"{r.name}: {r.detail}" for r in fatal
        )
        log.error("Refusing to load the pipeline — %s", app.state.load_error)
    else:
        try:
            _load_pipeline(app)
        except (SettingsError, FileNotFoundError, OSError) as exc:
            app.state.load_error = str(exc)
            log.error("Pipeline failed to load: %s", exc)
    pipeline = app.state.pipeline
    if pipeline is not None and pipeline.cache_enabled:
        try:
            log.info("cache sweep on startup: %s", pipeline.cache.sweeper.sweep())
        except Exception as exc:  # housekeeping must never block startup
            log.warning("startup cache sweep failed: %s", exc)
        sweep_task = asyncio.create_task(
            pipeline.cache.sweeper.run_forever(pipeline.thresholds.cache.sweep_interval_seconds)
        )
    try:
        yield
    finally:
        if sweep_task is not None:
            sweep_task.cancel()


app = FastAPI(
    title="Advanced Multimodal RAG — NVIDIA FY2026 balance sheet analysis",
    version=__version__,
    lifespan=lifespan,
)
app.include_router(ask_router)
app.include_router(byok_router)
app.include_router(admin_router)
app.add_middleware(RateLimitMiddleware)  # a pass-through unless the lifespan installed a limiter

FRONTEND_DIR = PROJECT_ROOT / "frontend"


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"status": "ok", "version": __version__}


@app.middleware("http")
async def _localhost_only(request, call_next):  # noqa: ANN001
    """Two request-time guards (NFR-12): only loopback peers are served, and admin routes
    exist in the app but answer 404 unless the environment enables them."""
    peer = request.client.host if request.client else None
    public = getattr(app.state, "public_deploy", False)
    if not public and not is_loopback_client(peer):
        return JSONResponse(
            status_code=403, content={"detail": "this server answers localhost only (D-44)"}
        )
    if request.url.path.startswith("/api/admin") and not getattr(app.state, "admin_enabled", False):
        return JSONResponse(status_code=404, content={"detail": "admin routes are dev-only"})
    return await call_next(request)


@app.get("/readyz")
def readyz() -> JSONResponse:
    pipeline = getattr(app.state, "pipeline", None)
    checks = [c.as_dict() for c in getattr(app.state, "checks", [])]
    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "error": getattr(app.state, "load_error", None),
                "checks": checks,
            },
        )
    server = app.state.app_config.server
    return JSONResponse(
        content={
            "ready": True,
            "corpus_version": pipeline.store.corpus_version,
            "chunks": len(pipeline.store.chunks),
            "embedder": pipeline.store.manifest.embedder,
            "model_profile": pipeline.models.active_profile,
            "startup_seconds": app.state.startup_seconds,
            "cache_enabled": pipeline.cache_enabled,
            "byok_only": app.state.settings.byok_only,
            "public_deploy": app.state.settings.public_deploy,
            "admin_enabled": getattr(app.state, "admin_enabled", False),
            "clock_offset_s": pipeline.clock.offset_s,
            "limits": {
                "max_question_chars": server.max_question_chars,
                "request_timeout_seconds": server.request_timeout_seconds,
            },
            "checks": checks,
        }
    )


# The page is mounted last, at the root, so `/api/*`, `/healthz` and `/readyz` are matched
# first and the page's own relative asset paths (`./app.js`, `./vendor/…`) resolve. Same
# origin, so no CORS configuration anywhere.
class Frontend(StaticFiles):
    """Static files for the page, with the web-font media types pinned.

    `mimetypes` has no entry for `.woff2` on a stock Windows install, and any library that
    calls `mimetypes.init()` rebuilds its table and drops a registration made earlier, so the
    self-hosted faces went out as `application/octet-stream`. Setting the header here instead
    cannot be undone by anything else in the process.
    """

    MEDIA_TYPES = {".woff2": "font/woff2", ".woff": "font/woff"}

    def file_response(self, full_path, stat_result, scope, status_code: int = 200):  # noqa: ANN001
        response = super().file_response(full_path, stat_result, scope, status_code)
        forced = self.MEDIA_TYPES.get(Path(full_path).suffix.lower())
        if forced and "content-type" in response.headers:
            response.headers["content-type"] = forced
        return response


if (FRONTEND_DIR / "index.html").exists():
    app.mount("/", Frontend(directory=str(FRONTEND_DIR), html=True), name="frontend")


def main() -> None:
    import uvicorn

    cfg = load_app_config()
    uvicorn.run("rag.api.main:app", host=cfg.server.host, port=cfg.server.port, workers=1)


if __name__ == "__main__":
    main()


__all__ = ["app", "Path"]
