"""FastAPI app: API + the localhost HTML test page in one process (architecture §8.3, §14).

    make serve   →   uvicorn rag.api.main:app --host 127.0.0.1 --port 8000 --workers 1

Startup loads the index and the model client once (lifespan). `/readyz` reports whether both
are loaded. The server is meant for 127.0.0.1 only; the bind address is asserted at startup
(NFR-12). Admin and dev-clock routes arrive in Phase 6 and are enabled only in dev mode.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from rag import __version__
from rag.api.routes_ask import router as ask_router
from rag.core.config import load_app_config
from rag.core.logging import configure_logging, get_logger
from rag.core.settings import PROJECT_ROOT, SettingsError, get_settings

log = get_logger(__name__)


def _load_pipeline(app: FastAPI) -> None:
    from rag.graph import Pipeline

    started = time.perf_counter()
    app.state.pipeline = Pipeline()
    app.state.load_error = None
    app.state.startup_seconds = round(time.perf_counter() - started, 1)
    log.info("Pipeline ready in %.1fs", app.state.startup_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, secrets=settings.secret_values())
    app_cfg = load_app_config(settings=settings)
    app.state.settings = settings
    app.state.app_config = app_cfg
    app.state.figures_root = settings.data_dir / "index"
    app.state.pipeline = None
    app.state.load_error = None
    app.state.startup_seconds = None
    if app_cfg.server.host not in {"127.0.0.1", "localhost"}:
        log.warning(
            "app.yaml binds to %s — this build is localhost-only (D-44)", app_cfg.server.host
        )
    try:
        _load_pipeline(app)
    except (SettingsError, FileNotFoundError, OSError) as exc:
        app.state.load_error = str(exc)
        log.error("Pipeline failed to load: %s", exc)
    yield


app = FastAPI(
    title="Advanced Multimodal RAG — NVIDIA FY2026 balance sheet analysis",
    version=__version__,
    lifespan=lifespan,
)
app.include_router(ask_router)

FRONTEND_DIR = PROJECT_ROOT / "frontend"


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"status": "ok", "version": __version__}


@app.get("/readyz")
def readyz() -> JSONResponse:
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "error": getattr(app.state, "load_error", None)},
        )
    return JSONResponse(
        content={
            "ready": True,
            "corpus_version": pipeline.store.corpus_version,
            "chunks": len(pipeline.store.chunks),
            "embedder": pipeline.store.manifest.embedder,
            "model_profile": pipeline.models.active_profile,
            "startup_seconds": app.state.startup_seconds,
        }
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


if (FRONTEND_DIR / "app.js").exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def main() -> None:
    import uvicorn

    cfg = load_app_config()
    uvicorn.run("rag.api.main:app", host=cfg.server.host, port=cfg.server.port, workers=1)


if __name__ == "__main__":
    main()


__all__ = ["app", "Path"]
