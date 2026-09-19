"""Dev-mode admin endpoints (architecture §14.3, plan Phase 6): cache stats / entries / purge and
the dev clock. Mounted by `main.py` only when `app.yaml` `env` is in `dev_clock.enabled_in`.

    GET  /api/admin/stats           ops panel: cache hit rate over time, tokens by role, failures
    GET  /api/admin/cache/stats
    GET  /api/admin/cache/entries?offset=0&limit=50
    POST /api/admin/cache/purge     {"scope": all | class | slot | stale | l1, "value": "..."}
    POST /api/admin/cache/sweep
    GET  /api/admin/clock
    POST /api/admin/clock           {"offset_seconds": N} | {"advance_seconds": N} | {"reset": true}
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from rag.api.ops import ops_stats
from rag.core.clock import ClockError

router = APIRouter(prefix="/api/admin")


def _pipeline(request: Request):  # noqa: ANN202 - Pipeline
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline not ready")
    return pipeline


class PurgeRequest(BaseModel):
    scope: Literal["all", "class", "slot", "stale", "l1"] = "all"
    value: str | None = Field(default=None, description="TTL class, or slot `field=value`")


class ClockRequest(BaseModel):
    offset_seconds: int | None = None
    advance_seconds: int | None = None
    reset: bool = False


def _clock_state(pipeline) -> dict[str, Any]:  # noqa: ANN001
    c = pipeline.clock
    return {
        "offset_s": c.offset_s,
        "offset_days": round(c.offset_s / 86_400, 2),
        "now": c.now(),
        "now_iso": c.now_iso(),
        "allow_offset": c.allow_offset,
    }


@router.get("/stats")
def stats(request: Request, limit: int = Query(default=500, ge=1, le=5000)) -> dict[str, Any]:
    """Aggregate of the last `limit` trace lines plus the usage ledger and decision log."""
    pipeline = _pipeline(request)
    settings = pipeline.settings
    out = ops_stats(
        traces_path=pipeline.trace_writer.path,
        ledger_path=pipeline.ledger.path,
        decisions_path=pipeline.decision_log.path,
        limit=limit,
    )
    out["profile"] = pipeline.models.active_profile
    out["corpus_version"] = pipeline.store.corpus_version
    out["app_env"] = settings.app_env
    out["cache_live"] = pipeline.cache.stats() if pipeline.cache_enabled else None
    return out


@router.get("/cache/stats")
def cache_stats(request: Request) -> dict[str, Any]:
    pipeline = _pipeline(request)
    stats = pipeline.cache.stats()
    stats["enabled"] = pipeline.cache_enabled
    return stats


@router.get("/cache/entries")
def cache_entries(
    request: Request,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    pipeline = _pipeline(request)
    rows, total = pipeline.cache.l2.entries(offset=offset, limit=limit)
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "now": pipeline.clock.now(),
        "entries": [r.model_dump() for r in rows],
    }


@router.post("/cache/purge")
def cache_purge(body: PurgeRequest, request: Request) -> dict[str, Any]:
    pipeline = _pipeline(request)
    try:
        removed = pipeline.cache.purge(body.scope, body.value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "scope": body.scope,
        "value": body.value,
        **removed,
        "remaining": pipeline.cache.l2.count(),
    }


@router.post("/cache/sweep")
def cache_sweep(request: Request) -> dict[str, Any]:
    pipeline = _pipeline(request)
    return pipeline.cache.sweeper.sweep()


@router.get("/clock")
def clock_get(request: Request) -> dict[str, Any]:
    return _clock_state(_pipeline(request))


@router.post("/clock")
def clock_set(body: ClockRequest, request: Request) -> dict[str, Any]:
    pipeline = _pipeline(request)
    try:
        if body.reset:
            pipeline.clock.reset()
        elif body.offset_seconds is not None:
            pipeline.clock.set_offset(body.offset_seconds)
        elif body.advance_seconds is not None:
            pipeline.clock.advance(body.advance_seconds)
        else:
            raise HTTPException(
                status_code=422, detail="give offset_seconds, advance_seconds or reset"
            )
    except ClockError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return _clock_state(pipeline)
