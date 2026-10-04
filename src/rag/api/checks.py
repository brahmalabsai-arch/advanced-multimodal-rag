"""Startup checks and the loopback guard (Phase 8; architecture §13, §14.3; NFR-12).

Every check is a pure function returning a `CheckResult`, so the same code runs at startup
(`run_startup_checks`, reported by `/readyz`) and in tests without a server. Two of them are
fatal: a bind address that is not loopback (`bind`) and an index whose manifest does not agree
with itself (`index`). The rest are warnings, printed and exposed but never blocking.

    bind      app.yaml `server.host` is 127.0.0.1 / localhost / ::1 (D-44), unless this is the
              deployed container (`PUBLIC_DEPLOY=true`), where the platform is the boundary
    public    a public deployment carries no key of its own (`BYOK_ONLY`) and no admin routes
    admin     admin + dev-clock routes are enabled only when the environment is `dev`
    secrets   the active profile's providers all have a key in `.env` (names only, never values)
    index     manifest present, all sidecars present, `corpus_version` recomputes from the
              manifest's own `pdf_sha256` + `ingestion_config`, `chunks_total` equals the
              number of lines in `chunks.jsonl`, and the embedder alias matches `thresholds.yaml`
"""

from __future__ import annotations

import ipaddress
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rag.core.config import AppConfig, ModelsConfig, ThresholdsConfig
from rag.core.settings import PROVIDER_KEY_FIELDS, Settings

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
REQUIRED_INDEX_FILES = (
    "manifest.json",
    "bm25.pkl",
    "chunks.jsonl",
    "sentences.jsonl",
    "sentence_emb.npy",
)


class StartupCheckError(RuntimeError):
    """A fatal startup check failed; the server must not serve."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    fatal: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def is_loopback_client(host: str | None) -> bool:
    """Request-time guard: a peer with a routable IP address is rejected. Non-IP strings
    (Starlette's in-process `testclient`, a missing peer) are not network peers and pass; a
    real network client always arrives with an IP."""
    if host is None or host in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return True


def check_bind(host: str, *, public: bool = False) -> CheckResult:
    """Loopback unless this is the deployed container, where the platform is the boundary."""
    if public:
        return CheckResult("bind", True, f"server.host={host!r} (public deployment)", fatal=True)
    ok = host in LOOPBACK_HOSTS
    return CheckResult(
        "bind",
        ok,
        f"server.host={host!r}"
        + ("" if ok else " — this build is localhost-only (D-44); use 127.0.0.1"),
        fatal=True,
    )


def check_public_safety(settings: Settings, app_cfg: AppConfig) -> CheckResult:
    """A public server must carry no key of its own and no admin surface (F2).

    Fatal: getting this wrong means strangers spending the owner's model quota, or purging the
    cache and moving the clock from the open internet.
    """
    if not settings.public_deploy:
        return CheckResult("public", True, "localhost build: loopback guard active", fatal=True)
    problems = []
    if not settings.byok_only:
        problems.append(
            "PUBLIC_DEPLOY without BYOK_ONLY — every visitor would spend the server's key"
        )
    if app_cfg.dev_clock_enabled:
        problems.append(f"admin routes enabled in env={app_cfg.env!r}")
    if problems:
        return CheckResult("public", False, "; ".join(problems), fatal=True)
    return CheckResult(
        "public",
        True,
        "public deployment: bring-your-own-key enforced, admin routes disabled",
        fatal=True,
    )


def check_admin_gating(app_cfg: AppConfig) -> CheckResult:
    enabled = app_cfg.dev_clock_enabled
    if app_cfg.env == "dev":
        return CheckResult("admin", True, "env=dev: admin and dev-clock routes enabled")
    ok = not enabled
    state = "disabled" if ok else f"ENABLED via dev_clock.enabled_in={app_cfg.dev_clock.enabled_in}"
    return CheckResult("admin", ok, f"env={app_cfg.env}: admin routes {state}")


def check_secrets(settings: Settings, models: ModelsConfig) -> CheckResult:
    profile = models.active()
    missing = []
    for provider in sorted(profile.providers()):
        field = PROVIDER_KEY_FIELDS.get(provider)
        secret = getattr(settings, field, None) if field else None
        if secret is None or not secret.get_secret_value().strip():
            missing.append((field or provider).upper())
    ok = not missing
    return CheckResult(
        "secrets",
        ok,
        f"profile={models.active_profile}: "
        + ("all provider keys present" if ok else f"missing {', '.join(missing)} in .env"),
    )


def check_index(index_dir: Path, expected_embedder_alias: str | None = None) -> CheckResult:
    from rag.ingest.index import Manifest, compute_corpus_version

    index_dir = Path(index_dir)
    missing = [f for f in REQUIRED_INDEX_FILES if not (index_dir / f).exists()]
    if missing:
        return CheckResult(
            "index",
            False,
            f"{index_dir}: missing {', '.join(missing)} — run `make ingest`",
            fatal=True,
        )
    try:
        manifest = Manifest.model_validate_json(
            (index_dir / "manifest.json").read_text(encoding="utf-8")
        )
    except (ValueError, json.JSONDecodeError) as exc:
        return CheckResult("index", False, f"manifest.json invalid: {exc}"[:300], fatal=True)
    problems: list[str] = []
    expected = compute_corpus_version(manifest.pdf_sha256, manifest.ingestion_config)
    if expected != manifest.corpus_version:
        problems.append(
            f"corpus_version {manifest.corpus_version} does not recompute from the manifest "
            f"(expected {expected})"
        )
    with (index_dir / "chunks.jsonl").open("r", encoding="utf-8") as fh:
        lines = sum(1 for line in fh if line.strip())
    if lines != manifest.chunks_total:
        problems.append(f"chunks.jsonl has {lines} rows, manifest says {manifest.chunks_total}")
    if sum(manifest.chunk_counts.values()) != manifest.chunks_total:
        problems.append("chunk_counts do not sum to chunks_total")
    if expected_embedder_alias and manifest.embedder_alias != expected_embedder_alias:
        problems.append(
            f"index embedder {manifest.embedder_alias!r} != thresholds.retrieval.embedder "
            f"{expected_embedder_alias!r}"
        )
    if problems:
        return CheckResult("index", False, "; ".join(problems), fatal=True)
    return CheckResult(
        "index",
        True,
        f"corpus_version={manifest.corpus_version} chunks={manifest.chunks_total} "
        f"embedder={manifest.embedder_alias}",
        fatal=True,
    )


def run_startup_checks(
    settings: Settings,
    app_cfg: AppConfig,
    models: ModelsConfig,
    thresholds: ThresholdsConfig,
    *,
    index_dir: Path | None = None,
) -> list[CheckResult]:
    from rag.ingest.index import index_dir_for

    alias = thresholds.retrieval.embedder
    index_dir = index_dir or index_dir_for(settings.data_dir, alias)
    return [
        check_bind(app_cfg.server.host, public=settings.public_deploy),
        check_public_safety(settings, app_cfg),
        check_admin_gating(app_cfg),
        check_secrets(settings, models),
        check_index(index_dir, alias),
    ]


def fatal_failures(results: list[CheckResult]) -> list[CheckResult]:
    return [r for r in results if r.fatal and not r.ok]


__all__ = [
    "CheckResult",
    "StartupCheckError",
    "check_admin_gating",
    "check_bind",
    "check_public_safety",
    "check_index",
    "check_secrets",
    "fatal_failures",
    "is_loopback_client",
    "run_startup_checks",
]
