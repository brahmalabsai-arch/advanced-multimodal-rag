"""Pydantic schemas and loaders for the YAML configuration files (architecture §11).

Phase 0 covers `models.yaml`, `app.yaml`, `thresholds.yaml`. Glossary, formulas and the
fiscal calendar get schemas in later phases. All three profile templates in `models.yaml`
must validate now so that a later model switch fails fast on a typo (FR-6.3, NFR-11).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag.core.settings import Provider, Settings, get_settings

Role = Literal["small", "large", "vision"]
ROLES: tuple[Role, ...] = ("small", "large", "vision")

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def expand_env(value: str, env: dict[str, str] | None = None) -> str:
    """Expand `${VAR}` and `${VAR:default}` placeholders in a string."""
    source = os.environ if env is None else env

    def _sub(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        found = source.get(name)
        if found:
            return found
        if default is not None:
            return default
        raise KeyError(f"Environment variable {name!r} referenced in config is not set")

    return _ENV_PATTERN.sub(_sub, value)


def _expand_tree(node: Any, env: dict[str, str] | None) -> Any:
    if isinstance(node, str):
        return expand_env(node, env)
    if isinstance(node, dict):
        return {k: _expand_tree(v, env) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_tree(v, env) for v in node]
    return node


def load_yaml(path: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _expand_tree(raw, env)


# --------------------------------------------------------------------------- models.yaml


class Pacing(BaseModel):
    """Client-side rate limits for one model (values from the provider console)."""

    model_config = ConfigDict(extra="forbid")

    rpm: int = Field(gt=0, description="requests per minute")
    tpm: int = Field(gt=0, description="tokens per minute")
    otpm: int | None = Field(
        default=None,
        gt=0,
        description="output tokens per minute; Groq charges the *requested* max_tokens against it",
    )
    rpd: int | None = Field(default=None, gt=0, description="requests per day (informational)")
    tpd: int | None = Field(default=None, gt=0, description="tokens per day (informational)")


class Price(BaseModel):
    """USD per million tokens; used by the compression break-even test in price mode (§6.6)."""

    model_config = ConfigDict(extra="forbid")

    input: float = Field(ge=0)
    output: float = Field(ge=0)


class RoleModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Provider
    model: str = Field(min_length=1)
    pacing: Pacing | None = None
    price_usd_per_mtok: Price | None = None
    max_images_per_request: int | None = Field(default=None, gt=0)
    # Passed straight to the provider's chat-model constructor (e.g. Groq `reasoning_format`).
    provider_kwargs: dict[str, Any] = Field(default_factory=dict)


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    small: RoleModel
    large: RoleModel
    vision: RoleModel
    context_budget_tokens: int = Field(gt=0)
    structured_output: Literal["json_mode_validate", "native"]
    pacing: Pacing | None = None
    notes: str | None = None

    def role(self, name: Role) -> RoleModel:
        return getattr(self, name)

    def pacing_for(self, name: Role) -> Pacing | None:
        """Role-level pacing overrides the profile-level default."""
        return self.role(name).pacing or self.pacing

    def providers(self) -> set[str]:
        return {self.role(r).provider for r in ROLES}


class EnrichmentRoles(BaseModel):
    model_config = ConfigDict(extra="forbid")

    figures: Role
    table_summaries: Role


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_profile: str
    profiles: dict[str, Profile]
    ingestion_enrichment: EnrichmentRoles

    @model_validator(mode="after")
    def _active_exists(self) -> ModelsConfig:
        if self.active_profile not in self.profiles:
            raise ValueError(
                f"active_profile {self.active_profile!r} is not defined; "
                f"known profiles: {sorted(self.profiles)}"
            )
        return self

    def active(self) -> Profile:
        return self.profiles[self.active_profile]


# ----------------------------------------------------------------------------- app.yaml


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)


class DevClockConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled_in: list[str] = Field(default_factory=lambda: ["dev"])
    max_offset_days: int = Field(default=400, ge=0)


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw: Path = Path("data/raw")
    parsed: Path = Path("data/parsed")
    index: Path = Path("data/index")
    cache: Path = Path("data/cache")
    logs: Path = Path("data/logs")
    frontend: Path = Path("frontend")


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server: ServerConfig = Field(default_factory=ServerConfig)
    env: Literal["dev", "test", "prod"] = "dev"
    dev_clock: DevClockConfig = Field(default_factory=DevClockConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @property
    def dev_clock_enabled(self) -> bool:
        return self.env in self.dev_clock.enabled_in


# ---------------------------------------------------------------------- thresholds.yaml


class RetrievalThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dense_top_k: int = Field(gt=0)
    sparse_top_k: int = Field(gt=0)
    rrf_k: int = Field(gt=0)
    final_k: int = Field(gt=0)
    max_retrieval_queries: int = Field(gt=0)


class RerankThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    lightweight_alternative: str | None = None
    candidates: int = Field(gt=0)
    margin_skip_ratio: float = Field(ge=0, le=1)
    drop_floor_logit: float
    min_keep: int = Field(ge=0)


class StageBThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keep: float = Field(ge=0, le=1)
    llm: float = Field(ge=0, le=1)


class CompressionThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skip_budget_tokens: int = Field(gt=0)
    dedupe_cosine: float = Field(ge=0, le=1)
    sentence_relevance_tau: float = Field(ge=0, le=1)
    numeric_density_cap: float = Field(ge=0, le=1)
    min_reduction_for_llm: float = Field(ge=0, le=1)
    stage_b_thresholds: StageBThresholds


class L1Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_entries: int = Field(gt=0)
    ttl_seconds: int = Field(gt=0)
    policy: Literal["lru"]


class SimilarityThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_rich: float = Field(ge=0, le=1)
    slot_poor: float = Field(ge=0, le=1)


class TTLSeconds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filed_fact: int = Field(gt=0)
    analytical: int = Field(gt=0)
    time_anchored: int = Field(gt=0)
    negative: int = Field(gt=0)


class EvictionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: Literal["redis_volatile_lfu"]
    lfu_init_val: int = Field(ge=0)
    lfu_log_factor: int = Field(ge=0)
    lfu_decay_minutes: int = Field(gt=0)


class L2Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_entries: int = Field(gt=0)
    similarity: SimilarityThresholds
    ttl_seconds: TTLSeconds
    eviction: EvictionConfig


class CacheThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    l1: L1Config
    l2: L2Config
    sweep_interval_seconds: int = Field(gt=0)


class ThresholdsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retrieval: RetrievalThresholds
    rerank: RerankThresholds
    compression: CompressionThresholds
    cache: CacheThresholds


# ------------------------------------------------------------------------------ loaders


def _env_from_settings(settings: Settings | None) -> dict[str, str]:
    """Environment used for `${VAR:default}` expansion: process env overlaid with settings."""
    env = dict(os.environ)
    if settings is not None:
        env["MODEL_PROFILE"] = settings.model_profile
        env["APP_ENV"] = settings.app_env
    return env


def _config_path(name: str, path: Path | None, settings: Settings | None) -> Path:
    if path is not None:
        return path
    base = settings.config_dir if settings is not None else get_settings().config_dir
    return base / name


def load_models_config(path: Path | None = None, settings: Settings | None = None) -> ModelsConfig:
    return ModelsConfig.model_validate(
        load_yaml(_config_path("models.yaml", path, settings), _env_from_settings(settings))
    )


def load_app_config(path: Path | None = None, settings: Settings | None = None) -> AppConfig:
    return AppConfig.model_validate(
        load_yaml(_config_path("app.yaml", path, settings), _env_from_settings(settings))
    )


def load_thresholds_config(
    path: Path | None = None, settings: Settings | None = None
) -> ThresholdsConfig:
    return ThresholdsConfig.model_validate(
        load_yaml(_config_path("thresholds.yaml", path, settings), _env_from_settings(settings))
    )
