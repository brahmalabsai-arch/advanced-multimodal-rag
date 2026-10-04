"""Pydantic schemas and loaders for the YAML configuration files (architecture §11).

Phase 0 covers `models.yaml`, `app.yaml`, `thresholds.yaml`; Phase 3 added `formulas.yaml`;
Phase 4 adds `fiscal_calendar.yaml` and `glossary.yaml`. All three profile templates in
`models.yaml` must validate so that a later model switch fails fast on a typo (FR-6.3, NFR-11).
"""

from __future__ import annotations

import os
import re
from datetime import date
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
    itpm: int | None = Field(
        default=None,
        gt=0,
        description="input tokens per minute; Groq rejects (413) a single request above it",
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


class RateLimitConfig(BaseModel):
    """Per-address limits and an in-flight cap, applied only on a public deploy (D-70)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    ask_per_minute: int = Field(default=6, ge=0)  # 0 disables that window
    ask_per_hour: int = Field(default=60, ge=0)
    key_test_per_minute: int = Field(default=5, ge=0)
    max_in_flight: int = Field(default=2, ge=1)  # questions running at once, all visitors
    max_clients: int = Field(default=10000, ge=10)  # addresses tracked in memory
    # first header present wins; Cloudflare (in front of Render) sets both and overwrites any
    # client-supplied value. X-Forwarded-For is deliberately absent: Render appends to it, so
    # its first entry is whatever the client sent.
    client_ip_headers: list[str] = Field(
        default_factory=lambda: ["cf-connecting-ip", "true-client-ip"]
    )


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)
    # Phase 8 input limits: a question longer than this is rejected (422); a request that
    # outlives the timeout answers 504 (the worker thread finishes in the background).
    max_question_chars: int = Field(default=1000, ge=16, le=20000)
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)


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

    embedder: str = "bge-small"
    dense_top_k: int = Field(gt=0)
    sparse_top_k: int = Field(gt=0)
    rrf_k: int = Field(gt=0)
    final_k: int = Field(gt=0)
    max_retrieval_queries: int = Field(gt=0)


class ExpansionThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    glossary: bool = True
    filters: bool = True
    llm_fallback: bool = True
    min_rule_confidence: float = Field(default=0.8, ge=0, le=1)
    max_paraphrases: int = Field(default=2, ge=0)
    hyde: bool = True
    hyde_regenerate: bool = True


class RerankThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    lightweight_alternative: str | None = None
    candidates: int = Field(gt=0)
    margin_skip_ratio: float = Field(ge=0, le=1)
    drop_floor_logit: float
    min_keep: int = Field(ge=0)
    enabled: bool = True


class StageBThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keep: float = Field(ge=0, le=1)
    llm: float = Field(ge=0, le=1)


class CompressionThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    mode: Literal["classifier", "never", "always"] = "classifier"
    skip_budget_tokens: int = Field(gt=0)
    narrative_pressure_ratio: float = Field(default=0.5, ge=0)
    dedupe_cosine: float = Field(ge=0, le=1)
    sentence_relevance_tau: float = Field(ge=0, le=1)
    numeric_density_cap: float = Field(ge=0, le=1)
    min_reduction_for_llm: float = Field(ge=0, le=1)
    stage_b_thresholds: StageBThresholds
    stage_b: Literal["rules", "learned"] = "rules"


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
    embed_text: Literal["raw", "normalized", "canonical"] = "canonical"


class CacheThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    l1: L1Config
    l2: L2Config
    sweep_interval_seconds: int = Field(gt=0)


class CompletenessThresholds(BaseModel):
    """The completeness check after verification (`query/coverage.py`)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    self_check: bool = True  # one small-model call for questions with several asks


class ThresholdsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retrieval: RetrievalThresholds
    expansion: ExpansionThresholds = Field(default_factory=ExpansionThresholds)
    rerank: RerankThresholds
    compression: CompressionThresholds
    completeness: CompletenessThresholds = Field(default_factory=CompletenessThresholds)
    cache: CacheThresholds


# ------------------------------------------------------------------------ formulas.yaml


class FormulaInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    line_item: str
    optional: bool = False
    period: Literal["current", "prior"] = "current"


class Formula(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    kind: Literal["ratio", "amount", "pct"]
    round: int = Field(ge=0, le=6)
    inputs: dict[str, FormulaInput]
    expression: str
    keywords: list[str] = Field(default_factory=list)
    generic: bool = False
    statement: str | None = None


class FormulasConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    default_statement: str = "balance_sheet"
    formulas: dict[str, Formula]


# ----------------------------------------------------------------- fiscal_calendar.yaml


class FiscalYearSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> FiscalYearSpan:
        if self.end <= self.start:
            raise ValueError(f"fiscal year end {self.end} is not after start {self.start}")
        return self


class RelativePeriods(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latest: list[str] = Field(default_factory=list)
    prior: list[str] = Field(default_factory=list)


class FiscalCalendar(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    entity: str
    entity_aliases: list[str] = Field(min_length=1)
    latest_fiscal_year: int
    bare_year_offset: int = 1
    fiscal_years: dict[int, FiscalYearSpan]
    relative_periods: RelativePeriods = Field(default_factory=RelativePeriods)

    @model_validator(mode="after")
    def _consistent(self) -> FiscalCalendar:
        if self.latest_fiscal_year not in self.fiscal_years:
            raise ValueError(
                f"latest_fiscal_year {self.latest_fiscal_year} is not in fiscal_years "
                f"{sorted(self.fiscal_years)}"
            )
        years = sorted(self.fiscal_years)
        for earlier, later in zip(years, years[1:], strict=False):
            if self.fiscal_years[later].start <= self.fiscal_years[earlier].end:
                raise ValueError(f"fiscal years {earlier} and {later} overlap")
        return self

    def fiscal_year_of(self, day: date) -> int | None:
        for fy, span in self.fiscal_years.items():
            if span.start <= day <= span.end:
                return fy
        return None

    def year_end(self, fy: int) -> date | None:
        span = self.fiscal_years.get(fy)
        return span.end if span else None


# ------------------------------------------------------------------------ glossary.yaml


class GlossaryStatement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    synonyms: list[str] = Field(min_length=1)


class GlossaryMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str
    synonyms: list[str] = Field(min_length=1)


class GlossaryFormula(BaseModel):
    model_config = ConfigDict(extra="forbid")

    synonyms: list[str] = Field(min_length=1)


class DirectionLexicon(BaseModel):
    model_config = ConfigDict(extra="forbid")

    increase: list[str]
    decrease: list[str]


class Lexicons(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direction: DirectionLexicon
    aggregation: dict[str, list[str]]
    time_anchor: list[str]
    ask_type: dict[str, list[str]] = Field(default_factory=dict)
    topic_terms: dict[str, list[str]] = Field(default_factory=dict)


class GlossaryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    statements: dict[str, GlossaryStatement]
    metrics: dict[str, GlossaryMetric]
    formulas: dict[str, GlossaryFormula]
    lexicons: Lexicons

    @model_validator(mode="after")
    def _consistent(self) -> GlossaryConfig:
        for name, metric in self.metrics.items():
            if metric.statement not in self.statements:
                raise ValueError(f"metric {name!r} names unknown statement {metric.statement!r}")
        seen: dict[str, str] = {}
        for kind, table in (
            ("statement", self.statements),
            ("metric", self.metrics),
            ("formula", self.formulas),
        ):
            for name, entry in table.items():
                for syn in entry.synonyms:
                    key = syn.lower().strip()
                    if key in seen:
                        raise ValueError(
                            f"synonym {syn!r} maps to both {seen[key]} and {kind} {name!r}"
                        )
                    seen[key] = f"{kind} {name!r}"
        return self

    def synonym_count(self) -> int:
        return sum(
            len(e.synonyms)
            for table in (self.statements, self.metrics, self.formulas)
            for e in table.values()
        )


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


def load_formulas_config(
    path: Path | None = None, settings: Settings | None = None
) -> FormulasConfig:
    return FormulasConfig.model_validate(
        load_yaml(_config_path("formulas.yaml", path, settings), _env_from_settings(settings))
    )


def load_fiscal_calendar(
    path: Path | None = None, settings: Settings | None = None
) -> FiscalCalendar:
    return FiscalCalendar.model_validate(
        load_yaml(
            _config_path("fiscal_calendar.yaml", path, settings), _env_from_settings(settings)
        )
    )


def load_glossary_config(
    path: Path | None = None, settings: Settings | None = None
) -> GlossaryConfig:
    return GlossaryConfig.model_validate(
        load_yaml(_config_path("glossary.yaml", path, settings), _env_from_settings(settings))
    )
