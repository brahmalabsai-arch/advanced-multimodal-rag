"""Phase 0 tests — config: all three profiles validate; a broken template fails validation."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from rag.core.config import (
    ROLES,
    AppConfig,
    ModelsConfig,
    ThresholdsConfig,
    expand_env,
    load_app_config,
    load_models_config,
    load_thresholds_config,
    load_yaml,
)
from rag.core.settings import PROJECT_ROOT, Settings

CONFIG_DIR = PROJECT_ROOT / "config"


def test_all_three_profiles_validate(models_config: ModelsConfig) -> None:
    assert set(models_config.profiles) == {"groq_build", "anthropic", "gemini"}
    for name, profile in models_config.profiles.items():
        for role in ROLES:
            assert profile.role(role).model, f"{name}.{role} has no model id"
    assert models_config.active_profile == "groq_build"
    assert models_config.active().providers() == {"groq"}
    assert models_config.profiles["anthropic"].providers() == {"anthropic"}
    assert models_config.profiles["gemini"].providers() == {"google"}


def test_groq_profile_has_pacing_for_every_role(models_config: ModelsConfig) -> None:
    profile = models_config.active()
    for role in ROLES:
        pacing = profile.pacing_for(role)
        assert pacing is not None, role
        assert pacing.rpm > 0 and pacing.tpm > 0
    # The 2,500-token generation budget must fit under the per-minute token ceiling (§7.2).
    assert profile.context_budget_tokens < profile.pacing_for("large").tpm


def test_enrichment_roles_resolve(models_config: ModelsConfig) -> None:
    assert models_config.ingestion_enrichment.figures == "vision"
    assert models_config.ingestion_enrichment.table_summaries == "small"


def test_active_profile_follows_model_profile_setting(tmp_path: Path) -> None:
    s = Settings(_env_file=None, model_profile="gemini", config_dir=CONFIG_DIR, data_dir=tmp_path)
    cfg = load_models_config(settings=s)
    assert cfg.active_profile == "gemini"
    assert cfg.active().structured_output == "native"


def test_unknown_active_profile_fails(tmp_path: Path) -> None:
    s = Settings(_env_file=None, model_profile="nope", config_dir=CONFIG_DIR, data_dir=tmp_path)
    with pytest.raises(ValidationError, match="not defined"):
        load_models_config(settings=s)


def _raw_models() -> dict:
    return load_yaml(CONFIG_DIR / "models.yaml", {"MODEL_PROFILE": "groq_build"})


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda d: d["profiles"]["anthropic"].pop("vision"), "vision"),
        (lambda d: d["profiles"]["gemini"]["small"].update(provider="openai"), "provider"),
        (lambda d: d["profiles"]["anthropic"]["large"].update(model=""), "model"),
        (lambda d: d["profiles"]["gemini"].update(structured_output="maybe"), "structured_output"),
        (lambda d: d["profiles"]["groq_build"].update(context_budget_tokens=0), "context_budget"),
        (lambda d: d["profiles"]["groq_build"]["small"]["pacing"].update(rpm=-1), "rpm"),
        (lambda d: d["profiles"]["anthropic"].update(typo_field=1), "typo_field"),
        (lambda d: d["ingestion_enrichment"].update(figures="huge"), "figures"),
    ],
)
def test_broken_template_fails_validation(mutate, needle: str) -> None:
    raw = copy.deepcopy(_raw_models())
    mutate(raw)
    with pytest.raises(ValidationError) as exc_info:
        ModelsConfig.model_validate(raw)
    assert needle in str(exc_info.value)


def test_app_config_loads(settings: Settings) -> None:
    app = load_app_config(settings=settings)
    assert isinstance(app, AppConfig)
    assert app.server.host == "127.0.0.1"
    assert app.server.port == 8000
    assert app.server.workers == 1
    assert app.env == "test"  # from settings.app_env via ${APP_ENV:dev}
    assert not app.dev_clock_enabled
    assert app.paths.logs == Path("data/logs")


def test_app_config_dev_clock_enabled_in_dev(tmp_path: Path) -> None:
    s = Settings(_env_file=None, app_env="dev", config_dir=CONFIG_DIR, data_dir=tmp_path)
    assert load_app_config(settings=s).dev_clock_enabled


def test_thresholds_config_matches_architecture_11_1(settings: Settings) -> None:
    t = load_thresholds_config(settings=settings)
    assert isinstance(t, ThresholdsConfig)
    assert (t.retrieval.dense_top_k, t.retrieval.rrf_k, t.retrieval.final_k) == (30, 60, 8)
    assert t.rerank.min_keep == 3
    assert t.compression.dedupe_cosine == 0.95
    assert t.cache.l1.max_entries == 512
    assert t.cache.l2.similarity.slot_rich == 0.90
    assert t.cache.l2.similarity.slot_poor == 0.95
    assert t.cache.l2.ttl_seconds.filed_fact == 30 * 86400
    assert t.cache.l2.eviction.policy == "redis_volatile_lfu"
    assert t.cache.sweep_interval_seconds == 900


def test_thresholds_reject_unknown_keys(tmp_path: Path) -> None:
    raw = load_yaml(CONFIG_DIR / "thresholds.yaml")
    raw["retrieval"]["dense_topk"] = 5  # typo
    broken = tmp_path / "thresholds.yaml"
    broken.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError, match="dense_topk"):
        load_thresholds_config(path=broken)


def test_expand_env_placeholders() -> None:
    env = {"SET": "value", "EMPTY": ""}
    assert expand_env("${SET}", env) == "value"
    assert expand_env("${SET:fallback}", env) == "value"
    assert expand_env("${MISSING:fallback}", env) == "fallback"
    assert expand_env("${EMPTY:fallback}", env) == "fallback"
    assert expand_env("plain", env) == "plain"
    with pytest.raises(KeyError):
        expand_env("${MISSING}", env)
