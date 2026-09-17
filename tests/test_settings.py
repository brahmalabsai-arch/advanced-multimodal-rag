"""Phase 0 tests — settings: missing key -> clear error; APP_ENV parsing."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag.core.settings import Settings, SettingsError
from rag.llm import LLMClient


def test_missing_groq_key_is_a_clear_error() -> None:
    s = Settings(_env_file=None)
    with pytest.raises(SettingsError) as exc_info:
        s.api_key_for("groq")
    message = str(exc_info.value)
    assert "GROQ_API_KEY" in message
    assert "groq_build" in message
    assert ".env" in message


def test_blank_key_counts_as_missing() -> None:
    s = Settings(_env_file=None, groq_api_key="   ")
    with pytest.raises(SettingsError):
        s.api_key_for("groq")


def test_llm_client_fails_fast_without_key(models_config, tmp_path) -> None:
    s = Settings(_env_file=None, app_env="test", data_dir=tmp_path)
    with pytest.raises(SettingsError, match="GROQ_API_KEY"):
        LLMClient(s, models_config)


def test_unknown_provider_rejected() -> None:
    with pytest.raises(SettingsError, match="Unknown provider"):
        Settings(_env_file=None).api_key_for("openai")


@pytest.mark.parametrize("value", ["dev", "test", "prod"])
def test_app_env_accepts_known_values(value: str) -> None:
    assert Settings(_env_file=None, app_env=value).app_env == value


def test_app_env_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="staging")


def test_app_env_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("MODEL_PROFILE", "anthropic")
    s = Settings(_env_file=None)
    assert s.app_env == "prod"
    assert not s.is_dev
    assert s.model_profile == "anthropic"


def test_secret_never_appears_in_repr() -> None:
    s = Settings(_env_file=None, groq_api_key="gsk_supersecretvalue1234567890")
    assert "supersecret" not in repr(s)
    assert "supersecret" not in str(s)
    assert s.secret_values() == ["gsk_supersecretvalue1234567890"]
