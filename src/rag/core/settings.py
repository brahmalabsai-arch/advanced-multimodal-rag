"""Environment settings loaded from `.env` (architecture §11, problem statement NFR-12).

Only three variables matter during the build: `GROQ_API_KEY`, `MODEL_PROFILE`, `APP_ENV`.
Provider keys are `SecretStr` so they never appear in reprs or logs by accident.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/rag/core/settings.py -> parents[3] is the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

Provider = Literal["groq", "anthropic", "google"]
AppEnv = Literal["dev", "test", "prod"]

# Which settings field holds the key for each provider.
PROVIDER_KEY_FIELDS: dict[str, str] = {
    "groq": "groq_api_key",
    "anthropic": "anthropic_api_key",
    "google": "google_api_key",
}


class SettingsError(RuntimeError):
    """Raised when required configuration is missing or inconsistent."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    groq_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None

    model_profile: str = "groq_build"
    app_env: AppEnv = "dev"
    # F2: the deployed build serves nobody from its own quota. With BYOK_ONLY=true every
    # `/api/ask` must carry the caller's provider and key, and a key in the environment (if one
    # is even present) is never used to answer a question.
    byok_only: bool = False
    # F2: set only in the deployed container. It turns off the loopback-only request guard
    # (NFR-12), which exists for the localhost build and would refuse every visitor arriving
    # through a platform proxy. It is refused unless `byok_only` is also set — a public server
    # that still holds a key would spend the owner's quota on strangers.
    public_deploy: bool = False

    config_dir: Path = Field(default=PROJECT_ROOT / "config")
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    log_level: str = "INFO"

    @property
    def byok(self) -> bool:
        """True when requests must bring their own credentials."""
        return self.byok_only

    @property
    def is_dev(self) -> bool:
        return self.app_env == "dev"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    def api_key_for(self, provider: str) -> str:
        """Return the API key for a provider or fail fast with an actionable message."""
        field = PROVIDER_KEY_FIELDS.get(provider)
        if field is None:
            raise SettingsError(
                f"Unknown provider {provider!r}; expected one of {sorted(PROVIDER_KEY_FIELDS)}"
            )
        secret: SecretStr | None = getattr(self, field)
        if secret is None or not secret.get_secret_value().strip():
            raise SettingsError(
                f"{field.upper()} is not set but the active profile {self.model_profile!r} "
                f"uses provider {provider!r}. Add it to .env (see .env.example)."
            )
        return secret.get_secret_value()

    def secret_values(self) -> list[str]:
        """All configured secrets, for log redaction."""
        values = []
        for field in PROVIDER_KEY_FIELDS.values():
            secret: SecretStr | None = getattr(self, field)
            if secret is not None and secret.get_secret_value().strip():
                values.append(secret.get_secret_value())
        return values


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Tests construct `Settings(_env_file=None, ...)` directly instead."""
    return Settings()
