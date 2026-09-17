from __future__ import annotations

from pathlib import Path

import pytest

from rag.core.config import ModelsConfig, load_models_config
from rag.core.ledger import UsageLedger
from rag.core.settings import PROJECT_ROOT, Settings

FAKE_GROQ_KEY = "gsk_TESTKEY0123456789abcdefghijklmnop"
CONFIG_DIR = PROJECT_ROOT / "config"

ENV_VARS = (
    "GROQ_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "MODEL_PROFILE",
    "APP_ENV",
    "CONFIG_DIR",
    "DATA_DIR",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never see the developer's real .env or shell variables."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        groq_api_key=FAKE_GROQ_KEY,
        model_profile="groq_build",
        app_env="test",
        config_dir=CONFIG_DIR,
        data_dir=tmp_path / "data",
    )


@pytest.fixture
def models_config(settings: Settings) -> ModelsConfig:
    return load_models_config(settings=settings)


@pytest.fixture
def ledger(tmp_path: Path) -> UsageLedger:
    return UsageLedger(tmp_path / "llm_usage.jsonl")


class FakeClock:
    """Monotonic clock whose `sleep` advances time instead of waiting."""

    def __init__(self, start: float = 1000.0):
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()
