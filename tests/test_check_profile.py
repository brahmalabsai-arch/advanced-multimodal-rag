"""Phase 8 tests — model-swap dry run (NFR-11): the inactive templates validate with no network
and no keys; a broken profile is reported as an error rather than a crash."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from rag.core.settings import PROJECT_ROOT, Settings

_spec = importlib.util.spec_from_file_location(
    "check_profile", PROJECT_ROOT / "scripts" / "check_profile.py"
)
check_profile = importlib.util.module_from_spec(_spec)
sys.modules["check_profile"] = check_profile  # dataclasses resolve annotations via sys.modules
_spec.loader.exec_module(check_profile)  # type: ignore[union-attr]


@pytest.fixture
def no_keys() -> Settings:
    return Settings(_env_file=None, app_env="test", config_dir=PROJECT_ROOT / "config")


@pytest.mark.parametrize("profile", ["anthropic", "gemini", "groq_build"])
def test_templates_pass_dry_run_without_keys(no_keys: Settings, profile: str) -> None:
    rep = check_profile.check_profile(profile, settings=no_keys)
    assert rep.ok, [ln.detail for ln in rep.errors]
    checks = {ln.check for ln in rep.lines}
    assert {"config", "roles", "package", "key", "construct", "budgets", "structured"} <= checks
    assert set(rep.roles) == {"small", "large", "vision"}
    # no key → a warning naming the variable, never a value; the constructor still runs
    key_lines = [ln for ln in rep.lines if ln.check == "key"]
    assert key_lines and all(ln.level == "WARN" and "missing" in ln.detail for ln in key_lines)
    construct = [ln for ln in rep.lines if ln.check == "construct"]
    assert len(construct) == 3
    # package installed → built with a placeholder key; not installed (a fresh clone before
    # F1's `requirements-future.txt`) → skipped with the install hint, still a valid template
    assert all("placeholder key" in ln.detail or "not installed" in ln.detail for ln in construct)
    assert check_profile.PLACEHOLDER_KEY not in " ".join(ln.detail for ln in rep.lines)


def test_groq_profile_with_key_has_no_key_warning(settings: Settings) -> None:
    rep = check_profile.check_profile("groq_build", settings=settings)
    assert rep.ok
    assert all(ln.level == "OK" for ln in rep.lines if ln.check in {"key", "construct"})
    assert any("otpm" in v["pacing"] for v in rep.roles.values())


def test_broken_profile_is_an_error_not_a_crash(tmp_path: Path, no_keys: Settings) -> None:
    bad = tmp_path / "models.yaml"
    bad.write_text(
        "active_profile: ${MODEL_PROFILE:broken}\n"
        "profiles:\n"
        "  broken:\n"
        "    small: {provider: groq, model: x}\n"
        "    large: {provider: groq, model: y}\n"
        "    vision: {provider: groq, model: z}\n"
        "    context_budget_tokens: 9000\n"
        "    structured_output: native\n"
        "    pacing: {rpm: 30, tpm: 8000}\n"
        "ingestion_enrichment: {figures: vision, table_summaries: small}\n",
        encoding="utf-8",
    )
    rep = check_profile.check_profile("broken", settings=no_keys, models_path=bad)
    kinds = {ln.check: ln.level for ln in rep.lines if ln.level == "ERROR"}
    # budget over the TPM bucket and JSON mode missing on a Groq profile
    assert kinds == {"budgets": "ERROR", "structured": "ERROR"}
    assert not rep.ok

    bad.write_text("active_profile: nope\nprofiles: {}\n", encoding="utf-8")
    rep = check_profile.check_profile("nope", settings=no_keys, models_path=bad)
    assert not rep.ok and rep.lines[0].check == "config"


def test_unknown_provider_kwarg_is_flagged_at_construct(tmp_path: Path, no_keys: Settings) -> None:
    """LangChain forwards an unknown kwarg to `model_kwargs` instead of rejecting it (the
    provider would 400 at the first request); the dry run must say so."""
    bad = tmp_path / "models.yaml"
    bad.write_text(
        "active_profile: ${MODEL_PROFILE:kw}\n"
        "profiles:\n"
        "  kw:\n"
        "    small: {provider: anthropic, model: claude-haiku-4-5-20251001, "
        "provider_kwargs: {no_such_option: 1}}\n"
        "    large: {provider: anthropic, model: claude-sonnet-5}\n"
        "    vision: {provider: anthropic, model: claude-sonnet-5}\n"
        "    context_budget_tokens: 6000\n"
        "    structured_output: native\n"
        "    pacing: {rpm: 50, tpm: 40000}\n"
        "ingestion_enrichment: {figures: vision, table_summaries: small}\n",
        encoding="utf-8",
    )
    pytest.importorskip("langchain_anthropic")
    rep = check_profile.check_profile("kw", settings=no_keys, models_path=bad)
    construct = {ln.detail.split()[0]: ln for ln in rep.lines if ln.check == "construct"}
    assert construct["small"].level == "WARN" and "no_such_option" in construct["small"].detail
    assert construct["large"].level == "OK"
