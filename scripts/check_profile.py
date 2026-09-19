"""Model-swap readiness: validate a model profile with no network calls (plan Phase 8, NFR-11).

    .venv/Scripts/python scripts/check_profile.py --profile anthropic --dry-run
    .venv/Scripts/python scripts/check_profile.py --profile gemini --dry-run
    .venv/Scripts/python scripts/check_profile.py --profile all --dry-run --report
    .venv/Scripts/python scripts/check_profile.py --profile anthropic --live   # + one `small` call

What a dry run proves for a profile (architecture §7.1, D-27, D-45):

    config      `models.yaml` validates with that profile active (pydantic, `extra=forbid`)
    roles       small / large / vision each resolve to a provider + model id + pacing
    package     the provider's LangChain package is importable (version printed) — a template
                whose package is not installed is still a valid template; the line says which
                `requirements-future.txt` install it needs
    construct   the provider chat-model object is built with the profile's `provider_kwargs`
                (a placeholder key is used when `.env` has none) — this is the constructor the
                serving path calls, so unknown kwargs fail here, not at the first request
    key         the provider key is present in `.env` (name only; never printed)
    budgets     `context_budget_tokens` fits under the pacing TPM with room for output;
                the vision role's OTPM/ITPM caps are reported the way `generate.py` uses them
    structured  `structured_output` matches the provider (Groq → json_mode_validate; others →
                native or json_mode_validate)
    prices      per-token prices present → compression break-even runs in price mode; missing →
                quota mode (a warning, expected for the Groq free tier)
    enrichment  `ingestion_enrichment` roles resolve through this profile

Exit code 0 when no check is an ERROR (warnings are allowed and listed), 1 otherwise.
`--live` additionally makes one `small` text call ("PONG") and reports tokens and latency.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import sys
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import SecretStr, ValidationError

from rag.core.config import ModelsConfig, Profile, load_models_config, load_thresholds_config
from rag.core.console import utf8_console
from rag.core.settings import PROJECT_ROOT, PROVIDER_KEY_FIELDS, Settings, SettingsError

ROLES = ("small", "large", "vision")
PACKAGES = {
    "groq": ("langchain_groq", "langchain-groq", "requirements-serve.txt"),
    "anthropic": ("langchain_anthropic", "langchain-anthropic", "requirements-future.txt"),
    "google": ("langchain_google_genai", "langchain-google-genai", "requirements-future.txt"),
}
PLACEHOLDER_KEY = "dry-run-placeholder-key"


@dataclass
class Line:
    check: str
    level: str  # OK | WARN | ERROR
    detail: str


@dataclass
class ProfileReport:
    profile: str
    lines: list[Line] = field(default_factory=list)
    roles: dict[str, dict[str, str]] = field(default_factory=dict)

    def add(self, check: str, level: str, detail: str) -> None:
        self.lines.append(Line(check, level, detail))

    @property
    def errors(self) -> list[Line]:
        return [ln for ln in self.lines if ln.level == "ERROR"]

    @property
    def warnings(self) -> list[Line]:
        return [ln for ln in self.lines if ln.level == "WARN"]

    @property
    def ok(self) -> bool:
        return not self.errors


def _package_status(provider: str) -> tuple[bool, str]:
    module, dist, req = PACKAGES[provider]
    try:
        importlib.import_module(module)
    except ImportError:
        return False, f"{dist} not installed — `pip install -r {req}`"
    try:
        version = md.version(dist)
    except md.PackageNotFoundError:  # pragma: no cover - importable but no metadata
        version = "?"
    return True, f"{dist} {version}"


def check_profile(
    profile_name: str,
    *,
    settings: Settings | None = None,
    models_path: Path | None = None,
    live: bool = False,
) -> ProfileReport:
    rep = ProfileReport(profile_name)
    base = settings or Settings()
    settings = base.model_copy(update={"model_profile": profile_name})

    # 1. config validates with this profile active
    try:
        models: ModelsConfig = load_models_config(models_path, settings=settings)
    except (ValidationError, ValueError, FileNotFoundError) as exc:
        rep.add("config", "ERROR", f"models.yaml does not validate for {profile_name!r}: {exc}")
        return rep
    profile: Profile = models.active()
    rep.add(
        "config",
        "OK",
        f"models.yaml validates; active_profile={models.active_profile}; "
        f"structured_output={profile.structured_output}; "
        f"context_budget_tokens={profile.context_budget_tokens}",
    )

    # 2. roles resolve
    for role in ROLES:
        cfg = profile.role(role)  # type: ignore[arg-type]
        pacing = profile.pacing_for(role)  # type: ignore[arg-type]
        rep.roles[role] = {
            "provider": cfg.provider,
            "model": cfg.model,
            "pacing": (
                f"rpm {pacing.rpm} · tpm {pacing.tpm}"
                + (f" · otpm {pacing.otpm}" if pacing.otpm else "")
                + (f" · itpm {pacing.itpm}" if pacing.itpm else "")
                if pacing
                else "none"
            ),
            "price": (
                f"${cfg.price_usd_per_mtok.input:g} in / ${cfg.price_usd_per_mtok.output:g} out "
                "per Mtok"
                if cfg.price_usd_per_mtok
                else "—"
            ),
            "kwargs": ", ".join(f"{k}={v}" for k, v in cfg.provider_kwargs.items()) or "—",
        }
        rep.add("roles", "OK", f"{role:6} → {cfg.provider}:{cfg.model}")

    providers = sorted(profile.providers())

    # 3. packages
    for provider in providers:
        installed, detail = _package_status(provider)
        rep.add("package", "OK" if installed else "WARN", f"{provider}: {detail}")

    # 4. keys (names only)
    missing_key = set()
    for provider in providers:
        field_name = PROVIDER_KEY_FIELDS[provider]
        try:
            settings.api_key_for(provider)
            rep.add("key", "OK", f"{field_name.upper()} present in .env")
        except SettingsError:
            missing_key.add(provider)
            rep.add(
                "key",
                "WARN",
                f"{field_name.upper()} missing — add it to .env before "
                f"`MODEL_PROFILE={profile_name}`",
            )

    # 5. construct the chat models exactly as the serving path does (no request is sent)
    from rag.llm import LLMClient, ProviderNotInstalledError

    construct_settings = settings.model_copy(
        update={PROVIDER_KEY_FIELDS[p]: SecretStr(PLACEHOLDER_KEY) for p in missing_key}
    )
    client = LLMClient(construct_settings, models, pacing_enabled=False)
    for role in ROLES:
        cfg = profile.role(role)  # type: ignore[arg-type]
        kwargs = ", ".join(f"{k}={v!r}" for k, v in cfg.provider_kwargs.items())
        label = f"{role:6} {cfg.model!r}" + (f", {kwargs}" if kwargs else "")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = client._model(role)  # type: ignore[arg-type]
        except ProviderNotInstalledError as exc:
            rep.add("construct", "WARN", f"{role:6} skipped — package not installed ({exc})")
            continue
        except Exception as exc:  # constructor rejected the kwargs: that is the finding
            rep.add("construct", "ERROR", f"{role:6} {type(exc).__name__}: {str(exc)[:200]}")
            continue
        # LangChain does not reject an unknown kwarg: it moves it to `model_kwargs` with a
        # UserWarning and the provider rejects it at the first request. Surface that here.
        forwarded = [
            str(w.message).splitlines()[0].strip()
            for w in caught
            if "not default parameter" in str(w.message)
        ]
        if forwarded:
            rep.add(
                "construct",
                "WARN",
                f"{role:6} {type(model).__name__} built, but provider_kwargs forwarded unchecked "
                f"to the API: {'; '.join(forwarded)}",
            )
        else:
            rep.add(
                "construct",
                "OK",
                f"{role:6} {type(model).__name__}({label[7:]})"
                + (" [placeholder key]" if cfg.provider in missing_key else ""),
            )

    # 6. budgets
    for role in ROLES:
        pacing = profile.pacing_for(role)  # type: ignore[arg-type]
        if pacing is None:
            rep.add("budgets", "WARN", f"{role:6} no pacing → client-side limiter disabled")
            continue
        headroom = pacing.tpm - profile.context_budget_tokens
        if role == "large":
            level = "OK" if headroom >= 1200 else "ERROR" if headroom <= 0 else "WARN"
            rep.add(
                "budgets",
                level,
                f"large  context_budget {profile.context_budget_tokens} vs tpm {pacing.tpm} → "
                f"{headroom} tokens headroom for prompt scaffolding + output",
            )
        if role == "vision":
            out_cap = client.max_output_tokens("vision")
            in_cap = client.max_input_tokens("vision", out_cap or 0)
            rep.add(
                "budgets",
                "OK",
                f"vision max_tokens cap {out_cap or 'none'} (OTPM) · input cap {in_cap or 'none'} "
                f"· max_images {profile.vision.max_images_per_request or 'unbounded'}",
            )

    # 7. structured output mode vs provider
    if "groq" in providers and profile.structured_output != "json_mode_validate":
        rep.add("structured", "ERROR", "Groq profiles must use json_mode_validate (JSON mode)")
    else:
        rep.add(
            "structured",
            "OK",
            f"{profile.structured_output} for providers {', '.join(providers)}"
            + (
                " (schema instruction + validation; native structured output is an F1 step)"
                if profile.structured_output == "native"
                else ""
            ),
        )

    # 8. prices → break-even mode
    priced = [r for r in ROLES if profile.role(r).price_usd_per_mtok]  # type: ignore[arg-type]
    if len(priced) == len(ROLES):
        rep.add("prices", "OK", "all roles priced → compression break-even runs in price mode")
    else:
        rep.add(
            "prices",
            "WARN",
            "no per-token prices for "
            + ", ".join(r for r in ROLES if r not in priced)
            + " → break-even test runs in quota mode (§6.6); fill `price_usd_per_mtok` at F1",
        )

    # 9. enrichment roles
    enrich = models.ingestion_enrichment
    rep.add(
        "enrichment",
        "OK",
        f"figures → {enrich.figures} ({profile.role(enrich.figures).model}); "
        f"table_summaries → {enrich.table_summaries} "
        f"({profile.role(enrich.table_summaries).model})",
    )

    # 10. thresholds still load (they are profile-independent, but the swap touches config)
    try:
        th = load_thresholds_config(settings=settings)
        rep.add(
            "thresholds",
            "OK",
            f"thresholds.yaml validates; compression.mode={th.compression.mode}, "
            f"cache.enabled={th.cache.enabled}",
        )
    except (ValidationError, FileNotFoundError) as exc:
        rep.add("thresholds", "ERROR", f"thresholds.yaml does not validate: {exc}"[:300])

    # 11. optional live ping
    if live:
        if missing_key:
            rep.add("live", "ERROR", "cannot ping: provider key missing")
        else:
            try:
                live_client = LLMClient(settings, models)
                import time

                t0 = time.perf_counter()
                reply = live_client.text(
                    "Reply with exactly one word: PONG",
                    role="small",
                    request_id=f"check-profile-{profile_name}",
                    max_tokens=64,
                )
                rep.add(
                    "live",
                    "OK" if "PONG" in reply.upper() else "WARN",
                    f"small replied {reply.strip()!r} in {time.perf_counter() - t0:.1f}s",
                )
            except Exception as exc:
                rep.add("live", "ERROR", f"{type(exc).__name__}: {str(exc)[:200]}")
    return rep


def print_report(rep: ProfileReport) -> None:
    print(f"\n=== profile: {rep.profile} ===")
    for role, info in rep.roles.items():
        print(
            f"  {role:6} {info['provider']}:{info['model']}  pacing[{info['pacing']}]  "
            f"price[{info['price']}]  kwargs[{info['kwargs']}]"
        )
    for ln in rep.lines:
        print(f"  {ln.level:5} {ln.check:11} {ln.detail}")
    status = "PASS" if rep.ok else "FAIL"
    print(f"  → {status}: {len(rep.errors)} error(s), {len(rep.warnings)} warning(s)")


def write_report(reports: list[ProfileReport], path: Path, live: bool) -> None:
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    out = [
        "# Model-swap dry run (Phase 8, NFR-11)",
        "",
        f"Generated {stamp} · `scripts/check_profile.py --profile all --dry-run"
        + (" --live" if live else "")
        + "` · no network calls"
        + ("" if not live else " except the one `small` ping per profile")
        + ".",
        "",
        "A profile passes when `models.yaml` validates with it active, every role resolves to a "
        "provider + model, the provider chat-model constructor accepts the profile's "
        "`provider_kwargs`, the context budget fits under the pacing limits and the structured-"
        "output mode matches the provider. A missing key or an uninstalled package is a warning: "
        "the template is valid, the switch (F1) supplies them.",
        "",
        "| Profile | Result | Errors | Warnings | small | large | vision |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for rep in reports:
        r = rep.roles
        out.append(
            f"| `{rep.profile}` | {'✅ pass' if rep.ok else '❌ fail'} | {len(rep.errors)} | "
            f"{len(rep.warnings)} | "
            + " | ".join(
                f"{r[k]['provider']}:`{r[k]['model']}`" if k in r else "—"
                for k in ("small", "large", "vision")
            )
            + " |"
        )
    for rep in reports:
        out += ["", f"## `{rep.profile}`", "", "| Check | Level | Detail |", "|---|---|---|"]
        for ln in rep.lines:
            mark = {"OK": "✅", "WARN": "🟡", "ERROR": "❌"}[ln.level]
            detail = ln.detail.replace("|", "\\|")
            out.append(f"| {ln.check} | {mark} {ln.level} | {detail} |")
    out += [
        "",
        "Reproduce: `make check-profile` (all profiles) or "
        "`.venv/Scripts/python scripts/check_profile.py --profile anthropic --dry-run`.",
    ]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\nreport → {path.relative_to(PROJECT_ROOT)}")


def main() -> int:
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile", default="all", help="profile name from models.yaml, or `all`")
    ap.add_argument(
        "--dry-run", action="store_true", default=True, help="no network calls (default)"
    )
    ap.add_argument("--live", action="store_true", help="also make one tiny `small` call")
    ap.add_argument(
        "--report", action="store_true", help="write docs/reports/model_swap_dry_run.md"
    )
    args = ap.parse_args()

    try:
        settings = Settings()
    except ValidationError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        all_profiles = list(load_models_config(settings=settings).profiles)
    except (ValidationError, FileNotFoundError) as exc:
        print(f"CONFIG ERROR: models.yaml does not load: {exc}", file=sys.stderr)
        return 2
    names = all_profiles if args.profile == "all" else [args.profile]
    unknown = [n for n in names if n not in all_profiles]
    if unknown:
        print(f"unknown profile(s) {unknown}; known: {all_profiles}", file=sys.stderr)
        return 2

    reports = [check_profile(n, settings=settings, live=args.live) for n in names]
    for rep in reports:
        print_report(rep)
    if args.report:
        write_report(
            reports, PROJECT_ROOT / "docs" / "reports" / "model_swap_dry_run.md", args.live
        )
    failed = [r.profile for r in reports if not r.ok]
    print("\n" + ("ALL PROFILES PASS" if not failed else f"FAILED: {', '.join(failed)}"))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
