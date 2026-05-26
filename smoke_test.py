"""Pre-workshop verification script.

Default offline mode (no API calls, ~30 seconds):
    uv run python smoke_test.py

Online mode (makes one tiny LLM call + one provider-configured Anonymizer call):
    uv run python smoke_test.py --online

Exits 0 with "All checks passed." on success, non-zero with a clear remediation
hint per failure.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(REPO_ROOT / ".env", override=False)

PROVIDER_ENV_VARS = (
    "NVIDIA_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
)

# Per-provider config for the online ping in check_online(). Each provider exposes
# an OpenAI-compatible /chat/completions endpoint, so a single openai client class
# can talk to all three by swapping base_url + model id.
PROVIDER_CONFIG = {
    "NVIDIA_API_KEY": {
        "label": "NVIDIA Build",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    },
    "OPENROUTER_API_KEY": {
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "qwen/qwen3.6-flash",
    },
    "OPENAI_API_KEY": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5.4",
    },
}

EXPECTED_SEED_FILES = (
    REPO_ROOT / "data" / "wiki_seed.parquet",
    REPO_ROOT / "data" / "rich_document_seed.parquet",
    REPO_ROOT / "data" / "usage_logs_seed.parquet",
)


# ─── Pretty printing ─────────────────────────────────────────────────────────

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"{GREEN}✓{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}!{RESET} {msg}")


def fail(msg: str) -> str:
    print(f"{RED}✗{RESET} {msg}")
    return msg


def header(msg: str) -> None:
    print(f"\n{BOLD}{msg}{RESET}")


# ─── Checks ──────────────────────────────────────────────────────────────────


def check_python_version() -> str | None:
    major, minor = sys.version_info[:2]
    if major != 3 or minor < 11 or minor > 13:
        return fail(
            f"Python {major}.{minor} detected; the workshop requires Python 3.11, "
            f"3.12, or 3.13. Run `uv sync` -- it will provision the right interpreter."
        )
    ok(f"Python {major}.{minor}.{sys.version_info[2]}")
    return None


def check_imports() -> str | None:
    try:
        import data_designer  # noqa: F401
    except ImportError as exc:
        return fail(
            f"Could not import `data_designer` ({exc}). Run `uv sync` from the repo root."
        )
    ok("data_designer imported")

    try:
        import anonymizer  # noqa: F401
    except ImportError as exc:
        return fail(
            f"Could not import `anonymizer` ({exc}). Note: install name is "
            f"`nemo-anonymizer`; import name is `anonymizer`. Run `uv sync`."
        )
    ok("anonymizer imported (package: nemo-anonymizer)")
    return None


def check_api_key() -> str | None:
    detected = []
    for var in PROVIDER_ENV_VARS:
        value = os.environ.get(var)
        if value:
            masked = value[:6] + "…" + value[-4:] if len(value) > 12 else "set"
            detected.append((var, masked))

    if not detected:
        warn(
            "No provider API key detected. Set one of "
            f"{', '.join(PROVIDER_ENV_VARS)} in your shell or `.env` file before "
            "running the notebooks. (This is fine for the offline smoke test.)"
        )
        return None

    for var, masked in detected:
        ok(f"{var} = {masked}")
    return None


def check_local_pipeline() -> str | None:
    """Build and run a 3-row Data Designer config with sampler columns only.

    No LLM calls, no model providers required. Verifies the framework is
    wired up correctly end to end.
    """
    try:
        import data_designer.config as dd
        from data_designer.interface import DataDesigner
    except ImportError as exc:
        return fail(f"Could not import Data Designer modules: {exc}")

    try:
        config_builder = dd.DataDesignerConfigBuilder()

        config_builder.add_column(
            dd.SamplerColumnConfig(
                name="topic",
                sampler_type=dd.SamplerType.CATEGORY,
                params=dd.CategorySamplerParams(
                    values=["machine learning", "history", "biology"],
                ),
            )
        )
        config_builder.add_column(
            dd.SamplerColumnConfig(
                name="difficulty",
                sampler_type=dd.SamplerType.CATEGORY,
                params=dd.CategorySamplerParams(
                    values=["easy", "medium", "hard"],
                    weights=[0.3, 0.5, 0.2],
                ),
            )
        )

        data_designer = DataDesigner()
        preview = data_designer.preview(config_builder, num_records=3)
    except Exception as exc:
        return fail(
            f"Local pipeline failed: {exc}. This usually means a Data Designer "
            f"version mismatch -- try `uv lock --upgrade && uv sync`."
        )

    rows = len(preview.dataset)
    if rows != 3:
        return fail(f"Expected 3 rows from preview, got {rows}.")
    ok(f"Local pipeline produced {rows} sampler-only rows")
    return None


def check_seed_data() -> None:
    header("Optional data files")

    missing_seed = [p for p in EXPECTED_SEED_FILES if not p.exists()]
    if missing_seed:
        warn(
            "Seed data missing. Seed parquets should be checked in -- try "
            f"`git checkout -- data/` or pull the latest repo. "
            f"({len(missing_seed)} missing file(s).)"
        )
        for p in missing_seed:
            print(f"    {DIM}missing: {p.relative_to(REPO_ROOT)}{RESET}")
    else:
        ok("Seed data present")


def _ping_provider(env_var: str) -> str | None:
    """Send a tiny chat completion to one provider. Returns failure msg or None.

    Uses `max_completion_tokens` (newer name; required by OpenAI reasoning models
    like gpt-5.x and accepted by NVIDIA Build / OpenRouter as an alias for
    `max_tokens`). Budget is generous because reasoning-class models consume
    hidden tokens before producing visible output -- a 4-token budget yields
    empty `content` even on a successful call.
    """
    cfg = PROVIDER_CONFIG[env_var]
    label, base_url, model = cfg["label"], cfg["base_url"], cfg["model"]
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ[env_var], base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word 'ok'."}],
            max_completion_tokens=256,
            temperature=0,
        )
        # Any 200 OK with a choices array proves the endpoint + key + model id
        # are valid. Empty content on reasoning models (all budget went to
        # reasoning tokens) is still a successful call for connectivity purposes.
        text = (resp.choices[0].message.content or "").strip().lower()
        snippet = f" -> '{text[:40]}'" if text else " (empty content; reasoning model)"
        ok(f"{label} reachable via {env_var} (model: {model}){snippet}")
        return None
    except Exception as exc:
        return fail(
            f"{label} call via {env_var} failed: {exc}. "
            f"Verify the key is valid and you have available credit."
        )


def check_online() -> list[str]:
    """Verify every configured provider responds and Anonymizer can run.

    Each provider key set in the environment is pinged independently so attendees
    using more than one (e.g. NVIDIA_API_KEY for class + OPENAI_API_KEY personally)
    learn which keys actually work before the workshop starts.
    """
    header("Online checks (this may take ~30s per provider and incur a small charge)")

    failures: list[str] = []

    set_vars = [v for v in PROVIDER_ENV_VARS if os.environ.get(v)]
    if not set_vars:
        failures.append(fail(
            "No provider API key set. Cannot run online check. "
            f"Set one of {', '.join(PROVIDER_ENV_VARS)} and try again."
        ))
        return failures

    skipped = [v for v in PROVIDER_ENV_VARS if v not in set_vars]
    for var in skipped:
        print(f"{DIM}- {PROVIDER_CONFIG[var]['label']} ({var}) not set, skipping{RESET}")

    for var in set_vars:
        result = _ping_provider(var)
        if result:
            failures.append(result)

    try:
        import csv
        import tempfile

        from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Redact

        sys.path.insert(0, str(REPO_ROOT / "notebooks"))
        from notebook_helpers import build_anonymizer_model_setup, detect_provider

        # The Anonymizer API takes a file source rather than a raw string, so we
        # write a one-row CSV with a single PII-bearing sentence and run preview()
        # against it. This exercises the same hosted detector + LLM endpoints the
        # workshop notebooks will use, without writing any real attendee data.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as tmp:
            writer = csv.writer(tmp)
            writer.writerow(["text"])
            writer.writerow(["My name is Jane Doe and my email is jane@example.com."])
            tmp_path = tmp.name

        provider = detect_provider("auto")
        model_providers, model_configs = build_anonymizer_model_setup(provider)
        anon = Anonymizer(model_providers=model_providers, model_configs=model_configs)
        anon.preview(
            config=AnonymizerConfig(replace=Redact()),
            data=AnonymizerInput(source=tmp_path, text_column="text"),
            num_records=1,
        )
        ok(f"Anonymizer endpoints reachable via {provider.provider_name}")
    except Exception as exc:
        warn(
            f"Anonymizer check skipped or failed: {exc}. "
            f"This is not blocking for the workshop -- Notebook 3 configures "
            f"Anonymizer from whichever provider key is available. NVIDIA Build "
            f"uses hosted GLiNER + LLM stages; OpenRouter/OpenAI use the workshop "
            f"Brev GLiNER endpoint plus your selected LLM provider."
        )
    return failures


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-workshop environment verification."
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Also verify provider connectivity and Anonymizer hosted endpoints.",
    )
    args = parser.parse_args()

    failures: list[str] = []

    header("Environment")
    for check in (check_python_version, check_imports, check_api_key):
        result = check()
        if result:
            failures.append(result)

    header("Data Designer pipeline (offline)")
    result = check_local_pipeline()
    if result:
        failures.append(result)

    check_seed_data()

    if args.online:
        failures.extend(check_online())

    print()
    if failures:
        print(f"{RED}{BOLD}✗ {len(failures)} check(s) failed.{RESET}")
        print(f"{DIM}Fix the issues above and re-run. Need help? Open an issue.{RESET}")
        return 1

    print(f"{GREEN}{BOLD}✓ All checks passed.{RESET}")
    print(f"{DIM}You're ready for the workshop. See you in London!{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
