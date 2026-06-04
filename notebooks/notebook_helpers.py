"""Shared utilities for the PyData London 2026 workshop notebooks.

This module centralises:
  * Environment setup (loads .env, resolves DATA_DESIGNER_HOME, picks a model alias)
  * Model provider auto-detection (NVIDIA Build / OpenRouter / OpenAI)
  * Display helpers (image grids, QA review panels, provider info)
  * Local artifact and seed data paths used by the notebook sequence
  * Seed data loading (wiki excerpts, rich document images)

Prompt templates live in ``prompts.py``.
"""

from __future__ import annotations

import json
import os
import random
import re
import warnings
from dataclasses import dataclass
from html import escape
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from IPython.display import HTML, display
from rich.console import Console
from rich.panel import Panel

# ─── Repo paths ──────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
ARTIFACT_DIR = REPO_ROOT / "artifacts"


# ─── Provider detection ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved model provider for the current notebook session."""

    env_var: str
    api_key: str
    endpoint: str
    provider_name: str
    text_model: str
    vlm_model: str
    text_alias: str = "text-llm"
    judge_alias: str = "judge-llm"
    vlm_alias: str = "vlm"


_PROVIDER_REGISTRY: dict[str, dict[str, str]] = {
    "NVIDIA_API_KEY": {
        "endpoint": "https://integrate.api.nvidia.com/v1",
        "provider_name": "nvidia-build",
        "text_model": "nvidia/nemotron-3-super-120b-a12b",
        "vlm_model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    },
    "OPENROUTER_API_KEY": {
        "endpoint": "https://openrouter.ai/api/v1",
        "provider_name": "openrouter",
        "text_model": "qwen/qwen3.6-flash",
        "vlm_model": "qwen/qwen3.6-flash",
    },
    "OPENAI_API_KEY": {
        "endpoint": "https://api.openai.com/v1",
        "provider_name": "openai",
        "text_model": "gpt-5.4",
        "vlm_model": "gpt-5.4",
    },
}


# Short name → env var. One canonical alias per provider; "auto" means
# "first env var found in _PROVIDER_REGISTRY precedence order".
_PROVIDER_ALIASES: dict[str, str] = {
    "nvidia": "NVIDIA_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
}

_VALID_PROVIDER_NAMES = ("auto", *_PROVIDER_ALIASES)


def _build_config(env_var: str, api_key: str) -> ProviderConfig:
    defaults = _PROVIDER_REGISTRY[env_var]
    return ProviderConfig(
        env_var=env_var,
        api_key=api_key,
        endpoint=defaults["endpoint"],
        provider_name=defaults["provider_name"],
        text_model=defaults["text_model"],
        vlm_model=defaults["vlm_model"],
    )


def detect_provider(provider: str = "auto") -> ProviderConfig:
    """Resolve which model provider this notebook session should use.

    Args:
        provider: Either "auto" (default — first env var found in precedence
            order), or one of "nvidia" / "openrouter" / "openai" to force a
            specific backend regardless of which other keys happen to be set.

    Raises:
        ValueError: when `provider` is not a recognised name.
        EnvironmentError: when the required API key for the chosen provider
            (or any provider, in auto mode) is not set.
    """
    key = provider.lower().strip()
    if key not in _VALID_PROVIDER_NAMES:
        raise ValueError(
            f"Unknown provider {provider!r}. "
            f"Choose one of: {', '.join(_VALID_PROVIDER_NAMES)}."
        )

    if key == "auto":
        for env_var in _PROVIDER_REGISTRY:
            api_key = os.environ.get(env_var)
            if api_key:
                return _build_config(env_var, api_key)
        raise OSError(
            "No model provider key detected. Set one of "
            f"{', '.join(_PROVIDER_REGISTRY)} in your environment or .env file."
        )

    target_env_var = _PROVIDER_ALIASES[key]
    api_key = os.environ.get(target_env_var)
    if not api_key:
        raise OSError(
            f"You requested provider={provider!r} but {target_env_var} is not set. "
            f"Add it to your .env file (see .env.example) and restart the kernel, "
            f'or pass provider="auto" to use whichever key is available.'
        )
    return _build_config(target_env_var, api_key)


# ─── Environment setup ───────────────────────────────────────────────────────


def environment_setup(provider: str = "auto") -> ProviderConfig:
    """Load .env, resolve DATA_DESIGNER_HOME, silence noisy warnings.

    Args:
        provider: Pass `"auto"` (default) to use whichever provider key is set,
            or one of `"nvidia"`, `"openrouter"`, `"openai"` to flip between
            backends explicitly without editing .env.

    Returns the resolved provider config so notebooks can show what they
    will use without re-importing this module.
    """
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)

    os.environ["PROJECT_ROOT"] = str(REPO_ROOT)

    raw_home = os.environ.get("DATA_DESIGNER_HOME", ".data-designer")
    resolved = Path(os.path.expandvars(os.path.expanduser(raw_home)))
    if not resolved.is_absolute():
        resolved = (REPO_ROOT / resolved).resolve()
    os.environ["DATA_DESIGNER_HOME"] = str(resolved)
    resolved.mkdir(parents=True, exist_ok=True)

    # Pandas-style noise from upstream packages — keep notebooks clean.
    warnings.filterwarnings(
        "ignore",
        message=r".*doesn't match a supported version!",
        module=r"requests(\..*)?$",
    )
    warnings.filterwarnings("ignore", message=r"IProgress not found")

    try:
        resolved_provider = detect_provider(provider)
    except (OSError, ValueError) as exc:
        emoji = random.choice(["❌", "🛑", "⚠️"])
        print(f"{emoji} {exc}")
        print("   Copy .env.example to .env and fill in your API key, then restart the kernel.")
        raise

    emoji = random.choice(["🚀", "🔥", "🙌", "⭐️", "✅"])
    mode = "explicit" if provider != "auto" else "auto-detected"
    print(
        f"{emoji} Environment ready  "
        f"({resolved_provider.env_var} {mode} → {resolved_provider.provider_name})"
    )
    return resolved_provider


# ─── Data Designer model providers ───────────────────────────────────────────


def build_model_providers(provider: ProviderConfig):
    """Return the shared ``list[ModelProvider]`` used by *both* DD and Anonymizer.

    ``anonymizer.ModelProvider`` is literally re-exported from
    ``data_designer.config.models`` -- the same class. So the same provider list
    works for ``DataDesigner(model_providers=...)`` and
    ``Anonymizer(model_providers=...)`` without translation.
    """
    import data_designer.config as dd

    return [
        dd.ModelProvider(
            name=provider.provider_name,
            endpoint=provider.endpoint,
            provider_type="openai",
            api_key=provider.env_var,
        ),
    ]


def build_dd_model_setup(provider: ProviderConfig):
    """Return ([model_providers], [model_configs]) wired to the detected provider.

    Used by every notebook that calls an LLM/VLM column. Keeps
    inference_parameters in one place so we can tune them workshop-wide.

    On NVIDIA Build the default text and vision models are Nemotron models.
    We pass ``enable_thinking=True`` explicitly so the provider configuration
    shown in notebooks matches the intended reasoning-mode behavior.
    """
    import data_designer.config as dd

    model_providers = build_model_providers(provider)

    extra_body: dict | None = None
    if provider.provider_name == "nvidia-build":
        extra_body = {"chat_template_kwargs": {"enable_thinking": True}}

    def chat_inference_params(model: str, *, temperature: float | None = None):
        params: dict = {
            "timeout": 120,
            "max_parallel_requests": 8,
        }
        if provider.provider_name == "openai" and model == "gpt-5.4":
            params["extra_body"] = {"max_completion_tokens": 4096}
        else:
            params["max_tokens"] = 4096
            if extra_body:
                params["extra_body"] = extra_body
        if temperature is not None:
            params["temperature"] = temperature
        return dd.ChatCompletionInferenceParams(**params)

    inference = chat_inference_params(provider.text_model)
    judge_inference = chat_inference_params(provider.text_model, temperature=0.1)
    vlm_inference = chat_inference_params(provider.vlm_model)

    model_configs = [
        dd.ModelConfig(
            alias=provider.text_alias,
            model=provider.text_model,
            provider=provider.provider_name,
            inference_parameters=inference,
        ),
        dd.ModelConfig(
            alias=provider.judge_alias,
            model=provider.text_model,
            provider=provider.provider_name,
            inference_parameters=judge_inference,
        ),
        dd.ModelConfig(
            alias=provider.vlm_alias,
            model=provider.vlm_model,
            provider=provider.provider_name,
            inference_parameters=vlm_inference,
        ),
    ]
    return model_providers, model_configs


# ─── Anonymizer model providers ──────────────────────────────────────────────


# Workshop-hosted GLiNER endpoint on Brev. Attendees who don't use NVIDIA
# Build for their LLM stages (i.e. OpenAI / OpenRouter users) point the
# Anonymizer entity-detector at this endpoint so they don't need a Build
# API key just for PII detection. Build users keep using the bundled
# ``nvidia/gliner-pii`` on the Build endpoint they already have a key for.
_BREV_GLINER_ENDPOINT = "https://gliner-qaqtckhiy.brevlab.com/v1"
_BREV_GLINER_PROVIDER_NAME = "nvidia/gliner-pii"

# ─── Anonymizer model pool (Notebook 3 — Substitute only) ───────────────────
#
# Two LLM aliases in ``model_configs``; roles in ``selected_models`` point at them.
# Rewrite mode is not configured here — bundled defaults apply if ever enabled.
#
# Role mapping (workshop):
#   entity_validator        → gpt-oss-120b
#   entity_augmenter        → gpt-oss-120b
#   replacement_generator   → gpt-oss-120b
#   latent_detector         → gpt-oss-120b
#   evaluate.*_judge        → per provider (see ``_ANON_MODEL_ID_EVAL_JUDGE``)

_ANON_ALIAS_GPT_OSS = "gpt-oss-120b"
_ANON_ALIAS_EVAL_JUDGE = "eval-judge"

_ANON_MODEL_ID_GPT_OSS: dict[str, str] = {
    "nvidia-build": "openai/gpt-oss-120b",
    "openrouter": "openai/gpt-oss-120b",
    "openai": "gpt-5.4",
}
# Nemotron 3 Ultra for eval judges (Kimi K2.6 hit Build/OpenRouter rate limits).
_ANON_MODEL_ID_EVAL_JUDGE: dict[str, str] = {
    "nvidia-build": "nvidia/nemotron-3-ultra-550b-a55b",
    "openrouter": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "openai": "gpt-5.4",
}

_ANON_INFERENCE_GPT_OSS = """\
      max_parallel_requests: 16
      max_tokens: 16384
      temperature: 0.3
      top_p: 0.95
      timeout: 300"""
_ANON_INFERENCE_OPENAI = """\
      max_parallel_requests: 1
      temperature: 0.3
      timeout: 300"""
# Eval judges: structured JSON; lower parallelism to ease rate limits on evaluate().
_ANON_INFERENCE_NEMOTRON_ULTRA_EVAL_JUDGE_BUILD = """\
      max_parallel_requests: 4
      max_tokens: 16384
      temperature: 0.3
      top_p: 0.95
      timeout: 300
      extra_body:
        chat_template_kwargs:
          enable_thinking: false"""
_ANON_INFERENCE_NEMOTRON_ULTRA_EVAL_JUDGE_OPENROUTER = """\
      max_parallel_requests: 4
      max_tokens: 16384
      temperature: 0.3
      top_p: 0.95
      timeout: 300
      extra_body:
        reasoning:
          effort: none"""


def _anon_model_id(alias: str, provider: ProviderConfig) -> str:
    registries = {
        _ANON_ALIAS_GPT_OSS: _ANON_MODEL_ID_GPT_OSS,
        _ANON_ALIAS_EVAL_JUDGE: _ANON_MODEL_ID_EVAL_JUDGE,
    }
    try:
        return registries[alias][provider.provider_name]
    except KeyError as exc:
        if alias not in registries:
            raise ValueError(f"Unknown Anonymizer alias {alias!r}") from None
        raise ValueError(
            f"No Anonymizer model id for alias {alias!r} on provider "
            f"{provider.provider_name!r}."
        ) from exc


def _anon_inference_block(provider: ProviderConfig, model_id: str) -> str:
    if provider.provider_name == "openai" and model_id == "gpt-5.4":
        return _ANON_INFERENCE_OPENAI
    if model_id == "openai/gpt-oss-120b":
        return _ANON_INFERENCE_GPT_OSS
    if "nemotron-3-ultra" in model_id:
        if provider.provider_name == "nvidia-build":
            return _ANON_INFERENCE_NEMOTRON_ULTRA_EVAL_JUDGE_BUILD
        return _ANON_INFERENCE_NEMOTRON_ULTRA_EVAL_JUDGE_OPENROUTER
    return (
        "      max_parallel_requests: 8\n"
        "      max_tokens: 16384\n"
        "      temperature: 0.3\n"
        "      timeout: 300"
    )


def _anon_model_config_lines(
    *,
    alias: str,
    provider: ProviderConfig,
    extra_lines: list[str] | None = None,
    skip_health_check: bool = False,
) -> list[str]:
    model_id = _anon_model_id(alias, provider)
    lines = [
        f"  - alias: {alias}",
    ]
    if skip_health_check:
        lines.append("    skip_health_check: true")
    lines.extend(
        [
            f"    model: {model_id}",
            f"    provider: {provider.provider_name}",
            "    inference_parameters:",
            *_anon_inference_block(provider, model_id).splitlines(),
        ]
    )
    if extra_lines:
        lines.extend(extra_lines)
    return lines


def _build_anonymizer_yaml(provider: ProviderConfig) -> str:
    """YAML model pool + ``selected_models`` for Substitute (detection + replace)."""
    gpt_oss = _ANON_ALIAS_GPT_OSS
    eval_judge = _ANON_ALIAS_EVAL_JUDGE

    if provider.provider_name == "nvidia-build":
        gliner_provider = "nvidia-build"
    else:
        gliner_provider = _BREV_GLINER_PROVIDER_NAME

    openai_extra: list[str] = []
    if provider.provider_name == "openai":
        openai_extra = [
            "      extra_body:",
            "        max_completion_tokens: 4096",
        ]

    lines = [
        "model_configs:",
        "  - alias: gliner-pii-detector",
        "    model: nvidia/gliner-pii",
        f"    provider: {gliner_provider}",
        "    inference_parameters:",
        "      max_parallel_requests: 1",
        "      timeout: 120",
        "",
        *_anon_model_config_lines(
            alias=gpt_oss,
            provider=provider,
            extra_lines=openai_extra or None,
        ),
        "",
        # Eval judges run only inside ``evaluate()``; skip DD's startup probe.
        *_anon_model_config_lines(
            alias=eval_judge,
            provider=provider,
            skip_health_check=True,
        ),
        "",
        "selected_models:",
        "  detection:",
        "    entity_detector: gliner-pii-detector",
        f"    entity_validator: [{gpt_oss}]",
        f"    entity_augmenter: {gpt_oss}",
        f"    latent_detector: {gpt_oss}",
        "  replace:",
        f"    replacement_generator: {gpt_oss}",
        "  evaluate:",
        f"    detection_validity_judge: {eval_judge}",
        f"    replace_type_fidelity_judge: {eval_judge}",
        f"    replace_relational_consistency_judge: {eval_judge}",
        f"    replace_attribute_fidelity_judge: {eval_judge}",
    ]
    return "\n".join(lines) + "\n"


def _anonymizer_gpt_oss_model(provider: ProviderConfig) -> str:
    return _anon_model_id(_ANON_ALIAS_GPT_OSS, provider)


def _anonymizer_eval_judge_model(provider: ProviderConfig) -> str:
    return _anon_model_id(_ANON_ALIAS_EVAL_JUDGE, provider)


def _anonymizer_kimi_model(provider: ProviderConfig) -> str:
    """Back-compat alias for benchmark scripts."""
    return _anonymizer_eval_judge_model(provider)


def _brev_gliner_provider():
    """``ModelProvider`` for the workshop's Brev-hosted GLiNER endpoint."""
    import data_designer.config as dd

    return dd.ModelProvider(
        name=_BREV_GLINER_PROVIDER_NAME,
        endpoint=_BREV_GLINER_ENDPOINT,
        provider_type="openai",
    )


def build_anonymizer_model_providers(provider: ProviderConfig):
    """``ModelProvider`` list for ``Anonymizer(...)`` — not always the same as DD.

    Build: one provider (NVIDIA Build) — GLiNER and LLMs both hit integrate.api.
    OpenRouter / OpenAI: attendee provider plus the workshop Brev GLiNER endpoint.
    """
    import data_designer.config as dd

    providers = [
        dd.ModelProvider(
            name=provider.provider_name,
            endpoint=provider.endpoint,
            provider_type="openai",
            api_key=provider.env_var,
        ),
    ]
    if provider.provider_name != "nvidia-build":
        providers.append(_brev_gliner_provider())
    return providers


def build_anonymizer_model_setup(provider: ProviderConfig):
    """Return ``(model_providers, model_configs_yaml)`` ready for ``Anonymizer(...)``.

    Substitute-only workshop mapping:

    - **Validator / augmenter / substitutions** → ``gpt-oss-120b`` (``gpt-5.4`` on OpenAI)
    - **Eval judges** → Nemotron 3 Ultra (``gpt-5.4`` on OpenAI)

    Rewrite roles are not configured; bundled defaults apply if ever enabled.
    """
    model_providers = build_anonymizer_model_providers(provider)
    return model_providers, _build_anonymizer_yaml(provider)


_COLORS = {
    "teal": "#2aa198",
    "green": "#859900",
    "violet": "#6c71c4",
    "orange": "#cb4b16",
    "blue": "#268bd2",
    "magenta": "#d33682",
    "red": "#dc322f",
    "nv_green": "#76b900",
}


# ─── Regex baseline (offline scripts) ────────────────────────────────────────
# N3 inlines the same patterns in-notebook for teaching; these helpers are for
# ``_run_n3_full.py`` and ``_benchmark_n3_providers.py`` only.

REGEX_BASELINE_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL_ADDRESS": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE_NUMBER": re.compile(r"\+?\d[\d\s().-]{7,}\d"),
    "DATE_TIME": re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    "US_ZIP_CODE": re.compile(r"\b\d{5}(?:-\d{4})?\b"),
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}


def regex_baseline_counts(texts: list[str]) -> dict[str, int]:
    """Count regex matches per closed-set PII type across a list of texts."""
    return {
        label: sum(len(pattern.findall(text)) for text in texts)
        for label, pattern in REGEX_BASELINE_PATTERNS.items()
    }


def anonymizer_row_entities(row: object) -> list[object]:
    """Normalize one ``final_entities`` trace cell across Anonymizer versions."""
    if row is None:
        return []
    if isinstance(row, list):
        return row
    if isinstance(row, dict):
        entities = row.get("entities")
        return list(entities) if entities is not None else []
    entities = getattr(row, "entities", None)
    return list(entities) if entities is not None else []


# ─── Display helpers ─────────────────────────────────────────────────────────


def _display_text(value: object) -> str:
    """Format nested values compactly for notebook HTML displays."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def _has_display_text(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().lower() not in {"nan", "<na>"}
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def display_base64_image(png_base64: str, width: int = 600) -> None:
    """Render a base64-encoded PNG inline in the notebook."""
    html = (
        f'<img src="data:image/png;base64,{png_base64}" '
        f'style="max-width:{width}px; border:1px solid #ccc; border-radius:4px;" />'
    )
    display(HTML(html))


def display_image_with_qa(
    png_base64: str,
    question: str,
    answer: object,
    *,
    label: str = "document",
    failure_review: object | None = None,
    show_failure_review: bool = False,
    max_height: int = 620,
) -> None:
    """Render a document image next to a generated QA pair for review."""
    label_html = escape(str(label))
    question_html = escape(_display_text(question))
    answer_html = escape(_display_text(answer))

    failure_review_html = ""
    if show_failure_review or _has_display_text(failure_review):
        if _has_display_text(failure_review):
            failure_review_body = escape(_display_text(failure_review))
        else:
            failure_review_body = "N/A"
        failure_review_html = f"""
        <div style="background:#ffffff; color:#111827; border:1px solid #d1d5db; padding:14px; border-radius:6px; margin-top:10px;">
          <div style="font-weight:700; color:#b45309; font-size:12px; margin-bottom:6px; letter-spacing:0.04em;">FAILURE REVIEW</div>
          <div style="white-space:pre-wrap; line-height:1.45;">{failure_review_body}</div>
        </div>
        """

    html = f"""
    <div style="display:flex; gap:16px; align-items:flex-start; font-family:system-ui, sans-serif;">
      <div style="flex:0 0 auto;">
        <img src="data:image/png;base64,{png_base64}"
             style="max-height:{max_height}px; border:1px solid #cbd5e1; border-radius:4px; background:#fff;" />
        <div style="font-size:12px; color:#64748b; margin-top:6px;">{label_html}</div>
      </div>
      <div style="flex:1 1 auto; min-width:280px; max-width:560px;">
        <div style="background:#ffffff; color:#111827; border:1px solid #d1d5db; padding:14px; border-radius:6px; margin-bottom:10px;">
          <div style="font-weight:700; color:#4d7c0f; font-size:12px; margin-bottom:6px; letter-spacing:0.04em;">QUESTION</div>
          <div style="white-space:pre-wrap; line-height:1.45;">{question_html}</div>
        </div>
        <div style="background:#ffffff; color:#111827; border:1px solid #d1d5db; padding:14px; border-radius:6px;">
          <div style="font-weight:700; color:#0369a1; font-size:12px; margin-bottom:6px; letter-spacing:0.04em;">ANSWER</div>
          <div style="white-space:pre-wrap; line-height:1.45;">{answer_html}</div>
        </div>
        {failure_review_html}
      </div>
    </div>
    """
    display(HTML(html))


# ─── Seed data loading helpers ───────────────────────────────────────────────


def load_wiki_excerpts(limit: int | None = None) -> pd.DataFrame:
    """Load wiki excerpts (checked into the repo at ``data/wiki_seed.parquet``)."""
    path = DATA_DIR / "wiki_seed.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.relative_to(REPO_ROOT)} missing. "
            f"Try `git checkout -- data/` or pull the latest repo."
        )
    df = pd.read_parquet(path)
    if limit:
        df = df.head(limit)
    return df


def load_usage_logs_seed() -> pd.DataFrame:
    """Load VLM deployment logs in production ChatML shape (checked into the repo).

    Each row is one logged session from a document-QA copilot deployment
    (training documents of the kind Notebook 2 generates from
    ``data/rich_document_seed.parquet``). Three text columns
    teach the "PII at four surfaces" point:

    - ``system_prompt`` -- application-built system message naming the
      authenticated user (name, email, employer, role, project codename,
      colleagues).
    - ``attached_document_context`` -- plain-text rendering of one
      uploaded business document (clinical trial status report, financial
      variance memo, market research brief, product launch readiness plan,
      operations dashboard export, or customer support incident review).
    - ``conversation_trace`` -- a multi-turn ChatML JSON string with user
      messages, assistant tool calls (with JSON arguments), and tool-result
      messages (with JSON returned by the model's tools).

    Generated by ``bonus_generate_usage_logs.ipynb``.
    """
    path = DATA_DIR / "usage_logs_seed.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.relative_to(REPO_ROOT)} missing. "
            f"Run notebooks/bonus_generate_usage_logs.ipynb to regenerate it, "
            f"or pull from the repo (it should be checked in)."
        )
    return pd.read_parquet(path)


def load_document_seed() -> pd.DataFrame:
    """Load the rich document image seed parquet (checked into the repo).

    The parquet contains a ``png_base64`` column with each document image already
    encoded, plus four orientation fields: ``primary_visual``,
    ``secondary_visual``, ``layout_style``, and ``document_condition``.
    """
    path = DATA_DIR / "rich_document_seed.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.relative_to(REPO_ROOT)} missing. "
            "Pull the latest repo; this seed file should be checked in."
        )
    return pd.read_parquet(path)


# ─── Pretty info panel ───────────────────────────────────────────────────────


def show_provider_info(provider: ProviderConfig) -> None:
    """Print a Rich panel summarising the resolved provider for the notebook."""
    console = Console()
    body = (
        f"[bold]Provider:[/] {provider.provider_name}\n"
        f"[bold]Endpoint:[/] {provider.endpoint}\n"
        f"[bold]Text model alias:[/] [cyan]{provider.text_alias}[/] "
        f"-> {provider.text_model}\n"
        f"[bold]Judge model alias:[/] [cyan]{provider.judge_alias}[/] "
        f"-> {provider.text_model} (temperature=0.1)\n"
        f"[bold]VLM model alias:[/] [cyan]{provider.vlm_alias}[/] "
        f"-> {provider.vlm_model}"
    )
    console.print(
        Panel(
            body,
            title="[bold]Model setup[/]",
            border_style=_COLORS["nv_green"],
            padding=(1, 2),
        )
    )


__all__ = [
    "ARTIFACT_DIR",
    "DATA_DIR",
    "REPO_ROOT",
    "ProviderConfig",
    "build_anonymizer_model_setup",
    "build_dd_model_setup",
    "build_model_providers",
    "REGEX_BASELINE_PATTERNS",
    "anonymizer_row_entities",
    "display_base64_image",
    "display_image_with_qa",
    "environment_setup",
    "regex_baseline_counts",
    "load_document_seed",
    "load_usage_logs_seed",
    "load_wiki_excerpts",
    "show_provider_info",
]
