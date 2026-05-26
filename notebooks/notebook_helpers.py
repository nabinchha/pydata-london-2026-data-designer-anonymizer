"""Shared utilities for the PyData London 2026 workshop notebooks.

This module centralises:
  * Environment setup (loads .env, resolves DATA_DESIGNER_HOME, picks a model alias)
  * Model provider auto-detection (NVIDIA Build / OpenRouter / OpenAI)
  * Display helpers (column traces, image grids, anonymizer side-by-side)
  * Local artifact and seed data paths used by the notebook sequence
  * Seed data loading (wiki excerpts, rich document images)

Prompt templates live in ``prompts.py``.
"""

from __future__ import annotations

import json
import os
import random
import warnings
from dataclasses import dataclass
from html import escape
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from IPython.display import HTML, display
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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


def build_dd_model_setup(provider: ProviderConfig):
    """Return ([model_providers], [model_configs]) wired to the detected provider.

    Used by every notebook that calls an LLM/VLM column. Keeps
    inference_parameters in one place so we can tune them workshop-wide.
    """
    import data_designer.config as dd

    model_providers = [
        dd.ModelProvider(
            name=provider.provider_name,
            endpoint=provider.endpoint,
            provider_type="openai",
            api_key=provider.env_var,
        ),
    ]

    def chat_inference_params(model: str, *, temperature: float | None = None):
        params = {
            "timeout": 120,
            "max_parallel_requests": 8,
        }
        if provider.provider_name == "openai" and model == "gpt-5.4":
            params["extra_body"] = {"max_completion_tokens": 4096}
        else:
            params["max_tokens"] = 4096
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



# ─── Display helpers ─────────────────────────────────────────────────────────

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


# ─── Anonymizer side-by-side display (Notebook 3) ────────────────────────────


def display_anonymizer_comparison(
    original: list[str],
    by_strategy: dict[str, list[str]],
    max_rows: int = 5,
) -> None:
    """Render original vs each strategy's output for visual comparison."""
    console = Console()
    rows = min(max_rows, len(original))
    for i in range(rows):
        table = Table(
            title=f"[bold]Row {i + 1}[/]",
            border_style="dim",
            show_lines=True,
            title_style="",
        )
        table.add_column("Strategy", style=_COLORS["nv_green"], no_wrap=True)
        table.add_column("Output")
        table.add_row("[bold]ORIGINAL[/]", original[i])
        for strategy, outputs in by_strategy.items():
            if i < len(outputs):
                table.add_row(strategy, outputs[i])
        console.print(table)
        print()


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


def load_document_seed() -> pd.DataFrame:
    """Load the rich document image seed parquet (checked into the repo).

    The parquet contains a ``png_base64`` column with each document image already
    encoded, plus metadata columns (document_type, primary_visual, layout_style,
    etc.).
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
        f"-> {provider.text_model} (temperature=0.0)\n"
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
    "build_dd_model_setup",
    "display_anonymizer_comparison",
    "display_base64_image",
    "display_image_with_qa",
    "environment_setup",
    "load_document_seed",
    "load_wiki_excerpts",
    "show_provider_info",
]
