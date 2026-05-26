# Build Training and Evaluation Datasets That Actually Work

> Hands-on synthetic data pipeline workshop -- PyData London 2026

A 90-minute hands-on tutorial that treats synthetic dataset creation as
engineering. You'll build three progressive datasets using two open-source
NVIDIA NeMo tools:

- **[NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner)** --
  declarative synthetic data generation
- **[NeMo Anonymizer](https://github.com/NVIDIA-NeMo/Anonymizer)** -- LLM-powered
  PII detection and replacement

By the end you'll have a working text QA dataset, a multimodal document VQA
dataset, and a privacy workflow for production-style usage data.

## Notebooks

Run the first two notebooks in order if you want the full synthetic-data arc.
Notebook 3 is standalone and uses production-style usage data for the
anonymization workflow.

|   | Notebook | What you'll learn |
|---|----------|-------------------|
| 1 | [`01_text_qa_pipeline.ipynb`](notebooks/01_text_qa_pipeline.ipynb) | Sampler columns, seed datasets, LLM-generated text + structured output columns with Jinja templating, LLM-as-a-judge scoring, quality filtering |
| 2 | [`02_document_visual_qa.ipynb`](notebooks/02_document_visual_qa.ipynb) | Multimodal generation -- VLM-generated summaries, structured questions, and answers grounded in rich synthetic document images, boolean structured judge |
| 3 | [`03_privacy_with_anonymizer.ipynb`](notebooks/03_privacy_with_anonymizer.ipynb) | PII detection across production usage text, side-by-side comparison of `Substitute` vs. `Redact` strategies, the privacy/utility tradeoff |
| bonus | [`bonus_generate_documents.ipynb`](notebooks/bonus_generate_documents.ipynb) | Image generation columns and Nemotron Personas for rich business documents with charts, tables, KPI cards, annotations, and scan artifacts |

Shared utilities live in [`notebooks/notebook_helpers.py`](notebooks/notebook_helpers.py).
Prompt templates and judge rubrics live in [`notebooks/prompts.py`](notebooks/prompts.py).

## Requirements

- macOS, Linux, or Windows (WSL works well)
- Python 3.11 through 3.13; `uv` will provision Python 3.12 automatically from `.python-version`
- 8 GB RAM and roughly 2 GB free disk for the Python environment
- No local model weights or separate Jupyter install required
- Comfort with Python and pandas; one prior LLM API call under your belt

Install `uv` if you do not already have it:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell, if you are not using WSL
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## API Keys

Pick **one** provider. The notebooks auto-detect whichever variable you set in
your shell environment or `.env`:

| Provider | Env var | Why | Sign-up |
|----------|---------|-----|---------|
| **NVIDIA Build** (recommended) | `NVIDIA_API_KEY` | Free-tier credits, hosts the VLM used in Notebook 2 | [build.nvidia.com](https://build.nvidia.com) |
| **OpenRouter** | `OPENROUTER_API_KEY` | Alternative hosted provider | [openrouter.ai](https://openrouter.ai) |
| **OpenAI** | `OPENAI_API_KEY` | Works for all three notebooks | [platform.openai.com](https://platform.openai.com) |

Costs vary by provider and model. Check your provider account before running
larger jobs.

## Quickstart

Run these commands to get set up before running the notebooks. Make sure **one**
provider key is available, either in your shell environment or in `.env`:
`NVIDIA_API_KEY`, `OPENROUTER_API_KEY`, or `OPENAI_API_KEY`.

```bash
git clone https://github.com/nabinchha/pydata-london-2026-data-designer-anonymizer.git
cd pydata-london-2026-data-designer-anonymizer

cp .env.example .env                     # optional if your shell already exports a key
make setup                               # installs deps + runs offline checks
make lab                                 # opens JupyterLab with workshop dark-theme settings
```

`make setup` runs `uv sync`, then `uv run python smoke_test.py`. The smoke test
checks Python, imports, masked API key detection, a tiny local Data Designer
pipeline, and the checked-in seed files. It does not make API calls.

To test your provider key before arriving, run:

```bash
make smoke-online
```

The online check makes one tiny LLM call per configured provider and one small
Anonymizer preview call. It may take about 30 seconds per provider and can incur
a small API charge.

We don't assume you've used Data Designer or Anonymizer before.

## Repo Layout

```
pydata-london-2026-data-designer-anonymizer/
├── notebooks/
│   ├── 01_text_qa_pipeline.ipynb
│   ├── 02_document_visual_qa.ipynb
│   ├── 03_privacy_with_anonymizer.ipynb
│   ├── bonus_generate_documents.ipynb
│   ├── notebook_helpers.py          # env setup, display helpers, data loading
│   └── prompts.py                   # all prompt templates and judge rubrics
├── data/                            # seed parquets (both checked in)
│   ├── wiki_seed.parquet
│   └── rich_document_seed.parquet
├── pyproject.toml + uv.lock         # pinned environment
├── smoke_test.py                    # offline readiness check
└── Makefile                         # short memorable commands
```

## Make Targets

```bash
make help            # list all targets
make setup           # install deps + run offline readiness checks
make install         # install pinned deps
make smoke           # offline smoke test
make smoke-online    # online smoke test (verifies provider key + Anonymizer endpoints)
make lab             # launch JupyterLab with workshop dark-theme settings
```

## Tools and Versions

| Tool | Version | Docs |
|------|---------|------|
| NeMo Data Designer | `data-designer>=0.6.0` | [nvidia-nemo.github.io/DataDesigner](https://nvidia-nemo.github.io/DataDesigner/latest/) |
| NeMo Anonymizer | `nemo-anonymizer>=0.2.0` | [nvidia-nemo.github.io/Anonymizer](https://nvidia-nemo.github.io/Anonymizer/latest/) |

## Background Reading

- [Training a VLM to Understand Long Documents](https://nvidia-nemo.github.io/DataDesigner/latest/devnotes/training-a-vlm-to-understand-long-documents-an-iterative-sdg-story/) -- the
  blog post behind Notebook 2's pipeline pattern
- [Introducing NeMo Anonymizer: Text Anonymization for the Reasoning Era](https://nvidia-nemo.github.io/Anonymizer/latest/devnotes/introducing-nemo-anonymizer-text-anonymization-for-the-reasoning-era/) -- background for Notebook 3's anonymization workflow
- [Data Designer recipes](https://github.com/NVIDIA-NeMo/DataDesigner/tree/main/docs/assets/recipes) -- production-grade SDG recipes including the full VLM long-document pipeline

## Notes

Code: Apache 2.0. Seed data: Wikipedia text excerpts (CC BY-SA 4.0;
source URLs included) and synthetic document images generated by Gemini
through OpenRouter, subject to the applicable provider terms. All generated
outputs are synthetic.
