.DEFAULT_GOAL := help

JUPYTERLAB_SETTINGS_DIR := $(CURDIR)/.jupyter/lab/user-settings
UV := $(shell command -v uv 2>/dev/null)

# ─── Help ────────────────────────────────────────────────────────────────────

help:                          ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ─── Attendee targets ────────────────────────────────────────────────────────

check-uv:                      ## Verify uv is installed
	@if [ -z "$(UV)" ]; then \
		echo "uv is not installed or not on PATH."; \
		echo ""; \
		echo "Install uv, then rerun make:"; \
		echo "  macOS/Linux:"; \
		echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		echo "  Windows PowerShell:"; \
		echo '    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'; \
		echo ""; \
		echo "More options: https://docs.astral.sh/uv/getting-started/installation/"; \
		exit 1; \
	else \
		echo "uv found: $$($(UV) --version)"; \
	fi

install: check-uv              ## Install pinned deps into .venv
	uv sync

setup: install smoke           ## Install deps and run offline readiness checks

smoke: check-uv                ## Fast offline smoke test (~30s, no API calls)
	uv run python smoke_test.py

smoke-test: smoke

smoke-online: check-uv         ## Online smoke test (verifies provider key + Anonymizer endpoints)
	uv run python smoke_test.py --online

lab: check-uv                  ## Launch JupyterLab with workshop settings
	uv run jupyter lab --LabApp.user_settings_dir="$(JUPYTERLAB_SETTINGS_DIR)"

# ─── Maintainer targets ──────────────────────────────────────────────────────

lock: check-uv                 ## Upgrade and re-lock dependencies
	uv lock --upgrade

clean:                         ## Remove .venv, generated artifacts, and caches
	rm -rf .venv .data-designer artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +

.PHONY: help check-uv install setup smoke smoke-test smoke-online lab lock clean
