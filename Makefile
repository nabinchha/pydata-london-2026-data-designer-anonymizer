.DEFAULT_GOAL := help

JUPYTERLAB_SETTINGS_DIR := $(CURDIR)/.jupyter/lab/user-settings

# ─── Help ────────────────────────────────────────────────────────────────────

help:                          ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ─── Attendee targets ────────────────────────────────────────────────────────

install:                       ## Install pinned deps into .venv
	uv sync

setup: install smoke           ## Install deps and run offline readiness checks

smoke:                         ## Fast offline smoke test (~30s, no API calls)
	uv run python smoke_test.py

smoke-online:                  ## Online smoke test (verifies provider key + Anonymizer endpoints)
	uv run python smoke_test.py --online

lab:                           ## Launch JupyterLab with workshop settings
	uv run jupyter lab --LabApp.user_settings_dir="$(JUPYTERLAB_SETTINGS_DIR)"

# ─── Maintainer targets ──────────────────────────────────────────────────────

lock:                          ## Upgrade and re-lock dependencies
	uv lock --upgrade

clean:                         ## Remove .venv, generated artifacts, and caches
	rm -rf .venv .data-designer artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +

.PHONY: help install setup smoke smoke-online lab lock clean
