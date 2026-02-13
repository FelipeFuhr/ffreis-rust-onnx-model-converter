.DEFAULT_GOAL := help

SHELL := /usr/bin/env bash

CONTAINER_COMMAND ?= podman
PYTHON_VERSION ?= 3.12
VENV_DIR ?= .venv

PREFIX ?= ffreis
BASE_DIR ?= .
CONTAINER_DIR ?= container

UV_VENV_IMAGE ?= $(PREFIX)/onnx-converter-uv-venv
PACKAGE_IMAGE ?= $(PREFIX)/onnx-converter-package
CLI_IMAGE ?= $(PREFIX)/onnx-converter-cli
EXTRAS ?= all

BASE_IMAGE_VALUE := $(shell grep '^BASE_IMAGE=' $(CONTAINER_DIR)/digests.env | cut -d= -f2)
BASE_DIGEST_VALUE := $(shell grep '^BASE_DIGEST=' $(CONTAINER_DIR)/digests.env | cut -d= -f2)

.PHONY: help
help: ## Show help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: all
all: check ## Run lint and tests

# ------------------------------------------------------------------------------
# Local development
# ------------------------------------------------------------------------------

.PHONY: env
env: ## Create virtual environment
	python$(PYTHON_VERSION) -m venv $(VENV_DIR)
	@echo "Activate with: . $(VENV_DIR)/bin/activate"

.PHONY: install
install: ## Install runtime dependencies
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -e "./[${EXTRAS}]"

.PHONY: install-dev
install-dev: ## Install dev tooling
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -e "./[all,dev]"

.PHONY: format
format: ## Format code (ruff + black + isort)
	$(VENV_DIR)/bin/ruff format src tests
	$(VENV_DIR)/bin/black src tests
	$(VENV_DIR)/bin/isort src tests

.PHONY: lint
lint: ## Lint code (ruff + flake8 + mypy)
	$(VENV_DIR)/bin/ruff check src tests
	$(VENV_DIR)/bin/flake8 src tests
	$(VENV_DIR)/bin/mypy src

.PHONY: test
test: ## Run tests
	$(VENV_DIR)/bin/pytest -q

.PHONY: test-unit
test-unit: ## Run unit tests only
	$(VENV_DIR)/bin/pytest -q -m "not integration"

.PHONY: test-integration
test-integration: ## Run integration tests only
	$(VENV_DIR)/bin/pytest -q -m integration

.PHONY: check
check: lint test-unit ## Run lint and fast tests

.PHONY: coverage
coverage: ## Generate coverage report
	mkdir -p coverage
	$(VENV_DIR)/bin/pytest -m "not integration" --cov=onnx_converter --cov-report=xml:coverage.xml --cov-report=html:coverage/html --cov-report=term

.PHONY: deps-sync-check
deps-sync-check: ## Verify requirements.txt is synced with pyproject.toml
	$(VENV_DIR)/bin/python scripts/check_dependencies_sync.py

.PHONY: deps-sync-generate
deps-sync-generate: ## Regenerate requirements.txt from pyproject.toml
	$(VENV_DIR)/bin/python scripts/generate_requirements.py

.PHONY: architecture-check
architecture-check: ## Run architecture and complexity checks
	$(VENV_DIR)/bin/python scripts/check_architecture.py
	$(VENV_DIR)/bin/python scripts/check_orchestrator_complexity.py
	$(VENV_DIR)/bin/mypy src/onnx_converter/application

.PHONY: ci-local
ci-local: deps-sync-check architecture-check lint test-unit coverage ## Approximate default CI checks locally

.PHONY: clean
clean: ## Remove caches and venv
	rm -rf $(VENV_DIR) .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov coverage.xml coverage
	find . -type d -name '__pycache__' -exec rm -r {} +
	find . -type f -name '*.py[cod]' -delete

# ------------------------------------------------------------------------------
# Container builds
# ------------------------------------------------------------------------------

.PHONY: build-base
build-base: ## Build base image (pinned by digest env)
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.base -t $(PREFIX)/base $(BASE_DIR) \
		--build-arg BASE_IMAGE="$(BASE_IMAGE_VALUE)" \
		--build-arg BASE_DIGEST="$(BASE_DIGEST_VALUE)"

.PHONY: build-base-runner
build-base-runner: build-base ## Build base-runner image
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.base-runner -t $(PREFIX)/base-runner -t $(PREFIX)/base-runner:local $(BASE_DIR)

.PHONY: build-uv-venv
build-uv-venv: build-base ## Build uv venv image
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.uv-builder -t $(UV_VENV_IMAGE) $(BASE_DIR)

.PHONY: build-package
build-package: build-uv-venv ## Build package image (installs converter)
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.package -t $(PACKAGE_IMAGE) $(BASE_DIR) \
		--build-arg UV_VENV_IMAGE="$(UV_VENV_IMAGE)" \
		--build-arg EXTRAS="$(EXTRAS)"

.PHONY: build-cli
build-cli: build-base-runner build-package ## Build CLI image
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.cli -t $(CLI_IMAGE) $(BASE_DIR) \
		--build-arg PACKAGE_IMAGE="$(PACKAGE_IMAGE)"

.PHONY: build build-converter-images
build: build-converter-images ## Build all converter images (alias used by CI)
build-converter-images: build-uv-venv build-package build-cli ## Build all converter images
.PHONY: run-cli
run-cli: ## Run converter CLI image (use RUN_ARGS=...)
	$(CONTAINER_COMMAND) run --rm $(CLI_IMAGE) $(RUN_ARGS)

.PHONY: clean-images
clean-images: ## Remove converter images
	$(CONTAINER_COMMAND) rmi $(UV_VENV_IMAGE) $(PACKAGE_IMAGE) $(CLI_IMAGE) || true
