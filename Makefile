.DEFAULT_GOAL := help

SHELL := /usr/bin/env bash

CONTAINER_COMMAND ?= podman
PYTHON_VERSION ?= 3.13
VENV_DIR ?= .venv

PREFIX ?= ffreis
IMAGE_PROVIDER ?=
IMAGE_TAG ?= api-grpc-smoke
SMOKE_TIMEOUT ?= 20m
BASE_DIR ?= .
CONTAINER_DIR ?= container

IMAGE_PREFIX := $(if $(IMAGE_PROVIDER),$(IMAGE_PROVIDER)/,)$(PREFIX)
IMAGE_ROOT := $(IMAGE_PREFIX)
BASE_IMAGE ?= $(IMAGE_PREFIX)/base
BASE_RUNNER_IMAGE ?= $(IMAGE_PREFIX)/base-runner
UV_VENV_IMAGE ?= $(IMAGE_PREFIX)/onnx-converter-uv-venv
PACKAGE_IMAGE ?= $(IMAGE_PREFIX)/onnx-converter-package
CLI_IMAGE ?= $(IMAGE_PREFIX)/onnx-converter-cli
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
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment already exists at $(VENV_DIR)"; \
	else \
		python$(PYTHON_VERSION) -m venv $(VENV_DIR); \
	fi
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
	$(VENV_DIR)/bin/pytest -q tests/unit_tests

.PHONY: test-integration
test-integration: ## Run integration tests only
	$(VENV_DIR)/bin/pytest -q tests/integration_tests

.PHONY: test-e2e
test-e2e: ## Run end-to-end tests only
	$(VENV_DIR)/bin/pytest -q tests/e2e_tests

.PHONY: grpc-generate
grpc-generate: ## Regenerate gRPC protobuf stubs
	./scripts/generate_grpc_stubs.sh

.PHONY: grpc-check
grpc-check: ## Verify gRPC protobuf stubs are in sync
	./scripts/check_grpc_stubs.sh

.PHONY: grpc-clean
grpc-clean: ## Remove generated gRPC protobuf stubs
	rm -f src/converter_grpc/converter_pb2.py src/converter_grpc/converter_pb2_grpc.py

.PHONY: smoke-api-grpc
smoke-api-grpc: ## Run docker-compose HTTP + gRPC smoke test
	@set -euo pipefail; \
	cleanup() { \
		IMAGE_ROOT="$(IMAGE_ROOT)" IMAGE_TAG="$(IMAGE_TAG)" docker compose -f examples/docker-compose.api-grpc.yml down --remove-orphans || true; \
	}; \
	trap cleanup EXIT; \
	IMAGE_ROOT="$(IMAGE_ROOT)" IMAGE_TAG="$(IMAGE_TAG)" timeout --foreground "$(SMOKE_TIMEOUT)" docker compose -f examples/docker-compose.api-grpc.yml up --build --abort-on-container-exit --exit-code-from smoke

.PHONY: examples-autosklearn
examples-autosklearn: ## Build and run autosklearn example containers
	docker build -f container/examples/Dockerfile.example-base \
		--build-arg BASE_RUNNER_IMAGE="$(BASE_RUNNER_IMAGE)" \
		-t "$(IMAGE_PREFIX)/onnx-converter-examples-base" .
	docker build -f container/examples/Dockerfile.example-autosklearn1 \
		--build-arg EXAMPLES_BASE_IMAGE="$(IMAGE_PREFIX)/onnx-converter-examples-base" \
		-t example-autosklearn-v1 .
	docker run --rm example-autosklearn-v1
	docker build -f container/examples/Dockerfile.example-autosklearn2 \
		--build-arg EXAMPLES_BASE_IMAGE="$(IMAGE_PREFIX)/onnx-converter-examples-base" \
		-t example-autosklearn-v2 .
	docker run --rm example-autosklearn-v2

.PHONY: test-grpc-parity
test-grpc-parity: ## Run gRPC/API parity tests
	$(VENV_DIR)/bin/pytest -q tests/integration_tests/test_grpc_parity.py

.PHONY: test-grpc-parity-property
test-grpc-parity-property: ## Run gRPC/API parity property tests (Hypothesis)
	$(VENV_DIR)/bin/pytest -q tests/integration_tests/test_grpc_parity.py -m property

.PHONY: openapi-check
openapi-check: ## Validate OpenAPI contract and verify runtime drift
	env -u VIRTUAL_ENV uv run --project . --extra server --with openapi-spec-validator --with pyyaml python scripts/check_openapi.py

.PHONY: check
check: grpc-check lint test-unit ## Run lint and fast tests

.PHONY: coverage
coverage: ## Generate coverage report
	mkdir -p coverage
	$(VENV_DIR)/bin/pytest tests/unit_tests --cov=onnx_converter --cov-report=xml:coverage.xml --cov-report=html:coverage/html --cov-report=term

.PHONY: architecture-check
architecture-check: ## Run architecture and complexity checks
	$(VENV_DIR)/bin/python scripts/check_architecture.py
	$(VENV_DIR)/bin/python scripts/check_orchestrator_complexity.py
	$(VENV_DIR)/bin/mypy src/onnx_converter/application

.PHONY: ci-local
ci-local: architecture-check lint test-unit coverage ## Approximate default CI checks locally

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
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.base \
		-t $(BASE_IMAGE) \
		$(BASE_DIR) \
		--build-arg BASE_IMAGE="$(BASE_IMAGE_VALUE)" \
		--build-arg BASE_DIGEST="$(BASE_DIGEST_VALUE)"

.PHONY: build-base-runner
build-base-runner: build-base ## Build base-runner image
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.base-runner -t $(BASE_RUNNER_IMAGE) $(BASE_DIR) \
		--build-arg BASE_IMAGE="$(BASE_IMAGE)"

.PHONY: build-uv-venv
build-uv-venv: build-base ## Build uv venv image
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.uv-builder -t $(UV_VENV_IMAGE) $(BASE_DIR) \
		--build-arg BASE_IMAGE="$(BASE_IMAGE)" \
		--build-arg PYTHON_VERSION="$(PYTHON_VERSION)"

.PHONY: build-package
build-package: build-uv-venv ## Build package image (installs converter)
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.package -t $(PACKAGE_IMAGE) $(BASE_DIR) \
		--build-arg UV_VENV_IMAGE="$(UV_VENV_IMAGE)" \
		--build-arg EXTRAS="$(EXTRAS)"

.PHONY: build-cli
build-cli: build-base-runner build-package ## Build CLI image
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.cli -t $(CLI_IMAGE) $(BASE_DIR) \
		--build-arg PACKAGE_IMAGE="$(PACKAGE_IMAGE)" \
		--build-arg BASE_RUNNER_IMAGE="$(BASE_RUNNER_IMAGE)"

.PHONY: build build-converter-images
build: build-converter-images ## Build all converter images (alias used by CI)
build-converter-images: build-uv-venv build-package build-cli ## Build all converter images
.PHONY: run-cli
run-cli: ## Run converter CLI image (use RUN_ARGS=...)
	$(CONTAINER_COMMAND) run --rm $(CLI_IMAGE) $(RUN_ARGS)

.PHONY: clean-images
clean-images: ## Remove converter images
	$(CONTAINER_COMMAND) rmi $(UV_VENV_IMAGE) $(PACKAGE_IMAGE) $(CLI_IMAGE) || true

.PHONY: ci-grpc
ci-grpc: grpc-check openapi-check lint test-grpc-parity ## Run gRPC sync + parity quality gate
