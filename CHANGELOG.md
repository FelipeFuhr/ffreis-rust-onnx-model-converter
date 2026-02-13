# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
<!-- This section will be updated with a version number and date when released (e.g., [1.0.0] - YYYY-MM-DD) -->

### Major Architectural Refactoring

This release represents a complete architectural overhaul of the ONNX Model Converter project. The changes are extensive and touch nearly every aspect of the codebase.

#### Architecture Changes

**New Layered Architecture:**
- **Application Layer** (`src/onnx_converter/application/`):
  - Introduced use-case-driven orchestration pattern
  - Added typed configuration objects (`ConversionOptions`, `ParityOptions`, `PostprocessOptions`)
  - Implemented port-adapter pattern for framework independence
  - New modules: `use_cases.py`, `options.py`, `ports.py`, `results.py`

- **Adapter Layer** (`src/onnx_converter/adapters/`):
  - Framework-specific adapter implementations
  - Separated concerns: model loading, conversion, and parity checking
  - New modules: `loaders.py`, `converters.py`, `parity_checkers.py`

- **Infrastructure Layer** (`src/onnx_converter/infrastructure/`):
  - ONNX post-processing implementation details
  - New module: `postprocessing.py`

- **Plugin System** (`src/onnx_converter/plugins/`):
  - Extensible plugin protocol for custom model families
  - Plugin registry with discovery and validation
  - Built-in plugins for common frameworks
  - New modules: `base.py`, `registry.py`, `builtins.py`

**Refactored Core Modules:**
- `src/onnx_converter/api.py`: Simplified to be a thin wrapper over application use-cases (320 lines removed, cleaner interface)
- `src/onnx_converter/converters/`: Updated to work with new architecture
  - `pytorch_converter.py`: Enhanced with better error handling
  - `sklearn_converter.py`: Improved integration with adapter layer
  - `tensorflow_converter.py`: Updated for consistency

#### New Features

**Parity Checking:**
- Automatic validation that ONNX model outputs match source framework outputs
- Configurable tolerance and comparison strategies
- New module: `src/onnx_converter/parity.py` (129 lines)

**Post-Processing:**
- ONNX model metadata management
- Optimization passes
- Dynamic quantization support
- New module: `src/onnx_converter/postprocess.py` (76 lines)

**Plugin System:**
- Support for custom model families (e.g., AutoML frameworks)
- Runtime plugin discovery and loading
- Built-in plugins for sklearn, pytorch, tensorflow
- Example plugin for AutoSklearn integration
- Security warnings for plugin loading

**Enhanced CLI:**
- New `custom` command for plugin-based conversions
- New `doctor` command for environment diagnostics
- Improved command structure and help messages
- Better error messages and user guidance
- Enhanced: `src/onnx_converter/cli/cli.py` (+387 lines)

**Validation and Error Handling:**
- Comprehensive input validation with Pydantic schemas
- Better error messages with actionable guidance
- Type-safe configuration objects
- Enhanced: `src/onnx_converter/schemas.py` and `src/onnx_converter/errors.py`

#### Documentation

**New Documentation:**
- `docs/architecture.md`: Comprehensive architecture overview with sequence diagrams, layer boundaries, and design principles (119 lines)
- `docs/plugin-authoring.md`: Complete guide for writing custom plugins (148 lines)
- `README.md`: Complete rewrite with architecture overview, request flow, and updated usage examples (271 lines modified)

**Updated Documentation:**
- Installation instructions with feature extras
- CLI usage with new commands
- Python API examples updated for new patterns
- Developer workflow documentation
- Security and serialization notes

#### Examples

**New Example Scripts:**
- `examples/autosklearn_plugin.py`: Plugin implementation for AutoSklearn (77 lines)
- `examples/autosklearn_roundtrip.py`: End-to-end AutoSklearn conversion workflow (145 lines)
- `examples/compare_sklearn_vs_onnx.py`: Parity checking demonstration (100 lines)
- `examples/optuna_sklearn_example.py`: Optuna hyperparameter optimization integration (117 lines)

**Updated Example Scripts:**
- `examples/pytorch_example.py`: Enhanced with parity checks and post-processing (+126 lines modified)
- `examples/sklearn_example.py`: Updated for new API and validation (+155 lines modified)
- `examples/sklearn_custom_cli_example.py`: Demonstrates custom CLI usage (+73 lines modified)
- `examples/tensorflow_example.py`: Updated with new features (+106 lines modified)

#### Testing

**New Test Suites:**
- `tests/unit_tests/application/test_use_case_contracts.py`: Application layer contract tests (131 lines)
- `tests/unit_tests/adapters/test_adapter_contracts.py`: Adapter interface compliance tests (82 lines)
- `tests/unit_tests/plugins/test_builtin_sklearn_plugin.py`: Plugin system tests (135 lines)
- `tests/conftest.py`: Shared test fixtures and helpers (16 lines)

**Updated Tests:**
- Refactored unit tests for new architecture
- Enhanced test coverage for edge cases
- Integration test improvements

#### CI/CD Workflows

**New Workflows:**
- `.github/workflows/architecture.yml`: Validates architectural boundaries, dependency sync, and complexity checks (44 lines)
- `.github/workflows/examples.yml`: Dockerized example execution (fast PR set + scheduled comprehensive set) (170 lines)
- `.github/workflows/integration.yml`: Scheduled/manual integration test runs (35 lines)
- `.github/workflows/lint.yml`: Ruff and mypy linting (38 lines)

**Updated Workflows:**
- `.github/workflows/build-python.yml`: Enhanced matrix testing (3.10, 3.11, 3.12)
- `.github/workflows/coverage.yml`: Updated coverage configuration
- `.github/workflows/code-security.yml`: Updated for new structure
- `.github/workflows/commit-checks.yml`: Updated validation

#### Container Support

**New Dockerfiles:**
- `container/Dockerfile.example-base`: Base image for all examples (16 lines)
- `container/Dockerfile.example-pytorch`: PyTorch example container (7 lines)
- `container/Dockerfile.example-tensorflow`: TensorFlow example container (7 lines)
- `container/Dockerfile.example-sklearn`: Sklearn example container (6 lines)
- `container/Dockerfile.example-sklearn-custom-cli`: Custom CLI example container (6 lines)
- `container/Dockerfile.example-compare-sklearn`: Comparison example container (6 lines)
- `container/Dockerfile.example-optuna-sklearn`: Optuna example container (6 lines)
- `container/Dockerfile.autosklearn1`: AutoSklearn 1.x runtime (18 lines)
- `container/Dockerfile.autosklearn2`: AutoSklearn 2.x runtime (18 lines)

#### Build and Development Tools

**New Scripts:**
- `scripts/check_architecture.py`: Validates layer boundaries and imports (53 lines)
- `scripts/check_dependencies_sync.py`: Ensures requirements.txt matches pyproject.toml (61 lines)
- `scripts/check_orchestrator_complexity.py`: Monitors use-case complexity (32 lines)
- `scripts/generate_requirements.py`: Generates requirements.txt from pyproject.toml (37 lines)

**Updated Build Configuration:**
- `pyproject.toml`: Updated dependencies, added new optional feature groups (optuna, runtime), updated metadata (45 lines modified)
- `requirements.txt`: Synchronized with new dependencies (29 lines modified)
- `Makefile`: New targets for architecture checks, dependency sync, CI simulation (34 lines modified)

#### Dependency Changes

**New Dependencies:**
- Plugin system dependencies
- Parity checking libraries
- Development tools for architecture validation

**Updated Dependencies:**
- Framework version updates
- Testing library updates
- Build tool updates

#### Breaking Changes

⚠️ **API Changes:**
- Module structure completely reorganized
- Import paths changed (e.g., new `application`, `adapters`, `plugins` modules)
- Function signatures updated in some cases
- Configuration objects now use typed dataclasses/Pydantic models

⚠️ **CLI Changes:**
- New command structure with `custom` and `doctor` commands
- Some flag names may have changed
- Enhanced validation may reject previously accepted inputs

#### Migration Guide

**For Library Users:**
1. Update import statements to use new module structure
2. Replace direct converter calls with application use-cases or API wrappers
3. Update configuration to use typed options objects
4. Review error handling for new exception types

**For CLI Users:**
1. Review help text for updated command syntax
2. Use `convert-to-onnx doctor` to diagnose environment issues
3. Use `convert-to-onnx custom` for plugin-based conversions
4. Update scripts to use new flag names if changed

**For Contributors:**
1. Review `docs/architecture.md` for design principles
2. Use `make architecture-check` to validate layer boundaries
3. Follow new patterns in `src/onnx_converter/application/` for orchestration
4. Create plugins for new model families rather than modifying core

#### Statistics

- **69 files changed**
- **7,354 insertions**
- **686 deletions**
- **4 new directories** (application, adapters, infrastructure, plugins)
- **20+ new modules**
- **9 new Dockerfiles**
- **4 new CI workflows**
- **4 new development scripts**
- **2 new documentation files**
- **4 new example scripts**

### Commits

- `bf9b10c`: feat: add improvements
- `ec9e490`: feat: more tests (note: commit message does not reflect full scope of changes)

---

## Notes on This Release

This release represents a fundamental shift in how the project is structured and should be considered a major version change. The previous commit messages ("feat: more tests", "feat: add improvements") do not adequately capture the scope of these changes, which include:

1. **Complete architectural refactoring** with new layered structure
2. **New plugin system** for extensibility
3. **New features** (parity checking, post-processing, validation)
4. **Major CLI enhancements** with new commands
5. **Comprehensive documentation** additions
6. **CI/CD infrastructure** expansion
7. **Container definitions** for all examples
8. **Extensive testing** additions

This changelog provides the comprehensive overview that the commit messages did not capture.
