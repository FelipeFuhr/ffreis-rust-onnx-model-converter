# Contributing

## Documentation Standard

All new or changed production code in `src/` must be documented with NumPy-style
docstrings.

### Required

- Module docstrings for public modules.
- Class docstrings for public classes.
- Function/method docstrings for public callables.
- `Parameters` and `Returns` sections when applicable.
- `Raises` section for user-facing errors or important failure modes.

### Commenting Rules

- Prefer clear code over excessive comments.
- Use inline comments only for non-obvious intent, invariants, or tradeoffs.
- Do not add comments that restate obvious code behavior.
- Keep comments short and factual.

### Tooling Enforcement

- Ruff enforces docstrings in `src/` using pydocstyle (`D*`) with NumPy
  convention.
- Pre-commit and CI run Ruff; docstring violations in production code fail checks.

## Local Validation

Run these before opening a PR:

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest -q -m unit
```
