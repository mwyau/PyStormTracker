# Contributing to PyStormTracker

Thank you for your interest in contributing to PyStormTracker! We welcome contributions from the community to help improve this cyclone tracking framework.

## How to Contribute

### Reporting Bugs
If you find a bug, please [open an issue](https://github.com/mwyau/PyStormTracker/issues) on GitHub. Include:
- A clear description of the issue.
- Steps to reproduce the bug.
- Any relevant logs or error messages.
- Information about your environment (OS, Python version, PyStormTracker version).

### Suggesting Enhancements
We welcome ideas for new features or improvements. Please [open an issue](https://github.com/mwyau/PyStormTracker/issues) to discuss your proposal before starting implementation.

### Submitting Pull Requests
1.  **Fork the repository** and create your branch from `main`.
2.  **Install development dependencies**:
    ```bash
    uv sync
    ```
3.  **Make your changes**:
    - Ensure your code follows the project's style (use `ruff` for formatting and linting).
    - Add or update tests for your changes.
    - Ensure all tests pass.
4.  **Run Quality Control Checks**:
    ```bash
    uv run ruff check . --fix
    uv run ruff format .
    uv run mypy src/
    uv run pytest
    ```
5.  **Submit the Pull Request**:
    - Provide a clear description of the changes.
    - Reference any related issues.

## Development Standards

- **Code Style**: We use `ruff` for linting and formatting.
- **Type Safety**: All new code should be type-hinted and pass `mypy` strict checks.
- **Testing**: We use `pytest`. Ensure that your changes are covered by unit tests and, if applicable, integration tests.
- **Documentation**: Update the documentation in `docs/` if you introduce new features or change existing behavior.

## Governance

This project is currently maintained by [Albert M. W. Yau](https://github.com/mwyau). Decisions regarding pull requests and the project's roadmap are made by the maintainer in consultation with the community.

## Conduct

Please be respectful and professional in all interactions within the PyStormTracker community.
