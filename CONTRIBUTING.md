# Contributing to LoopShield™

Thank you for your interest in contributing to LoopShield! We welcome community contributions, bug reports, framework adapters, and architectural optimizations.

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/gabiUlianitski/loopshield.git
   cd loopshield
   ```
3. Set up a local virtual environment and install the development dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -e ".[all]"
   ```

## Running the Simulation Suite

We maintain strict test-driven development standards. Before submitting any pull request, ensure that the entire simulation suite passes flawlessly:

```bash
pytest
```
Your changes must not break any of the 68 automated verification simulations checking directed graph cycles, semantic thought loops, proxy interceptions, or framework ecosystem middleware layers.

## Pull Request Guidelines

- **Branching:** Create a descriptive feature branch (e.g., `feature/add-autogen-wrapper` or `bugfix/fix-proxy-stream-edgecase`).
- **Type Hints:** All new Python code must include strict type hints using Pydantic validation boundaries where applicable.
- **Documentation:** If you add configuration variables or framework hooks, update the `README.md` file accordingly.
- **Symmetry:** Any structural parsing updates must follow our clean transcript contract guidelines (never leave orphaned tool calls).
