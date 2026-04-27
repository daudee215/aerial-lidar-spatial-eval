# Contributing

1. Fork the repository and create a feature branch.
2. Install dev dependencies: `pip install -e ".[dev]"`
3. Run the full quality bar before opening a PR:
   ```bash
   ruff check src tests
   mypy --strict src/aerial_lidar_spatial_eval
   pytest -q
   ```
4. Add or update tests for any new public API.
5. Open a pull request against `main` with a clear description.

See `docs/adr/` for architectural decisions that guide design choices.
