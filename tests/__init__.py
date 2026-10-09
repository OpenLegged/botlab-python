# Makes `tests` a regular package so test modules can share helpers via
# `from tests.conftest import ...` under every invocation. Without this file
# the import only resolves when the caller happens to put the repo root on
# sys.path (e.g. `python -m pytest`); the bare `pytest` console script that
# CI uses (`uv run pytest`) does not, and collection fails with
# ModuleNotFoundError: No module named 'tests'.
