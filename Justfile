# Add a new notebook to the project
add path:
  uv run jupytext --set-formats ipynb,py:percent {{path}}

# Sync all notebooks
sync:
  uv run jupytext --sync **/*.ipynb

# Run the linters
check: sync
  uv run ruff check
  uv run mypy continuing_education

# Fix the code based on ruff
fix: sync
  uv run ruff check --fix
  uv run ruff format
