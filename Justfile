# Add a new notebook to the project
add path:
  poetry run jupytext --set-formats ipynb,py:percent {{path}}

# Sync all notebooks
sync:
  poetry run jupytext --sync **/*.ipynb

# Run the linters
check: sync
  poetry run ruff check
  poetry run mypy continuing_education

# Fix the code based on ruff
fix: sync
  poetry run ruff check --fix
  poetry run ruff format
