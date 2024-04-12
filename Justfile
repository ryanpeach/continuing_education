# Add a new notebook to the project
add path:
  poetry run --set-formats ipynb,py:percent {{path}}

# Sync all notebooks
sync:
  poetry run jupytext --sync **/*.ipynb

# Run the tests
test:
  poetry run pytest continuing_education

# Run the linters
check:
  poetry run ruff check
  poetry run mypy .

# Fix the code based on ruff
fix:
  poetry run ruff check --fix
  poetry run ruff format
