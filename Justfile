sync:
  poetry run jupytext --sync **/*.ipynb

test: sync
  poetry run pytest .

lint: sync
  poetry run ruff check
  poetry run mypy .

fmt: sync
  poetry run ruff format
