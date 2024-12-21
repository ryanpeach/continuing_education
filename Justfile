# Add a new notebook to the project
add path:
  uv run jupytext --set-formats ipynb,py:percent {{path}}

# Sync all notebooks
sync:
  uv run jupytext --sync **/*.ipynb
