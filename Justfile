# Sync all notebooks with their python files
jupytext-sync:
  poetry run jupytext --sync **/*.ipynb

# Create the python file from the notebook
jupytext-add file:
  poetry run jupytext --to py {{file}}