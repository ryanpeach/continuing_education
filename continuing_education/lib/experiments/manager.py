from pathlib import Path
import os
from pprint import pformat
import warnings
from git import Repo

class ExperimentManager:
    """
    A simple git based experiment manager.
    Put this at the end of your training script, and when it runs, it will commit the changes to the current git branch with the results.
    """
    def __init__(self, *, name: str, file: Path, description: str = "", primary_metric: str = "") -> None:
        self.name = name
        self.description = description
        self.file = __file__ if file is None else file
        self.repo = Repo(search_parent_directories=True)
        self.primary_metric = primary_metric

    @property
    def is_jupytext(self) -> bool:
        return self.file.suffix.endswith(".ipynb") and Path(self.file).with_suffix(".py").exists()
    
    def run_jupytext_sync(self):
        if self.is_jupytext:
            os.system(f"jupytext --sync {self.file}")

    def commit(self, metrics: dict[str, float] | None = None):
        if metrics is None:
            metrics = {}
        self.run_jupytext_sync()

        # Staging files
        self.repo.git.add(self.file)
        if self.is_jupytext:
            self.repo.git.add(Path(self.file).with_suffix('.py'))

        # Check for unstaged changes
        if self.repo.is_dirty(untracked_files=True):
            warnings.warn("There are unstaged changes in the repository. Please commit or stage them before running the experiment manager.")
      
        # Committing changes
        if self.primary_metric in metrics:
            commit_message = f"Experiment: {self.name}, {self.primary_metric}: {metrics[self.primary_metric]}"
        else:
            commit_message = f"Experiment: {self.name}"
        detailed_message = f"{self.description}\n\nResults:\n{pformat(metrics)}".strip()
        self.repo.git.commit('--allow-empty', '-m', commit_message, '-m', detailed_message)
    
