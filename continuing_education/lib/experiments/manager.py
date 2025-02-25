from pathlib import Path
from pprint import pformat
from typing import Any
import warnings
from git import Repo
import subprocess


class ExperimentManager:
    """
    A simple git based experiment manager.
    Put this at the end of your training script, and when it runs, it will commit the changes to the current git branch with the results.
    """

    def __init__(
        self, *, name: str, file: Path, description: str = "", primary_metric: str = ""
    ) -> None:
        self.name = name
        self.description = description
        self.file = file
        self.primary_metric = primary_metric

        # Check for unstaged changes
        # This is being moved to __init__ because we usually generate some notebook changes when running experiments, like plots and the like
        self.repo = Repo(
            path=self.file.absolute().parent, search_parent_directories=True
        )
        if self.repo.is_dirty(untracked_files=True):
            warnings.warn(
                "There are unstaged changes in the repository. Please commit or stage them before running the experiment manager."
            )

    @property
    def is_jupytext(self) -> bool:
        return (
            self.file.suffix.endswith(".ipynb")
            and Path(self.file).with_suffix(".py").exists()
        )

    def run_jupytext_sync(self):
        if self.is_jupytext:
            cmd = ["jupytext", "--sync", str(self.file.absolute())]
            print("Running: ", " ".join(cmd))
            subprocess.run(cmd, check=True)

    def commit(self, metrics: dict[str, Any] | None = None):
        if metrics is None:
            metrics = {}
        self.run_jupytext_sync()

        # Staging files
        files = [self.file.relative_to(self.repo.working_dir)]
        if self.is_jupytext:
            files.append(
                self.file.relative_to(self.repo.working_dir).with_suffix(".py")
            )
        self.repo.index.add(files)

        # Committing changes
        if self.primary_metric in metrics:
            commit_message = f"Experiment: {self.name}, {self.primary_metric}: {metrics[self.primary_metric]}"
        else:
            commit_message = f"Experiment: {self.name}"
        detailed_message = f"{self.description}\n\nResults:\n{pformat(metrics)}".strip()
        self.repo.index.commit(
            message=commit_message + "\n\n" + detailed_message, skip_hooks=True
        )
