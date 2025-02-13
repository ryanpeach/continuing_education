Great job making a new implementation!

Here are some tasks to complete before merging this PR:

## Task List for all Jupyter Notebooks
- [ ] Add the proper headers to the notebook:

```
%load_ext autoreload
%autoreload 2
```

- [ ] Make sure the jupyter notebook is importable without running any code. Most cells should look like this:

```python
def foo():
    pass

if __name__ == '__main__':
    foo()
```

- [ ] APA style citations for all sources at the bottom.

- [ ] Actually read the paper you cite, not just the tutorial

- [ ] Make logseq compatible flash cards. Save them to a readme in the same folder as the notebook.

- [ ] In natural language, explain your understanding of the solution.

- [ ] In mathematical language, document all implementations of all equations in the same notation as the source paper. Preferably you would do this in the functions docstrings, and cite the equation number in the paper. If it's a single line expression you can instead add a comment in the code.

- [ ] Comment the shapes of any numpy arrays or torch tensors. Make assertions on the output.

- [ ] Ask ChatGPT to review your work.

- [ ] Functions that have more than one parameter should have a `*` before the first or second parameter to force the user to use named arguments, unless there is only one parameter.

- [ ] Do not set default values, you might forget to pass parameters up the stack.

- [ ] Make sure it runs on cpu and gpu
