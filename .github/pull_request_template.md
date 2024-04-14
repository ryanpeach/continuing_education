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

- [ ] In natural language, explain your understanding of the solution.

- [ ] In mathematical language, document all implementations of all equations in the same notation as the source paper. Preferably you would do this in the functions docstrings, and cite the equation number in the paper. If it's a single line expression you can instead add a comment in the code.