Great job making a new implementation!

Here are some tasks to complete before merging this PR:

# Styleguide

## Jupyter

- [ ] Make sure it runs on cpu and gpu
- [ ] Comment the shapes of any numpy arrays or torch tensors. Make assertions on the output.
- [ ] Functions that have more than one parameter should have a `*` before the first or second parameter to force the user to use named arguments, unless there is only one parameter.
- [ ] Do not set default values, you might forget to pass parameters up the stack.
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

## Logseq

- [ ] Make logseq notes and flash cards.
- [ ] Do not use logseq aliases so that the graph looks clean and its more navigable.
- [ ] Use singular nouns for tags.
- [ ] Use spaces in filenames instead of `-` or `_` just so that you don't have to use aliases (ugly I know).
