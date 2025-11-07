Great job making a new implementation!

Here are some tasks to complete before merging this PR:

# Styleguide

- [ ] Did you update the version in Cargo.toml?

## Jupyter

- [ ] Add references to the bottom of the notebook.
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

## Marksman

- [ ] Make marksman notes and anki flash cards.
- [ ] Use singular nouns for filenames.
- [ ] Use `-` in filenames instead of spaces.
- [ ] For any new notes, you need to go back in other notes and link to them using project search.
- [ ] Put sources on each flashcard. Mark AI as the source if you used AI to generate the flashcard. These can be revisited later. Mark yourself as the source if you wrote the flashcard yourself.
