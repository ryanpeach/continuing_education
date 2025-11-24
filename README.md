# Continuing Education

Some projects I'm using to learn AI since my MS

# Installation

First follow specific instructions for your operating system listed in subheaders below.

Consider modifying pyproject.toml to handle your necessary cuda version as an optional dependency.

Run `uv sync` to install all dependencies.

# Technologies

## jupytext

Jupyter is great to work in, but it has several shortcomings:

1. A lack of a proper git diff when making a PR
2. You can't import jupyter notebooks from other jupyter notebooks
3. They can't be linted or formatted by tools like ruff or pyright

So I am using [jupytext](https://github.com/mwouts/jupytext) to make syncronized copies of the jupyter notebooks in plain text `.py` format.

If you then wrap cells (the code you run) in `if __name__ == "__main__"` you also now gain the ability to use these notebooks as importable libraries in future work.

## Logseq

I use [marksman](https://github.com/artempyanykh/marksman) to manage my markdown notes and link them to each other. I then use [mdanki](https://github.com/ashlinchak/mdanki) to convert them to flashcards.

# Styleguide

See [.github/pull_request_template.md](.github/pull_request_template.md) for the styleguide.

## MacOSX

`brew install sdl sdl_ttf sdl_image sdl_mixer portmidi`

## Ubuntu

`sudo apt-get install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libfreetype6-dev libportmidi-dev libjpeg-dev python3-setuptools python3-dev python3-numpy`

# TODO

https://ryanpeach.com/Publish/Machine+Learning/Research+Papers

## Sources for further work

* [Key Papers in Deep RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html)
* [Spinning Up](https://spinningup.openai.com/en/latest/index.html)
* [Lightning Tutorials](https://lightning.ai/docs/pytorch/stable/notebooks.html)
* [Hugging Face RL](https://huggingface.co/learn/deep-rl-course/unit0/introduction)
* [Pytorch](https://pytorch.org/tutorials)
