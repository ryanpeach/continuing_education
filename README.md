# Continuing Education

Some projects I'm using to learn AI since my MS

# TODO

* Reinforcement Learning
    * [X] Value Based Methods - It's not implemented here, but from my MS I feel I'm already an "expert" in Q-Learning up to and including RAINBOW.
    * Policy Based Methods
        * [X] REINFORCE
        * [ ] Actor-Critic 2
        * [ ] Proximal Policy Optimization (PPO)
        * [ ] RLHF
    * [ ] Model Based Reinforcement Learning
    * [ ] Exploration in RL
    * [ ] Multi Agent RL
* Transformers
    * [ ] Tokenization
    * [ ] Word Embeddings
    * [ ] Transformers
    * [ ] Self Implemented LLM
    * [ ] Multimodality
* [ ] Diffusion Models
* [ ] Graph Neural Networks (GNN)

## Sources for further work

* [Key Papers in Deep RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html)

# Installation

First follow specific instructions for your operating system listed in subheaders below.

Then make sure you have cloned the submodules:

`git submodule update --init --recursive`

Then install the dependencies:

`poetry install`

After you run `poetry install`, `pip` install pytorch the way it describes [here](https://pytorch.org/get-started/locally/#start-locally) for your system. This ensures you get the best performance. Always use the version listed in [.github/workflows/ci.yaml](.github/workflows/ci.yaml) for the best compatibility.

## MacOSX

`brew install sdl sdl_ttf sdl_image sdl_mixer portmidi`

## Ubuntu

`sudo apt-get install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libfreetype6-dev libportmidi-dev libjpeg-dev python3-setuptools python3-dev python3-numpy`
