# Continuing Education

Some projects I'm using to learn AI since my MS

# TODO

* Reinforcement Learning
    * Value Based Methods - I'm pretty much up to date with these methods, but might as well implement them. I may go into less explanation though.
        * [ ] Monte-Carlo Methods
        * [ ] $TD(\lambda)$
        * [X] Deep Q Learning
        * [ ] Prioritized Experience Replay
        * [ ] Double Q Learning
        * [ ] Dueling Q Learning
        * [ ] Multi Step Learning
        * [ ] Distributional DQN
        * [ ] Noisy Nets
        * [ ] RAINBOW
    * Policy Based Methods
        * [X] REINFORCE
        * [ ] Actor-Critic
        * [ ] Proximal Policy Optimization (PPO)
    * Model Based Reinforcement Learning
        * [ ] AlphaZero
    * [ ] Exploration in RL
    * Multi Agent RL
        * [ ] Emergent Communication through Negotiation https://arxiv.org/abs/1804.03980
* Transformers
    * [ ] Tokenization
    * [ ] Word Embeddings
    * [ ] Transformers
    * [ ] Fine Tuning
    * [ ] RLHF
    * [ ] Self Implemented LLM
    * [ ] Multimodality
* [ ] Diffusion Models
* [ ] Graph Neural Networks (GNN)
* Techniques
    * Distribution
        * [ ] pytorch.distributed
        * [ ] rllib
    * Profiling
    * Debugging Metrics

## Other interests slightly related

* linguistic philosophy 
* cognitive science
* economics and game theory

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
