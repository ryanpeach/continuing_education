# Continuing Education

Some projects I'm using to learn AI since my MS

# TODO

`*` Means priority

* Reinforcement Learning
    * Value Based Methods - I'm pretty much up to date with these methods, but might as well implement them. I may go into less explanation though.
        * [ ] Monte-Carlo Methods
        * [ ] $TD(\lambda)$
        * [X] Deep Q Learning
          * https://lightning.ai/docs/pytorch/LTS/notebooks/lightning_examples/reinforce-learning-DQN.html
        * [ ] Prioritized Experience Replay
        * [ ] Double Q Learning
        * [ ] Dueling Q Learning
        * [ ] Multi Step Learning
        * [ ] Distributional DQN
        * [ ] Noisy Nets
        * [ ] RAINBOW
    * Policy Based Methods
        * [X] REINFORCE *
        * [ ] Actor-Critic *
        * [ ] Trust Region Policy Optimization (TRPO) 
        * [ ] Proximal Policy Optimization (PPO) *
        * [ ] Deep Deterministic Policy Gradient (DDPG) 
    * Model Based Reinforcement Learning
        * [ ] AlphaZero
    * [ ] Exploration in RL
    * Multi Agent RL
        * [ ] [Emergent Communication through Negotiation](https://arxiv.org/abs/1804.03980)
        * [ ] [Warp Drive](https://lightning.ai/docs/pytorch/LTS/notebooks/lightning_examples/warp-drive.html)
    * Distributed RL
        * [ ] [Survey](https://arxiv.org/pdf/2011.11012)
        * [ ] [RLLib](https://docs.ray.io/en/master/rllib.html)
* Transformers
    * [ ] [Tokenization](https://huggingface.co/learn/nlp-course/en/chapter6/1?fw=pt) *
    * [ ] [Word Embeddings](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html) *
    * [ ] Transformers *
      * https://pytorch.org/tutorials/beginner/transformer_tutorial.html
      * https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html
    * [ ] [Fine Tuning](https://huggingface.co/learn/nlp-course/en/chapter3/1?fw=pt) *
    * [ ] [RLHF](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo) *
    * [ ] [Multimodality](https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/11-vision-transformer.html)
    * [ ] [Mamba and SSM's](https://towardsdatascience.com/mamba-ssm-theory-and-implementation-in-keras-and-tensorflow-32d6d4b32546)
    * [ ] [Sentence Transformers](https://medium.com/@vipra_singh/building-llm-applications-sentence-transformers-part-3-a9e2529f99c1)
    * [ ] [Multi token prediction](https://arxiv.org/pdf/2404.19737)
    * [ ] Time Series https://www.datadoghq.com/blog/datadog-time-series-foundation-model/
* [ ] Diffusion Models
  * https://lightning.ai/lightning-community-labs/studios/build-diffusion-models-with-pytorch-lightning-hf-diffusers
* [ ] Graph Neural Networks (GNN)
  * https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/06-graph-neural-networks.html
* [ ] [Conformal Prediction](https://blog.dataiku.com/measuring-models-uncertainty-conformal-prediction?utm_source=pocket_saves)
* Techniques
    * Profiling
    * Debugging Metrics

## Other interests slightly related

* linguistic philosophy
  * [ ] Wittgenstein
  * [ ] Chompsky
* cognitive science
  * [ ] Predictive Coding Models
* economics and game theory
  * [ ] https://web.stanford.edu/~jacksonm/Networks-Online-Syllabus.pdf

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
