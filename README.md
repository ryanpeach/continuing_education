# Continuing Education

Some projects I'm using to learn AI since my MS

# Installation

First follow specific instructions for your operating system listed in subheaders below.

Consider modifying pyproject.toml to handle your necessary cuda version as an optional dependency.

Run `uv sync` to install all dependencies.

# Styleguide

See [.github/pull_request_template.md](.github/pull_request_template.md) for the styleguide.

## MacOSX

`brew install sdl sdl_ttf sdl_image sdl_mixer portmidi`

## Ubuntu

`sudo apt-get install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libfreetype6-dev libportmidi-dev libjpeg-dev python3-setuptools python3-dev python3-numpy`

# TODO

* Emoji Meanings
  * ❗ Indicates Priority
  * 📖 Paper Read
  * 📓 Notes Taken
  * 💻 Implementation Completed


* Reinforcement Learning
    * Value Based Methods - I'm pretty much up to date with these methods, but might as well implement them. I may go into less explanation though.
        * 📖📓💻 [$TD(\lambda)$](https://web.stanford.edu/class/cs234/notes/cs234-notes7.pdf)
        * 📖📓💻❗ [Deep Q Learning](https://arxiv.org/abs/1312.5602) 
          * <https://lightning.ai/docs/pytorch/LTS/notebooks/lightning_examples/reinforce-learning-DQN.html>
        * 📖❗[Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
        * 📖❗[Double Q Learning](https://arxiv.org/abs/1509.06461)
        * [ ] [Dueling Q Learning](https://arxiv.org/abs/1511.06581)
        * [ ] [Multi Step Learning](https://arxiv.org/abs/1901.02876)
        * [ ] [Distributional DQN](https://arxiv.org/abs/1707.06887)
        * [ ] [Noisy Nets](https://arxiv.org/abs/1706.10295)
        * 📖 [RAINBOW](https://arxiv.org/abs/1710.02298)
    * Policy Based Methods
        * 📖📓💻 [REINFORCE](https://arxiv.org/abs/2010.11364) *
        * 📖❗ [Actor-Critic](https://arxiv.org/pdf/1602.01783v2) (A2C, A3C) *
        * [ ] [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477) (TRPO)
        * [ ]❗[Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (PPO) *
        * [ ] [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971v6) (DDPG)
    * Model Based Reinforcement Learning
        * 📖❗[AlphaZero](https://arxiv.org/abs/1712.01815)
        * [ ] [MuZero](https://www.nature.com/articles/s41586-020-03051-4.epdf?sharing_token=kTk-xTZpQOF8Ym8nTQK6EdRgN0jAjWel9jnR3ZoTv0PMSWGj38iNIyNOw_ooNp2BvzZ4nIcedo7GEXD7UmLqb0M_V_fop31mMY9VBBLNmGbm0K9jETKkZnJ9SgJ8Rwhp3ySvLuTcUr888puIYbngQ0fiMf45ZGDAQ7fUI66-u7Y%3D)
            * <https://deepmind.google/discover/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules/>
        * [ ] [Dreamer](https://arxiv.org/pdf/1912.01603)
            * <https://research.google/blog/introducing-dreamer-scalable-reinforcement-learning-using-world-models/>
        * [ ] [Efficient Zero](https://arxiv.org/abs/2111.00210)
        * [ ] [Efficient Zero V2](https://arxiv.org/abs/2403.00564)
        * [ ] [SIMA](https://arxiv.org/abs/2404.10179)
            * <https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/>
        * [ ] [Genie 1](https://arxiv.org/abs/2402.15391)
            * <https://deepmind.google/research/publications/60474/>
        * [ ] [Genie 2](https://arxiv.org/pdf/2405.15489)
            * <https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/>
    * [ ] [Exploration in RL](https://github.com/opendilab/awesome-exploration-rl)
         * [ ] [Go-Explore](https://www.nature.com/articles/s41586-020-03157-9)
         * [ ] [NoisyNet](https://openreview.net/pdf?id=rywHCPkAW)
         * [ ] [DQN-PixelCNN](https://arxiv.org/abs/1606.01868)
         * [ ] [#Exploration](http://papers.neurips.cc/paper/6868-exploration-a-study-of-count-based-exploration-for-deep-reinforcement-learning.pdf) 
         * [ ] [EX2](https://papers.nips.cc/paper/2017/file/1baff70e2669e8376347efd3a874a341-Paper.pdf) 
         * [ ] [ICM](https://arxiv.org/abs/1705.05363) 
         * [ ] [RND](https://arxiv.org/abs/1810.12894) 
         * [ ] [NGU](https://arxiv.org/abs/2002.06038) 
         * [ ] [Agent57](https://arxiv.org/abs/2003.13350) 
         * [ ] [VIME](https://arxiv.org/abs/1605.09674) 
         * [ ] [EMI](https://openreview.net/forum?id=H1exf64KwH) 
         * [ ] [DIYAN](https://arxiv.org/abs/1802.06070) 
         * [ ] [SAC](https://arxiv.org/abs/1801.01290) 
         * [ ] [BootstrappedDQN](https://arxiv.org/abs/1602.04621) 
         * [ ] [PSRL](https://arxiv.org/pdf/1306.0940.pdf) 
         * [ ] [HER](https://arxiv.org/pdf/1707.01495.pdf) 
         * [ ] [DQfD](https://arxiv.org/abs/1704.03732) 
         * [ ] [R2D3](https://arxiv.org/abs/1909.01387) 
    * Multi Agent RL
        * [ ] [Emergent Communication through Negotiation](https://arxiv.org/abs/1804.03980)
        * [ ] Warp Drive
           * <https://lightning.ai/docs/pytorch/LTS/notebooks/lightning_examples/warp-drive.html>
    * [Human-Timescale Adaptation in an Open-Ended Task Space](https://sites.google.com/view/adaptive-agent/)
        * [ ] [Muesli](https://arxiv.org/pdf/2104.06159)
        * [ ] [Transformer-XL](https://arxiv.org/abs/1901.02860)
        * [ ] [Robust PLR](https://arxiv.org/pdf/2110.02439)
    * Distributed RL
        * [ ] [Survey](https://arxiv.org/pdf/2011.11012)
        * [ ] [RLLib](https://docs.ray.io/en/master/rllib.html)
* Transformers
    * [ ] [Tokenization](https://huggingface.co/learn/nlp-course/en/chapter6/1?fw=pt)
    * [ ] [Word Embeddings](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)
    * 📖❗[Transformers](https://arxiv.org/abs/1706.03762) 
      * <https://pytorch.org/tutorials/beginner/transformer_tutorial.html>
      * <https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html>
    * 📖❗[BERT](https://arxiv.org/abs/1810.04805) 
    * [ ]❗[Sentence-BERT](https://arxiv.org/pdf/1908.10084) 
    * [ ] [Fine Tuning](https://huggingface.co/learn/nlp-course/en/chapter3/1?fw=pt)
    * [ ] [RLHF](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo)
    * [ ] [Direct Preference Optimization](https://arxiv.org/pdf/2305.18290)
    * [ ] [Multimodality](https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/11-vision-transformer.html)
    * [ ] [Mamba and SSM's](https://towardsdatascience.com/mamba-ssm-theory-and-implementation-in-keras-and-tensorflow-32d6d4b32546)
    * [ ] [Sentence Transformers](https://medium.com/@vipra_singh/building-llm-applications-sentence-transformers-part-3-a9e2529f99c1)
    * [ ] [Multi token prediction](https://arxiv.org/pdf/2404.19737)
    * [ ] Time Series
        * <https://www.datadoghq.com/blog/datadog-time-series-foundation-model/>
* RAG
    * 📖 [Survey on RAG](https://arxiv.org/abs/2405.06211)
        * [ ]❗REALM
        * [ ]❗Hyde
        * [ ]❗DPR
        * [ ]❗Raft
        * [ ] PRCA
        * [ ] EAE
        * [ ] MIPS
        * [ ] Self reinforce
    * [Survey on Graph RAG](https://arxiv.org/abs/2408.08921)
* [ ] Diffusion Models
  * <https://lightning.ai/lightning-community-labs/studios/build-diffusion-models-with-pytorch-lightning-hf-diffusers>
* [ ]❗Graph Neural Networks (GNN) 
  * <https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/06-graph-neural-networks.html>
* Cognitive Science
   * [ ] [Hopfield Network](https://www.youtube.com/watch?v=1WPJdAW-sFo)
   * [ ] [Boltzman Machine](https://www.youtube.com/watch?v=_bqa_I5hNAo)
   * [ ] [Conformal Prediction](https://blog.dataiku.com/measuring-models-uncertainty-conformal-prediction?utm_source=pocket_saves)
   * [ ] [Predictive Coding Models](https://arxiv.org/abs/2202.09467)
   * [ ] [Liquid Neural Networks](https://arxiv.org/pdf/2006.04439)
* Techniques
    * Profiling
    * Debugging Metrics

## Sources for further work

* [Key Papers in Deep RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html)
* [Spinning Up](https://spinningup.openai.com/en/latest/index.html)
* [Lightning Tutorials](https://lightning.ai/docs/pytorch/stable/notebooks.html)
* [Hugging Face RL](https://huggingface.co/learn/deep-rl-course/unit0/introduction)
* [Pytorch](https://pytorch.org/tutorials)
