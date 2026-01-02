# Inspirations

- [Kheperax](https://github.com/adaptive-intelligent-robotics/Kheperax): requirements management 
- [QDax](https://github.com/adaptive-intelligent-robotics/QDax): evolution, ANNs representations
- [EvoSAX](https://github.com/RobertTLange/evosax): evolutionary strategies
- [Lexicase](https://github.com/ryanboldi/lexicase): transitions between JAX and plain numpy
- [Kozax](https://github.com/sdevries0/Kozax): GP trees in JAX
- [Stoix](https://github.com/EdanToledo/Stoix): RL algorithms in JAX

# TODOs

- try WANN-like experiment do GA: eval each genome with 5 random weighting constants
- try experiment with ant and ANNs to see how filled the archive gets
- implement RL based constant optimization (PPO?) 
- do actor-critic for constants (eg SAC) and keep critics through optimization

- add back to weight init
- "variance": jnp.zeros((1,), dtype=jnp.float32),

# GECCO todos
- track results of constants optimization -> https://arxiv.org/pdf/1810.04119
- try classification on given datasets
- implement sparsity of constants