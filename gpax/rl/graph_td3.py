from dataclasses import dataclass

import jax
import jax.numpy as jnp
from qdax.custom_types import RNGKey, Params
from qdax.baselines.td3 import TD3, TD3Config
from qdax.custom_types import Genotype

from gpax.gp.graph_genetic_programming import GGP


@dataclass
class GraphPolicy:

    def __init__(self, graph_structure: GGP, genotype: Genotype = None, rnd_key: RNGKey = None):
        self.graph_structure = graph_structure
        self.genotype = genotype or self.graph_structure.init(rnd_key or jax.random.key(0))

    def init(self, rnd_key: RNGKey, *args) -> Params:
        init_weights = self.graph_structure.init_weights(rnd_key)
        fake_genome = {
            "weights": init_weights,
        }
        # get weights filters out the ones which we actually want to train
        return self.graph_structure.get_weights(fake_genome)

    def apply(self, policy_params: Params, obs: jnp.ndarray) -> jnp.ndarray:
        return jax.jit(jax.vmap(self.graph_structure.apply, in_axes=(None, 0, None)))(self.genotype, obs, policy_params)


@dataclass
class GraphTD3Config(TD3Config):
    """Configuration for the TD3 algorithm"""

    policy_graph_structure: GGP = None
    policy_graph_genome: Genotype = None


class GraphTD3(TD3):
    def __init__(self, config: GraphTD3Config, action_size: int):
        super().__init__(config, action_size)
        self._policy = GraphPolicy(config.policy_graph_structure, config.policy_graph_genome)
        # TODO mettere vmap su apply (cosi parallelo su obs)
