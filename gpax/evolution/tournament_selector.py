from functools import partial

from flax import struct
import jax
import jax.numpy as jnp
from jax import vmap
from qdax.core.emitters.repertoire_selectors.selector import Selector, GARepertoireT, unfold_repertoire
from qdax.custom_types import RNGKey, Genotype, Fitness


@struct.dataclass
class TournamentSelector(Selector):
    """A selector using tournament to sample individuals from the population."""

    tournament_size: int = 3

    def select(
            self,
            repertoire: GARepertoireT,
            key: RNGKey,
            num_samples: int,
    ) -> GARepertoireT:
        def _tournament(sample_key: RNGKey, genomes: Genotype, fitness_values: Fitness, ) -> jnp.ndarray:
            indexes = jax.random.choice(sample_key,
                                        jnp.arange(start=0, stop=len(genomes)), shape=[self.tournament_size],
                                        replace=True)
            mask = -jnp.inf * jnp.ones_like(fitness_values)
            mask = mask.at[indexes].set(1)
            fitness_values_for_selection = fitness_values * mask
            return jnp.argmax(fitness_values_for_selection)

        sample_keys = jax.random.split(key, num_samples)
        partial_single_tournament = partial(_tournament, genomes=repertoire.genotypes,
                                            fitness_values=repertoire.fitnesses,)
        vmap_tournament = vmap(partial_single_tournament)
        selected_indexes = vmap_tournament(sample_key=sample_keys)

        repertoire_unfolded = unfold_repertoire(repertoire)
        selected: GARepertoireT = jax.tree.map(
            lambda x: x[selected_indexes],
            repertoire_unfolded,
        )

        return selected
