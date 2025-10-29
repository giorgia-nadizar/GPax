import jax
import jax.numpy as jnp
from qdax.core.containers.ga_repertoire import GARepertoire

from gpax.evolution.tournament_selector import TournamentSelector
from gpax.graphs.cartesian_genetic_programming import CGP


def test_tournament_with_cgp() -> None:
    """Test that tournament selection works with CGP.
        """

    # Init a random key
    key = jax.random.key(seed=0)

    # Init the CGP policy graph with default values
    policy_graph = CGP(
        n_inputs=3,
        n_outputs=2,
        weighted_functions=True,
    )

    # Init the population of CGP genomes and store in a GA repertoire
    pop_size = 10
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=pop_size)
    init_cgp_genomes = jax.vmap(policy_graph.init)(keys)
    ga_repertoire = GARepertoire.init(init_cgp_genomes, jnp.ones((pop_size,1), ), population_size=pop_size)

    # Create selector
    selector = TournamentSelector(tournament_size=3)
    key, subkey = jax.random.split(key)
    selected = selector.select(ga_repertoire, subkey, num_samples=3)
    print(selected)
