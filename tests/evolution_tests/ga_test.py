from functools import partial

import jax
from qdax.core.emitters.standard_emitters import MixingEmitter
import jax.numpy as jnp

from gpax.evolution.genetic_algorithm_extra_scores import GeneticAlgorithmWithExtraScores
from gpax.evolution.evolution_metrics import custom_ga_metrics


def test_extra_scores_storage():
    pop_size = 10
    fake_scoring_fn = partial(lambda x, k: (jnp.ones((pop_size, 1)),
                                            {
                                                "extra": jnp.ones((pop_size, 1)),
                                                "updated_params": jnp.ones((pop_size, 1)) * 2
                                            }))
    fake_mutate_fn = lambda x, k: x
    mixing_emitter = MixingEmitter(
        mutation_fn=fake_mutate_fn,
        variation_fn=None,
        variation_percentage=0.0,  # note: CGP works with mutation only
        batch_size=pop_size,
    )
    for extra_scores_metrics in ["extra", ["extra"], {"extra": jnp.ravel}]:
        ga_metrics = partial(
            custom_ga_metrics,
            extra_scores_metrics=extra_scores_metrics
        )
        fake_genomes = jnp.ones((pop_size, 1))

        # Instantiate GA
        ga = GeneticAlgorithmWithExtraScores(
            scoring_function=fake_scoring_fn,
            emitter=mixing_emitter,
            metrics_function=ga_metrics,
        )
        key = jax.random.key(0)
        repertoire, emitter_state, init_metrics = ga.init(genotypes=fake_genomes, population_size=pop_size, key=key)
        assert "extra" in repertoire.extra_scores
        assert "extra" in init_metrics

        repertoire, emitter_state, current_metrics = ga.update(repertoire=repertoire, emitter_state=emitter_state,
                                                               key=key)
        assert "extra" in repertoire.extra_scores
        assert "extra" in current_metrics


def test_lamarckian_evolution():
    pop_size = 10
    fake_scoring_fn = partial(lambda x, k: (jnp.ones((pop_size, 1)),
                                            {
                                                "extra": jnp.ones((pop_size, 1)),
                                                "updated_params": jnp.ones((pop_size, 1)) * 2
                                            }))
    fake_mutate_fn = lambda x, k: x
    mixing_emitter = MixingEmitter(
        mutation_fn=fake_mutate_fn,
        variation_fn=None,
        variation_percentage=0.0,  # note: CGP works with mutation only
        batch_size=pop_size,
    )
    for lamarckian in [True, False]:
        ga_metrics = partial(
            custom_ga_metrics,
            extra_scores_metrics="extra"
        )
        fake_genomes = jnp.ones((pop_size, 1))

        # Instantiate GA
        ga = GeneticAlgorithmWithExtraScores(
            scoring_function=fake_scoring_fn,
            emitter=mixing_emitter,
            metrics_function=ga_metrics,
        )
        key = jax.random.key(0)
        repertoire, emitter_state, init_metrics = ga.init(genotypes=fake_genomes, population_size=pop_size, key=key,
                                                          lamarckian=lamarckian)
        assert lamarckian != jnp.allclose(fake_genomes, repertoire.genotypes)
        assert lamarckian == jnp.allclose(repertoire.extra_scores["updated_params"], repertoire.genotypes)
        repertoire, emitter_state, current_metrics = ga.update(repertoire=repertoire, emitter_state=emitter_state,
                                                               key=key, lamarckian=lamarckian)

        assert lamarckian == jnp.allclose(repertoire.extra_scores["updated_params"], repertoire.genotypes)
