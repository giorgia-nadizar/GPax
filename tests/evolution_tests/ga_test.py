from functools import partial

import jax
import pytest
from qdax.core.emitters.standard_emitters import MixingEmitter
import jax.numpy as jnp

from gpax.evolution.genetic_algorithm_extra_scores import GeneticAlgorithmWithExtraScores
from gpax.evolution.evolution_metrics import custom_ga_metrics


def test_extra_scores_storage():
    pop_size = 10
    target_values = jnp.arange(pop_size, dtype=float).reshape(-1, 1)
    target_new_params = jnp.arange(pop_size, dtype=float).reshape(-1, 1) * 2
    fake_scoring_fn = partial(lambda x, k: (target_values,
                                            {
                                                "extra": target_values,
                                                "updated_params": target_new_params
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
        assert jnp.allclose(repertoire.fitnesses, repertoire.extra_scores["extra"])

        repertoire, emitter_state, current_metrics = ga.update(repertoire=repertoire, emitter_state=emitter_state,
                                                               key=key)
        assert "extra" in repertoire.extra_scores
        assert "extra" in current_metrics
        assert jnp.allclose(repertoire.fitnesses, repertoire.extra_scores["extra"])


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


def dummy_scoring_fn(genotype, rng):
    return 1.0, {"extra": 0.5}


def another_scoring_fn(genotype, rng):
    return 2.0, {"extra": 1.0}


@pytest.fixture
def ga():
    # Create a simple instance with a dummy emitter + metrics function
    emitter = object()
    metrics_fn = lambda *args, **kwargs: None
    return GeneticAlgorithmWithExtraScores(dummy_scoring_fn, emitter, metrics_fn)


def test_ga_scoring_fn_replacement_returns_new_instance(ga):
    new_ga = ga.replace_scoring_fn(another_scoring_fn)

    assert isinstance(new_ga, GeneticAlgorithmWithExtraScores)
    assert new_ga is not ga  # ensure a new object is created


def test_ga_scoring_fn_replacement_scoring_fn_is_replaced(ga):
    new_ga = ga.replace_scoring_fn(another_scoring_fn)
    assert new_ga._scoring_function is another_scoring_fn


def test_ga_scoring_fn_replacement_other_components_are_preserved(ga):
    new_ga = ga.replace_scoring_fn(another_scoring_fn)

    assert new_ga._emitter is ga._emitter
    assert new_ga._metrics_function is ga._metrics_function
