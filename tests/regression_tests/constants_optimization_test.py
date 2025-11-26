from functools import partial

import jax
import jax.numpy as jnp

from gpax.graphs.cartesian_genetic_programming import CGP
from gpax.symbolicregression.constants_optimization import optimize_constants_with_adam_sgd
from gpax.symbolicregression.scoring_functions import predict_regression_output


def test_adam_sgd_output_shape_and_type():
    """ Ensure the constants optimization with adam and sgd have the correct shape. """
    n_genomes, n_features, n_samples = 3, 5, 20
    X = jnp.ones((n_samples, n_features))
    y = jnp.ones((n_samples,))

    cgp = CGP(
        n_inputs=n_features,
        n_outputs=1,
        n_nodes=5,
        weighted_functions=True,
    )
    key = jax.random.key(42)

    # init genomes
    init_key, key = jax.random.split(key)
    init_keys = jax.random.split(init_key, n_genomes)
    genotypes = jax.vmap(jax.jit(cgp.init))(init_keys)

    graph_weights = cgp.get_weights(genotypes)
    prediction_fn = jax.jit(partial(predict_regression_output, graph_structure=cgp))
    optimized_weights = optimize_constants_with_adam_sgd(graph_weights, genotypes, key, X, y, prediction_fn)

    # same keys
    assert set(optimized_weights.keys()) == set(graph_weights.keys())
    # check array shapes
    for k in optimized_weights:
        assert optimized_weights[k].shape == graph_weights[k].shape
        assert isinstance(optimized_weights[k], jnp.ndarray)
