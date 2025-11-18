from functools import partial

import jax
import jax.numpy as jnp

from gpax.graphs.cartesian_genetic_programming import CGP
from gpax.symbolicregression.scoring_functions import predict_regression_output, regression_accuracy_evaluation, \
    regression_scoring_fn, regression_accuracy_evaluation_with_sgd


def test_prediction_shape():
    """
    Ensure that `_predict_regression_output` returns predictions with the
    correct shape.
    """
    n_inputs = 2
    n_data_points = 10
    n_outputs = 1
    cgp = CGP(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_nodes=5,
        weighted_functions=True,
        weighted_inputs=False
    )
    key = jax.random.key(42)

    # init genome
    key, init_key = jax.random.split(key)
    cgp_genome = cgp.init(init_key)

    # generate some random points
    random_X = jax.random.uniform(key, (n_data_points, n_inputs))

    # simulate prediction
    jit_predict = jax.jit(partial(predict_regression_output, graph_structure=cgp))
    prediction = jit_predict(random_X, cgp_genome)
    assert prediction.shape == (n_data_points, n_outputs)

    # simulate prediction with random weights
    key, weights_key = jax.random.split(key)
    cgp_weights = jax.random.uniform(key=weights_key, shape=(cgp.n_nodes,)) * 2 - 1
    weighted_prediction = jit_predict(random_X, cgp_genome, graph_weights={"functions": cgp_weights})
    assert weighted_prediction.shape == (n_data_points, n_outputs)


def test_regression_accuracy_evaluation_shape():
    """Test that function returns accuracies with shape matching number of genotypes."""
    n_genotypes = 3
    n_samples = 5
    n_inputs = 2

    cgp = CGP(
        n_inputs=n_inputs,
        n_outputs=1,
        n_nodes=5,
    )
    key = jax.random.key(42)

    # init genomes
    init_key, key = jax.random.split(key)
    init_keys = jax.random.split(init_key, n_genotypes)
    genotypes = jax.vmap(jax.jit(cgp.init))(init_keys)

    # generate random data points
    x_key, y_key, key = jax.random.split(key, 3)
    X = jax.random.uniform(x_key, (n_samples, n_inputs))
    y = jax.random.uniform(y_key, (n_samples,))

    accuracies, returned_genotypes = regression_accuracy_evaluation(
        genotype=genotypes,
        key=key,
        graph_structure=cgp,
        X=X,
        y=y,
    )

    assert accuracies.shape[0] == n_genotypes
    assert returned_genotypes == genotypes


def test_regression_accuracy_evaluation_with_sgd_shape():
    """Test that function returns accuracies with shape matching number of genotypes."""
    n_genotypes = 3
    n_samples = 10
    n_inputs = 2

    cgp = CGP(
        n_inputs=n_inputs,
        n_outputs=1,
        n_nodes=5,
        weighted_functions=True,
        weighted_inputs=False
    )
    key = jax.random.key(42)

    # init genomes
    init_key, key = jax.random.split(key)
    init_keys = jax.random.split(init_key, n_genotypes)
    genotypes = jax.vmap(jax.jit(cgp.init))(init_keys)

    # generate random data points
    x_key, y_key, key = jax.random.split(key, 3)
    X = jax.random.uniform(x_key, (n_samples, n_inputs))
    y = jax.random.uniform(y_key, (n_samples,))

    accuracies, returned_genotypes = regression_accuracy_evaluation_with_sgd(
        genotype=genotypes,
        key=key,
        graph_structure=cgp,
        X=X,
        y=y,
        batch_size=4
    )

    non_sgd_accuracies, non_sgd_returned_genotypes = regression_accuracy_evaluation(
        genotype=genotypes,
        key=key,
        graph_structure=cgp,
        X=X,
        y=y,
    )

    assert accuracies.shape[0] == n_genotypes
    assert all(jax.tree_leaves(jax.tree_map(lambda x, y: jnp.allclose(x, y),
                                            returned_genotypes["genes"], genotypes["genes"])))
    assert all(accuracies > non_sgd_accuracies)


def test_regression_scoring_fn():
    """Test that the regression scoring function works."""
    n_genotypes = 3
    n_samples = 5
    n_inputs = 2

    cgp = CGP(
        n_inputs=n_inputs,
        n_outputs=1,
        n_nodes=5,
    )
    key = jax.random.key(42)

    # init genomes
    init_key, key = jax.random.split(key)
    init_keys = jax.random.split(init_key, n_genotypes)
    genotypes = jax.vmap(jax.jit(cgp.init))(init_keys)

    # generate random data points
    x_key, y_key, key = jax.random.split(key, 3)
    X = jax.random.uniform(x_key, (n_samples, n_inputs))
    y = jax.random.uniform(y_key, (n_samples,))

    # define train and test fn
    train_fn = partial(regression_accuracy_evaluation, graph_structure=cgp, X=X, y=y)
    test_fn = train_fn

    # compute scoring fn
    fitness, extra_scores = regression_scoring_fn(genotypes, key, train_fn, test_fn)
    assert len(fitness) == n_genotypes
    assert jnp.array_equal(fitness, extra_scores["test_accuracy"])
    assert genotypes == extra_scores["updated_params"]


def test_regression_scoring_fn_with_sgd():
    """Test that the regression scoring function works with SGD."""
    n_genotypes = 3
    n_samples = 10
    n_inputs = 2

    cgp = CGP(
        n_inputs=n_inputs,
        n_outputs=1,
        n_nodes=5,
        weighted_functions=True,
    )
    key = jax.random.key(42)

    # init genomes
    init_key, key = jax.random.split(key)
    init_keys = jax.random.split(init_key, n_genotypes)
    genotypes = jax.vmap(jax.jit(cgp.init))(init_keys)

    # generate random data points
    x_key, y_key, key = jax.random.split(key, 3)
    X = jax.random.uniform(x_key, (n_samples, n_inputs))
    y = jax.random.uniform(y_key, (n_samples,))

    # define train and test fn
    for reset_weights in [True, False]:
        for batch_size in [4, None]:
            train_fn = partial(regression_accuracy_evaluation_with_sgd, graph_structure=cgp, X=X, y=y,
                               batch_size=batch_size, reset_weights=reset_weights)
            test_fn = partial(regression_accuracy_evaluation, graph_structure=cgp, X=X, y=y)

            # compute scoring fn
            fitness, extra_scores = regression_scoring_fn(genotypes, key, train_fn, test_fn)
            assert len(fitness) == n_genotypes
            assert jnp.array_equal(fitness, extra_scores["test_accuracy"])
            assert not all(jax.tree_leaves(
                jax.tree_map(lambda x, y: jnp.allclose(x, y), genotypes, extra_scores["updated_params"])))
