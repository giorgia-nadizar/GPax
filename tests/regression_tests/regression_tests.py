import jax
import jax.numpy as jnp

from gpax.graphs.cartesian_genetic_programming import CGP
from gpax.symbolicregression.scoring_functions import predict_regression_output


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
    prediction = predict_regression_output(cgp_genome, cgp, random_X)
    assert prediction.shape == (n_data_points, n_outputs)

    # simulate prediction with random weights
    key, weights_key = jax.random.split(key)
    cgp_weights = jax.random.uniform(key=weights_key, shape=(cgp.n_nodes,)) * 2 - 1
    weighted_prediction = predict_regression_output(cgp_genome, cgp, random_X, {"functions": cgp_weights})
    assert weighted_prediction.shape == (n_data_points, n_outputs)
