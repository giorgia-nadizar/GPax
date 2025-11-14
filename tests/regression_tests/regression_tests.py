import jax

from gpax.graphs.cartesian_genetic_programming import CGP
from gpax.symbolicregression.scoring_functions import _predict_regression_output


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
    )
    key = jax.random.key(42)

    # init genome
    key, init_key = jax.random.split(key)
    initial_cgp_genome = cgp.init(init_key)

    # generate some random points
    random_X = jax.random.uniform(key, (n_data_points, n_inputs))

    # simulate prediction
    prediction = _predict_regression_output(initial_cgp_genome, cgp, random_X)
    assert prediction.shape == (n_data_points, n_outputs)
