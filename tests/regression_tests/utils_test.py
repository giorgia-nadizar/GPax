import functools

import jax
import jax.numpy as jnp
import pytest

from gpax.graphs.cartesian_genetic_programming import CGP
import numpy as np
from gpax.symbolicregression.constants_optimization import optimize_constants_with_sgd, optimize_constants_with_cmaes, \
    optimize_constants_with_lbfgs
from gpax.symbolicregression.scoring_functions import regression_accuracy_evaluation, \
    regression_accuracy_evaluation_with_constants_optimization, regression_scoring_fn
from gpax.symbolicregression.utils import prepare_train_test_evaluation_fns, prepare_scoring_fn, prepare_rescoring_fn, \
    load_dataset


@pytest.fixture
def sample_data():
    X_train = jnp.ones((4, 3))
    y_train = jnp.ones(4)
    X_test = jnp.ones((2, 3))
    y_test = jnp.ones(2)
    return X_train, y_train, X_test, y_test


def test_prepare_eval_fns_default_behavior(sample_data):
    X_train, y_train, X_test, y_test = sample_data

    train_fn, test_fn = prepare_train_test_evaluation_fns(
        X_train, y_train, X_test, y_test, graph_structure=None
    )

    assert train_fn.func is regression_accuracy_evaluation
    assert test_fn.func is regression_accuracy_evaluation


@pytest.mark.parametrize("opt", ["automl0", "mutation"])
def test_prepare_eval_fns_no_optimizer_aliases(sample_data, opt):
    X_train, y_train, X_test, y_test = sample_data

    train_fn, _ = prepare_train_test_evaluation_fns(
        X_train, y_train, X_test, y_test, const_optimizer=opt, graph_structure=None
    )

    assert train_fn.func is regression_accuracy_evaluation


def test_prepare_eval_fns_adam_optimizer(sample_data):
    X_train, y_train, X_test, y_test = sample_data

    train_fn, _ = prepare_train_test_evaluation_fns(
        X_train, y_train, X_test, y_test, const_optimizer="adam", graph_structure=None
    )

    assert train_fn.func is regression_accuracy_evaluation_with_constants_optimization
    opt_fn = train_fn.keywords["constants_optimization_fn"]
    assert opt_fn.func is optimize_constants_with_sgd
    assert opt_fn.keywords["n_gradient_steps"] == 100


def test_prepare_eval_fns_rmsprop_optimizer(sample_data):
    X_train, y_train, X_test, y_test = sample_data

    train_fn, _ = prepare_train_test_evaluation_fns(
        X_train, y_train, X_test, y_test, const_optimizer="rmsprop", graph_structure=None
    )

    assert train_fn.func is regression_accuracy_evaluation_with_constants_optimization
    opt_fn = train_fn.keywords["constants_optimization_fn"]
    assert opt_fn.func is optimize_constants_with_sgd
    assert opt_fn.keywords["n_gradient_steps"] == 120


def test_prepare_eval_fns_cmaes_optimizer(sample_data):
    X_train, y_train, X_test, y_test = sample_data

    train_fn, _ = prepare_train_test_evaluation_fns(
        X_train, y_train, X_test, y_test, const_optimizer="cmaes", graph_structure=None
    )

    assert train_fn.func is regression_accuracy_evaluation_with_constants_optimization
    opt_fn = train_fn.keywords["constants_optimization_fn"]
    assert opt_fn.func is optimize_constants_with_cmaes
    assert opt_fn.keywords["max_iter"] == 20


def test_prepare_eval_fns_lbfgs_optimizer(sample_data):
    X_train, y_train, X_test, y_test = sample_data

    train_fn, _ = prepare_train_test_evaluation_fns(
        X_train, y_train, X_test, y_test, const_optimizer="lbfgs", graph_structure=None
    )

    assert train_fn.func is regression_accuracy_evaluation_with_constants_optimization
    opt_fn = train_fn.keywords["constants_optimization_fn"]
    assert opt_fn.func is optimize_constants_with_lbfgs
    assert opt_fn.keywords["max_iter"] == 5


def test_prepare_eval_fns_test_fn_always_simple(sample_data):
    X_train, y_train, X_test, y_test = sample_data

    _, test_fn = prepare_train_test_evaluation_fns(
        X_train, y_train, X_test, y_test, const_optimizer="adam", graph_structure=None
    )

    assert test_fn.func is regression_accuracy_evaluation


def test_prepare_scoring_fn_returns_partial(sample_data):
    X_train, y_train, X_test, y_test = sample_data

    scoring_fn = prepare_scoring_fn(
        X_train, y_train, X_test, y_test,
        graph_structure=None
    )

    assert isinstance(scoring_fn, functools.partial)
    assert scoring_fn.func is regression_scoring_fn


def test_prepare_scoring_fn_partial_contains_train_and_test_fns(sample_data):
    X_train, y_train, X_test, y_test = sample_data

    scoring_fn = prepare_scoring_fn(
        X_train, y_train, X_test, y_test,
        graph_structure=None
    )

    train_fn = scoring_fn.keywords["train_set_evaluation_fn"]
    test_fn = scoring_fn.keywords["test_set_evaluation_fn"]

    # default: no constants optimizer → simple evaluation
    assert train_fn.func is regression_accuracy_evaluation
    assert test_fn.func is regression_accuracy_evaluation


def test_prepare_scoring_fn_constants_optimizer_flow(sample_data):
    X_train, y_train, X_test, y_test = sample_data

    scoring_fn = prepare_scoring_fn(
        X_train, y_train, X_test, y_test,
        graph_structure=None,
        const_optimizer="adam",
    )

    train_fn = scoring_fn.keywords["train_set_evaluation_fn"]
    test_fn = scoring_fn.keywords["test_set_evaluation_fn"]

    assert train_fn.func is regression_accuracy_evaluation_with_constants_optimization
    assert test_fn.func is regression_accuracy_evaluation  # always simple


def test_prepare_rescoring_fn(sample_data):
    X_train, y_train, _, _ = sample_data

    cgp = CGP(
        n_inputs=X_train.shape[1],
        n_outputs=1
    )
    rescoring_fn = prepare_rescoring_fn(
        X_train, y_train, graph_structure=cgp,
    )

    key = jax.random.key(0)
    key, pop_key = jax.random.split(key)
    pop_keys = jax.random.split(pop_key, 10)
    population = jax.vmap(cgp.init)(pop_keys)

    resulting_fitness = rescoring_fn(population, key)
    assert jnp.allclose(resulting_fitness, 0)


@pytest.mark.parametrize("dataset_name", ["diabetes", "feynman_I_6_2", "nikuradse_1"])
@pytest.mark.parametrize("scale_x", [True, False])
@pytest.mark.parametrize("scale_y", [True, False])
def test_load_dataset_shapes(dataset_name, scale_x, scale_y, monkeypatch):
    """
    Test that load_dataset returns arrays of correct shape.
    """
    # Mock reading files for feynman and custom datasets
    if "feynman" in dataset_name:
        import pandas as pd
        df_mock = pd.DataFrame(np.random.rand(100, 5), columns=[f"x{i}" for i in range(4)] + ["y"])
        monkeypatch.setattr("pandas.read_csv", lambda *args, **kwargs: df_mock)
    elif "nikuradse" in dataset_name:
        import pandas as pd
        df_train = pd.DataFrame(np.random.rand(80, 5), columns=[f"x{i}" for i in range(4)] + ["target"])
        df_test = pd.DataFrame(np.random.rand(20, 5), columns=[f"x{i}" for i in range(4)] + ["target"])

        def mock_read_csv(path, *args, **kwargs):
            if "train" in path:
                return df_train
            else:
                return df_test

        monkeypatch.setattr("pandas.read_csv", mock_read_csv)

    X_train, X_test, y_train, y_test = load_dataset(
        dataset_name=dataset_name,
        scale_x=scale_x,
        scale_y=scale_y,
        test_split=0.2,
        random_state=42
    )

    # Check shapes
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train.shape[1] == X_test.shape[1]
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]
    assert y_train.shape[1] == 1
    assert y_test.shape[1] == 1

    # Check scaling if requested
    if scale_x:
        assert np.isclose(np.mean(X_train, axis=0), 0, atol=1e-6).all()
        assert np.isclose(np.std(X_train, axis=0), 1, atol=1e-6).all()
    if scale_y:
        assert np.isclose(np.mean(y_train), 0, atol=1e-6)
        assert np.isclose(np.std(y_train), 1, atol=1e-6)


def test_invalid_dataset():
    """
    Test that an invalid dataset raises an error (FileNotFoundError or similar)
    """
    with pytest.raises(Exception):
        load_dataset("nonexistent_dataset", scale_x=False, scale_y=False)
