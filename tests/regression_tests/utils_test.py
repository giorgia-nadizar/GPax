import functools

import jax
import jax.numpy as jnp
import pytest

from gpax.graphs.cartesian_genetic_programming import CGP
from gpax.graphs.graph_genetic_programming import GGP
from gpax.symbolicregression.constants_optimization import optimize_constants_with_sgd, optimize_constants_with_cmaes, \
    optimize_constants_with_lbfgs
from gpax.symbolicregression.scoring_functions import regression_accuracy_evaluation, \
    regression_accuracy_evaluation_with_constants_optimization, regression_scoring_fn
from gpax.symbolicregression.utils import prepare_train_test_evaluation_fns, prepare_scoring_fn, prepare_rescoring_fn


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
    assert opt_fn.keywords["max_iter"] == 8


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
