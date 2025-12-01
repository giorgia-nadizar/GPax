import functools
from typing import Callable, Tuple

import optax

from gpax.graphs.graph_genetic_programming import GGP
import jax.numpy as jnp

from gpax.symbolicregression.constants_optimization import optimize_constants_with_sgd, optimize_constants_with_cmaes, \
    optimize_constants_with_lbfgs
from gpax.symbolicregression.scoring_functions import regression_accuracy_evaluation, \
    regression_accuracy_evaluation_with_constants_optimization


def prepare_train_test_evaluation_fns(
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        X_test: jnp.ndarray,
        y_test: jnp.ndarray,
        graph_structure: GGP,
        const_optimizer: str = None,
) -> Tuple[Callable, Callable]:
    """
        Prepare training and testing evaluation functions for symbolic regression models.

        This utility constructs two callable evaluation functions, one for training data
        and one for test data, based on the provided dataset and configuration. Depending
        on the selected constant-optimization strategy, the training function may also
        include an inner optimization routine to refine numeric constants within the
        model’s computational graph.

        Parameters
        ----------
        X_train : jnp.ndarray
            Input features for the training set.
        y_train : jnp.ndarray
            Target values corresponding to `X_train`.
        X_test : jnp.ndarray
            Input features for the test set.
        y_test : jnp.ndarray
            Target values corresponding to `X_test`.
        graph_structure : GGP
            A graph-based program (GGP) structure representing the symbolic model.
            This is passed through to the evaluation functions.
        const_optimizer : str, optional
            Specifies the strategy for optimizing constants within the symbolic model.
            Supported values include:
            - ``"adam"``: optimize constants via SGD with Adam-like settings.
            - ``"lbfgs"``: optimize using L-BFGS.
            - ``"rmsprop"``: optimize using RMSProp.
            - ``"cmaes"``: optimize using CMA-ES.

            Any unrecognized value defaults to no constants optimization.

        Returns
        -------
        (train_fn, test_fn) : tuple of callables
            `train_fn` is a partially applied evaluation function for the training set,
            optionally including a constant-optimization subroutine.
            `test_fn` is a partially applied evaluation function for evaluating regression
            accuracy on the test set.

        Notes
        -----
        - The returned functions are already bound to the provided data and model
          structure and can be called directly with a candidate program representation.
        - If a constant optimizer is specified, training proceeds by first optimizing the
          constants of the program and then evaluating accuracy.
        """

    if const_optimizer == "adam":
        constants_optimizer = functools.partial(optimize_constants_with_sgd, batch_size=32,
                                                n_gradient_steps=100)
    elif const_optimizer == "rmsprop":
        constants_optimizer = functools.partial(optimize_constants_with_sgd, batch_size=32,
                                                n_gradient_steps=120,
                                                optimizer=optax.rmsprop(1e-3, momentum=.9))
    elif const_optimizer == "cmaes":
        constants_optimizer = functools.partial(optimize_constants_with_cmaes, max_iter=8)
    elif const_optimizer == "lbfgs":
        constants_optimizer = functools.partial(optimize_constants_with_lbfgs, max_iter=5)
    else:
        constants_optimizer = None
    if constants_optimizer:
        train_fn = functools.partial(regression_accuracy_evaluation_with_constants_optimization,
                                     graph_structure=graph_structure,
                                     X=X_train, y=y_train, reset_weights=False,
                                     constants_optimization_fn=constants_optimizer)
    else:
        train_fn = functools.partial(regression_accuracy_evaluation, graph_structure=graph_structure, X=X_train,
                                     y=y_train)

    test_fn = functools.partial(regression_accuracy_evaluation, graph_structure=graph_structure, X=X_test, y=y_test)
    return train_fn, test_fn
