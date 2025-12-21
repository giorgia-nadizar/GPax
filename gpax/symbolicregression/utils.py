import functools
from typing import Callable, Tuple

import optax
import pandas as pd
import numpy as np
import scipy
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from gpax.graphs.graph_genetic_programming import GGP
import jax.numpy as jnp

from gpax.symbolicregression.constants_optimization import optimize_constants_with_sgd, optimize_constants_with_cmaes, \
    optimize_constants_with_lbfgs
from gpax.symbolicregression.scoring_functions import regression_accuracy_evaluation, \
    regression_accuracy_evaluation_with_constants_optimization, regression_scoring_fn


def load_dataset(dataset_name: str,
                 scale_x: bool = False,
                 scale_y: bool = False,
                 test_split: float = 0.25,
                 random_state: int = 0,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a dataset, split into train/test sets, and optionally scale features and targets.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset ('diabetes', Feynman TSV, or custom CSV).
    scale_x : bool, default=False
        If True, standardize input features.
    scale_y : bool
        If True, standardize target values, default=False.
    test_split : float, default=0.25
        Fraction of data for testing (ignored for pre-split CSVs).
    random_state : int, default=0
        Seed for reproducible train/test split.

    Returns
    -------
    X_train, X_test : np.ndarray
        Training and testing features.
    y_train, y_test : np.ndarray
        Training and testing targets, reshaped to (-1, 1).
    """
    if "diabetes" in dataset_name:
        X, y = load_diabetes(return_X_y=True)
        y = y.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=random_state)
    elif "feynman" in dataset_name:
        df = pd.read_csv(f"../datasets/{dataset_name}.tsv", sep="\t")
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        y = y.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=random_state)
    elif "mtr" in dataset_name:
        statistics = pd.read_csv("../datasets/mtr/statistics.csv")
        n_targets = statistics.loc[statistics["name"] == dataset_name.replace("mtr/", ""), "targets"].iloc[0]
        raw_data = scipy.io.arff.loadarff(f"../datasets/{dataset_name}.arff")
        df = pd.DataFrame(raw_data[0])
        X = df.iloc[:, :-n_targets].to_numpy()
        y = df.iloc[:, -n_targets:].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=random_state)
    else:
        df_train = pd.read_csv(f"../datasets/{dataset_name}_train.csv")
        df_test = pd.read_csv(f"../datasets/{dataset_name}_test.csv")
        X_train = df_train.drop(columns=["target"], inplace=False).to_numpy()
        X_test = df_test.drop(columns=["target"], inplace=False).to_numpy()
        y_train = df_train["target"].to_numpy().reshape(-1, 1)
        y_test = df_test["target"].to_numpy().reshape(-1, 1)

    # Create scalers
    if scale_x:
        X_scaler = StandardScaler()
        X_train = X_scaler.fit_transform(X_train)
        X_test = X_scaler.transform(X_test)
    if scale_y:
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train)
        y_test = y_scaler.transform(y_test)

    return X_train, X_test, y_train, y_test


def prepare_train_test_evaluation_fns(
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        X_test: jnp.ndarray,
        y_test: jnp.ndarray,
        graph_structure: GGP,
        const_optimizer: str = None,
        long_const_optimization: bool = False,
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
        long_const_optimization : bool, optional
            Whether the constants are optimizer for longer or not.

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
    multiplier = 100 if long_const_optimization else 1
    if const_optimizer == "adam":
        constants_optimizer = functools.partial(optimize_constants_with_sgd, batch_size=32,
                                                n_gradient_steps=100 * multiplier)
    elif const_optimizer == "rmsprop":
        constants_optimizer = functools.partial(optimize_constants_with_sgd, batch_size=32,
                                                n_gradient_steps=120 * multiplier,
                                                optimizer=optax.rmsprop(1e-3, momentum=.9))
    elif const_optimizer == "cmaes":
        constants_optimizer = functools.partial(optimize_constants_with_cmaes, max_iter=20 * multiplier)
    elif const_optimizer == "lbfgs":
        constants_optimizer = functools.partial(optimize_constants_with_lbfgs, max_iter=5 * multiplier)
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


def prepare_scoring_fn(
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        X_test: jnp.ndarray,
        y_test: jnp.ndarray,
        graph_structure: GGP,
        const_optimizer: str = None,
        long_const_optimization: bool = False,
) -> Callable:
    """
        Create a scoring function for symbolic regression model evaluation.

        This function constructs a combined scoring function that internally uses
        training and testing evaluation functions produced by
        `prepare_train_test_evaluation_fns`.
        The returned function is pre-configured with the provided datasets,
        graph structure, and optional constant-optimization strategy. It can be
        applied directly to a candidate program genotype (graph representation).

        Parameters
        ----------
        X_train : jnp.ndarray
            Training input features.
        y_train : jnp.ndarray
            Training target values.
        X_test : jnp.ndarray
            Test input features.
        y_test : jnp.ndarray
            Test target values.
        graph_structure : GGP
            A graph-based symbolic program structure to be used when evaluating
            candidate models.
        const_optimizer : str, optional
            Name of the constant-optimization strategy to use during training
            evaluation. See `prepare_train_test_evaluation_fns` for supported values.
        long_const_optimization : bool, optional
            Whether the constants are optimizer for longer or not.

        Returns
        -------
        Callable
            A partially applied scoring function that evaluates both training and
            test performance using internally prepared evaluation functions.

        Notes
        -----
        - The returned scoring function delegates actual evaluation to
          `regression_scoring_fn`.
        - The train/test evaluation functions used by the scorer are already bound
          to the supplied datasets and graph structure.
        """
    train_fn, test_fn = prepare_train_test_evaluation_fns(X_train, y_train, X_test, y_test, graph_structure,
                                                          const_optimizer, long_const_optimization)
    return functools.partial(
        regression_scoring_fn,
        train_set_evaluation_fn=train_fn,
        test_set_evaluation_fn=test_fn
    )


def prepare_rescoring_fn(
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        graph_structure: GGP,
) -> Callable:
    """
        Create a scoring function for symbolic regression model evaluation.

        This function constructs a combined scoring function that internally uses
        training and testing evaluation functions produced by
        `prepare_train_test_evaluation_fns`.
        The returned function is pre-configured with the provided datasets,
        graph structure, and optional constant-optimization strategy. It can be
        applied directly to a candidate program genotype (graph representation).

        Parameters
        ----------
        X_train : jnp.ndarray
            Training input features.
        y_train : jnp.ndarray
            Training target values.
        graph_structure : GGP
            A graph-based symbolic program structure to be used when evaluating
            candidate models.

        Returns
        -------
        Callable
            A partially applied scoring function that evaluates only training
            performance using internally prepared evaluation functions.

        Notes
        -----
        - The returned scoring function delegates actual evaluation to
          `regression_scoring_fn`.
        - The train/test evaluation functions used by the scorer are already bound
          to the supplied datasets and graph structure.
        """
    train_fn, _ = prepare_train_test_evaluation_fns(X_train, y_train, None, None, graph_structure)
    test_fn = lambda x, y: (None, None)
    rescoring_fn = functools.partial(
        regression_scoring_fn,
        train_set_evaluation_fn=train_fn,
        test_set_evaluation_fn=test_fn
    )
    return lambda x, y: rescoring_fn(x, y)[0]
