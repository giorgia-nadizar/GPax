from functools import partial
from typing import Tuple, Callable, Optional, Dict

import jax
import jax.numpy as jnp

from qdax.custom_types import (
    Params,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    RNGKey,
)

from gpax.graphs.graph_genetic_programming import GGP
from gpax.symbolicregression.metrics import r2_score, rmse


def predict_regression_output(
        X: jnp.ndarray,
        genotype: Genotype,
        graph_structure: GGP,
        graph_weights: Dict[str, jnp.ndarray] = None,
) -> jnp.ndarray:
    """
        Compute regression predictions for a batch of inputs.
        The batch of input is processed in parallel via vectorization.

        Parameters
        ----------
        X : jnp.ndarray
            Input data of shape (batch_size, input_dim), where each row is a
            separate sample (i.e., data points) to evaluate.
        genotype : Genotype
            The genotype parameters to be evaluated.
        graph_structure : GGP
            The structure defining how a genotype is encoded into a program.
        graph_weights : jnp.ndarray
            Optional weighting factors for the graph.

        Returns
        -------
        jnp.ndarray
            The predicted regression outputs for each input sample, of shape
            (batch_size,) or (batch_size, output_dim).
        """
    parallel_apply = jax.vmap(jax.jit(graph_structure.apply), in_axes=(None, 0, None))
    prediction = parallel_apply(genotype, X, graph_weights)
    return prediction


def regression_accuracy_evaluation(
        genotype: Genotype,
        key: RNGKey,
        graph_structure: GGP,
        X: jnp.ndarray,
        y: jnp.ndarray,
        accuracy_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = r2_score,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
        Evaluate regression accuracy for a batch of genotypes on a dataset.

        This function computes predictions for each genotype in the batch on
        the input data `X`, and then evaluates the predictions against the target
        outputs `y` using the provided `accuracy_fn`. All computations
        are vectorized and JIT-compiled.

        Parameters
        ----------
        genotype : Genotype
            A batch of genotypes to evaluate. Shape should allow vectorization
            over genotypes (e.g., (n_genotypes, ...)).
        key : RNGKey
            JAX random key for any stochastic operations during evaluation.
        graph_structure : GGP
            The computational graph defining how genotypes map inputs to outputs.
        X : jnp.ndarray
            Input features of shape (n_samples, n_features) to evaluate.
        y : jnp.ndarray
            Ground-truth target values corresponding to `X`.
        accuracy_fn : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], optional
            Function to compute accuracy between predictions and targets.
            Defaults to `r2_score`.

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            accuracies : jnp.ndarray
                Accuracy values for each genotype, shape (n_genotypes, ...)
                depending on `accuracy_fn`.
            genotype : Genotype
                The batch of genotypes that were evaluated, returned for convenience.
        """
    prediction_fn = partial(predict_regression_output, graph_structure=graph_structure)
    mapped_predict_regression_output = jax.vmap(jax.jit(prediction_fn), in_axes=(None, 0))
    predictions = mapped_predict_regression_output(X, genotype)
    mapped_accuracy_fn = jax.vmap(jax.jit(accuracy_fn), in_axes=(None, 0))
    accuracies = mapped_accuracy_fn(y, predictions)
    return accuracies, genotype
