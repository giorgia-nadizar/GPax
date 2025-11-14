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
