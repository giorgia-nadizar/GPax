from typing import Tuple, Callable, Optional

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
from gpax.symbolicregression.metrics import r2_score


def _predict_regression_output(
        genotype: Genotype,
        graph_structure: GGP,
        X: jnp.ndarray,
) -> jnp.ndarray:
    """
        Compute regression predictions for a batch of inputs.
        The batch of input is processed in parallel via vectorization.

        Parameters
        ----------
        genotype : Genotype
            The genotype parameters to be evaluated.
        graph_structure : GGP
            The structure defining how a genotype is encoded into a program.
        X : jnp.ndarray
            Input data of shape (batch_size, input_dim), where each row is a
            separate sample (i.e., data points) to evaluate.

        Returns
        -------
        jnp.ndarray
            The predicted regression outputs for each input sample, of shape
            (batch_size,) or (batch_size, output_dim).
        """
    parallel_apply = jax.vmap(graph_structure.apply, in_axes=(None, 0))
    prediction = parallel_apply(genotype, X)
    return prediction
