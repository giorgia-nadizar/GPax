from typing import Dict, Callable, Any
import jax
import jax.numpy as jnp
import optax
from optax import GradientTransformation
from qdax.custom_types import Genotype, RNGKey

from gpax.symbolicregression.metrics import rmse


def optimize_constants_with_adam_sgd(
        graph_weights: Dict,
        genotype: Genotype,
        key: RNGKey,
        X: jnp.ndarray,
        y: jnp.ndarray,
        prediction_fn: Callable,
        optimizer: GradientTransformation = optax.adam(1e-3),
        loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = rmse,
        n_gradient_steps: int = 100,
        batch_size: int = None,
) -> Dict:
    """
        Optimize the constant parameters (weights) of a batch of computational graphs
        using mini-batch stochastic gradient descent with an Optax optimizer
        (default Adam) and gradient clipping.

        This function operates over a batch of genomes in parallel via `jax.vmap`,
        performing iterative gradient updates on each genome’s weight dictionary.
        During each optimization step, a random mini-batch is sampled, loss and
        gradients are computed, and weights are updated.

        Parameters
        ----------
        graph_weights : Dict
            A PyTree-compatible dictionary containing the current constant/weight
            parameters for each genome in the batch. The leading dimension corresponds
            to the batch size (number of genomes).
        genotype : Genotype
            The batch of genotypes associated with the weight dictionaries. Used by
            the prediction function to interpret the weights.
        key : RNGKey
            JAX random key used for mini-batch sampling during optimization.
        X : jnp.ndarray
            Input feature matrix of shape (n_samples, n_features).
        y : jnp.ndarray
            Target regression outputs of shape (n_samples,) or (n_samples, n_outputs).
        prediction_fn : Callable
            A function with signature `(X, genotype, graph_weights) -> predictions`
            that computes regression model outputs given weights.
        optimizer : GradientTransformation, optional
            An Optax optimizer transformation used to update weights. Defaults to
            `optax.adam(1e-3)`. Gradient clipping is automatically prepended.
        loss_fn : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], optional
            A differentiable loss function comparing predicted and target outputs.
            Defaults to `rmse`.
        n_gradient_steps : int, optional
            Number of optimization steps to perform. Defaults to 100.
        batch_size : int or None, optional
            Mini-batch size. If None, uses the full dataset (i.e., full-batch gradient
            descent). Defaults to None.

        Returns
        -------
        Dict
            A dictionary of optimized graph/constant weights, with the same structure
            as the input `graph_weights`, containing updated values after SGD/Adam
            training.

        Notes
        -----
        - Optimization is parallelized across genomes using `jax.vmap`.
        - Gradient clipping (`clip_by_global_norm`) is automatically added to prevent
          exploding gradients and reduce the likelihood of NaNs.
        - Mini-batch sampling is stochastic and driven by the provided PRNG key.
        """
    # add gradient clipping to the pipeline to prevent nan values
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.),
        optimizer,
    )

    opt_states = jax.vmap(optimizer.init)(graph_weights)

    batch_size = batch_size if batch_size is not None else X.shape[0]
    num_samples = X.shape[0]

    @jax.jit
    def _single_genome_loss(single_weights: Dict[str, jnp.ndarray], single_genotype: Genotype, X_batch: jnp.ndarray,
                            y_batch: jnp.ndarray) -> jnp.ndarray:
        return loss_fn(y_batch, prediction_fn(X_batch, single_genotype, graph_weights=single_weights))

    @jax.jit
    def _single_genome_gradient_step(single_weights: Dict[str, jnp.ndarray], single_genotype: Genotype, opt_st: Any,
                                     X_batch: jnp.ndarray, y_batch: jnp.ndarray):
        loss, grads = jax.value_and_grad(_single_genome_loss)(single_weights, single_genotype, X_batch, y_batch)
        weights_updates, new_opt_st = optimizer.update(grads, opt_st)
        updated_weights = optax.apply_updates(single_weights, weights_updates)
        return updated_weights, new_opt_st, loss

    step_fn = jax.vmap(jax.jit(_single_genome_gradient_step), in_axes=(0, 0, 0, None, None))

    for i in range(n_gradient_steps):
        key, subkey = jax.random.split(key)
        # sample a mini-batch
        batch_idx = jax.random.choice(subkey, num_samples, shape=(batch_size,), replace=False)
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        graph_weights, opt_states, train_losses = step_fn(graph_weights, genotype, opt_states, X_batch, y_batch)

    return graph_weights
