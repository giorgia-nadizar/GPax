from functools import partial
from typing import Tuple, Callable, Optional, Dict, Any, Union

import jax
import jax.numpy as jnp
import optax
from optax._src.base import GradientTransformation

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
    prediction_fn = jax.jit(partial(predict_regression_output, graph_structure=graph_structure))

    def _accuracy_fn(single_genotype: Genotype):
        prediction = prediction_fn(X, single_genotype)
        return accuracy_fn(y, prediction)

    accuracies = jax.vmap(_accuracy_fn)(genotype)

    return jnp.expand_dims(accuracies, 1), genotype


def regression_accuracy_evaluation_with_sgd(
        genotype: Genotype,
        key: RNGKey,
        graph_structure: GGP,
        X: jnp.ndarray,
        y: jnp.ndarray,
        accuracy_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = r2_score,
        optimizer: GradientTransformation = optax.adam(1e-3),
        loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = rmse,
        n_gradient_steps: int = 100,
        batch_size: int = None,
        reset_weights: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """ Perform mini-batch stochastic gradient descent on a batch of genotypes.

        Args:
            genotype: A batch of genotypes to optimize.
            key: PRNG key for mini-batch sampling.
            graph_structure: The graph model.
            X, y: Training data.
            accuracy_fn: Function to compute regression accuracy.
            optimizer: Optax optimizer.
            loss_fn: Loss function.
            n_gradient_steps: Number of gradient steps.
            batch_size: Mini-batch size for SGD.
            reset_weights: whether the weights should be re-initialized.

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            accuracies : jnp.ndarray
                Accuracy values for each genotype, shape (n_genotypes, ...) after SGD
                optimization of weights, depending on `accuracy_fn`.
            genotype : Genotype
                The batch of genotypes that were evaluated with the updated weights.
        """
    graph_weights = graph_structure.get_weights(genotype)
    if reset_weights:
        n_genomes = jax.tree_util.tree_leaves(genotype)[0].shape[0]
        key, subkey = jax.random.split(key)
        weights_keys = jax.random.split(subkey, n_genomes)
        new_weights = jax.vmap(jax.jit(graph_structure.init_weights))(weights_keys)
        graph_weights = {k: new_weights[k] for k in graph_weights.keys() if k in new_weights}

    opt_states = jax.vmap(optimizer.init)(graph_weights)

    batch_size = batch_size if batch_size is not None else X.shape[0]
    num_samples = X.shape[0]

    prediction_fn = jax.jit(partial(predict_regression_output, graph_structure=graph_structure))

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

    # update the weights in the genomes
    updated_genotype = jax.vmap(jax.jit(graph_structure.update_weights), in_axes=(0, 0))(genotype, graph_weights)

    return regression_accuracy_evaluation(updated_genotype, key, graph_structure, X, y, accuracy_fn)


def regression_scoring_fn(
        functions_params: Genotype,
        key: RNGKey,
        train_set_evaluation_fn: Callable[[Params, RNGKey], Tuple[jnp.ndarray, jnp.ndarray]],
        test_set_evaluation_fn: Callable[[Params, RNGKey], Tuple[jnp.ndarray, jnp.ndarray]],
        descriptor_extractor: Optional[Callable[[Params], Descriptor]] = None,
) -> Union[Tuple[Fitness, ExtraScores], Tuple[Fitness, Descriptor, ExtraScores]]:
    """
    Evaluate a set of genotypes on training and test sets and optionally extract descriptors.

    This function computes a fitness score for a batch of genotypes by first evaluating them on a training set
    using `train_set_evaluation_fn`, which may also perform updates to the parameters (e.g., gradient steps).
    It then evaluates the updated parameters on a test set using `test_set_evaluation_fn`. Optionally, a
    `descriptor_extractor` can be provided to compute descriptors from the original genotype parameters.

    Parameters
    ----------
    functions_params : Genotype
        The genotype batch to evaluate.
    key : RNGKey
        JAX random key used to seed any stochastic operations in evaluation.
    train_set_evaluation_fn : Callable[[Params, RNGKey], Tuple[jnp.ndarray, jnp.ndarray]]
        Function that evaluates the genotype on the training set. Returns a tuple of
        `(train_accuracy, updated_params)`, where `updated_params` may correspond to new genotypes
        updated during the evaluation on the train set, e.g., via gradient descent.
    test_set_evaluation_fn : Callable[[Params, RNGKey], Tuple[jnp.ndarray, jnp.ndarray]]
        Function that evaluates the (possibly updated) genotypes on the test set.
        Returns `(test_accuracy, updated_params)`.
    descriptor_extractor : Optional[Callable[[Params], Descriptor]], optional
        A function to compute descriptors from the original parameters. If `None`, no descriptor is computed.

    Returns
    -------
    Union[Tuple[Fitness, ExtraScores], Tuple[Fitness, Descriptor, ExtraScores]]
        If `descriptor_extractor` is provided:
            - train_accuracy : Fitness
                Fitness score(s) computed on the training set.
            - descriptor : Descriptor
                Descriptor extracted from the original genotype parameters.
            - extra_scores : ExtraScores
                Dictionary containing additional metrics, including:
                - `"test_accuracy"`: test set accuracy computed from updated parameters.
                - `"updated_params"`: parameters after training evaluation.
        If `descriptor_extractor` is None:
            - train_accuracy : Fitness
                Fitness score(s) computed on the training set.
            - extra_scores : ExtraScores
                Dictionary containing additional metrics as above, but no descriptor is returned.

    Notes
    -----
    - `train_set_evaluation_fn` can optionally perform updates on the parameters.
    - Descriptors are only returned when `descriptor_extractor` is not None.
    """
    train_key, test_key = jax.random.split(key)
    # it can be a simple accuracy computation, but it can also include gradient steps
    train_accuracy, updated_params = train_set_evaluation_fn(functions_params, train_key)
    test_accuracy, _ = test_set_evaluation_fn(updated_params, test_key)
    if descriptor_extractor is not None:
        descriptor = descriptor_extractor(functions_params)
        return train_accuracy, descriptor, {"test_accuracy": test_accuracy, "updated_params": updated_params}
    else:
        return train_accuracy, {"test_accuracy": test_accuracy, "updated_params": updated_params}
