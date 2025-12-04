import jax.numpy as jnp
import jax
from qdax.custom_types import RNGKey


def downsample_dataset(
        X: jnp.ndarray,
        y: jnp.ndarray,
        random_key: RNGKey,
        ratio: float = None,
        size: int = None
):
    """
    Randomly downsample a dataset to a given size or ratio.

    You may specify either:
    - `ratio`: a float in (0, 1], representing the fraction of data to keep, or
    - `size`: the exact number of samples to keep.

    If neither is given, the full dataset is returned unchanged.

    Parameters
    ----------
    X : jnp.ndarray
        Input feature matrix of shape (N, ...).
    y : jnp.ndarray
        Target values of shape (N, ...).
    random_key : RNGKey
        JAX PRNG key used for sampling.
    ratio : float, optional
        Fraction of the dataset to retain. Ignored if `size` is provided.
    size : int, optional
        Exact number of samples to retain.

    Returns
    -------
    (X_sub, y_sub) : tuple of jnp.ndarray
        Downsampled feature matrix and target array.
    """
    if size is None:
        size = int(X.shape[0] * ratio) if ratio is not None else X.shape[0]

    size = min(size, X.shape[0])  # safety

    # Randomly choose indices without replacement
    indices = jax.random.choice(
        random_key,
        X.shape[0],
        shape=(size,),
        replace=False
    )

    X_batch = jnp.take(X, indices, axis=0)
    y_batch = jnp.take(y, indices, axis=0)

    return X_batch, y_batch
