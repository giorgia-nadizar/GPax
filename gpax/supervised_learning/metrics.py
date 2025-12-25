import jax.numpy as jnp


def r2_score(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Compute R² score.

    R² = 1 - (SS_res / SS_tot)
    """
    # Residual sum of squares
    ss_res = jnp.sum((y_true - y_pred) ** 2)

    # Total sum of squares (variance around the mean)
    ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)

    # If variance is zero, define R² as 1.0 when predictions are perfect, else 0.0
    r2 = jnp.where(ss_tot == 0,
                   jnp.where(ss_res == 0, 1.0, 0.0),
                   1 - ss_res / ss_tot)
    r2 = jnp.nan_to_num(r2, nan=-1e30, posinf=-1e30, neginf=-1e30)
    r2 = jnp.clip(r2, -1e30, 1.0)

    return r2


def mse(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Mean Squared Error (MSE).
    """
    mse_val = jnp.mean((y_true - y_pred) ** 2)
    return jnp.nan_to_num(mse_val, nan=1e6, posinf=1e6, neginf=1e6)


def rmse(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Root Mean Squared Error (RMSE).
    """
    mse_val = mse(y_true, y_pred)
    rmse_val = jnp.sqrt(mse_val)
    return jnp.nan_to_num(rmse_val, nan=1e6, posinf=1e6, neginf=1e6)


def rrmse_per_target(y_test: jnp.ndarray, y_pred: jnp.ndarray, y_train: jnp.ndarray) -> jnp.ndarray:
    """
    Compute RRMSE for each target in a multi-target regression problem.

    Parameters
    ----------
    y_test : jnp.ndarray, shape (N, T)
        True target values on the test set.
    y_pred : jnp.ndarray, shape (N, T)
        Predicted target values on the test set.
    y_train : jnp.ndarray, shape (N_train, T)
        Target values from the training set.

    Returns
    -------
    rrmse : jnp.ndarray, shape (T,)
        RRMSE value for each target.
    """

    # Mean of each target over the training set
    y_train_mean = jnp.mean(y_train, axis=0)  # (T,)

    # Numerator: sum of squared prediction errors per target
    num = jnp.sum((y_pred - y_test) ** 2, axis=0)

    # Denominator: sum of squared deviations from training mean per target
    den = jnp.sum((y_train_mean - y_test) ** 2, axis=0)

    # Small epsilon for numerical stability
    eps = 1e-12

    return jnp.sqrt(num / (den + eps))
