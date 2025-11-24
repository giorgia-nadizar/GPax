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
    r2 = jnp.nan_to_num(r2, nan=-10.0, posinf=-10.0, neginf=-10.0)
    r2 = jnp.clip(r2, -10.0, 1.0)

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
