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
    return jnp.where(ss_tot == 0,
                     jnp.where(ss_res == 0, 1.0, 0.0),
                     1 - ss_res / ss_tot)


def rmse(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Root Mean Squared Error (RMSE).
    """
    return jnp.sqrt(jnp.mean((y_true - y_pred) ** 2))
