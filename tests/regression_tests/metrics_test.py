import jax.numpy as jnp
from sklearn.metrics import root_mean_squared_error, r2_score as sklearn_r2_score, mean_squared_error

from gpax.symbolicregression.metrics import r2_score, rmse, mse


def test_r2_score_perfect():
    # Perfect prediction → R² should be 1.0
    y = jnp.array([1.0, 2.0, 3.0])
    y_pred = jnp.array([1.0, 2.0, 3.0])
    assert r2_score(y, y_pred) == 1.0


def test_r2_score_zero_variance_perfect():
    # Zero variance in y and perfect prediction → defined as 1.0
    y = jnp.array([5.0, 5.0, 5.0])
    y_pred = jnp.array([5.0, 5.0, 5.0])
    assert r2_score(y, y_pred) == 1.0


def test_r2_score_zero_variance_imperfect():
    # Zero variance in y but prediction is not perfect → defined as 0.0
    y = jnp.array([5.0, 5.0, 5.0])
    y_pred = jnp.array([4.0, 6.0, 5.0])
    assert r2_score(y, y_pred) == 0.0


def test_r2_score_known_value():
    # Known example used in scikit-learn documentation
    # R² should match sklearn's output closely
    y = jnp.array([3.0, -0.5, 2.0, 7.0])
    y_pred = jnp.array([2.5, 0.0, 2.0, 8.0])
    result = r2_score(y, y_pred)
    expected = sklearn_r2_score(y, y_pred)
    assert jnp.isclose(result, expected)


def test_mse_perfect():
    # Perfect prediction → MSE should be 0
    y = jnp.array([1.0, 2.0, 3.0])
    y_pred = jnp.array([1.0, 2.0, 3.0])
    assert mse(y, y_pred) == 0.0


def test_mse_known_value():
    # MSE for a known simple example:
    # Errors: [1, 0, 1] → MSE = 2/3
    y = jnp.array([1.0, 2.0, 3.0])
    y_pred = jnp.array([2.0, 2.0, 2.0])
    result = mse(y, y_pred)
    expected = mean_squared_error(y, y_pred)
    assert jnp.isclose(result, expected)


def test_rmse_perfect():
    # Perfect prediction → RMSE should be 0
    y = jnp.array([1.0, 2.0, 3.0])
    y_pred = jnp.array([1.0, 2.0, 3.0])
    assert rmse(y, y_pred) == 0.0


def test_rmse_known_value():
    # RMSE for a known simple example:
    # Errors: [1, 0, 1] → MSE = 2/3 → RMSE = sqrt(2/3)
    y = jnp.array([1.0, 2.0, 3.0])
    y_pred = jnp.array([2.0, 2.0, 2.0])
    result = rmse(y, y_pred)
    expected = root_mean_squared_error(y, y_pred)
    assert jnp.isclose(result, expected)
