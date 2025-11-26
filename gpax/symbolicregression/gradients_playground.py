import jax
import optax
import jax.numpy as jnp


def run_opt(init_params, fun, opt, max_iter, tol):
    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun
        )
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = optax.tree.get(state, 'count')
        grad = optax.tree.get(state, 'grad')
        err = optax.tree.norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (init_params, opt.init(init_params))
    final_params, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_params, final_state


def fun(w):
    return jnp.sum(100.0 * (w[1:] - w[:-1] ** 2) ** 2 + (1.0 - w[:-1]) ** 2)


opt = optax.lbfgs()


def init_fn(x):
    return {"params": jnp.zeros((8,)), "mypars": jnp.ones((8,))}


multiple_params = jax.vmap(init_fn)(jnp.zeros((2)))
print(multiple_params)

def ravel_single(tree):
    leaves, _ = jax.flatten_util.ravel_pytree(tree)
    return leaves

flatties = jax.vmap(ravel_single)(multiple_params)
print(flatties)
single_example = jax.tree.map(lambda x: x[0], multiple_params)
leaves_template, treedef = jax.flatten_util.ravel_pytree(single_example)

unflats = jax.vmap(treedef)(flatties)
print(unflats)
#
#
#
# multiple_flat_params, unravel_fn = jax.flatten_util.ravel_pytree(multiple_params)
# print(multiple_flat_params)

exit(5)
pytree_params = {"params": jnp.zeros((8,))}
init_params, unravel_fn = jax.flatten_util.ravel_pytree(pytree_params)
print(
    f'Initial pytree: {pytree_params}, '
)
print(init_params)
print(jnp.zeros((8,)))
print(
    f'Initial value: {fun(init_params):.2e} '
    f'Initial gradient norm: {optax.tree.norm(jax.grad(fun)(init_params)):.2e}'
)
final_params, _ = run_opt(init_params, fun, opt, max_iter=100, tol=1e-3)
print(
    f'Final value: {fun(final_params):.2e}, '
    f'Final gradient norm: {optax.tree.norm(jax.grad(fun)(final_params)):.2e}'
)
final_pytree = unravel_fn(final_params)
print(
    f'Final pytree: {final_pytree}, '
)
