import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax import random

from src.agent.components.networks.networks import LinearConfig, TorsoConfig, torso_builder

N = 1000
d = 3

key = random.key(1)
key, subkey = random.split(key)
X = random.uniform(subkey, shape=(N, d))

A = jnp.array([1.0, -0.5, 0.5])
b = jnp.array(1)

def f1(key: jax.Array, x: jax.Array):
    noise = 0.1 * random.normal(key)
    y = A @ x + b + noise
    return y

def f2(key: jax.Array, x: jax.Array):
    return jnp.sin(f1(key, x))


key, subkey = random.split(key)
keys_y1 = random.split(subkey, N)
Y1 = jax.vmap(f1, in_axes=(0, 0))(keys_y1, X)

key, subkey = random.split(key)
keys_y2 = random.split(subkey, N)
Y2 = jax.vmap(f1, in_axes=(0, 0))(keys_y2, X)

def sqerr(y1: jax.Array, y2: jax.Array) -> jax.Array:
    return jnp.square(y1 - y2)

def mse(Y1: jax.Array, Y2: jax.Array) -> jax.Array:
    return jax.vmap(sqerr, in_axes=(0,0))(Y1, Y2).mean()



layer_cfgs = [LinearConfig(size=32) for _ in range(2)] + [LinearConfig(size=1)]
nn_cfg = TorsoConfig(layers=layer_cfgs)
nn = hk.transform(lambda x: torso_builder(nn_cfg)(x))
nn = hk.without_apply_rng(nn)

key, subkey = random.split(key)

def nn_loss(params: optax.Params, X: jax.Array, Y: jax.Array):
    Y_hat = jax.vmap(nn.apply, in_axes=(None, 0))(params, X)
    loss = mse(Y_hat, Y)
    return loss




M = 1_000
alpha = 0.1
adam = optax.adam(alpha)

@jax.jit
def adam_update(params: optax.Params, opt_state: optax.OptState, X: jax.Array, Y: jax.Array):
    grad = jax.grad(nn_loss)(params, X, Y)
    updates, opt_state = adam.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

params = nn.init(subkey, X[0])
opt_state = adam.init(params)
for i in range(M):
    print(f'Iter {i} Objective function: {nn_loss(params, X, Y2):.2E}')
    params, opt_state = adam_update(params, opt_state, X, Y2)


lso = optax.chain(
    optax.adam(learning_rate=alpha),
    optax.scale_by_backtracking_linesearch(
        max_backtracking_steps=50, max_learning_rate=alpha, decrease_factor=0.9, slope_rtol=0.1
    ),
)

@jax.jit
def lso_update(params: optax.Params, opt_state: optax.OptState, X: jax.Array, Y: jax.Array):
    value, grad = jax.value_and_grad(nn_loss)(params, X, Y2)

    updates, opt_state = lso.update(
      grad,
      opt_state,
      params,
      value=value,
      grad=grad,
      value_fn=nn_loss,
      X=X,
      Y=Y2
    )
    params = optax.apply_updates(params, updates)
    return params, opt_state


print('Objective function: {:.2E}'.format(nn_loss(params, X, Y2)))
params = nn.init(subkey, X[0])
opt_state = lso.init(params)
for i in range(M):
    print(f'Iter {i} Objective function: {nn_loss(params, X, Y2):.2E}')
    params, opt_state = lso_update(params, opt_state, X, Y2)
