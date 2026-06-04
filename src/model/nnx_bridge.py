"""Bridge Flax NNX modules into GenJAX's explicit-parameter idiom.

GenJAX has no global parameter store: trainable weights are passed into
generative functions as ordinary arguments and owned by the learner (+ optax).
Flax NNX, conversely, keeps parameters *inside* a live module object. This is
the converse of NumPyro's ``nnx_module`` contrib, which extracts a module's
parameters *out* into NumPyro's global store; here (:func:`nnx_wrap`) we keep
parameters as the optimized pytree and reconstitute a live module from them on
demand.

The bridge is NNX's functional split/merge:

- ``nnx.split(module) -> (graphdef, state)`` separates the static structure
  (``graphdef``, closed over) from the parameter pytree (``state`` -- plain
  JAX arrays, the thing the learner optimizes).
- ``nnx.merge(graphdef, state)`` rebuilds a callable module from a parameter
  pytree -- "take a big pytree of parameters, get a concrete ``nnx.Module``".
"""

from typing import Callable, Tuple

import jax
import flax.nnx as nnx


def nnx_wrap(
    ctor: Callable[[jax.Array], nnx.Module],
) -> Tuple[Callable, Callable[[jax.Array], object]]:
    """Wrap an NNX module constructor as an ``(apply, init)`` pair.

    Args:
        ctor: ``ctor(key) -> nnx.Module``; builds the module with parameters
            initialized from a PRNG key, e.g.
            ``lambda k: Decoder(..., rngs=nnx.Rngs(k))``.

    Returns:
        ``(apply, init)`` where

        - ``init(key) -> params`` is the module's parameter pytree (NNX state,
          plain arrays) -- ready for optax and for threading into a ``@gen``
          program as an argument;
        - ``apply(params, *args, **kwargs) -> output`` merges ``params`` back
          into a live module and calls it.

    The static ``graphdef`` is captured once at wrap time and closed over, so it
    never enters the differentiated/traced argument path. ``graphdef`` is
    independent of parameter *values*, so building it from a throwaway key is
    safe.
    """
    graphdef, _ = nnx.split(ctor(jax.random.key(0)))

    def apply(params, *args, **kwargs):
        return nnx.merge(graphdef, params)(*args, **kwargs)

    def init(key):
        _, state = nnx.split(ctor(key))
        return state

    return apply, init
