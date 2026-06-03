"""Configurable variational objectives.

An *objective* is a reduction from the per-particle ELBO log-weights

    log w_k = log p(x, z_k) - log q(z_k | x),   z_k ~ q(. | x),

collected as a ``(num_particles,)`` array, down to a scalar bound on the
marginal log-likelihood ``log p(x)``. ``SviLearner`` collects the log-weights
inside its ADEV ``@expectation`` and applies the configured objective; which one
is used is selected by the ``objective`` Hydra config group
(``configs/learner/objective``).

All objectives reduce along axis 0 (the particle axis) and are built from
reparameterization-friendly primitives so they differentiate cleanly through
GenJAX/ADEV's ``grad_estimate``.
"""

import jax.numpy as jnp
from jax.scipy.special import logsumexp


def elbo(log_weights):
    """Multi-sample ELBO: the mean of the log-weights.

    This averages ``num_particles`` independent single-sample ELBOs; it equals
    the standard ELBO when ``num_particles == 1``. It is the looser of the two
    bounds and does not tighten as particles are added (only its variance does).
    """
    return jnp.mean(log_weights, axis=0)


def iwae(log_weights):
    """Importance-weighted bound (IWAE): ``log (1/K) * sum_k w_k``.

    A log-mean-exp of the log-weights. Tighter than :func:`elbo`, and
    monotonically tightens towards ``log p(x)`` as ``num_particles`` grows.
    """
    n = log_weights.shape[0]
    return logsumexp(log_weights, axis=0) - jnp.log(float(n))
