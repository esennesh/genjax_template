"""Amortized stochastic variational inference learner, in GenJAX.

Port of the NumPyro ``SviLearner``. The ELBO is built from GenJAX's generative
function interface -- ``guide.simulate`` + ``model.assess`` -- wrapped in an
ADEV ``@expectation`` so that ``grad_estimate`` yields reparameterized gradient
estimates. Parameters (encoder + decoder neural-network weights) are held
explicitly and optimized with optax (GenJAX has no global parameter store).

The ``model`` and ``guide`` passed in are *factories* ``f(out_dim) -> ...`` (see
``src.model.model``); the data dimensionality is only known at ``setup_step``
time, so the generative functions and parameters are built there.
"""

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as random
import optax
from typing import Any, Dict

from genjax.adev import expectation
from genjax.pjax import seed

from .learner import ParamLearner
from .objectives import elbo
from src.data import DataModule


def _flatten_batch(data) -> jnp.ndarray:
    data = jnp.asarray(data)
    return jnp.reshape(data, (data.shape[0], -1)).astype(jnp.float32)


class SviLearner(ParamLearner):
    def __init__(self, data_shape, guide, model, optim, num_particles=1, rng=0,
                 objective=elbo):
        if not isinstance(rng, jax.Array):
            rng = random.key(rng)
        self._rng = rng
        self._model_factory = model
        self._guide_factory = guide
        self.num_particles = num_particles
        self.optimizer = optim
        # Reduction from per-particle log-weights to the scalar bound (e.g. ELBO
        # or IWAE); selected via the `objective` Hydra config group.
        self._objective_fn = objective

        self._model = None
        self._guide = None
        self._params = None
        self._opt_state = None
        self._value_and_grad = None  # jitted, batched (elbo_value, elbo_grad)

    # -- construction (deferred until the data shape is known) ----------------

    def _build(self, out_dim):
        model, decoder_init = self._model_factory(out_dim)
        guide, encoder_init = self._guide_factory(out_dim)
        self._model, self._guide = model, guide

        n_particles = self.num_particles
        objective_fn = self._objective_fn

        def single_log_weight(x, params):
            # ELBO log-weight: log p(x, z) - log q(z | x), with z ~ q.
            tr = guide.simulate(x, params["encoder"])
            merged, _ = model.merge({"obs": x}, tr.get_choices())
            model_logp, _ = model.assess(merged, params["decoder"])
            return jnp.sum(model_logp) + tr.get_score()

        @expectation
        def objective(x, params):
            log_weights = jnp.stack(
                [single_log_weight(x, params) for _ in range(n_particles)]
            )
            # Reduce the per-particle log-weights to the configured bound.
            return objective_fn(log_weights)

        # One seeded pass producing both the bound's value and its gradient
        # w.r.t. params (shared randomness), vmapped across the batch.
        def value_and_grad(key, x, params):
            return seed(
                lambda x, p: (
                    objective.prog.source.value(x, p),
                    objective.grad_estimate(x, p)[1],
                )
            )(key, x, params)

        self._objective = objective
        self._value_and_grad = jax.jit(
            jax.vmap(value_and_grad, in_axes=(0, 0, None))
        )
        return decoder_init, encoder_init

    def setup_step(self, datamodule: DataModule):
        for batch in datamodule.test_dataloader():
            data = batch[0]
            break
        x = _flatten_batch(data)
        out_dim = int(x.shape[-1])

        decoder_init, encoder_init = self._build(out_dim)

        if self._params is None:
            self._rng, k_dec, k_enc = random.split(self._rng, 3)
            self._params = {
                "decoder": decoder_init(k_dec),
                "encoder": encoder_init(k_enc),
            }
            self._opt_state = self.optimizer.init(self._params)
        return self._params

    # -- ParamLearner interface ----------------------------------------------

    def __call__(self, data, *args, **kwargs):
        """Reconstruct a batch: encode each image, then decode the latent."""
        x = _flatten_batch(data)
        self._rng, key = random.split(self._rng)
        keys = random.split(key, x.shape[0])

        def reconstruct(key, xi):
            tr = seed(self._guide.simulate)(key, xi, self._params["encoder"])
            z = tr.get_choices()["z"]
            # model.assess returns (log density, retval); retval = decode(z).
            _, img_probs = self._model.assess(
                {"z": z, "obs": xi}, self._params["decoder"]
            )
            return img_probs

        return jax.vmap(reconstruct)(keys, x)

    def load(self, checkpoint: Dict[str, Any]):
        self._params = checkpoint["params"]
        self._opt_state = checkpoint["opt_state"]
        self._rng = checkpoint["rng"]

    @property
    def parameters(self):
        return self._params

    def save(self) -> Dict[str, Any]:
        return {
            "params": self._params,
            "opt_state": self._opt_state,
            "rng": self._rng,
        }

    def _epoch_keys(self, batch_size):
        self._rng, key = random.split(self._rng)
        return random.split(key, batch_size)

    def train_step(self, data, *args) -> Dict[str, float]:
        x = _flatten_batch(data)
        keys = self._epoch_keys(x.shape[0])
        elbos, grads = self._value_and_grad(keys, x, self._params)
        mean_grad = jax.tree.map(lambda g: jnp.mean(g, axis=0), grads)
        # Maximize the ELBO == minimize -ELBO; optax descends, so feed -gradient.
        loss_grad = jax.tree.map(lambda g: -g, mean_grad)
        updates, self._opt_state = self.optimizer.update(
            loss_grad, self._opt_state, self._params
        )
        self._params = optax.apply_updates(self._params, updates)
        return {"loss": -jnp.mean(elbos)}

    def _evaluate(self, data) -> Dict[str, float]:
        x = _flatten_batch(data)
        keys = self._epoch_keys(x.shape[0])
        elbos, _ = self._value_and_grad(keys, x, self._params)
        return {"loss": -jnp.mean(elbos)}

    def test_step(self, data, *args) -> Dict[str, float]:
        return self._evaluate(data)

    def valid_step(self, data, *args) -> Dict[str, float]:
        return self._evaluate(data)
