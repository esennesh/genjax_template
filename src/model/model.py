"""VAE model + guide for MNIST, expressed as GenJAX generative functions.

This is the GenJAX port of the NumPyro VAE example. The model and guide are
per-example ``@gen`` programs (the batch dimension is handled by vmapping in the
learner, following the GenJAX amortized-inference idiom). Neural networks are
Flax NNX modules; because GenJAX has no global parameter store (unlike
``numpyro.module``), their parameters are managed explicitly by the learner and
threaded in as generative-function arguments. The ``nnx_wrap`` bridge
(:mod:`src.model.nnx_bridge`) converts each NNX module into an ``(apply, init)``
pair: ``init`` yields the parameter pytree the learner optimizes, and ``apply``
reconstitutes a live module from a parameter pytree via ``nnx.merge``.
"""

import jax.numpy as jnp
import flax.nnx as nnx

from genjax import gen, tfp_distribution
from genjax.adev import multivariate_normal_diag_reparam

import tensorflow_probability.substrates.jax as tfp

from .nnx_bridge import nnx_wrap

tfd = tfp.distributions

# Float-valued, probs-parameterized Bernoulli for pixel likelihoods. (GenJAX's
# built-in ``bernoulli`` is logits-parameterized and boolean-valued; the decoder
# emits probabilities in [0, 1], so we want a probs parameterization here.)
bernoulli_probs = tfp_distribution(
    lambda probs: tfd.Bernoulli(probs=probs, dtype=jnp.float32),
    name="BernoulliProbs",
)


class Encoder(nnx.Module):
    """Amortized inference network: image -> (z_loc, z_scale)."""

    def __init__(self, in_dim, hidden_dim, z_dim, *, rngs):
        self.hidden = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.loc = nnx.Linear(hidden_dim, z_dim, rngs=rngs)
        self.log_scale = nnx.Linear(hidden_dim, z_dim, rngs=rngs)

    def __call__(self, x):
        h = nnx.softplus(self.hidden(x))
        # Scale is positive; parameterize it in log-space (cf. the stax `Exp`).
        return self.loc(h), jnp.exp(self.log_scale(h))


class Decoder(nnx.Module):
    """Generative network: latent z -> per-pixel Bernoulli probabilities."""

    def __init__(self, z_dim, hidden_dim, out_dim, *, rngs):
        self.hidden = nnx.Linear(z_dim, hidden_dim, rngs=rngs)
        self.out = nnx.Linear(hidden_dim, out_dim, rngs=rngs)

    def __call__(self, z):
        h = nnx.softplus(self.hidden(z))
        return nnx.sigmoid(self.out(h))


def make_mnist_model(out_dim, hidden_dim=400, z_dim=100):
    """Build the per-example VAE model ``p(x, z)``.

    Returns ``(model, decoder_init)`` where ``model(decoder_params)`` is a
    ``@gen`` program sampling ``z ~ N(0, I)`` and ``x ~ Bernoulli(decode(z))``,
    and ``decoder_init(key)`` initializes the decoder parameters.
    """
    decode, decoder_init = nnx_wrap(
        lambda key: Decoder(z_dim, hidden_dim, out_dim, rngs=nnx.Rngs(key))
    )
    z_loc = jnp.zeros((z_dim,), dtype=jnp.float32)
    z_scale = jnp.ones((z_dim,), dtype=jnp.float32)

    @gen
    def model(decoder_params):
        z = multivariate_normal_diag_reparam(z_loc, z_scale) @ "z"
        img_probs = decode(decoder_params, z)
        bernoulli_probs(img_probs) @ "obs"
        return img_probs

    return model, decoder_init


def make_mnist_guide(out_dim, hidden_dim=400, z_dim=100):
    """Build the per-example amortized guide ``q(z | x)``.

    Returns ``(guide, encoder_init)`` where ``guide(x, encoder_params)`` encodes a
    single (flattened) image to ``(z_loc, z_scale)`` and samples ``z`` via the
    reparameterized normal, and ``encoder_init(key)`` initializes the encoder.
    """
    encode, encoder_init = nnx_wrap(
        lambda key: Encoder(out_dim, hidden_dim, z_dim, rngs=nnx.Rngs(key))
    )

    @gen
    def guide(x, encoder_params):
        z_loc, z_scale = encode(encoder_params, x)
        multivariate_normal_diag_reparam(z_loc, z_scale) @ "z"

    return guide, encoder_init
