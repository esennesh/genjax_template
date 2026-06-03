"""VAE model + guide for MNIST, expressed as GenJAX generative functions.

This is the GenJAX port of the NumPyro VAE example. The model and guide are
per-example ``@gen`` programs (the batch dimension is handled by vmapping in the
learner, following the GenJAX amortized-inference idiom). Neural networks are
built with ``jax.example_libraries.stax``; their parameters are managed
explicitly by the learner and threaded in as generative-function arguments
(GenJAX has no global parameter store, unlike ``numpyro.module``).
"""

import jax.numpy as jnp
from jax.example_libraries import stax

from genjax import gen, tfp_distribution
from genjax.adev import multivariate_normal_diag_reparam

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

# Float-valued, probs-parameterized Bernoulli for pixel likelihoods. (GenJAX's
# built-in ``bernoulli`` is logits-parameterized and boolean-valued; the decoder
# emits probabilities in [0, 1], so we want a probs parameterization here.)
bernoulli_probs = tfp_distribution(
    lambda probs: tfd.Bernoulli(probs=probs, dtype=jnp.float32),
    name="BernoulliProbs",
)


def encoder(hidden_dim, z_dim):
    return stax.serial(
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Softplus,
        stax.FanOut(2),
        stax.parallel(
            stax.Dense(z_dim, W_init=stax.randn()),
            stax.serial(stax.Dense(z_dim, W_init=stax.randn()), stax.Exp),
        ),
    )


def decoder(hidden_dim, out_dim):
    return stax.serial(
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Softplus,
        stax.Dense(out_dim, W_init=stax.randn()),
        stax.Sigmoid,
    )


def make_mnist_model(out_dim, hidden_dim=400, z_dim=100):
    """Build the per-example VAE model ``p(x, z)``.

    Returns ``(model, decoder_init, z_dim)`` where ``model(decoder_params)`` is a
    ``@gen`` program sampling ``z ~ N(0, I)`` and ``x ~ Bernoulli(decode(z))``,
    and ``decoder_init(key)`` initializes the decoder parameters.
    """
    dec_init, decode = decoder(hidden_dim, out_dim)
    z_loc = jnp.zeros((z_dim,), dtype=jnp.float32)
    z_scale = jnp.ones((z_dim,), dtype=jnp.float32)

    @gen
    def model(decoder_params):
        z = multivariate_normal_diag_reparam(z_loc, z_scale) @ "z"
        img_probs = decode(decoder_params, z)
        bernoulli_probs(img_probs) @ "obs"
        return img_probs

    def decoder_init(key):
        _, params = dec_init(key, (z_dim,))
        return params

    return model, decoder_init, z_dim


def make_mnist_guide(out_dim, hidden_dim=400, z_dim=100):
    """Build the per-example amortized guide ``q(z | x)``.

    Returns ``(guide, encoder_init)`` where ``guide(x, encoder_params)`` encodes a
    single (flattened) image to ``(z_loc, z_scale)`` and samples ``z`` via the
    reparameterized normal, and ``encoder_init(key)`` initializes the encoder.
    """
    enc_init, encode = encoder(hidden_dim, z_dim)

    @gen
    def guide(x, encoder_params):
        z_loc, z_scale = encode(encoder_params, x)
        multivariate_normal_diag_reparam(z_loc, z_scale) @ "z"

    def encoder_init(key):
        _, params = enc_init(key, (out_dim,))
        return params

    return guide, encoder_init
