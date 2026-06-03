"""Tests for static graph-structure capture (`capture_model_graph`).

These verify that dependency edges are recovered for both continuous and
*discrete* parents (the latter is where an autodiff-based detector fails), that
independent sites get no edge, and that node metadata (latent/observed, shape)
is annotated correctly.
"""

import jax.numpy as jnp
import jax.random as jrand

from genjax import gen, normal, categorical

from src.learner.graphical import capture_model_graph
from src.model.model import make_mnist_model


def _edges(model, model_args=(), *, key=0, latent=()):
    g = capture_model_graph(
        model, model_args, key=jrand.key(key), latent_addresses=latent
    )
    return g, set(g.edges())


def test_continuous_chain_edges():
    # a -> b (b's mean is a); c is independent.
    @gen
    def chain():
        a = normal(0.0, 1.0) @ "a"
        normal(a, 1.0) @ "b"
        normal(0.0, 1.0) @ "c"

    _, edges = _edges(chain)
    assert ("a", "b") in edges
    # No spurious edges to/from the independent site or backwards.
    assert ("b", "a") not in edges
    assert not any("c" in e for e in edges)


def test_discrete_parent_edge():
    # k ~ Categorical (integer-valued) indexes x's mean: autodiff cannot see
    # this edge, but dataflow provenance must.
    @gen
    def discrete_parent():
        k = categorical(jnp.zeros(3)) @ "k"
        mu = jnp.array([-2.0, 0.0, 2.0])[k]
        normal(mu, 1.0) @ "x"

    g, edges = _edges(discrete_parent, latent=("k", "x"))
    assert ("k", "x") in edges
    assert ("x", "k") not in edges
    assert g.nodes["k"]["distribution"] == "Categorical"


def test_independent_sites_have_no_edges():
    @gen
    def indep():
        normal(0.0, 1.0) @ "a"
        normal(0.0, 1.0) @ "b"

    _, edges = _edges(indep)
    assert edges == set()


def test_vae_structure_nodes_and_edge():
    out_dim, hidden, z_dim = 24, 16, 4
    model, decoder_init, _ = make_mnist_model(out_dim, hidden, z_dim)
    g, edges = _edges(
        model, (decoder_init(jrand.key(0)),), key=1, latent=("z",)
    )
    # Continuous latent z feeds the decoder -> observation likelihood.
    assert ("z", "obs") in edges
    assert ("obs", "z") not in edges
    # Node annotations.
    assert g.nodes["z"]["observed"] is False
    assert g.nodes["obs"]["observed"] is True
    assert g.nodes["z"]["shape"] == (z_dim,)
    assert g.nodes["obs"]["shape"] == (out_dim,)
