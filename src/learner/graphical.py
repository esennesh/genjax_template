"""A learner that captures and reasons about a GenJAX model's static graph.

``GraphicalModelLearner`` trains exactly like :class:`SviLearner` (ELBO via
GenJAX's GFI + ADEV), but additionally introspects the generative model's
*static graph structure* at setup time and exposes it as a ``networkx`` digraph
for reasoning and rendering.

GenJAX keeps random-choice addresses in the handler that interprets a ``@gen``
program (not in the staged jaxpr), so we recover the structure by running the
model under small custom handlers pushed onto ``genjax.core.handler_stack``:

- **Nodes** (one per sample-site address) are recorded by a handler that wraps
  ``gen_fn.simulate`` and notes each site's distribution family and value shape.
- **Edges** are recovered by *autodiff provenance*: a second handler computes the
  per-site log-densities ``log p_B(value_B; args_B(upstream values))`` as a
  function of all choice values, and we add an edge ``A -> B`` whenever
  ``d log p_B / d value_A`` is nonzero. This captures dependencies through
  continuous parents; dependencies routed only through a *discrete* parent are
  not detected by autodiff (a documented limitation -- extend with a provenance
  interpreter if you need them).

Latent vs. observed is annotated from the guide: addresses the guide samples are
latent, the remaining model addresses are treated as observed.
"""

import jax
import jax.numpy as jnp
import networkx as nx
from typing import Any, Dict

from genjax.core import handler_stack
from genjax.pjax import seed

from .svi import SviLearner
from src.data import DataModule


class _RecordSites:
    """Handler that records each sample site's distribution and value."""

    def __init__(self):
        self.sites: Dict[str, Dict[str, Any]] = {}

    def __call__(self, addr, gen_fn, args, kwargs=None):
        tr = gen_fn.simulate(*args, **(kwargs or {}))
        value = tr.get_retval()
        name = getattr(getattr(gen_fn, "name", None), "value", None)
        self.sites[addr] = {
            "distribution": name or type(gen_fn).__name__,
            "shape": tuple(jnp.shape(value)),
        }
        return value


class _PerSiteLogp:
    """Handler that records each site's log-density given fixed choices."""

    def __init__(self, choices: Dict[str, Any]):
        self.choices = choices
        self.logps: Dict[str, Any] = {}

    def __call__(self, addr, gen_fn, args, kwargs=None):
        x = self.choices[addr]
        logp, _ = gen_fn.assess(x, *args, **(kwargs or {}))
        self.logps[addr] = jnp.sum(logp)
        return x


def _run_with_handler(handler, fn, *args, **kwargs):
    handler_stack.append(handler)
    try:
        fn(*args, **kwargs)
    finally:
        handler_stack.pop()
    return handler


def capture_model_graph(model, model_args, *, key, latent_addresses=()):
    """Build a ``networkx.DiGraph`` of a GenJAX model's static structure.

    Args:
        model: a ``@gen`` generative function.
        model_args: arguments to call the model with.
        key: PRNG key (sampling is needed to record a reference execution).
        latent_addresses: addresses considered latent (typically the guide's
            sample sites); all other recorded addresses are marked observed.

    Returns:
        A ``networkx.DiGraph`` whose nodes carry ``distribution``, ``shape`` and
        ``observed`` attributes, and whose edges ``A -> B`` mean site ``B``'s
        distribution depends on site ``A``'s value.
    """
    source = model.source.value

    # -- nodes: record each site's distribution + shape. We run the recorder
    # as a side effect during `seed` tracing (the metadata we keep -- dist
    # family name and value shape -- is available at trace time), returning a
    # dummy JAX value so staging succeeds.
    recorder = _RecordSites()

    def _record(*a):
        handler_stack.append(recorder)
        try:
            source(*a)
        finally:
            handler_stack.pop()
        return jnp.array(0.0)

    seed(_record)(key, *model_args)
    sites = recorder.sites

    # Reference (concrete) choice values, used to stage the edge jaxpr below.
    ref_trace = seed(model.simulate)(key, *model_args)
    choices = {a: ref_trace.get_choices()[a] for a in sites}

    addrs = list(sites.keys())
    latent = set(latent_addresses)

    graph = nx.DiGraph()
    for a in addrs:
        graph.add_node(
            a,
            distribution=sites[a]["distribution"],
            shape=sites[a]["shape"],
            observed=a not in latent,
        )

    # -- edges via jaxpr provenance (dataflow reachability) -----------------
    # Pass each site's value as a separate argument so each maps to its own
    # jaxpr invar (model params are closed over -> constants, not sources), and
    # return per-site log-densities as separate outputs. Then propagate a
    # provenance set through the jaxpr: an edge A -> B exists iff site A's value
    # reaches site B's log-density. Unlike autodiff, this is pure dataflow and
    # so captures dependencies through *discrete* parents too.
    def per_site_logps(*choice_values):
        ch = {a: v for a, v in zip(addrs, choice_values)}
        handler = _run_with_handler(_PerSiteLogp(ch), source, *model_args)
        return tuple(handler.logps[a] for a in addrs)

    ordered = [choices[a] for a in addrs]
    leaf_counts = [len(jax.tree_util.tree_leaves(v)) for v in ordered]
    # invar -> address, in flattened-argument order
    invar_addr = [a for a, n in zip(addrs, leaf_counts) for _ in range(n)]

    jaxpr = jax.make_jaxpr(per_site_logps)(*ordered).jaxpr
    for A, B in _provenance_edges(jaxpr, invar_addr, addrs):
        graph.add_edge(A, B)

    return graph


def _provenance_edges(jaxpr, invar_addr, out_addr):
    """Reachability-based provenance over a jaxpr.

    Args:
        jaxpr: the staged jaxpr of the per-site log-density function.
        invar_addr: address tagging each ``jaxpr.invars`` entry (flattened order).
        out_addr: address for each ``jaxpr.outvars`` entry (the per-site logps).

    Yields ``(A, B)`` pairs meaning site ``B``'s log-density depends on ``A``.
    """
    from jax.extend.core import Literal

    deps: Dict[Any, frozenset] = {}
    for var, addr in zip(jaxpr.invars, invar_addr):
        deps[var] = frozenset({addr})

    def prov(v):
        if isinstance(v, Literal):
            return frozenset()
        return deps.get(v, frozenset())

    for eqn in jaxpr.eqns:
        srcs = frozenset().union(*(prov(v) for v in eqn.invars)) if eqn.invars \
            else frozenset()
        for ov in eqn.outvars:
            deps[ov] = srcs

    for outvar, b in zip(jaxpr.outvars, out_addr):
        for a in prov(outvar):
            if a != b:
                yield (a, b)


class GraphicalModelLearner(SviLearner):
    """SVI learner that also captures the model's static graph structure."""

    def __init__(self, data_shape, guide, lr, model, num_particles=1, rng=0):
        super().__init__(data_shape, guide, lr, model, num_particles, rng)
        self._graph = None
        self._guide_addresses = ()

    def setup_step(self, datamodule: DataModule):
        params = super().setup_step(datamodule)

        # Latent addresses = the guide's sample sites (run the guide once on a
        # representative input from the test batch).
        self._rng, gkey = jax.random.split(self._rng)
        for batch in datamodule.test_dataloader():
            x0 = jnp.reshape(jnp.asarray(batch[0])[0], (-1,)).astype(jnp.float32)
            break
        guide_trace = seed(self._guide.simulate)(gkey, x0, self._params["encoder"])
        self._guide_addresses = tuple(guide_trace.get_choices().keys())

        # Capture the model's static structure.
        self._rng, mkey = jax.random.split(self._rng)
        self._graph = capture_model_graph(
            self._model,
            (self._params["decoder"],),
            key=mkey,
            latent_addresses=self._guide_addresses,
        )
        return params

    @property
    def graph(self) -> nx.DiGraph:
        """The captured static-structure digraph (nodes + dependency edges)."""
        return self._graph

    @property
    def relations(self) -> Dict[str, Any]:
        """A summary of the captured structure for reasoning."""
        g = self._graph
        return {
            "nodes": dict(g.nodes(data=True)),
            "edges": list(g.edges()),
            "latent": [n for n, d in g.nodes(data=True) if not d["observed"]],
            "observed": [n for n, d in g.nodes(data=True) if d["observed"]],
            "topological_order": list(nx.topological_sort(g)),
        }

    def render_model(self, filename: str | None = None):
        """Render the structure with graphviz (requires the ``graphviz`` package)."""
        from graphviz import Digraph

        dot = Digraph()
        for node, data in self._graph.nodes(data=True):
            shape = "doublecircle" if data["observed"] else "circle"
            dot.node(node, f"{node}\n{data['distribution']}", shape=shape)
        for a, b in self._graph.edges():
            dot.edge(a, b)
        if filename is not None:
            from pathlib import Path

            path = Path(filename)
            dot.render(path.with_suffix(""), format=path.suffix[1:] or "pdf",
                       view=False, cleanup=True)
        return dot
