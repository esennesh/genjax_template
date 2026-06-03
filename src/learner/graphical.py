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

    # Reference (concrete) choice values, used for the edge autodiff below.
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

    # -- edges via autodiff provenance over per-site log-densities ----------
    # Differentiate only through real-valued choices; discrete choices are held
    # fixed (autodiff cannot trace dependencies through them). We flatten with
    # an explicit, fixed ordering so jacobian columns map back to addresses
    # unambiguously.
    float_addrs = [
        a for a in addrs
        if jnp.issubdtype(jnp.result_type(choices[a]), jnp.floating)
    ]
    fixed = {a: choices[a] for a in addrs if a not in float_addrs}
    shapes = {a: jnp.asarray(choices[a]).shape for a in float_addrs}
    offsets, start = {}, 0
    for a in float_addrs:
        size = int(jnp.asarray(choices[a]).size)
        offsets[a] = (start, start + size)
        start += size

    def make_choices(flat_choices):
        ch = dict(fixed)
        for a in float_addrs:
            lo, hi = offsets[a]
            ch[a] = jnp.reshape(flat_choices[lo:hi], shapes[a])
        return ch

    def per_site_logps(flat_choices):
        handler = _run_with_handler(
            _PerSiteLogp(make_choices(flat_choices)), source, *model_args
        )
        return jnp.stack([handler.logps[a] for a in addrs])

    if start:  # there is at least one differentiable choice dimension
        flat0 = jnp.concatenate(
            [jnp.ravel(jnp.asarray(choices[a])) for a in float_addrs]
        )
        jac = jax.jacrev(per_site_logps)(flat0)  # (n_sites, n_float_dims)
        for j, b in enumerate(addrs):
            for a in float_addrs:
                if a == b:
                    continue
                lo, hi = offsets[a]
                if bool(jnp.any(jnp.abs(jac[j, lo:hi]) > 1e-9)):
                    graph.add_edge(a, b)

    return graph


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
