# GenJAX Template Project

Deep probabilistic programming in JAX with [GenJAX](https://github.com/Astera-org/genjax), made easi**er**.

A template for configurable probabilistic-programming projects: Hydra configs, a
PyTorch-Lightning-style learner abstraction, amortized variational inference, and
static-graph-structure introspection — all on top of GenJAX's generative-function
interface (GFI) and ADEV gradient estimators.

## Requirements

- Python >= 3.12
- [`uv`](https://docs.astral.sh/uv/) for environment management
- `genjax` (and therefore `jax >= 0.7.2, < 0.8` — GenJAX's PJAX internals are not
  yet compatible with `jax >= 0.9`)
- `optax`, `hydra-core`, `networkx`, `tensorboard`, `torch`/`torchvision` (data),
  plus the rest declared in `pyproject.toml`

By default `genjax` is referenced as a **local editable checkout** at `../genjax`
(see `[tool.uv.sources]` in `pyproject.toml`). To use a shared checkout instead,
switch that entry to the git source noted there.

## Setup

```bash
uv sync                         # create the environment
uv run python -m src.train      # train (Hydra entrypoint; see configs/train.yaml)
uv run pytest                   # run the tests
```

Hydra makes every component swappable from the command line, e.g.:

```bash
uv run python -m src.train learner=graphical model=vae guide=vae data=mnist
```

## Features

- **GenJAX models & guides** written as `@gen` generative functions
  (`src/model/model.py` — a VAE: `p(x, z)` and an amortized guide `q(z | x)`).
- **`.yaml` config files** (Hydra) for models, guides, learners, data, and
  trainers, with command-line overrides.
- **Abstract base classes for fast iteration:**
  - `ParamLearner` — the trainable-module interface (`train_step` / `valid_step`
    / `test_step`, `save`/`load`, `parameters`).
  - `DataModule` — data shuffling and train/val/test splitting.
  - `Trainer` — the training loop, checkpointing, and TensorBoard logging.
- **Learners (`src/learner/`):**
  - `SviLearner` — amortized stochastic variational inference. The ELBO is built
    from `guide.simulate` + `model.assess` wrapped in an ADEV `@expectation`, so
    `grad_estimate` yields reparameterized gradients; parameters are optimized
    with `optax`. Supports an IWAE bound (`num_particles > 1`).
  - `GraphicalModelLearner` — trains exactly like `SviLearner`, and additionally
    **captures the model's static graph structure** at setup (see below).
- **Checkpoint saving / resuming** and **TensorBoard** visualization.

## Static graph-structure capture

`GraphicalModelLearner` introspects the generative model and exposes a
`networkx` digraph (`.graph`, `.relations`, `.render_model()`):

- **Nodes** — one per sample-site address, annotated with its distribution
  family, value shape, and whether it is latent or observed (observed = model
  sites the guide does not sample).
- **Edges** — `A -> B` iff site `A`'s value influences site `B`'s distribution.
  Edges are recovered by **dataflow provenance** over the per-site log-density
  jaxpr (not autodiff), so dependencies routed through **discrete** parents are
  captured too — e.g. a `Categorical` index feeding a child's mean. This mirrors
  NumPyro's `eval_provenance` and is sound across `cond`/`scan`/`pjit`.

See `tests/test_structure.py` for continuous, discrete, and independence cases.

## Folder structure

```
genjax_template/
│
├── src/
│   ├── train.py            - training entrypoint (Hydra)
│   ├── test.py             - evaluation entrypoint
│   ├── data/               - DataModule + dataset implementations (e.g. MNIST)
│   ├── inference/          - GenJAX-native inference helpers
│   ├── learner/            - ParamLearner ABC, SviLearner, GraphicalModelLearner
│   ├── logger/             - logging / TensorBoard writer
│   ├── model/              - GenJAX @gen models and guides
│   ├── trainer/            - Trainer (loop, checkpointing, logging)
│   └── utils/              - small utilities
│
├── configs/                - Hydra configs
│   ├── data/   guide/   learner/   model/   trainer/
│   ├── experiment/   extras/   hydra/   paths/
│   └── train.yaml   eval.yaml
│
├── notebooks/vae.ipynb     - VAE example
├── tests/                  - pytest tests (run with `uv run pytest`)
├── pyproject.toml          - project + uv configuration
└── data/                   - default input-data directory
```

## License

This project is licensed under the MIT License. See `LICENSE` for more details.

## Acknowledgements

Ported to GenJAX from a NumPyro template, itself inspired by
[Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template)
by [Mahmoud Gemy](https://github.com/MrGemy95).
