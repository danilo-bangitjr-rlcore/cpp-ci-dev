# Monorepo

[Wiki](https://github.com/rlcoretech/core-rl/wiki) - [benchmarks](https://fuzzy-guacamole-5k1rj63.pages.github.io/dev/bench/) - [bsuite wireguard](http://10.123.10.55:3003/d/feg5m3ku2qt4wf/bsuite-drill-down) - [bsuite local](http://workstation:3003/d/feg5m3ku2qt4wf/bsuite-drill-down)

This is the main repo containing code for the agent, the surrounding microservices, the research codebase, and project specific code.

System architecture diagrams are contained in [docs](docs/diagrams/), including our [plan for MVP](docs/diagrams/mvp.md).

## Python

We are using the [`uv` Python package and project manager](https://docs.astral.sh/uv/) for python projects.
Each microservice / subrepo maintains its own dependencies and project structure. Each microservice, therefore,
should have its own virtual environment.

```bash
cd corerl/
uv sync

# lint and type check
uv run ruff check .
uv run pyright

# run tests
uv run pytest test/small
``
