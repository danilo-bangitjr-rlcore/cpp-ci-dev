# core-rl

[Wiki](https://github.com/rlcoretech/core-rl/wiki)

This is the main repo containing code for our agents, environments, state constructors and interactions.

## Installation

We are using the [`uv` Python package and project manager](https://docs.astral.sh/uv/).

```bash
# create virtual environment
uv venv --python 3.12
source .venv/bin/activate

# install project requirements
uv sync

# install projects/ as editable packages
uv pip install -e projects/

# run linter
uv run ruff check

# run static type checker
uv run pyright

# run tests
uv run pytest -n auto test/small
uv run pytest -n auto test/medium
uv run pytest -n auto test/large
```

If using VSCode, ensure that the python interpreter is set to the virtual environment initialized in the above steps.
- `ctrl+shift+p`, "Python: Select Interpreter" > `./.venv/bin/python`.

## Style
This repo uses the following code style:
1. Classes: camel case. E.g. `GreedyActorCritic`
2. Python modules: lowercase with underscores. E.g. `greedy_actor_critic.py`
3. Python variables: lowercase with underscores. E.g. `agent = GreedyActorCritic()`
4. Config files: lowercase with underscores. E.g. `greedy_actor_critic.yaml`
5. String arguments in configs: lowercase with underscores. E.g. `agent: greedy_actor_critic`
6. Paths: please use `pathlib` instead of `os`
